"""Run a queue of experiments on one GPU in N parallel lanes.

Each non-comment line of the queue file is either
    <run_name> <seed> [key=value ...]    a training run
    ! <shell command>                    a command run in order (e.g. an evaluation)
Finished runs (metrics.json present) and finished commands are skipped, so the queue
can be restarted.
"""
import argparse
import os
import subprocess
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("queue")
ap.add_argument("--lanes", type=int, default=2)
ap.add_argument("--config", default="configs/base.yaml")
args = ap.parse_args()

def read_queue():
    # Re-read on every pop so the queue file can be edited while it runs.
    jobs = []
    with open(args.queue) as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if line.startswith("!"):
                # Commands are keyed by line number: the same command may appear several times.
                jobs.append(("!", f"{lineno}:{line[1:].strip()}", []))
            elif line:
                name, seed, *overrides = line.split()
                jobs.append((name, int(seed), overrides))
    return jobs


def is_done(job):
    if job[0] == "!":
        return os.path.exists(DONE_CMDS) and job[1] in open(DONE_CMDS).read().splitlines()
    return os.path.exists(f"runs/{job[0]}_s{job[1]}/metrics.json")


DONE_CMDS = "logs/queue_done_cmds.txt"


started = set()
lock = threading.Lock()
env = {**os.environ, "PYTHONPATH": "src"}
os.makedirs("logs", exist_ok=True)


def lane(lane_id):
    while True:
        with lock:
            todo = [j for j in read_queue() if (j[0], j[1]) not in started and not is_done(j)]
            if not todo:
                return
            name, seed, overrides = todo[0]
            started.add((name, seed))
        if name == "!":
            with open("logs/queue_cmds.log", "a") as log:
                rc = subprocess.call(seed.split(":", 1)[1], shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
            with lock, open(DONE_CMDS, "a") as d, open("logs/queue.log", "a") as q:
                if rc == 0:
                    d.write(seed + "\n")
                q.write(f"{time.strftime('%H:%M:%S')} lane{lane_id} CMD rc={rc}: {seed}\n")
            continue
        out = f"runs/{name}_s{seed}"
        cmd = ["python", "-m", "pssnet.train", "--config", args.config, "--seed", str(seed),
               "--out", out, "--set", f"name={name}", *overrides]
        t0 = time.time()
        with open(f"logs/{name}_s{seed}.log", "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        with lock, open("logs/queue.log", "a") as q:
            q.write(f"{time.strftime('%H:%M:%S')} lane{lane_id} {name}_s{seed} rc={rc} "
                    f"{(time.time() - t0) / 3600:.2f}h\n")


threads = [threading.Thread(target=lane, args=(i,)) for i in range(args.lanes)]
for t in threads:
    t.start()
for t in threads:
    t.join()
with open("logs/queue.log", "a") as q:
    q.write(f"{time.strftime('%H:%M:%S')} QUEUE DONE\n")
