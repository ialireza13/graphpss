"""Fetch, for every entry in the store, the polymer entity that contains label chain 'A'.

The graphs were built from the first chain of each entry (label_asym_id 'A'), so that
entity's sequence cluster is what decides train/test leakage. Writes
data/clusters/chain_a_entity.json: {"1ABC": "1ABC_1", ...}.
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")
from pssnet.data import GraphStore  # noqa: E402

QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id
 polymer_entities{rcsb_id rcsb_polymer_entity_container_identifiers{asym_ids auth_asym_ids}}}}"""


def fetch(ids):
    body = json.dumps({"query": QUERY, "variables": {"ids": ids}}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request("https://data.rcsb.org/graphql", body,
                                         {"Content-Type": "application/json"})
            data = json.load(urllib.request.urlopen(req, timeout=120))["data"]["entries"]
            break
        except Exception:
            time.sleep(5 * (attempt + 1))
    else:
        return {}, {}
    chain_a, first = {}, {}
    for e in data or []:
        ents = e.get("polymer_entities") or []
        for ent in ents:
            ci = ent["rcsb_polymer_entity_container_identifiers"]
            if "A" in (ci.get("asym_ids") or []):
                chain_a[e["rcsb_id"]] = ent["rcsb_id"]
        if ents:
            first[e["rcsb_id"]] = ents[0]["rcsb_id"]
    return chain_a, first


s = GraphStore.load("data/graphpss.npz")
ids = [str(x) for x in s.ids]
batches = [ids[i:i + 500] for i in range(0, len(ids), 500)]
chain_a, first = {}, {}
with ThreadPoolExecutor(8) as ex:
    for a, f in ex.map(fetch, batches):
        chain_a.update(a)
        first.update(f)
# Fall back to the first polymer entity when no entity holds label chain 'A'.
mapping = {pid: chain_a.get(pid) or first.get(pid) for pid in ids if chain_a.get(pid) or first.get(pid)}
json.dump(mapping, open("data/clusters/chain_a_entity.json", "w"))
print(f"entries {len(ids)}, chain A entity {len(chain_a)}, fallback first entity "
      f"{len(mapping) - len(chain_a)}, missing {len(ids) - len(mapping)}")
