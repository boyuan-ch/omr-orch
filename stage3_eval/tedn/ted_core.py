"""
TEDn scoring, copied VERBATIM from the LEGATO reference implementation (scripts/compute_TEDn.py) (cut_xml, LIMIT,
compute_score) -- the exact functions the original run_eva_{1..8}.py runners imported.

The only difference from that file is what is NOT imported: compute_TEDn.py pulls in `datasets`,
`tqdm`, `argparse` and `concurrent.futures` at module level for its own __main__ block, none of
which compute_score uses. Dropping them keeps the package light on the new server; the scoring
code path is unchanged (TEDn_xml_xml(..., flavor='lmx') with LIMIT = 6000).
"""
import xml.etree.ElementTree as ET
from utils.TEDn_eval.evaluation.TEDn_xml_xml import TEDn_xml_xml

LIMIT = 6000
def cut_xml(xml, limit=LIMIT):
    root = ET.fromstring(xml)
    num_nodes = sum(1 for _ in root.iter())
    initial_num_nodes = num_nodes

    while num_nodes > limit:
        parts = root.findall('part')
        for part in parts:
            part.remove(part[-1])
        num_nodes = sum(1 for _ in root.iter())

    return ET.tostring(root, encoding='unicode'), initial_num_nodes

def compute_score(input_data):
    idx, pred_xml, gold_xml = input_data
    pred_xml, pred_init_num_nodes = cut_xml(pred_xml, LIMIT)
    gold_xml, gold_init_num_nodes = cut_xml(gold_xml, LIMIT)
    print(f"init num nodes: {pred_init_num_nodes}")
    if pred_init_num_nodes > LIMIT:
        print(f"Index {idx}: Pred XML initial nodes {pred_init_num_nodes}, after cut {sum(1 for _ in ET.fromstring(pred_xml).iter())}")
    if gold_init_num_nodes > LIMIT:
        print(f"Index {idx}: Gold XML initial nodes {gold_init_num_nodes}, after cut {sum(1 for _ in ET.fromstring(gold_xml).iter())}")
    score = TEDn_xml_xml(pred_xml, gold_xml, flavor='lmx')
    return idx, score.edit_cost / score.gold_cost * 100
