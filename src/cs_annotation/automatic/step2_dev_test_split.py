#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reads the agreed JSONL from step 1.
Creates dev/test: ~30% per subset (min. 5 per subset, if possible).
Writes IDs to JSON (reproducible).
Writes --out-dev and --out-test.
Writes small Excel/CSV stats (counts for required_knowledge_agreement + derived CS/CA).
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL from step1 (agreed only)")
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--out-test", required=True)
    ap.add_argument("--out-ids", required=True, help="Output JSON with dev/test IDs and split settings")
    ap.add_argument("--out-xlsx", required=True, help="Output Excel with split counts")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dev-frac", type=float, default=0.30)
    ap.add_argument("--min-per-subset", type=int, default=5)
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    rows = read_jsonl(Path(args.in_path))
    logging.info("Loaded %d agreed rows", len(rows))

    rng = random.Random(args.seed)

    # group by subset
    by_subset: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_subset.setdefault(str(r.get("subset") or "UNKNOWN"), []).append(r)

    dev_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for subset, items in sorted(by_subset.items()):
        n = len(items)
        if n == 0:
            continue

        n_dev = int(round(args.dev_frac * n))
        if n >= args.min_per_subset:
            n_dev = max(n_dev, args.min_per_subset)
        n_dev = min(n_dev, n)
        if n > 1:
            n_dev = min(n_dev, n - 1)  # keep at least 1 for test

        rng.shuffle(items)
        dev_subset = items[:n_dev]
        test_subset = items[n_dev:]

        dev_rows.extend(dev_subset)
        test_rows.extend(test_subset)
        logging.info("Subset=%s n=%d dev=%d test=%d", subset, n, len(dev_subset), len(test_subset))

    write_jsonl(Path(args.out_dev), dev_rows)
    write_jsonl(Path(args.out_test), test_rows)

    dev_ids = [r.get("id") for r in dev_rows]
    test_ids = [r.get("id") for r in test_rows]

    out_ids = {
        "seed": args.seed,
        "dev_frac": args.dev_frac,
        "min_per_subset": args.min_per_subset,
        "n_total": len(rows),
        "n_dev": len(dev_rows),
        "n_test": len(test_rows),
        "dev_ids": dev_ids,
        "test_ids": test_ids,
    }
    out_ids_path = Path(args.out_ids)
    out_ids_path.parent.mkdir(parents=True, exist_ok=True)
    out_ids_path.write_text(json.dumps(out_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Wrote split IDs: %s", out_ids_path)

    def split_counts(rr: List[Dict[str, Any]]) -> Tuple[Counter, Counter]:
        rk = Counter()
        csl = Counter()
        for r in rr:
            ref = r.get("culture_ann_ref") or {}
            rk.update([ref.get("required_knowledge_agreement")])
            csl.update([ref.get("cultural_sensitivity_label_agreement")])
        return rk, csl

    dev_rk, dev_csl = split_counts(dev_rows)
    test_rk, test_csl = split_counts(test_rows)

    stats_rows = []
    for split_name, rk_cnt, csl_cnt in [
        ("dev", dev_rk, dev_csl),
        ("test", test_rk, test_csl),
    ]:
        for k, v in sorted(rk_cnt.items(), key=lambda x: str(x[0])):
            stats_rows.append({"split": split_name, "field": "required_knowledge_agreement", "label": k, "count": int(v)})
        for k, v in sorted(csl_cnt.items(), key=lambda x: str(x[0])):
            stats_rows.append({"split": split_name, "field": "csl_agreement", "label": k, "count": int(v)})

    stats_df = pd.DataFrame(stats_rows)

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="SplitCounts", index=False)

    logging.info("Wrote dev/test JSONL: %s | %s", args.out_dev, args.out_test)
    logging.info("Wrote split stats: %s", out_xlsx)


if __name__ == "__main__":
    main()


