#!/usr/bin/env python
import argparse, json
from pathlib import Path

KEY = ("suite","task","id","subset","split","language")

def score(row: dict) -> int:
    ca = row.get("culture_ann") or {}
    if not isinstance(ca, dict):
        return 0
    s = 0
    # prefer non-default / informative
    rk = ca.get("required_knowledge")
    if rk and rk != "none": s += 3
    if ca.get("cultural_sensitivity_label") == "CS": s += 2
    if ca.get("time_sensitive") == "Yes": s += 2
    if ca.get("subject_category") and ca.get("subject_category") != "Other": s += 1
    s += min(len(ca.get("references") or []), 5)
    s += min(len(ca.get("other_references") or []), 3)
    expl = ca.get("explanation") or ""
    s += 1 if len(expl.strip()) >= 20 else 0
    return s

def dedup_rows(rows):
    best = {}
    seen_counts = {}
    for r in rows:
        k = tuple(r.get(c) for c in KEY)
        seen_counts[k] = seen_counts.get(k, 0) + 1
        sc = score(r)
        if k not in best or sc > best[k][0]:
            best[k] = (sc, r)
    out = [best[k][1] for k in best]
    return out, seen_counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    inp = Path(args.in_path)
    rows = [json.loads(l) for l in inp.open(encoding="utf-8") if l.strip()]
    out, counts = dedup_rows(rows)

    # quick report
    max_rep = max(counts.values()) if counts else 1
    print(f"{inp.name}: rows={len(rows)} unique={len(out)} max_repetition={max_rep}")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()


