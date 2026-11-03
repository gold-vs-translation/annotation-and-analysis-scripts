#!/usr/bin/env python3
"""Camera-ready native GPT-5.2 meta-evaluation against 3-annotator majority gold.

Reports:
1) explicit human CS/CA majority vs explicit GPT cultural_sensitivity_label
2) explicit human CS/CA majority vs GPT CS/CA derived from required_knowledge
3) human CS/CA derived from majority required_knowledge vs GPT CS/CA derived from required_knowledge
4) 4-way required_knowledge on resolved majority items
5) conservative 4-way accuracy lower bound with 1-1-1 ties counted as errors
6) per-language tie counts/rates.

Prediction file requires language, sample_id, cultural_sensitivity_label, required_knowledge.
"""
import argparse, csv, json
from pathlib import Path
from collections import defaultdict

RK_CLASSES = ["cultural", "dialect", "none", "regional"]
CSCA_CLASSES = ["CA", "CS"]

def norm_csca(x):
    if x is None: return None
    s = str(x).strip().upper()
    return s if s in {"CA","CS"} else None

def norm_rk(x):
    if x is None: return None
    s = str(x).strip().lower()
    return s if s in RK_CLASSES else None

def derive_csca_from_rk(rk):
    rk = norm_rk(rk)
    if rk is None: return None
    return "CA" if rk == "none" else "CS"

def load_rows(path):
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        with p.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    delim = "\t" if p.suffix.lower() in {".tsv",".tab"} else ","
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delim))

def metric_table(y_true, y_pred, classes):
    cm = {t:{p:0 for p in classes} for t in classes}
    for t,p in zip(y_true,y_pred):
        cm[t][p] += 1
    per = []
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[t][c] for t in classes if t != c)
        fn = sum(cm[c][p] for p in classes if p != c)
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per.append((c, precision, recall, f1, tp, fp, fn))
    correct = sum(t == p for t,p in zip(y_true,y_pred))
    return correct/len(y_true), correct, per, cm

def write_metrics(out, name, pairs, classes):
    acc, correct, per, cm = metric_table([x[0] for x in pairs], [x[1] for x in pairs], classes)
    with (out/f"{name}_per_class.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["class","precision","recall","f1","tp","fp","fn"])
        w.writerows(per)
    with (out/f"{name}_confusion.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["human\\gpt"] + classes)
        for t in classes:
            w.writerow([t] + [cm[t][p] for p in classes])
    return acc, correct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds = load_rows(args.predictions)
    gold = load_rows(args.gold)

    # Deduplicate predictions by language + sample_id.
    predmap = {}
    conflicts = []
    tmp = defaultdict(lambda: {"explicit":set(), "rk":set()})
    for r in preds:
        key = (str(r.get("language","")).strip(), str(r.get("sample_id","")).strip())
        if not all(key): continue
        c = norm_csca(r.get("cultural_sensitivity_label"))
        rk = norm_rk(r.get("required_knowledge"))
        if c: tmp[key]["explicit"].add(c)
        if rk: tmp[key]["rk"].add(rk)
    for key, vals in tmp.items():
        if len(vals["explicit"]) != 1 or len(vals["rk"]) != 1:
            conflicts.append((key, vals))
        else:
            rk = next(iter(vals["rk"]))
            predmap[key] = {
                "explicit_csca": next(iter(vals["explicit"])),
                "rk": rk,
                "derived_csca": derive_csca_from_rk(rk),
            }
    if conflicts:
        raise SystemExit(f"Conflicting/missing prediction labels for {len(conflicts)} keys; first={conflicts[:3]}")

    joined = []
    for g in gold:
        key = (g["language"].strip(), g["sample_id"].strip())
        if key not in predmap:
            raise SystemExit(f"Missing GPT prediction for gold item {key}")
        joined.append((g, predmap[key]))

    explicit_explicit = []
    explicit_derived = []
    derived_derived = []
    rk_pairs = []
    tie_rows = []
    for g,p in joined:
        he = norm_csca(g.get("explicit_csca_majority_gold"))
        hd = norm_csca(g.get("derived_csca_from_rk_gold"))
        rg = norm_rk(g.get("required_knowledge_majority_gold"))
        if he:
            explicit_explicit.append((he, p["explicit_csca"]))
            explicit_derived.append((he, p["derived_csca"]))
        if hd:
            derived_derived.append((hd, p["derived_csca"]))
        if rg:
            rk_pairs.append((rg, p["rk"]))
        tie_rows.append((g["language"], g.get("required_knowledge_tie_type","")))

    results = []
    for name, pairs, classes, note in [
        ("csca_explicit_human_vs_explicit_gpt", explicit_explicit, CSCA_CLASSES,
         "Primary candidate: independently annotated human CS/CA majority vs independently predicted GPT CS/CA"),
        ("csca_explicit_human_vs_gpt_derived_rk", explicit_derived, CSCA_CLASSES,
         "Sensitivity: explicit human CS/CA majority vs GPT CS/CA derived from GPT required_knowledge"),
        ("csca_human_derived_rk_vs_gpt_derived_rk", derived_derived, CSCA_CLASSES,
         "Legacy-comparable construction: CS/CA derived from required_knowledge on both sides"),
        ("required_knowledge_resolved", rk_pairs, RK_CLASSES,
         "4-way required_knowledge; 1-1-1 human majority ties excluded"),
    ]:
        acc, correct = write_metrics(out, name, pairs, classes)
        results.append([name, len(pairs), correct, acc, note])

    # Conservative RK lower bound: unresolved ties count as errors.
    rk_correct = sum(t == p for t,p in rk_pairs)
    results.append([
        "required_knowledge_ties_as_errors_lower_bound",
        len(gold), rk_correct, rk_correct/len(gold),
        "Same correct count as resolved set; all unresolved 1-1-1 human ties counted as errors"
    ])

    with (out/"meta_eval_summary.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["analysis","n","correct","accuracy","note"])
        w.writerows(results)

    # Tie rates by language
    langs = sorted({x[0] for x in tie_rows})
    with (out/"required_knowledge_ties_by_language.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["language","n_total","n_complex_ties","tie_rate"])
        for lang in langs:
            rows = [t for l,t in tie_rows if l == lang]
            n_ties = sum(t not in {"", "none", "nan", "None"} for t in rows)
            w.writerow([lang, len(rows), n_ties, n_ties/len(rows)])

    for row in results:
        print(f"{row[0]}: {row[2]}/{row[1]} = {row[3]:.4f}")

if __name__ == "__main__":
    main()
