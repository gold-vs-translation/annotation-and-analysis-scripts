#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vergleicht Modell-Performances (EM) auf SQuAD-Varianten (Suiten), ohne subset/split zu verwenden.

Eingabe:  Verzeichnis mit JSONL/NDJSON-Dateien (auch .gz/.xz/.bz2/.zip), rekursiv.
Ausgabe:  CSVs mit Scores, Rankings, und Kendall-τb-Rankingvergleichen (suite-pairs).

Nutzung:
    python compare_rankings.py /pfad/zur/eingabe /pfad/zur/ausgabe
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

# Konfiguration: welche Suite-Paare vergleichen (Rankingvergleich)
SUITE_PAIRS = [
    ("GermanQuAD", "XQuAD", "DE"),
    ("SberQuAD",   "XQuAD", "RU"),
    ("FQuADv2",    "SQuAD-fr", "FR"),
    # weiterhin möglich (sprache automatisch je language-Gruppen):
    # ("SberQuAD", "XQuAD"),
]

# Hilfsfunktionen
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95%-Wilson-Konfidenzintervall für Anteilswerte."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (center - half, center + half)

def iter_jsonl_files(logdir: Path):
    """Finde rekursiv JSONL/NDJSON-Dateien (unkomprimiert & komprimiert)."""
    for p in logdir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if ".jsonl" in name or ".ndjson" in name:
            yield p

def read_jsonl_any(path: Path) -> pd.DataFrame:
    """Liest JSONL zeilenweise; Kompression wird automatisch erkannt."""
    return pd.read_json(path, lines=True, compression="infer")

def load_logs_from_dir(logdir: Path) -> pd.DataFrame:
    files = list(iter_jsonl_files(logdir))
    if not files:
        raise FileNotFoundError(
            f"Keine JSONL/NDJSON-Dateien unter {logdir} gefunden (inkl. *.jsonl.gz, *.jsonl.xz, ...)."
        )
    dfs = []
    for f in files:
        try:
            df = read_jsonl_any(f)
            df["__source_file"] = str(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Überspringe {f}: {e}")
    if not dfs:
        raise RuntimeError("Konnte keine Logs laden.")
    return pd.concat(dfs, ignore_index=True, sort=False)

def coerce_sample_score(series: pd.Series) -> pd.Series:
    """Robuste Umwandlung von sample_score nach float 0/1 (bool, int, str)."""
    s = series.copy()
    if s.dtype == bool:
        return s.astype(float)
    s = s.replace({
        True: 1.0, False: 0.0,
        "true": 1.0, "false": 0.0,
        "True": 1.0, "False": 0.0,
        "1": 1.0, "0": 0.0,
    })
    s = pd.to_numeric(s, errors="coerce")
    return s.fillna(0.0)

def _parse_suite_pair(pair: tuple) -> tuple[str, str, str | None]:
    """Erlaubt SUITE_PAIRS als (A,B) oder (A,B,LANG)."""
    if len(pair) == 2:
        return pair[0], pair[1], None
    if len(pair) == 3:
        return pair[0], pair[1], str(pair[2]) if pair[2] is not None else None
    raise ValueError(f"Ungültiger SUITE_PAIRS-Eintrag: {pair!r}")

def _normalize_language_col(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisiert language (z.B. 'de', 'DE', 'de-DE' -> 'DE', leer -> 'UNKNOWN')."""
    out = df.copy()
    if "language" not in out.columns:
        out["language"] = "UNKNOWN"
        return out
    lang = out["language"].fillna("UNKNOWN").astype(str)
    # heuristische Normalisierung: ISO-ähnliche Codes groß schreiben, regionalteile entfernen
    lang = lang.str.strip()
    lang = lang.str.replace(r"[_\-].*$", "", regex=True)  # 'de-DE' -> 'de'
    lang = lang.where(lang != "", "UNKNOWN").str.upper()
    out["language"] = lang
    return out


# Score + CI pro (suite, task, language, model)
def build_model_scores_em(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Sample-Logs zu EM + Wilson-CI je (suite, task, language, model_name).
    subset/split werden bewusst ignoriert (Mikro-Aggregation über alle Items).
    """
    need_cols = ["suite", "task", "language", "model_name", "sample_score"]
    for col in need_cols:
        if col not in df.columns:
            df[col] = "UNKNOWN" if col != "sample_score" else np.nan

    df = _normalize_language_col(df)  # <<--- NEU

    df["suite"] = df["suite"].astype(str)
    df["task"] = df["task"].astype(str)
    df["model_name"] = df["model_name"].astype(str)
    df["sample_score"] = coerce_sample_score(df["sample_score"])

    keys = ["suite", "task", "language", "model_name"]
    agg = df.groupby(keys, dropna=False)["sample_score"].agg(n="count", correct="sum").reset_index()
    agg["em"] = agg["correct"] / agg["n"]
    agg[["ci_lo", "ci_hi"]] = agg.apply(
        lambda r: pd.Series(wilson_ci(int(r["correct"]), int(r["n"]))), axis=1
    )
    return agg


# Ranking innerhalb einer Suite
def rank_within_suite(scores: pd.DataFrame) -> pd.DataFrame:
    """Fügt Rang je (suite, task, language) basierend auf EM (absteigend) hinzu."""
    out = scores.copy()
    out["rank_in_suite"] = out.groupby(["suite", "task", "language"])["em"] \
                              .rank(method="average", ascending=False)
    return out

def make_alignment_table(scores_ranked: pd.DataFrame,
                         suite_A: str, suite_B: str, language: str) -> pd.DataFrame:
    """Erzeuge Alignment-Tabelle für genau eine Sprache."""
    A = scores_ranked[(scores_ranked["suite"] == suite_A) & (scores_ranked["language"] == language)]
    B = scores_ranked[(scores_ranked["suite"] == suite_B) & (scores_ranked["language"] == language)]

    cols = ["task","language","model_name","n","correct","em","ci_lo","ci_hi","rank_in_suite"]
    A2 = A[cols].rename(columns={"n":"n_A","correct":"correct_A","em":"em_A",
                                 "ci_lo":"ci_lo_A","ci_hi":"ci_hi_A","rank_in_suite":"rank_A"})
    B2 = B[cols].rename(columns={"n":"n_B","correct":"correct_B","em":"em_B",
                                 "ci_lo":"ci_lo_B","ci_hi":"ci_hi_B","rank_in_suite":"rank_B"})
    align = pd.merge(A2, B2, how="inner", on=["task","language","model_name"])
    align.insert(0, "suite_A", suite_A)
    align.insert(1, "suite_B", suite_B)
    return align

# Kendall τb (SciPy), Rankingvergleich zwischen Suiten
def kendall_tables_for_pairs(scores_ranked: pd.DataFrame,
                             suite_pairs: list[tuple[str,str,str]]
                             ) -> tuple[pd.DataFrame, dict]:
    rows = []
    alignments = {}

    for suite_A, suite_B, lang in suite_pairs:
        align = make_alignment_table(scores_ranked, suite_A, suite_B, lang)
        alignments[(suite_A, suite_B, lang)] = align
        if align.empty:
            continue

        for task, sub in align.groupby("task", dropna=False):
            x = sub["rank_A"].to_numpy(float)
            y = sub["rank_B"].to_numpy(float)
            tau, p = kendalltau(x, y, variant="b", nan_policy="omit",
                                method="auto", alternative="greater")
            rows.append({
                "suite_A": suite_A,
                "suite_B": suite_B,
                "task": task,
                "language": lang,
                "n_models_common": int(len(sub)),
                "kendall_tau_b_rank": tau,
                "p_value": p,
                "models_used": ",".join(sub["model_name"].astype(str))
            })

    summary_df = pd.DataFrame(rows).sort_values(["suite_A","suite_B","task","language"])
    return summary_df, alignments

# =========================
# Main
# =========================

def main():
    if len(sys.argv) != 3:
        print("Nutzung: python compare_rankings.py <eingabe_verzeichnis> <ausgabe_verzeichnis>")
        sys.exit(1)

    in_dir = Path(sys.argv[1]).expanduser().resolve()
    out_dir = Path(sys.argv[2]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logs laden
    df_raw = load_logs_from_dir(in_dir)

    # Scores (EM + Wilson-CI) pro (suite, task, language, model)
    scores = build_model_scores_em(df_raw)
    scores = scores.sort_values(["suite", "task", "language", "em"], ascending=[True, True, True, False])

    # Ranking innerhalb Suite
    scores_ranked = rank_within_suite(scores)
    scores_ranked = scores_ranked.sort_values(
        ["suite", "task", "language", "rank_in_suite", "model_name"]
    )
    scores_ranked.to_csv(out_dir / "scores_ranked.csv", index=False)

    # Kendall-τb Rankingvergleiche für definierte Suite-Paare
    kendall_summary, alignments = kendall_tables_for_pairs(scores_ranked, SUITE_PAIRS)
    kendall_summary.to_csv(out_dir / "kendall_summary.csv", index=False)

    # Ausrichtungstabellen je Paar exportieren (EM/CI/Rank nebeneinander)
    for (suite_A, suite_B, lang), df_align in alignments.items():
        if df_align.empty:
            continue
        safe_A = suite_A.replace("/", "-")
        safe_B = suite_B.replace("/", "-")
        df_out = df_align.sort_values(["task","language","rank_A","rank_B","model_name"])
        df_out.to_csv(out_dir / f"alignment_{safe_A}_vs_{safe_B}_{lang}.csv", index=False)

    print(f"[OK] Exportiert nach: {out_dir}")
    print("  - scores_ranked.csv         (EM + CI + Rang je suite/task/language/model)")
    print("  - kendall_summary.csv       (τb der Ränge je (suite_A,suite_B,task,language))")
    print("  - alignment_*_vs_*.csv      (nebeneinander: EM/CI/Rank je Suite für gemeinsame Modelle)")

if __name__ == "__main__":
    main()


