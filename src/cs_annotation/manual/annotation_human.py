#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argilla as rg
import pandas as pd
import pycountry

# -----------------------------
# Config
# -----------------------------
ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")
WORKSPACE = os.getenv("ARGILLA_WORKSPACE", "argilla")
BASE_DATASET_NAME = "cultural_annotation"

# -----------------------------
# Label sets (stored values must match schema enums)
# -----------------------------

LANGS = ["SV", "ET", "LT", "NB", "DA"]

REQUIRED_KNOWLEDGE = {
    "regional": "Yes, geographic",
    "cultural": "Yes, cultural",
    "dialect": "Yes, dialect or regional language vocabulary",
    "none": "None",
}

TIME_SENSITIVE = {"Yes": "Yes", "No": "None"}  # store Yes/No, display Yes/None
CS_LABEL = {"CS": "CS", "CA": "CA"}

SUBJECT_CATEGORY = {
    "STEM": "STEM",
    "Humanities": "Humanities",
    "Social Sciences": "Social Sciences",
    "Medical": "Medical",
    "Business": "Business",
    "Other": "Other",
}

CULTURE_OPTIONS = {
    "Western Culture": "Western Culture",
    "Eastern Asian Culture": "Eastern Asian Culture",
    "Middle Eastern Culture": "Middle Eastern Culture",
    "South Asian Culture": "South Asian Culture",
    "African Culture": "African Culture",
    "Latin American Culture": "Latin American Culture",
    "Other": "Other",
}

REGION_OPTIONS = {
    "North America": "North America",
    "South America": "South America",
    "Europe": "Europe",
    "Asia": "Asia",
    "Africa": "Africa",
    "Australia and Oceania": "Australia and Oceania",
    "Antarctica": "Antarctica",
}

SPAN_LABELS = {
    "Geographic": "Geographic",
    "Cultural": "Cultural",
    "Dialect": "Dialect",
}

_COMMON_NAME_OVERRIDES = {
    "BO": "Bolivia",
    "VE": "Venezuela",
    "TZ": "Tanzania",
    "IR": "Iran",
    "LA": "Laos",
    "MD": "Moldova",
    "RU": "Russia",
    "SY": "Syria",
    "VN": "Vietnam",
    "KR": "South Korea",
    "KP": "North Korea",
    "CD": "Democratic Republic of the Congo",
    "CG": "Republic of the Congo",
    "GB": "United Kingdom",
    "US": "United States",
    "TW": "Taiwan",
}

# -----------------------------
# Guidelines
# -----------------------------
GUIDELINES = """
# Cultural Sensitivity Annotation Guide

You annotate **one exam-style multiple-choice item** (question + possibly answer options) in language **{LANG}**.

The text may include an answer key marker like `=> (1)`.  
**Do NOT use the answer key to decide labels or spans. Ignore it.**

## Language requirements
- **All answers (all labels + the explanation) must be in English.**
- Only highlighted **span strings** are copied verbatim from the item text (so they may be in {LANG}).

---

# Overview: what you must provide

You will answer:
1) Required knowledge type (choose exactly ONE)  
2) Time sensitivity (Yes / None)  
3) Highlight trigger spans (only when applicable)  
4) If cultural: choose culture  
5) If geographic: choose region  
6) If geographic: optionally choose country  
Plus:
- Cultural sensitivity label (CS/CA)  
- Subject category (always required)  
- Short explanation (≤ 250 characters)

---

# Q1. Required knowledge (choose exactly ONE)

**Question:** Does answering correctly require **geographic**, **cultural**, or **dialect / regional vocabulary** knowledge?

Choose exactly one:
- **Yes, geographic** → requires place/region/country/jurisdiction-specific knowledge.
- **Yes, cultural** → requires culture-specific facts, norms, religion, canonical cultural anchors.
- **Yes, dialect or regional language vocabulary** → requires understanding language variants, slang, idioms in the item’s language.
- **None** → none of the above is necessary.

## “Necessary to answer” rule (VERY IMPORTANT)
Only select a knowledge type if it is **necessary to answer correctly**, not just mentioned in passing.

## Tie-break rule (rare, but required)
If more than one type truly seems necessary, still choose exactly one using:
**Dialect > Geographic > Cultural > None**

---

# Q2. Time-sensitive?

**Question:** Does the correct answer change depending on **when** the question is asked?

- Mark **Yes** if the item contains explicit time/trend/change language such as:
  “currently”, “today”, “in recent years”, “has increased”, “is evolving”, etc.
- Otherwise mark **None**.

Note: For time-sensitivity you can mark **Yes** based on explicit time/trend cues even if the fact might still be broadly true.

---

# Q3. Highlight trigger spans (only if Q1 is not “None”)

Highlight the **minimal parts of the item** that trigger the required knowledge.

## Span rules
- Highlight **minimal** spans (shortest text that justifies your decision).
- Spans must be **non-overlapping** and should be added **left → right**.
- If knowledge is **implicit** and there is no clear trigger phrase, it is OK to leave spans empty — but **do NOT choose “None”** just because you cannot find an explicit span.

## Span labels
Use exactly one label per span:
- **Geographic**: places/regions/countries, local jurisdictions, country-specific regimes, local units/currencies as cues.
- **Cultural**: cultural anchors (religion when relevant, canonical figures/terms, norms/traditions).
- **Dialect**: idioms/slang/region-specific vocabulary in the item language.

If a span could be both **Cultural** and **Geographic**, prefer **Geographic** when the span refers to a real place/country/region.

---

# How to decide Q1 (detailed)

## A) Cultural knowledge (Q1 = “Yes, cultural”)
Use this when answering requires culture-specific facts, norms, values, customs, religion, arts/literature/pop culture, or historically rooted traditions tied to a cultural group.

### Strong cultural anchors
- **Religion / religious law/ritual** when relevant to answering (e.g., Sharia, Halakha, canon law, Buddhist precepts).
- **Named canonical philosophers/scholars/texts/schools/traditions** (especially philosophy / political theory / psychology),
  e.g., Freud, Kant, Aristotle, Mill, Rawls, Bentham, Hobbes, Locke, Marx, Weber, Durkheim.
- **Canonical Western philosophy / logic / philosophy-of-mind terminology** counts as cultural anchoring even without a named person:
  e.g., consequentialism, virtue ethics, deontology, ad hominem, substance dualism, materialism.
  → In these cases select **Western Culture** in Q4.

### Important caveats
- “According to Author & Author (Year)” is usually **NOT** cultural (that’s domain-specific academic recall),
  unless it is clearly a widely recognized canonical anchor (e.g., Kant, Marx, Rawls).
- **Do NOT map religions mechanically to a region.**
  If religion is discussed generally without a clear region/country anchor, choose culture **Other** (not a region).

---

## B) Geographic / regional knowledge (Q1 = “Yes, geographic”)
Use this when answering requires knowledge tied to a specific **place/region/country/jurisdiction**, such as:
- Real places that matter for solving (countries/regions/US states).
- Country/region-specific laws, institutions, schemes (e.g., GDPR, Sarbanes–Oxley, EMAS).
- Local measurement/jurisdiction conventions that imply a region.

### Important cues (treat as geographic when central)
**Units / jurisdiction cues**
- Imperial / US customary units without explanation (inch, mile, feet, yard, °F, lb, oz, mph)
  → typically region = **North America**.
- **$ is ambiguous alone**: do NOT infer a country/region from “$” without additional anchors.
  Treat as geographic only when combined with clear anchors (US states, “United States”, mph/oz/inch/mile, “World Series”, etc.).
  If the text explicitly specifies USD/AUD/CAD (US$, A$, C$, USD), that can be a geographic cue.

**Named local regimes**
- Named regulations or institutions that are clearly tied to a country/region (GDPR, Sarbanes–Oxley, etc.) → geographic.

**Place in scenario**
- If a real place is part of the scenario setup and affects reasoning (travel, jurisdiction, local context), mark geographic.
- Do NOT over-label: do not mark geographic just because a country is mentioned in historical/political background if it is not needed to answer.

---

## C) Dialect / regional language vocabulary (Q1 = “Yes, dialect or regional language vocabulary”)
Use this when understanding/answering depends on **language variation** in the item language:
- Slang, idioms, dialect forms, region-specific vocabulary.
- Treat common idioms as Dialect (e.g., “Mom & Pop”) even if the underlying concept is general.

Note: Dialect here is about language variation, **not** geography metadata.

---

# Q4–Q6 (only answer when applicable)

## Q4. Culture (only if Q1 = cultural)
Choose exactly one:
- Western Culture / Eastern Asian Culture / Middle Eastern Culture / South Asian Culture / African Culture / Latin American Culture / Other

## Q5. Region (only if Q1 = geographic)
Choose exactly one:
- North America / South America / Europe / Asia / Africa / Australia and Oceania / Antarctica

## Q6. Country (only if Q1 = geographic; optional)
Choose the **primary country** if country-specific knowledge is required.
Format: “Country Name (ISO2)” (e.g., “Germany (DE)”, “United States (US)”).
If multiple countries matter, pick the primary one and mention others briefly in the explanation or as secondary spans.

---

# Cultural sensitivity label (CS vs CA)

This label is **not strictly determined** by Q1.

- Set **CA** ONLY if the item is clearly free of cultural/geographic/dialect cues:
  - no named places/countries/US states,
  - no canonical culture/religion/ideology anchors,
  - no dialect/idiom cues,
  - no local regime/unit cues.
- Otherwise set **CS**, even if you selected Q1 = None (borderline sensitivity).

---

# Subject category (always required)
Choose exactly one:
- **STEM**: math, physics, chemistry, biology (non-medical), engineering, computer science
- **Humanities**: literature, philosophy, history, arts, languages, religion
- **Social Sciences**: politics, law, economics (non-business), sociology, psychology, education, public policy
- **Medical**: medicine/clinical, anatomy/physiology (clinical), pharmacology, pathology, nursing
- **Business**: accounting, finance, management, marketing, business ethics, operations
- **Other**: mixed/unclear/general knowledge

Rule: choose what a person must know to answer correctly; if ambiguous, choose **Other**.

---

# Explanation (≤ 250 characters)
Write a concise English explanation:
- main reason for your Q1 decision and CS/CA label,
- if time-sensitive = Yes, mention the time/trend cue.

""".strip()

GUIDELINES_SHORT = """
# Quick Guide (Cheat Sheet)

**Ignore answer keys** like `=> (1)`.  
**All answers in English**; only highlighted spans are copied verbatim (may be in {LANG}).

## Q1 Required knowledge (choose exactly ONE)
- **Yes, geographic**: place/region/country/jurisdiction needed (laws like GDPR; US units mph/°F; US states).
- **Yes, cultural**: culture/religion/canonical anchors needed (Kant/Marx/Rawls; terms like deontology/ad hominem ⇒ Western Culture).
- **Yes, dialect**: idioms/slang/variant vocabulary needed (e.g., “Mom & Pop”).
- **None**: none needed.

**Necessary-to-answer rule:** only label if needed to answer, not just mentioned.  
**Tie-break:** Dialect > Geographic > Cultural > None.

## Q2 Time-sensitive
**Yes** only if explicit time/trend wording (“currently”, “in recent years”, “is evolving”…). Otherwise **None**.

## Q3 Spans (only if Q1 ≠ None)
Mark **minimal trigger spans**, **non-overlapping**, **left→right**.  
If implicit knowledge: spans may be empty (don’t pick None just because no span).  
Labels: Geographic / Cultural / Dialect.  
If both: prefer **Geographic** for real places.

## Q4–Q6 (only if applicable)
- Q4 Culture (if cultural): pick one (Western/E Asian/Middle Eastern/S Asian/African/Latin American/Other)
- Q5 Region (if geographic): pick one continent region
- Q6 Country (if geographic, optional): “Country (ISO2)”

## CS vs CA
**CA only if no C/G/D cues at all** (no places, no canonical anchors, no idioms, no local units/regimes).  
Otherwise **CS** (even if Q1=None).

## Subject category (required)
Pick one: STEM / Humanities / Social Sciences / Medical / Business / Other.

## Explanation (≤ 250 chars)
Short reason for Q1 + CS/CA; if time-sensitive=Yes mention the cue.
""".strip()


def build_country_options() -> Dict[str, str]:
    """
    Build ISO 3166-1 alpha-2 country labels as "Country Name (ISO2)".
    Keys == stored values, values == display strings.
    """
    opts: Dict[str, str] = {}
    for c in pycountry.countries:
        code = c.alpha_2
        name = getattr(c, "common_name", None) or getattr(c, "official_name", None) or c.name
        name = _COMMON_NAME_OVERRIDES.get(code, name)
        label = f"{name} ({code})"
        opts[label] = label

    # Kosovo is not official ISO-3166-1; XK is widely used in practice
    opts["Kosovo (XK)"] = "Kosovo (XK)"

    return dict(sorted(opts.items(), key=lambda kv: kv[0].lower()))


COUNTRY_OPTIONS = build_country_options()


def build_settings(*, lang: Optional[str] = None) -> rg.Settings:
    lang_str = lang or "the item's language"

    questions: List[rg.Question] = [
        rg.LabelQuestion(
            name="required_knowledge",
            title="1. Does answering the question correctly require geographic, cultural or dialect specific knowledge?",
            labels=REQUIRED_KNOWLEDGE,
            description=(
                "Choose exactly one.\n"
                "Only mark a type if it is NECESSARY to answer correctly (not just mentioned).\n"
                "Tie-break: dialect > regional > cultural > none."
            ),
            required=True,
        ),
        rg.LabelQuestion(
            name="time_sensitive",
            title="2. Does the correct answer to this question change depending on when it is asked?",
            labels=TIME_SENSITIVE,
            description="Yes if the item includes explicit time/trend cues and the correct answer may change over time.",
            required=True,
        ),
        rg.SpanQuestion(
            name="references",
            title="3. If the question requires specific knowledge, highlight the parts of the question that refer to geographic, dialect, or cultural specific knowledge.",
            description=(
                "Highlight MINIMAL trigger spans. Non-overlapping, left→right.\n"
                "- If Q1=Yes, geographic: use only Geographic\n"
                "- If Q1=Yes, cultural: use only Cultural\n"
                "- If Q1=Yes, dialect: use only Dialect\n"
                "- If Q1=None: leave empty\n"
                "If knowledge is implicit, references may be empty."
            ),
            field="item_text",
            labels=SPAN_LABELS,
            allow_overlapping=False,
            required=False,
            visible_labels=3,
        ),
        rg.LabelQuestion(
            name="culture",
            title="4. If answering the question requires culture specific knowledge, what culture?",
            labels=CULTURE_OPTIONS,
            visible_labels=7,
            required=False,
        ),
        rg.LabelQuestion(
            name="region",
            title="5. If answering the question requires geographic specific knowledge, what region?",
            labels=REGION_OPTIONS,
            visible_labels=7,
            required=False,
        ),
        rg.LabelQuestion(
            name="country",
            title="6. If the question requires country specific knowledge, what country?",
            labels=COUNTRY_OPTIONS,
            visible_labels=12,
            required=False,
        ),
        rg.LabelQuestion(
            name="cultural_sensitivity_label",
            title="Cultural sensitivity label",
            labels=CS_LABEL,
            description="Set CA ONLY if clearly free of cultural/geographic/dialect cues; otherwise CS.",
            required=True,
        ),
        rg.LabelQuestion(
            name="subject_category",
            title="Subject category",
            labels=SUBJECT_CATEGORY,
            required=True,
        ),
    ]

    questions.append(
        rg.TextQuestion(
            name="explanation",
            title="Explanation (<= 250 chars, English)",
            description="Concise reason for the decisions (mention time cue if time_sensitive=Yes).",
            required=True,
        )
    )

    return rg.Settings(
        guidelines=GUIDELINES.format(LANG=lang_str),
        fields=[
            rg.TextField(name="sample_overview", title="Sample info", use_markdown=True),
            rg.TextField(name="item_text", title="Item text", use_markdown=False),
        ],
        questions=questions,
        metadata=[
            rg.TermsMetadataProperty(name="sample_id", title="Sample ID"),
            rg.TermsMetadataProperty(name="suite", title="Suite"),
            rg.TermsMetadataProperty(name="task", title="Task"),
            rg.TermsMetadataProperty(name="subset", title="Subset"),
            rg.TermsMetadataProperty(name="language", title="Language"),
        ],
    )


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    df = pd.read_json(path, lines=True, compression="infer")
    return df.to_dict(orient="records")


def make_sample_overview(rec: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"**ID:** `{rec.get('id','')}`",
            f"**SUITE:** `{rec.get('suite','')}`",
            f"**TASK:** `{rec.get('task','')}`",
            f"**SUBSET:** `{rec.get('subset','')}`",
            f"**LANG:** `{rec.get('language','')}`",
        ]
    )


def build_records(path: str) -> List[rg.Record]:
    rows = load_jsonl(path)
    records: List[rg.Record] = []

    for r in rows:
        fields = {
            "sample_overview": make_sample_overview(r),
            "item_text": r.get("text") or "",
        }
        metadata = {
            "sample_id": str(r.get("id", "")),
            "suite": str(r.get("suite", "")),
            "task": str(r.get("task", "")),
            "subset": str(r.get("subset", "")),
            "language": str(r.get("language", "")),
        }
        records.append(rg.Record(fields=fields, metadata=metadata))

    return records


def create_dataset(*, dataset_name: str, jsonl_path: str, lang: str) -> None:
    client = rg.Argilla(api_url=ARGILLA_API_URL, api_key=ARGILLA_API_KEY)
    settings = build_settings(lang=lang)

    ds = rg.Dataset(name=dataset_name, workspace=WORKSPACE, settings=settings, client=client)
    ds.create()

    recs = build_records(jsonl_path)
    BATCH = 500
    for i in range(0, len(recs), BATCH):
        ds.records.log(recs[i : i + BATCH])
    print(f"Created {dataset_name} ({lang}) with {len(recs)} records from {jsonl_path}.")


# -------- Export helpers --------
def _pick_response_value(responses: Dict[str, Any], qname: str) -> Any:
    items = responses.get(qname)
    if items is None:
        return None
    if isinstance(items, dict) and "value" in items:
        return items.get("value")
    if isinstance(items, list) and items:
        for it in items:
            if isinstance(it, dict) and it.get("status") == "submitted":
                return it.get("value")
        last = items[-1]
        return last.get("value") if isinstance(last, dict) else last
    return None


def _spans_to_schema(spans: Any, text: str, *, enforce_label: Optional[str] = None) -> List[Dict[str, Any]]:
    if not spans:
        return []
    out: List[Dict[str, Any]] = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        label = s.get("label")
        if enforce_label and label != enforce_label:
            continue
        start = int(s.get("start", 0))
        end = int(s.get("end", 0))
        out.append({"start": start, "end": end, "span": text[start:end], "label": label})
    out.sort(key=lambda x: x["start"])
    return out


def export_dataset_to_jsonl(*, dataset_name: str, out_path: str, workspace: str = WORKSPACE) -> None:
    client = rg.Argilla(api_url=ARGILLA_API_URL, api_key=ARGILLA_API_KEY)
    dataset = client.datasets(name=dataset_name, workspace=workspace)
    if dataset is None:
        raise ValueError(f"Dataset not found: {dataset_name} (workspace={workspace})")

    exported = dataset.records.to_list(flatten=False)  # includes fields/metadata/responses :contentReference[oaicite:2]{index=2}
    rows_out: List[Dict[str, Any]] = []

    for r in exported:
        fields = r.get("fields") or {}
        meta = r.get("metadata") or {}
        responses = r.get("responses") or {}
        text = fields.get("item_text") or ""

        required_knowledge = _pick_response_value(responses, "required_knowledge") or "none"
        time_sensitive = _pick_response_value(responses, "time_sensitive") or "No"
        cs_label = _pick_response_value(responses, "cultural_sensitivity_label") or "CA"
        subject_category = _pick_response_value(responses, "subject_category") or "Other"

        culture = _pick_response_value(responses, "culture") or None
        region = _pick_response_value(responses, "region") or None
        country = _pick_response_value(responses, "country")
        country = (country or "").strip() or None

        # Enforce primary-only rules (schema expectations)
        if required_knowledge != "cultural":
            culture = None
        if required_knowledge != "regional":
            region = None
            country = None
        if required_knowledge in ("dialect", "none"):
            culture = None
            region = None
            country = None

        refs_raw = _pick_response_value(responses, "references") or []
        other_raw = _pick_response_value(responses, "other_references") or []

        enforce = None
        if required_knowledge == "regional":
            enforce = "Geographic"
        elif required_knowledge == "cultural":
            enforce = "Cultural"
        elif required_knowledge == "dialect":
            enforce = "Dialect"
        elif required_knowledge == "none":
            enforce = "__NONE__"

        references = [] if enforce == "__NONE__" else _spans_to_schema(refs_raw, text, enforce_label=enforce)
        other_references = _spans_to_schema(other_raw, text)

        explanation = (_pick_response_value(responses, "explanation") or "").strip()
        if len(explanation) > 250:
            explanation = explanation[:250]

        culture_ann = {
            "required_knowledge": required_knowledge,
            "time_sensitive": time_sensitive,
            "cultural_sensitivity_label": cs_label,
            "subject_category": subject_category,
            "culture": culture,
            "region": region,
            "country": country,
            "references": references,
            "other_references": other_references,
            "explanation": explanation,
        }

        rows_out.append(
            {
                "id": meta.get("sample_id"),
                "suite": meta.get("suite"),
                "task": meta.get("task"),
                "subset": meta.get("subset"),
                "language": meta.get("language"),
                "text": text,
                "culture_ann": culture_ann,
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Exported {len(rows_out)} rows to {out_path}")


def find_lang_file(base_dir: Path, lang: str) -> Path:
    # matches your pattern: cult_ann_split_SV.jsonl
    p = base_dir / f"cult_ann_split_{lang}.jsonl"
    if p.exists():
        return p
    # fallback: search
    matches = sorted(base_dir.glob(f"*{lang}*.jsonl*"))
    if not matches:
        raise FileNotFoundError(f"No JSONL found for lang={lang} in {base_dir}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, help="Directory with cult_ann_split_XX.jsonl files")
    parser.add_argument("--langs", default="SV,ET,LT,NB,DA", help="Comma-separated language codes")
    parser.add_argument("--prefix", default=BASE_DATASET_NAME, help="Dataset name prefix")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    langs = [x.strip().upper() for x in args.langs.split(",") if x.strip()]

    dataset_names = {
        "ET": f"{args.prefix}_estonian",
        "LT": f"{args.prefix}_lithuanian",
        "NB": f"{args.prefix}_norwegian_bokmal",
        "SV": f"{args.prefix}_swedish",
        "DA": f"{args.prefix}_danish",
    }

    for lang in langs:
        in_path = find_lang_file(base_dir, lang)
        ds_name = dataset_names.get(lang, f"{args.prefix}_{lang.lower()}")
        print(f"\n=== Creating dataset '{ds_name}' for {lang} from {in_path} ===")
        create_dataset(dataset_name=ds_name, jsonl_path=str(in_path), lang=lang)

    print("\nDone.")

if __name__ == "__main__":
    main()


