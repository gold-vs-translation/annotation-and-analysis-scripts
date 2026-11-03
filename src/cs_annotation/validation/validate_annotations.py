#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ValidationError

SYSTEM_PROMPT_TEMPLATE = """You are a strict QA reviewer for human annotations of cultural sensitivity in short exam-style items.

You will be given ONE annotation instance (one item + one user_id), with:
- metadata: sample_id, suite, task, subset, language
- item_text
- annotation fields (may be missing): required_knowledge, cultural_sensitivity_label, time_sensitive, subject_category, culture, region, country, references, explanation

Validate the annotation against the rules below, flag issues, and propose minimal corrections.

Output requirements:
- Output ONLY a JSON object that matches the provided JSON schema exactly.
- Do NOT output CSV, markdown, or any extra text.

Your job:
1) Validate each annotation instance against the annotation schema and guideline rules.
2) Detect missing fields, contradictory labels, invalid formats, and suspicious/unusual entries.
3) Propose concrete corrections and short improvement suggestions.
4) Output ONLY a CSV table (no markdown, no explanations outside CSV).

General rules and schema expectations (core):
A) required_knowledge must be exactly one of: none | cultural | regional | dialect
B) cultural_sensitivity_label must be:
   - CA if required_knowledge == none
   - CS otherwise
C) Primary-only metadata constraints:
   - If required_knowledge == cultural:
       culture SHOULD be set (one of: Western Culture, Eastern Asian Culture, Middle Eastern Culture, South Asian Culture, African Culture, Latin American Culture, Other)
       region MUST be empty/null
       country MUST be empty/null
   - If required_knowledge == regional:
       region MAY be set (North America, South America, Europe, Asia, Africa, Australia and Oceania, Antarctica)
       country MAY be set (format "Country Name (ISO2)")
       culture MUST be empty/null
   - If required_knowledge == dialect:
       culture MUST be empty/null
       region MUST be empty/null
       country MUST be empty/null
   - If required_knowledge == none:
       culture MUST be empty/null
       region MUST be empty/null
       country MUST be empty/null
D) time_sensitive must be exactly Yes | No
   - Mark Yes ONLY if the ITEM TEXT contains explicit time/trend/change framing (e.g., “currently”, “today”, “now”, “in recent years”, “has increased”, “is evolving”, “as of”, “until”, “since”, “this year”, “last year”, “next year”, or a phrasing that clearly implies a changing situation).
   - Do NOT mark Yes just because a fact could change in principle; require explicit time/trend wording in the text.
E) subject_category must be exactly one of:
   STEM | Humanities | Social Sciences | Medical | Business | Other
F) references format:
   - references SHOULD be a list of span objects.
   - Each span object SHOULD include: start (int), end (int), label (Geographic|Cultural|Dialect), and ideally span (string).
   - If span is missing, it is acceptable; you should reconstruct span = TEXT[start:end] and flag it as a minor issue (implementation/schema mismatch), not an annotator mistake.
   - start must be >=0, end > start, end <= len(TEXT)
   - Spans should be non-overlapping and sorted by start (flag if not).
   - If required_knowledge == regional: all references labels should be Geographic
     If required_knowledge == cultural: all references labels should be Cultural
     If required_knowledge == dialect: all references labels should be Dialect
     If required_knowledge == none: references should be empty []
   - If knowledge is implicit and no clear trigger span exists, references may be [] even when required_knowledge != none; flag as “missing_span_optional” (WARN), and suggest a minimal trigger span if one exists.

Guideline interpretation (label-choice sanity checks):
- Use regional for country/region-specific institutions, legal regimes, government bodies, local policies, local history facts tied to a place.
- Use cultural for culture-/religion-/tradition-/arts-/literature-/canon-knowledge that is tied to a cultural group.
- Use dialect for idioms, slang, dialectal words/phrases necessary to answer.
- Use none for general domain knowledge not tied to a specific culture/region/dialect.

Output CSV requirements:
- Output exactly one header row + one row per annotation instance.
- If an item contains multiple annotators (multiple user_id entries), output one row per (item_id, user_id).
- Never output markdown or additional text.

CSV columns (exact names and order):
item_id,user_id,sample_id,suite,task,subset,language,
annot_required_knowledge,annot_csl,annot_time_sensitive,annot_subject_category,annot_culture,annot_region,annot_country,
missing_fields,contradictions,format_issues,span_issues,content_sanity_flags,
needs_fix,severity,
suggest_required_knowledge,suggest_csl,suggest_time_sensitive,suggest_subject_category,suggest_culture,suggest_region,suggest_country,
suggest_span_fix,comment

Column conventions:
- missing_fields / contradictions / format_issues / span_issues / content_sanity_flags:
  use pipe-separated short codes (or empty).
- needs_fix: YES/NO
- severity: OK / WARN / ERROR
- suggest_span_fix: either empty or a compact reconstruction like:
  "reconstruct_span: start-end='...'; label=Geographic" or "add_span_suggestion: '...' [start:end] label=..."
- comment: 1–2 concise sentences, focusing on why it is flagged and what to do.
- If you propose changes, keep them minimal and guideline-based.

Issue code vocabulary (use these where relevant):
Missing fields:
- missing_required_knowledge
- missing_csl
- missing_time_sensitive
- missing_subject_category
- missing_references
Contradictions:
- csl_vs_required_knowledge
- cultural_has_region_or_country
- regional_has_culture
- dialect_has_geo_or_culture
- none_has_any_geo_or_culture
Format issues:
- invalid_enum
- invalid_country_format
- invalid_region_enum
- explanation_too_long
Span issues:
- span_missing_text
- span_out_of_bounds
- span_not_sorted
- span_overlapping
- span_label_mismatch
Content sanity flags:
- likely_regional_not_cultural
- likely_none_not_cultural
- likely_none_not_regional
- time_sensitive_without_time_cues
- time_cues_but_time_sensitive_no
- weak_explanation

How to decide “needs_fix”:
- YES for ERROR, or for WARN when it affects correctness (e.g., csl mismatch, wrong required_knowledge type).
- NO for minor WARNs (e.g., missing span text but reconstructable) if labels are otherwise consistent.

Important:
- Do not “grade” the annotator; just flag issues and suggest corrections.
- Use the item_text to justify your sanity checks.""".strip()


# ----------------------------
# Structured output model (diagnostics only)
# ----------------------------

NeedsFix = Literal["YES", "NO"]
Severity = Literal["OK", "WARN", "ERROR"]


class DiagnosticsOutput(BaseModel):
    missing_fields: str = ""
    contradictions: str = ""
    format_issues: str = ""
    span_issues: str = ""
    content_sanity_flags: str = ""
    needs_fix: NeedsFix
    severity: Severity
    suggest_required_knowledge: str = ""
    suggest_csl: str = ""
    suggest_time_sensitive: str = ""
    suggest_subject_category: str = ""
    suggest_culture: str = ""
    suggest_region: str = ""
    suggest_country: str = ""
    suggest_span_fix: str = ""
    comment: str = ""


def diagnostics_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "missing_fields": {"type": "string"},
            "contradictions": {"type": "string"},
            "format_issues": {"type": "string"},
            "span_issues": {"type": "string"},
            "content_sanity_flags": {"type": "string"},
            "needs_fix": {"type": "string", "enum": ["YES", "NO"]},
            "severity": {"type": "string", "enum": ["OK", "WARN", "ERROR"]},
            "suggest_required_knowledge": {"type": "string"},
            "suggest_csl": {"type": "string"},
            "suggest_time_sensitive": {"type": "string"},
            "suggest_subject_category": {"type": "string"},
            "suggest_culture": {"type": "string"},
            "suggest_region": {"type": "string"},
            "suggest_country": {"type": "string"},
            "suggest_span_fix": {"type": "string"},
            "comment": {"type": "string"},
        },
        "required": [
            "missing_fields",
            "contradictions",
            "format_issues",
            "span_issues",
            "content_sanity_flags",
            "needs_fix",
            "severity",
            "suggest_required_knowledge",
            "suggest_csl",
            "suggest_time_sensitive",
            "suggest_subject_category",
            "suggest_culture",
            "suggest_region",
            "suggest_country",
            "suggest_span_fix",
            "comment",
        ],
    }


# ----------------------------
# Cache (SQLite)
# ----------------------------

OUTPUT_SCHEMA_VERSION = "cs-qa-v1"


class SQLiteCache:
    def __init__(self, path: Path):
        if path.exists() and path.is_dir():
            path = path / "cs_validation_cache.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cs_validation_cache (
              fingerprint TEXT PRIMARY KEY,
              result_json TEXT NOT NULL
            )
            """)
        self.conn.commit()

    def get(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT result_json FROM cs_validation_cache WHERE fingerprint = ?",
            (fingerprint,),
        )
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def put(self, fingerprint: str, result: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cs_validation_cache (fingerprint, result_json) VALUES (?, ?)",
            (fingerprint, json.dumps(result, ensure_ascii=False)),
        )
        self.conn.commit()


def fingerprint_instance(
    inst: Dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    prompt_cache_key: str,
) -> str:
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "system_prompt": SYSTEM_PROMPT_TEMPLATE,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_cache_key": prompt_cache_key,
        "instance": inst,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ----------------------------
# JSONL I/O
# ----------------------------

JSONObj = Union[Dict[str, Any], List[Any]]


def iter_jsonl_objects(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: JSONObj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON parse error in {path} at line {i}: {e}")

            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for el in obj:
                    if isinstance(el, dict):
                        yield el


# ----------------------------
# Argilla -> instances
# ----------------------------

EXPECTED_FIELDS = [
    "required_knowledge",
    "cultural_sensitivity_label",
    "time_sensitive",
    "subject_category",
    "culture",
    "region",
    "country",
    "references",
    "explanation",
]


def _collect_user_ids(item: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    responses = item.get("responses", {})
    if not isinstance(responses, dict):
        return out
    for entries in responses.values():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if isinstance(e, dict):
                uid = e.get("user_id")
                if isinstance(uid, str) and uid:
                    out.add(uid)
    return out


def _filter_responses_to_user(item: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    responses = item.get("responses", {})
    filtered: Dict[str, Any] = {}
    if not isinstance(responses, dict):
        return filtered
    for k, entries in responses.items():
        if not isinstance(entries, list):
            continue
        kept = [e for e in entries if isinstance(e, dict) and e.get("user_id") == user_id]
        if kept:
            filtered[k] = kept
    return filtered


def _get_first_value(filtered_responses: Dict[str, Any], field: str) -> Any:
    entries = filtered_responses.get(field)
    if isinstance(entries, list) and entries:
        e0 = entries[0]
        if isinstance(e0, dict) and "value" in e0:
            return e0["value"]
    return ""


def build_instances_from_item(item: Dict[str, Any], only_user_id: str = "") -> List[Dict[str, Any]]:
    item_id = str(item.get("id") or "")
    fields = item.get("fields", {}) if isinstance(item.get("fields"), dict) else {}
    metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
    text = str(fields.get("item_text") or "")

    user_ids = sorted(_collect_user_ids(item))
    if only_user_id:
        user_ids = [u for u in user_ids if u == only_user_id]

    instances: List[Dict[str, Any]] = []
    for uid in user_ids:
        filtered = _filter_responses_to_user(item, uid)

        inst = {
            "item_id": item_id,
            "user_id": uid,
            "metadata": {
                "sample_id": str(metadata.get("sample_id") or ""),
                "suite": str(metadata.get("suite") or ""),
                "task": str(metadata.get("task") or ""),
                "subset": str(metadata.get("subset") or ""),
                "language": str(metadata.get("language") or ""),
            },
            "item_text": text,
            "annotation": {
                "required_knowledge": _get_first_value(filtered, "required_knowledge"),
                "cultural_sensitivity_label": _get_first_value(filtered, "cultural_sensitivity_label"),
                "time_sensitive": _get_first_value(filtered, "time_sensitive"),
                "subject_category": _get_first_value(filtered, "subject_category"),
                "culture": _get_first_value(filtered, "culture"),
                "region": _get_first_value(filtered, "region"),
                "country": _get_first_value(filtered, "country"),
                "references": _get_first_value(filtered, "references"),
                "explanation": _get_first_value(filtered, "explanation"),
            },
        }
        instances.append(inst)

    return instances


def _try_parse_json_from_text(t: str) -> Optional[Dict[str, Any]]:
    t = t.strip()
    # direct
    if t.startswith("{"):
        try:
            obj = json.loads(t)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    # embedded JSON
    i = t.find("{")
    j = t.rfind("}")
    if 0 <= i < j:
        try:
            obj = json.loads(t[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


# ----------------------------
# Responses parsing (Structured Outputs)
# ----------------------------


def extract_output_object(resp_body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 0) refusal detection (for logging elsewhere; could also return None)
    # 1) output_json blocks
    for item in resp_body.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_json":
                v = c.get("json") or c.get("value") or c.get("data")
                if isinstance(v, dict):
                    return v
                if isinstance(v, str):
                    obj = _try_parse_json_from_text(v)
                    if obj:
                        return obj

    # 2) output_text blocks
    for item in resp_body.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                obj = _try_parse_json_from_text(c["text"])
                if obj:
                    return obj
            if c.get("type") == "refusal" and isinstance(c.get("refusal"), str):
                # caller should log this; we return None here
                return None

    # 3) output_parsed
    if isinstance(resp_body.get("output_parsed"), dict):
        return resp_body["output_parsed"]

    # 4) output_text convenience
    ot = resp_body.get("output_text")
    if isinstance(ot, str):
        obj = _try_parse_json_from_text(ot)
        if obj:
            return obj

    return None


def extract_output_object_old(resp_body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Prefer Structured Outputs:
      output[].type == "message"
        content[].type == "output_json" -> c["json"] (dict) or stringified JSON
    Fallbacks:
      - content[].type == "output_text" -> parse as JSON if it looks like JSON
      - resp_body["output_parsed"] (if present)
      - resp_body["output_text"] (SDK convenience)
    """
    # 1) Best: output_json blocks
    for item in resp_body.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_json":
                for k in ("json", "value", "data"):
                    v = c.get(k)
                    if isinstance(v, dict):
                        return v
                    if isinstance(v, str) and v.strip().startswith("{"):
                        try:
                            obj = json.loads(v)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            return None

    # 2) Parse output_text blocks (still inside message content)
    for item in resp_body.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                t = c["text"].strip()
                if t.startswith("{"):
                    try:
                        obj = json.loads(t)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        return None

    # 3) Some SDKs provide parsed output directly
    if isinstance(resp_body.get("output_parsed"), dict):
        return resp_body["output_parsed"]

    # 4) Last resort: aggregated output_text
    ot = resp_body.get("output_text")
    if isinstance(ot, str):
        t = ot.strip()
        if t.startswith("{"):
            try:
                obj = json.loads(t)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None

    return None


# ----------------------------
# Request building
# ----------------------------


def build_user_content(inst: Dict[str, Any]) -> str:
    md = inst.get("metadata", {})
    ann = inst.get("annotation", {})
    parts = [
        f"item_id: {inst.get('item_id','')}",
        f"user_id: {inst.get('user_id','')}",
        f"sample_id: {md.get('sample_id','')}",
        f"suite: {md.get('suite','')}",
        f"task: {md.get('task','')}",
        f"subset: {md.get('subset','')}",
        f"language: {md.get('language','')}",
        "",
        "ITEM_TEXT:",
        str(inst.get("item_text") or ""),
        "",
        "ANNOTATION:",
        f"required_knowledge: {ann.get('required_knowledge','')}",
        f"cultural_sensitivity_label: {ann.get('cultural_sensitivity_label','')}",
        f"time_sensitive: {ann.get('time_sensitive','')}",
        f"subject_category: {ann.get('subject_category','')}",
        f"culture: {ann.get('culture','')}",
        f"region: {ann.get('region','')}",
        f"country: {ann.get('country','')}",
        "references:",
        json.dumps(ann.get("references", ""), ensure_ascii=False),
        "explanation:",
        str(ann.get("explanation", "")),
    ]
    return "\n".join(parts).strip()


def build_responses_body(
    *,
    inst: Dict[str, Any],
    model: str,
    max_output_tokens: int,
    reasoning_effort: str,
    prompt_cache_key: str,
    prompt_cache_retention: str,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": build_user_content(inst)},
        ],
        "store": False,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "cs_validation_diagnostics",
                "strict": True,
                "schema": diagnostics_json_schema(),
            }
        },
    }
    if prompt_cache_key:
        body["prompt_cache_key"] = prompt_cache_key
    if prompt_cache_retention:
        body["prompt_cache_retention"] = prompt_cache_retention
    return body


def build_batch_request_line(custom_id: str, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
    return {"custom_id": custom_id, "method": "POST", "url": endpoint, "body": body}


def make_custom_id(idx: int, inst: Dict[str, Any]) -> str:
    raw = f"{idx}__{inst.get('item_id','')}__{inst.get('user_id','')}"
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return raw[:120]


# ----------------------------
# Batch execution + download
# ----------------------------


def run_batch(
    client: OpenAI,
    batch_input_path: Path,
    *,
    batch_endpoint: str,
    completion_window: str,
    poll_seconds: int,
) -> str:
    up = client.files.create(file=batch_input_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint=batch_endpoint,
        completion_window=completion_window,
    )
    logging.info("Batch created: %s", batch.id)

    while True:
        b = client.batches.retrieve(batch.id)
        logging.info("Batch status: %s", b.status)
        if b.status in ("completed", "failed", "cancelled", "expired"):
            if b.status != "completed":
                raise RuntimeError(f"Batch ended with status={b.status}")
            return b.output_file_id
        time.sleep(poll_seconds)


def download_file_text(client: OpenAI, file_id: str) -> str:
    content = client.files.content(file_id)
    data = content.read() if hasattr(content, "read") else content
    return data.decode("utf-8")


def write_batch_input_jsonl(path: Path, request_lines: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in request_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ----------------------------
# CSV output
# ----------------------------

CSV_HEADER = [
    "item_id",
    "user_id",
    "sample_id",
    "suite",
    "task",
    "subset",
    "language",
    "annot_required_knowledge",
    "annot_csl",
    "annot_time_sensitive",
    "annot_subject_category",
    "annot_culture",
    "annot_region",
    "annot_country",
    "missing_fields",
    "contradictions",
    "format_issues",
    "span_issues",
    "content_sanity_flags",
    "needs_fix",
    "severity",
    "suggest_required_knowledge",
    "suggest_csl",
    "suggest_time_sensitive",
    "suggest_subject_category",
    "suggest_culture",
    "suggest_region",
    "suggest_country",
    "suggest_span_fix",
    "comment",
]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_HEADER})


def discover_jsonl_files(in_path: Path) -> List[Path]:
    if in_path.is_file():
        return [in_path]
    if in_path.is_dir():
        return sorted([p for p in in_path.rglob("*.jsonl") if p.is_file()])
    raise SystemExit(f"--in path not found: {in_path}")


def maybe_dump(dump_dir: Optional[Path], name: str, payload: Any) -> None:
    if not dump_dir:
        return
    try:
        dump_dir.mkdir(parents=True, exist_ok=True)
        out = dump_dir / name
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning("Failed to dump %s: %s", name, e)


def summarize_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    status = raw.get("status")
    incomplete_reason = (raw.get("incomplete_details") or {}).get("reason")
    out_types = [o.get("type") for o in (raw.get("output") or [])]

    content_types = []
    refusal_text = None
    for o in raw.get("output") or []:
        if o.get("type") != "message":
            continue
        for c in o.get("content") or []:
            content_types.append(c.get("type"))
            if c.get("type") == "refusal":
                refusal_text = c.get("refusal")

    preview = raw.get("output_text")
    if isinstance(preview, str):
        preview = preview[:300]

    return {
        "status": status,
        "incomplete_reason": incomplete_reason,
        "output_item_types": out_types,
        "message_content_types": content_types,
        "refusal_preview": (refusal_text[:300] if isinstance(refusal_text, str) else None),
        "output_text_preview": preview,
    }


# ----------------------------
# Main
# ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input Argilla JSONL (per user or mixed)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV")
    ap.add_argument("--only-user-id", default="", help="If set, validate only this user_id even if input is mixed")

    ap.add_argument("--mode", choices=["batch", "sync"], default="batch", help="batch (prod) or sync (debug)")
    ap.add_argument("--service-tier", choices=["default", "auto", "priority"], default="default", help="Only used in --mode sync")

    ap.add_argument("--model", default="gpt-5.2", help="Model name")
    ap.add_argument("--max-output-tokens", type=int, default=2000)
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"], default="high")

    ap.add_argument("--batch-endpoint", default="/v1/responses")
    ap.add_argument("--completion-window", default="24h")
    ap.add_argument("--poll-seconds", type=int, default=10)

    ap.add_argument("--prompt-cache-key", default="cs-qa-v1")
    ap.add_argument("--prompt-cache-retention", default="24h", help='Use "" to disable if unsupported')
    ap.add_argument("--cache-path", default="cs_validation_cache.sqlite", help="SQLite cache file OR directory")

    ap.add_argument("--dump-responses-dir", default="", help="Dump raw responses on parse failures (sync only).")
    ap.add_argument("--dump-all", action="store_true", help="Dump ALL raw responses (sync only).")

    ap.add_argument("--log", default="INFO")
    ap.add_argument("--dry-run", action="store_true", help="Build batch input but do not call API")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    cache = SQLiteCache(Path(args.cache_path))

    jsonl_files = discover_jsonl_files(in_path)

    dump_dir = Path(args.dump_responses_dir) if args.dump_responses_dir.strip() else None

    # Build instances
    instances = []
    inst_source = []

    for fpath in jsonl_files:
        for item in iter_jsonl_objects(fpath):
            insts = build_instances_from_item(item, only_user_id=args.only_user_id)
            for inst in insts:
                instances.append(inst)
                inst_source.append(fpath)

    logging.info("Built %d annotation instances", len(instances))
    if not instances:
        write_csv(out_path, [])
        logging.info("No instances. Wrote empty CSV: %s", out_path)
        return

    # Cache hits/misses
    misses: List[Tuple[int, Dict[str, Any], str, str]] = []  # (idx, inst, fp, cid)
    diag_by_fp: Dict[str, Dict[str, Any]] = {}

    for i, inst in enumerate(instances):
        fp = fingerprint_instance(inst, model=args.model, reasoning_effort=args.reasoning_effort, prompt_cache_key=args.prompt_cache_key)
        cached = cache.get(fp)
        if cached:
            diag_by_fp[fp] = cached
        else:
            cid = make_custom_id(i, inst)
            misses.append((i, inst, fp, cid))

    logging.info("Cache hits: %d | misses: %d", len(instances) - len(misses), len(misses))

    client = OpenAI(timeout=15.0 * 60)

    # Prepare request bodies for misses
    bodies_by_cid: Dict[str, Dict[str, Any]] = {}
    fp_by_cid: Dict[str, str] = {}
    inst_by_cid: Dict[str, Dict[str, Any]] = {}

    for idx, inst, fp, cid in misses:
        bodies_by_cid[cid] = build_responses_body(
            inst=inst,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            prompt_cache_key=args.prompt_cache_key,
            prompt_cache_retention=args.prompt_cache_retention,
        )
        fp_by_cid[cid] = fp
        inst_by_cid[cid] = inst

    diag_by_cid: Dict[str, DiagnosticsOutput] = {}

    if misses and args.mode == "sync":
        if args.dry_run:
            logging.info("Dry-run enabled; stopping before API calls.")
            return

        for cid, body in bodies_by_cid.items():
            try:
                resp = client.responses.create(**body, service_tier=args.service_tier)

                raw = resp.model_dump()

                if args.dump_all and dump_dir:
                    maybe_dump(dump_dir, f"{cid}.response.json", raw)

                obj = extract_output_object(raw)
                if not obj:
                    info = summarize_response(raw)
                    logging.error("No structured output for cid=%s | %s", cid, info)
                    if dump_dir:
                        maybe_dump(dump_dir, f"{cid}.response.json", raw)
                        # Optional: auch den Request dumpen, um exakt reproduzieren zu können
                        maybe_dump(dump_dir, f"{cid}.request.json", body)
                    continue

                diag_by_cid[cid] = DiagnosticsOutput.model_validate(obj)
            except BadRequestError as e:
                msg = str(e)
                if ("prompt_cache_retention" in msg) and ("not supported" in msg or "unsupported" in msg) and ("prompt_cache_retention" in body):
                    logging.warning("cid=%s: prompt_cache_retention unsupported; retrying without it", cid)
                    body2 = dict(body)
                    body2.pop("prompt_cache_retention", None)
                    resp = client.responses.create(**body2, service_tier=args.service_tier)
                    raw = resp.model_dump()
                    obj = extract_output_object(raw)
                    if obj:
                        diag_by_cid[cid] = DiagnosticsOutput.model_validate(obj)
                    continue
                logging.error("BadRequest cid=%s: %s", cid, e)
            except ValidationError as e:
                logging.error("Schema validation cid=%s: %s", cid, e)
                if dump_dir:
                    maybe_dump(dump_dir, f"{cid}.response.json", raw)
                    maybe_dump(dump_dir, f"{cid}.request.json", body)
            except Exception as e:
                logging.error("Error cid=%s: %s", cid, e)

    elif misses:
        # batch
        request_lines: List[Dict[str, Any]] = []
        for cid, body in bodies_by_cid.items():
            request_lines.append(build_batch_request_line(custom_id=cid, endpoint=args.batch_endpoint, body=body))

        batch_input_path = out_path.with_suffix(".batch_input.jsonl")
        write_batch_input_jsonl(batch_input_path, request_lines)
        logging.info("Wrote batch input: %s", batch_input_path)

        if args.dry_run:
            logging.info("Dry-run enabled; stopping before API calls.")
            return

        output_file_id = run_batch(
            client,
            batch_input_path,
            batch_endpoint=args.batch_endpoint,
            completion_window=args.completion_window,
            poll_seconds=args.poll_seconds,
        )
        raw_out = download_file_text(client, output_file_id)

        for line in raw_out.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = str(obj.get("custom_id"))
            resp = obj.get("response") or {}
            status_code = resp.get("status_code")
            body = resp.get("body") or {}
            if status_code != 200:
                logging.error("Non-200 for cid=%s status_code=%s", cid, status_code)
                continue
            parsed = extract_output_object(body)
            if not parsed:
                logging.error("No structured output for cid=%s", cid)
                continue
            try:
                diag_by_cid[cid] = DiagnosticsOutput.model_validate(parsed)
            except ValidationError as e:
                logging.error("Validation error for cid=%s: %s", cid, e)

    # Merge diagnostics into cache
    for cid, diag in diag_by_cid.items():
        fp = fp_by_cid.get(cid)
        if fp:
            diag_by_fp[fp] = diag.model_dump()
            cache.put(fp, diag.model_dump())

    # Build final CSV rows
    out_rows: List[Dict[str, Any]] = []
    for inst in instances:
        fp = fingerprint_instance(inst, model=args.model, reasoning_effort=args.reasoning_effort, prompt_cache_key=args.prompt_cache_key)
        diag = diag_by_fp.get(fp)

        md = inst["metadata"]
        ann = inst["annotation"]

        row = {
            "item_id": inst["item_id"],
            "user_id": inst["user_id"],
            "sample_id": md.get("sample_id", ""),
            "suite": md.get("suite", ""),
            "task": md.get("task", ""),
            "subset": md.get("subset", ""),
            "language": md.get("language", ""),
            "annot_required_knowledge": ann.get("required_knowledge", ""),
            "annot_csl": ann.get("cultural_sensitivity_label", ""),
            "annot_time_sensitive": ann.get("time_sensitive", ""),
            "annot_subject_category": ann.get("subject_category", ""),
            "annot_culture": ann.get("culture", ""),
            "annot_region": ann.get("region", ""),
            "annot_country": ann.get("country", ""),
        }

        if not diag:
            # fallback if no model output
            row.update(
                {
                    "missing_fields": "",
                    "contradictions": "",
                    "format_issues": "",
                    "span_issues": "",
                    "content_sanity_flags": "",
                    "needs_fix": "YES",
                    "severity": "ERROR",
                    "suggest_required_knowledge": "",
                    "suggest_csl": "",
                    "suggest_time_sensitive": "",
                    "suggest_subject_category": "",
                    "suggest_culture": "",
                    "suggest_region": "",
                    "suggest_country": "",
                    "suggest_span_fix": "",
                    "comment": "No model output (request/parse/validation failure). Re-run this instance.",
                }
            )
        else:
            row.update(diag)

        out_rows.append(row)

    write_csv(out_path, out_rows)
    logging.info("Done. Wrote: %s", out_path)


if __name__ == "__main__":
    main()


