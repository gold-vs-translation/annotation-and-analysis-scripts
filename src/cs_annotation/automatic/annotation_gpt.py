"""

JSONL -> sqlite dedupe cache -> OpenAI Responses API
  - Sync mode (debug): /v1/responses
      * robust Structured Outputs parsing (prefer output_json)
      * retries on transient errors
      * auto-retry with higher max_output_tokens when output is truncated/incomplete
      * optional raw response dumps
  - Batch mode (production): /v1/responses via Batches
      * same request body + same parsing logic as sync
      * writes .batch_input.jsonl alongside your --out

Requirements:
  pip install openai pydantic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, Field, ValidationError

# ----------------------------
# Pydantic models (must match JSON schema exactly)
# ----------------------------

CSLabel = Literal["CS", "CA"]
YesNo = Literal["Yes", "No"]
RequiredKnowledge = Literal["none", "cultural", "regional", "dialect"]
SubjectCategory = Literal["STEM", "Humanities", "Social Sciences", "Medical", "Business", "Other"]
RefLabel = Literal["Geographic", "Cultural", "Dialect"]


class ReferenceSpan(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    span: str
    label: RefLabel


class CultureAnn(BaseModel):
    required_knowledge: RequiredKnowledge
    time_sensitive: YesNo
    cultural_sensitivity_label: CSLabel
    subject_category: SubjectCategory

    # primary-only (single string or null)
    culture: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None

    # real JSON arrays
    references: List[ReferenceSpan] = Field(default_factory=list)
    other_references: List[ReferenceSpan] = Field(default_factory=list)

    # keep it short for token budget + evaluation convenience
    explanation: str = Field(max_length=250)


class CultureModelOutput(BaseModel):
    culture_ann: CultureAnn


# ----------------------------
# Prompt
# ----------------------------

SYSTEM_PROMPT_TEMPLATE = """
You are an annotator for cultural sensitivity in exam-style multiple-choice questions.

You will receive ONE item as plain text in language: {LANG}.
The text may include the answer key marker like "=> (1)". Do NOT use the answer key to decide spans; you may ignore it.

Goal:
Create a GlobalMMLU-style annotation that captures whether answering correctly requires:
1) Cultural knowledge
2) Geographic (regional) knowledge
3) Dialect or regional language vocabulary knowledge (language-variant knowledge in the item's language)
4) Temporal (time-sensitive) knowledge

Language requirement:
- All annotation fields and the explanation must be in English.
- Only the "span" strings in references/other_references must be copied verbatim from the provided TEXT (and may be in {LANG}).

Definitions:
- Cultural knowledge: Culture-specific facts, norms, values, customs, religion, arts/literature, pop culture, historically rooted traditions tied to a cultural group.
  Label as cultural if there is a clear cultural anchor such as:
  * religion or religious law/ritual (e.g., Sharia, Halakha, canon law, Buddhist precepts) when it is relevant for answering,
  * a named canonical philosopher/scholar/text/school/tradition (esp. in philosophy/psychology/political theory),
  * culturally specific norms or traditions.

  IMPORTANT: Treat named canonical figures/schools in philosophy/psychology/political theory as a cultural anchor (Western Culture), especially when the question asks about their theories/views (e.g., Freud, Kant, Aristotle, Mill, Rawls, Bentham, Hobbes, Locke, Marx, Weber, Durkheim).
  Do NOT treat "According to Author & Author (Year)" as cultural; this is domain-specific academic recall. 
  Only label cultural if the referenced figure/framework is a widely recognized canonical anchor (e.g., Kant, Marx, Rawls).

  IMPORTANT: Also treat standard terminology from Western moral philosophy, logic, and philosophy of mind
  (e.g., consequentialism, virtue ethics, deontology, argument from ignorance, ad hominem, substance dualism, materialism)
  as a cultural anchor. In such cases, label cultural and set culture="Western Culture", even if no person/text is named.

  Religion note:
  - Do NOT map religions mechanically to a region. If the item is about a religion/religious law in a general way without a clear region,
    set culture="Other". If the item explicitly anchors the religious practice to a specific region/country, you may reflect that via required_knowledge="regional"
    (or keep cultural with culture="Other", depending on what is necessary for the answer).

- Geographic/regional knowledge: Knowledge tied to specific regions/countries (places, institutions, laws/policies, local landmarks, region-specific context).
  IMPORTANT cues (treat as regional when present and central):

  Measurement/currency/jurisdiction cues:
  * Imperial/US customary units or US-specific conventions without explanation
    (inch, mile, feet, yard, Fahrenheit/°F, pounds/lb, ounces/oz, miles per hour/mph, "m. p. h.")
    -> label regional, region="North America".
  * Currency marker "$" in numeric problems is ambiguous by itself.
    If "$" appears together with a clear US/North-America anchor (e.g., US state names, "United States", mph/oz/inch/mile, "World Series"), treat it as regional and set region="North America" (and country="United States (US)" or "Canada (CA)" only if clearly implied).
    If "$" appears without any additional regional anchor, do NOT infer a region/country from "$" alone.
    If the text explicitly specifies USD/AUD/CAD (e.g., 'US$', 'A$', 'C$', 'USD'), you may treat that as a regional/country anchor.
  * US state names (e.g., New Jersey, Utah) or US-specific events (e.g., World Series)
    -> label regional (region="North America", country="United States (US)").

  Named local regimes/schemes:
  * Named country/region-specific regulations, institutions, or schemes
    (e.g., EMAS, GDPR, Sarbanes-Oxley Act)
    -> label regional.

  Place-in-scenario:
  * If the item explicitly mentions a real country/region/US state as part of the scenario setup (e.g., traveling from X to Y, offices in New Jersey/Utah, at the Atlanta airport), label required_knowledge="regional" even if the underlying principle is general, UNLESS the place is mentioned only as historical/political background and is not used in the reasoning.
  
  Do not over-label regional:
  * Do NOT label regional just because a country is mentioned in a historical or political passage.
    Label regional only if the place/country is necessary to determine the answer
    (spatial reasoning, legal regime/jurisdiction, local measurement/currency, or a named local institution/regulation).

- Dialect knowledge: Understanding region/community-specific language variations, slang, idioms, or dialect forms in the item's language
  that are necessary to understand/answer. This is about language variation, NOT geography metadata.
  IMPORTANT: Treat common region-specific idioms/colloquial phrases as Dialect when they appear in the question
  (e.g., "Mom & Pop"). If such an idiom is present, label dialect and include the idiom as a Dialect span,
  even if the underlying concept is general.

- Temporal knowledge: The correct answer may change over time (e.g., current leaders, up-to-date statistics, current events, changing regulations).
  Mark Yes if the item contains explicit time references or trend/change language (e.g., "in recent years", "currently", "has increased", "is evolving"),
  even if the fact might still be broadly true.

Subject category annotation (always required):
Assign exactly one of:
- STEM: math, physics, chemistry, biology (non-medical), engineering, computer science.
- Humanities: literature, philosophy, history, arts, languages, religion (as humanities).
- Social Sciences: politics, law, economics (if not business-focused), sociology, psychology, education, public policy.
- Medical: medicine, clinical knowledge, anatomy/physiology (clinical), pharmacology, pathology, nursing.
- Business: accounting, finance, management, marketing, business ethics, operations.
- Other: general knowledge, trivia, mixed/unclear.

Decision rule:
Choose the category that best matches what a person/student must know to answer correctly.
If ambiguous or mixed, prefer Other.

IMPORTANT decision rule:
Only mark a knowledge type if it is necessary to answer correctly, not merely a mention that does not have to be understood for answering the question.
Note:
- The "necessary to answer" rule applies to Cultural/Geographic/Dialect.
- For Temporal, mark Yes based on explicit time references or trend/change language as defined above, even if the time reference is contextual.

Span highlighting ("references"):
If any of Cultural/Geographic/Dialect knowledge is required, highlight minimal text spans in the provided TEXT that trigger it.
Return character offsets into the provided TEXT:
- start: 0-indexed
- end: exclusive
Offsets must match the TEXT exactly, including newline characters.
Spans must be non-overlapping and sorted by start.
Use labels exactly: "Geographic", "Cultural", "Dialect".
If a span could be both Cultural and Geographic, prefer "Geographic" when it is a reference to a place/country/region that exists (or historically existed) in reality.

IMPORTANT:
Do NOT choose required_knowledge="none" just because you cannot find an explicit trigger span.
If the knowledge is implicit, set references=[] and still choose the correct required_knowledge label.

Culture / Region / Country selection:
- If answering the question requires culture specific knowledge, choose the single best matching from; otherwise culture must be null:
  "Western Culture", "Eastern Asian Culture", "Middle Eastern Culture", "South Asian Culture",
  "African Culture", "Latin American Culture", "Other".
- If answering the question requires geographic specific knowledge, choose the single best matching region from:
  "North America", "South America", "Europe", "Asia", "Africa", "Australia and Oceania", "Antarctica".
- If the question requires country specific knowledge, list the primary country as string (or null).
  Use common English country names plus the corresponding ISO 3166-1 alpha-2 code, exactly "Country Name (ISO2)"
  (e.g., "United States (US)", "Germany (DE)").

Output format:
Return ONLY JSON that matches the schema.
Return one top-level object with exactly this structure:
{
  "culture_ann": {
    "required_knowledge": "none" | "cultural" | "regional" | "dialect",
    "time_sensitive": "Yes" | "No",
    "cultural_sensitivity_label": "CS" | "CA",
    "subject_category": "STEM" | "Humanities" | "Social Sciences" | "Medical" | "Business" | "Other",

    "culture": string or null,
    "region": string or null,
    "country": string or null,

    "references": [
      {"start": int, "end": int, "span": string, "label": "Geographic" | "Cultural" | "Dialect"}
    ],
    "other_references": [
      {"start": int, "end": int, "span": string, "label": "Geographic" | "Cultural" | "Dialect"}
    ],

    "explanation": string
  }
}

Rules for required_knowledge (must choose exactly ONE, mirroring the UI):
- "regional" = answering requires geographic/regional/country-specific knowledge.
- "cultural" = answering requires culture-specific knowledge (religion/religious law when relevant, canonical named figures/schools, canonical terminology, culturally specific norms).
- "dialect"  = answering requires dialect/regional vocabulary or idioms in the item's language.
- "none"     = none of the above is required.

Tie-break rule (only if more than one type truly seems necessary; still choose exactly ONE):
dialect > regional > cultural > none.

Note: In this schema, "regional" refers to geographic/regional world knowledge (Geographic label), not dialect.

Rules for cultural_sensitivity_label:
- Cultural_sensitivity_label is NOT strictly determined by required_knowledge.
- Set cultural_sensitivity_label="CA" ONLY if the item is clearly free of cultural/geographic/dialect cues:
  * no named places/countries/US states,
  * no canonical culture/religion/ideology anchors,
  * no dialect/idiom cues,
  * no local regime/unit cues.
- Otherwise set cultural_sensitivity_label="CS", even if required_knowledge="none" (borderline sensitivity).

Primary-only fields (single string, not lists):
- If required_knowledge == "cultural": set "culture" to the single best matching culture label; set "region" and "country" to null.
- If required_knowledge == "regional": set "region" and/or "country" (primary only); set "culture" to null.
  If multiple countries/regions are required: prefer broader region; country=primary or null; mention others briefly in explanation/other_references.
- If required_knowledge == "dialect": set "culture", "region", and "country" to null (dialect is about language variation, not geography metadata here).
- If required_knowledge == "none": set "culture", "region", and "country" to null.

Span annotation rules:
- references MUST contain only spans that directly justify the chosen required_knowledge.
  * If required_knowledge == "regional": references labels must be only "Geographic".
  * If required_knowledge == "cultural": references labels must be only "Cultural".
  * If required_knowledge == "dialect": references labels must be only "Dialect".
  * If required_knowledge == "none": references must be [].
- other_references may contain secondary helpful spans of any of the three labels, but keep it short (can be []).
- Each span must satisfy: span == TEXT[start:end] exactly.
- Offsets are 0-indexed, end is exclusive, and must match the provided TEXT including newlines.
- Spans must be non-overlapping and sorted by start.
- If the knowledge is implicit and no specific trigger span exists, references can be [] even if required_knowledge is not "none".

Explanation rules:
- explanation must be concise (max. 250 characters).
- Mention the main reason for the CS/CA decision.
- If time_sensitive == "Yes", briefly mention the time/trend trigger (e.g., "in recent years", "currently", "has increased", "is evolving").
""".strip()


def build_user_content(rec: Dict[str, Any], *, lang: str) -> str:
    item_id = rec.get("id", "")
    suite = rec.get("suite", "")
    task = rec.get("task", "")
    subset = rec.get("subset", "")
    text = rec.get("text") or ""

    parts = [
        f"ID: {item_id}",
        f"SUITE: {suite}",
        f"TASK: {task}",
        f"SUBSET: {subset}",
        f"LANG: {rec.get('language') or lang}",
        "",
        "TEXT:\n" + str(text),
    ]
    return "\n".join(parts).strip()


def output_json_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "culture_ann": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "required_knowledge": {
                        "type": "string",
                        "enum": ["none", "cultural", "regional", "dialect"],
                    },
                    "time_sensitive": {"type": "string", "enum": ["Yes", "No"]},
                    "cultural_sensitivity_label": {"type": "string", "enum": ["CS", "CA"]},
                    "subject_category": {
                        "type": "string",
                        "enum": ["STEM", "Humanities", "Social Sciences", "Medical", "Business", "Other"],
                    },
                    "culture": {"type": ["string", "null"]},
                    "region": {"type": ["string", "null"]},
                    "country": {"type": ["string", "null"]},
                    "references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "start": {"type": "integer", "minimum": 0},
                                "end": {"type": "integer", "minimum": 0},
                                "span": {"type": "string"},
                                "label": {"type": "string", "enum": ["Geographic", "Cultural", "Dialect"]},
                            },
                            "required": ["start", "end", "span", "label"],
                        },
                    },
                    "other_references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "start": {"type": "integer", "minimum": 0},
                                "end": {"type": "integer", "minimum": 0},
                                "span": {"type": "string"},
                                "label": {"type": "string", "enum": ["Geographic", "Cultural", "Dialect"]},
                            },
                            "required": ["start", "end", "span", "label"],
                        },
                    },
                    "explanation": {"type": "string", "maxLength": 250},
                },
                "required": [
                    "required_knowledge",
                    "time_sensitive",
                    "cultural_sensitivity_label",
                    "subject_category",
                    "culture",
                    "region",
                    "country",
                    "references",
                    "other_references",
                    "explanation",
                ],
            }
        },
        "required": ["culture_ann"],
    }


# ----------------------------
# Cache (SQLite only, to keep things stable & simple)
# ----------------------------

OUTPUT_SCHEMA_VERSION = "gpt52-cultureann-v1"


class SQLiteCache:
    def __init__(self, path: Path):
        # If user passed a directory, create a file inside it
        if path.exists() and path.is_dir():
            path = path / "mqm_cache.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)

        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mqm_cache (
              fingerprint TEXT PRIMARY KEY,
              result_json TEXT NOT NULL
            )
            """)
        self.conn.commit()

    def get(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT result_json FROM mqm_cache WHERE fingerprint = ?", (fingerprint,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def put(self, fingerprint: str, result: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO mqm_cache (fingerprint, result_json) VALUES (?, ?)",
            (fingerprint, json.dumps(result, ensure_ascii=False)),
        )
        self.conn.commit()


def fingerprint_record(
    rec: Dict[str, Any],
    *,
    lang: str,
    prompt_cache_key: str,
    model: str,
    reasoning_effort: str,
) -> str:
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "system_prompt_template": SYSTEM_PROMPT_TEMPLATE,
        "lang": lang,
        "prompt_cache_key": prompt_cache_key,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "id": rec.get("id"),
        "suite": rec.get("suite"),
        "task": rec.get("task"),
        "subset": rec.get("subset"),
        "language": rec.get("language"),
        "text": rec.get("text") or "",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ----------------------------
# JSONL I/O
# ----------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Responses parsing: prefer Structured Outputs (output_json)
# ----------------------------


def extract_output_object(resp_body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Prefer Structured Outputs:
      output[].type == "message"
        content[].type == "output_json"  -> c["json"] (dict) or stringified JSON
    Fallbacks:
      - content[].type == "output_text"  -> parse as JSON if possible
      - resp_body["output_parsed"] (if present)
      - resp_body["output_text"] (SDK convenience, last resort)
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

    # 3) Some SDKs provide parsed output
    if isinstance(resp_body.get("output_parsed"), dict):
        return resp_body["output_parsed"]

    # 4) Last resort: SDK convenience aggregated output_text
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


def build_responses_body(
    *,
    lang: str,
    rec: Dict[str, Any],
    model: str,
    prompt_cache_key: str,
    prompt_cache_retention: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> Dict[str, Any]:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{LANG}", lang)
    user_content = build_user_content(rec, lang=lang)

    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "store": False,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "culture_output",
                "strict": True,
                "schema": output_json_schema(),
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


def write_batch_input_jsonl(path: Path, request_lines: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in request_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


# ----------------------------
# Debugging: dump raw responses
# ----------------------------


def maybe_dump_response(dump_dir: Optional[Path], cid: str, payload: Any) -> None:
    if not dump_dir:
        return
    try:
        dump_dir.mkdir(parents=True, exist_ok=True)
        out_path = dump_dir / f"{cid}.response.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning("Failed to dump response for cid=%s: %s", cid, e)


# ----------------------------
# Custom ID hygiene (Batch + Sync keys)
# ----------------------------


def make_custom_id(row_idx: int, rec: Dict[str, Any]) -> str:
    """
    Batch custom_id should be a stable string. Your data sometimes has rec["id"] as list
    or contains '/', spaces, etc. We normalize to safe ASCII.
    """
    rid = rec.get("id")
    if isinstance(rid, list):
        raw = "__".join(map(str, rid))
    elif rid is None:
        raw = f"row_{row_idx}"
    else:
        raw = str(rid)

    raw = f"{row_idx}__{raw}"
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return raw[:120]


# ----------------------------
# Sync execution (debug) with robust retries
# ----------------------------


def _is_likely_truncation(err: Exception) -> bool:
    msg = str(err).lower()
    return ("eof while parsing" in msg) or ("json_invalid" in msg) or ("unterminated" in msg)


def run_sync_requests(
    client: OpenAI,
    *,
    bodies_by_custom_id: Dict[str, Dict[str, Any]],
    service_tier: str,
    max_retries: int = 4,
    dump_responses_dir: Optional[Path] = None,
) -> Dict[str, CultureModelOutput]:
    """
    Sync debug runner for /v1/responses.

    Retries:
      - transient errors with backoff
      - if prompt_cache_retention is rejected -> retry without it
      - if response is incomplete due to max_output_tokens -> bump tokens and retry
      - if JSON is truncated/invalid -> bump tokens and retry
    """
    out: Dict[str, CultureModelOutput] = {}

    for cid, original_body in bodies_by_custom_id.items():
        body = dict(original_body)  # per-cid mutable copy
        attempt = 0
        backoff = 1.5

        while True:
            try:
                resp = client.responses.create(**body, service_tier=service_tier)
                raw = resp.model_dump()
                maybe_dump_response(dump_responses_dir, cid, raw)

                status = raw.get("status")
                incomplete_reason = (raw.get("incomplete_details") or {}).get("reason")

                # Auto-bump tokens if incomplete due to budget
                if status == "incomplete" and incomplete_reason == "max_output_tokens":
                    old = int(body.get("max_output_tokens") or 0)
                    new = min(max(old * 2, old + 1000), 32000)
                    logging.warning("cid=%s: incomplete(max_output_tokens). bump max_output_tokens %d -> %d and retry", cid, old, new)
                    body = dict(body)
                    body["max_output_tokens"] = new
                    attempt += 1
                    if attempt > max_retries:
                        logging.error("cid=%s: exceeded retries after token bumps.", cid)
                        break
                    continue

                obj = extract_output_object(raw)
                if obj is None:
                    types = [it.get("type") for it in raw.get("output", [])]
                    logging.error(
                        "cid=%s: no parseable Structured Output. output_types=%s status=%s incomplete_reason=%s", cid, types, status, incomplete_reason
                    )
                    break

                parsed = CultureModelOutput.model_validate(obj)
                out[cid] = parsed
                break

            except ValidationError as e:
                # Often caused by truncated JSON when token budget is too small.
                logging.error("cid=%s: schema/JSON validation error: %s", cid, e)

                if _is_likely_truncation(e) and attempt < max_retries:
                    old = int(body.get("max_output_tokens") or 0)
                    new = min(max(old * 2, old + 1000), 32000)
                    logging.warning("cid=%s: likely truncation. bump max_output_tokens %d -> %d and retry", cid, old, new)
                    body = dict(body)
                    body["max_output_tokens"] = new
                    attempt += 1
                    continue
                break

            except BadRequestError as e:
                msg = str(e)
                # If retention is rejected by model/alias/account, drop it and retry once
                if ("prompt_cache_retention" in msg) and ("not supported" in msg or "unsupported" in msg) and ("prompt_cache_retention" in body):
                    logging.warning("cid=%s: prompt_cache_retention rejected; retrying without it.", cid)
                    body = dict(body)
                    body.pop("prompt_cache_retention", None)
                    attempt += 1
                    if attempt > max_retries:
                        logging.error("cid=%s: exceeded retries after dropping prompt_cache_retention.", cid)
                        break
                    continue

                logging.error("cid=%s: BadRequest: %s", cid, e)
                break

            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    logging.error("cid=%s: Sync failed after retries: %s", cid, e)
                    break
                sleep_s = min((backoff**attempt), 20.0)
                logging.warning("cid=%s: transient error: %s | retrying in %.1fs", cid, e, sleep_s)
                time.sleep(sleep_s)

    return out


# ----------------------------
# Main
# ----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Input JSONL")
    parser.add_argument("--out", dest="out_path", required=True, help="Output JSONL")
    parser.add_argument("--lang", required=True, help="Target language code (e.g., DE)")

    parser.add_argument("--mode", choices=["batch", "sync"], default="batch", help="batch (production) or sync (debug)")
    parser.add_argument(
        "--service-tier",
        choices=["default", "auto", "priority"],
        default="default",
        help="Only used in --mode sync",
    )

    # Focus is gpt-5.2, but keep as param
    parser.add_argument("--model", default="gpt-5.2", help="Model name (recommended: gpt-5.2)")
    parser.add_argument("--max-output-tokens", type=int, default=8000, help="Max output tokens (raise if you see truncation)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"], default="high")

    parser.add_argument("--batch-endpoint", default="/v1/responses", help="Batch endpoint (default: /v1/responses)")
    parser.add_argument("--completion-window", default="24h", help='Batch completion window (currently only "24h")')
    parser.add_argument("--poll-seconds", type=int, default=10, help="Batch polling interval in seconds")

    parser.add_argument("--prompt-cache-key", default="cultureann-v1", help="Prompt cache key (version tag)")
    parser.add_argument(
        "--prompt-cache-retention",
        default="24h",
        help='Prompt cache retention (e.g., "in_memory" or "24h"). Use "" to disable.',
    )

    parser.add_argument("--cache-path", default="mqm_cache.sqlite", help="SQLite cache file OR directory")
    parser.add_argument("--dump-responses-dir", default="", help="Optional dir to dump raw responses (sync only)")

    parser.add_argument("--log", default="INFO", help="Log level")
    parser.add_argument("--dry-run", action="store_true", help="Only build batch input, no API calls")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    cache = SQLiteCache(Path(args.cache_path))

    rows = read_jsonl(in_path)
    logging.info("Loaded %d rows", len(rows))

    # cache lookup
    misses: List[Tuple[int, Dict[str, Any], str, str]] = []  # (row_idx, rec, fp, cid)
    out_rows = rows[:]  # shallow copy

    for i, rec in enumerate(out_rows):
        fp = fingerprint_record(
            rec,
            lang=args.lang,
            prompt_cache_key=args.prompt_cache_key,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
        cached = cache.get(fp)
        if cached:
            rec.update(cached)
        else:
            cid = make_custom_id(i, rec)
            misses.append((i, rec, fp, cid))

    logging.info("Cache hits: %d | misses: %d", len(rows) - len(misses), len(misses))

    if not misses:
        write_jsonl(out_path, out_rows)
        logging.info("Done (all from cache). Wrote: %s", out_path)
        return

    client = OpenAI(timeout=15.0 * 60)

    # Build bodies for all misses (shared between batch and sync)
    bodies_by_custom_id: Dict[str, Dict[str, Any]] = {}
    for row_idx, rec, _fp, cid in misses:
        bodies_by_custom_id[cid] = build_responses_body(
            lang=args.lang,
            rec=rec,
            model=args.model,
            prompt_cache_key=args.prompt_cache_key,
            prompt_cache_retention=args.prompt_cache_retention,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )

    if args.mode == "sync":
        if args.dry_run:
            logging.info("Dry-run enabled; stopping before API calls.")
            return

        dump_dir = Path(args.dump_responses_dir) if args.dump_responses_dir.strip() else None
        by_custom_id = run_sync_requests(
            client,
            bodies_by_custom_id=bodies_by_custom_id,
            service_tier=args.service_tier,
            dump_responses_dir=dump_dir,
        )

    else:
        # Build batch input JSONL
        request_lines: List[Dict[str, Any]] = []
        for cid, body in bodies_by_custom_id.items():
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

        by_custom_id = {}
        for line in raw_out.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = str(obj.get("custom_id"))
            resp = obj.get("response") or {}
            status_code = resp.get("status_code")
            body = resp.get("body") or {}

            if status_code != 200:
                logging.error("Non-200 for custom_id=%s status_code=%s", cid, status_code)
                continue

            status = body.get("status")
            incomplete_reason = (body.get("incomplete_details") or {}).get("reason")

            parsed_obj = extract_output_object(body)
            if not parsed_obj:
                if status == "incomplete":
                    logging.error(
                        "Batch incomplete custom_id=%s reason=%s and no parseable output (rerun with higher --max-output-tokens)", cid, incomplete_reason
                    )
                else:
                    logging.error("Empty/unparseable output for custom_id=%s status=%s", cid, status)
                continue

            # Wir haben etwas Parsebares – auch wenn status==incomplete
            if status == "incomplete":
                logging.warning("Batch incomplete custom_id=%s reason=%s but output is parseable; using it.", cid, incomplete_reason)

            try:
                by_custom_id[cid] = CultureModelOutput.model_validate(parsed_obj)
            except ValidationError as e:
                logging.error("Validation error for custom_id=%s: %s | obj_preview=%r", cid, e, str(parsed_obj)[:400])

    # Merge + cache + indices
    DEFAULT_ANN = {
        "required_knowledge": "none",
        "time_sensitive": "No",
        "cultural_sensitivity_label": "CA",
        "subject_category": "Other",
        "culture": None,
        "region": None,
        "country": None,
        "references": [],
        "other_references": [],
        "explanation": "No cultural/regional/dialect-specific knowledge required.",
    }

    for row_idx, rec, fp, cid in misses:
        parsed = by_custom_id.get(cid)

        if not parsed:
            rec["culture_ann"] = dict(DEFAULT_ANN)
        else:
            rec["culture_ann"] = parsed.culture_ann.model_dump()

        cache.put(fp, {"culture_ann": rec["culture_ann"]})

    write_jsonl(out_path, out_rows)
    logging.info("Done. Wrote: %s", out_path)


if __name__ == "__main__":
    main()


