#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <users.tsv> <batch_paths.tsv> <out_dir> <path_to_filter_py>"
  echo "  users.tsv columns: Name<TAB>Batch<TAB>Credentials<TAB>Ids"
  echo "  batch_paths.tsv columns: Batch<TAB>Path   (Path MUST be a .jsonl file)"
  exit 1
fi

USERS_TSV="$1"
BATCH_PATHS_TSV="$2"
OUT_DIR="$3"
FILTER_PY="$4"

mkdir -p "$OUT_DIR"

# --- load batch->jsonl_path
declare -A BATCH_JSONL
tail -n +2 "$BATCH_PATHS_TSV" | while IFS=$'\t' read -r batch path rest; do
  batch="$(echo "${batch:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  path="$(echo "${path:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -z "$batch" || -z "$path" ]] && continue

  if [[ ! -f "$path" ]]; then
    echo "ERROR: batch_paths.tsv: Path for batch '$batch' does not exist or is not a file: $path" >&2
    exit 1
  fi
  BATCH_JSONL["$batch"]="$path"
done

# NOTE: associative array inside while subshell caveat
# We'll reload in a non-subshell way:
declare -A BATCH_JSONL
while IFS=$'\t' read -r batch path rest; do
  batch="$(echo "${batch:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  path="$(echo "${path:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ -z "$batch" || -z "$path" ]] && continue
  [[ "$batch" == "Batch" && "$path" == "Path" ]] && continue

  if [[ ! -f "$path" ]]; then
    echo "ERROR: batch_paths.tsv: Path for batch '$batch' does not exist or is not a file: $path" >&2
    exit 1
  fi
  BATCH_JSONL["$batch"]="$path"
done < "$BATCH_PATHS_TSV"

# --- iterate users.tsv
# expected columns: Name, Batch, Credentials, Ids
tail -n +2 "$USERS_TSV" | while IFS=$'\t' read -r name batch creds user_id rest; do
  name="$(echo "${name:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  batch="$(echo "${batch:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  creds="$(echo "${creds:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  user_id="$(echo "${user_id:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

  [[ -z "$batch" || -z "$creds" || -z "$user_id" ]] && continue

  in_jsonl="${BATCH_JSONL[$batch]:-}"
  if [[ -z "$in_jsonl" ]]; then
    echo "ERROR: No JSONL path configured for batch='$batch' (user '$name', creds '$creds')." >&2
    exit 1
  fi

  safe_creds="$(echo "$creds" | tr -c 'A-Za-z0-9_.-' '_' )"
  out_batch_dir="${OUT_DIR}/${batch}"
  mkdir -p "$out_batch_dir"
  out_path="${out_batch_dir}/${safe_creds}__${user_id}.jsonl"

  echo "==> Batch=${batch} | User=${name} | Creds=${creds} | user_id=${user_id}"
  echo "    IN : ${in_jsonl}"
  echo "    OUT: ${out_path}"

  audit_path="${out_path%.jsonl}.audit.tsv"

  python "$FILTER_PY" -i "$in_jsonl" -o "$out_path" -u "$user_id" \
    --stats \
    --audit-path "$audit_path"
    
done

