# Gold vs. Translation — release notes

Paper: https://aclanthology.org/2026.emnlp-main.32/

Data: https://github.com/gold-vs-translation/cs-annotation-data

This repository contains the original annotation/analysis package (`src/cs_annotation/`), demonstration scripts and synthetic example data, plus the final camera-ready analyses under `scripts/camera_ready/`.

## Camera-ready additions

- `scripts/camera_ready/compute_raw_iaa_plus_consensus_globalmmlu_style.py`
- `scripts/camera_ready/incremental_proxy_validity_bootstrap.py`
- `scripts/camera_ready/recompute_native_meta_eval_camera_ready.py`

The human-annotation names in the IAA script's mapping were replaced with pseudonyms in the public version. Its compatibility with the pseudonymized public TSV inputs **must be checked**; the original pipeline may require private source records for the initial data preparation.

## Installation and usage

See the existing `README.md`, `docs/PIPELINE.md`, and `pyproject.toml`. Before release, verify at least one documented pipeline/demo command and one camera-ready reproduction command, record prerequisites, and distinguish steps requiring private original inputs from those runnable with public release data.

## Credentials

API credentials are supplied through environment variables such as `OPENAI_API_KEY`, `ARGILLA_API_KEY`, and `ARGILLA_API_URL`. Never commit real keys or `.env` files. Existing README examples use placeholders.

## License

Intended license for original code: MIT, subject to confirming all contributor rights and third-party code notices. Insert an actual MIT license with the correct copyright owners before publishing; do not treat this note as a license grant.
