# Repository maintenance

- Read and write text as UTF-8 without BOM.
- Never add clinical data, annotations, patient/image lists, per-example predictions,
  attention arrays, model weights, credentials, or raw experiment logs.
- Only already composed paper figures and aggregate paper tables are public assets.
- `code/` and `research/` implement the regional architecture. The distinct
  `reproducibility/delivered_checkpoint/` source must use a separate import root.
- Preserve original/source hashes and report packaging changes in provenance.
- Do not claim numerical reproduction from software tests or manuscript builds.
- Training selects on validation, freezes the checkpoint/config/endpoint, then
  evaluates test. Report image, patient-max, patient-median, and center metrics.
- Run `python tools/check_release.py` and `python -m pytest` before publishing.
- After an intentional source edit, review the diff and regenerate the release
  inventory with `python tools/check_release.py --write-manifest`.
