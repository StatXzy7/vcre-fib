# Maintaining the public repository

Maintain public code and paper assets in this directory. Keep datasets and runs
outside the checkout. The original research directory remains a provenance source;
there is no automatic two-way synchronization or bulk copy from that directory.

Before a commit:

```bash
python -m pytest -q
python tools/build_paper.py --language all  # if manuscript sources changed
git diff --check
python tools/check_release.py --write-manifest
git add <reviewed-files> provenance/RELEASE_MANIFEST.json
python tools/check_release.py
```

Review the changed file list and manifest before committing. Stage explicit files.
The final check requires the staged bytes to equal the audited bytes; an unstaged
cleanup cannot conceal content still in Git's index. Manifest generation checks
the working tree only and deliberately reports DRAFT until that final check passes.
Do not bypass exclusions with `git add -f`. Do not publish raw experiment folders,
annotations, split lists, per-example outputs, checkpoint binaries or credentials.
New media requires a reviewed allowlist update; manuscript result changes require
source/checkpoint/metric provenance. Report test results before validation diagnostics
when reporting experiments; software-test success is not experiment completion.

New source changes do not update the historical `IMPORT_MANIFEST.json` identities.
Document the change and update the current `RELEASE_MANIFEST.json` instead.
