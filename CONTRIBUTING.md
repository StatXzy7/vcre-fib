# Contributing

Use English for documentation, issues, pull requests, commits, and new comments.
`README.md` is the primary entry point; `README.zh-CN.md` is an optional translation.
Preserve historical source comments rather than rewriting frozen snapshots for style.

Maintain public code here. Keep datasets, runs, manuscript text, result tables, and
project citation metadata outside the checkout. There is no automatic bulk import
or two-way synchronization with the private research directory.

Before a commit:

```bash
python -m pytest -q
git diff --check
python tools/check_release.py --write-manifest
git add <reviewed-files> provenance/RELEASE_MANIFEST.json
python tools/check_release.py
```

Review the changed file list and stage explicit paths. Manifest generation audits
the working tree and reports DRAFT. The final check requires the exact audited
bytes in Git's index, so an unstaged cleanup cannot conceal staged content.
Every imported source must remain in the inventory, including source packages
whose directory is named `data`. Deliberate release-scope changes must update the
import inventory; do not alter the original identities of retained source files.

Do not bypass exclusions with `git add -f`. Retain third-party licenses and notices.
Code edits update `RELEASE_MANIFEST.json`; historical source hashes do not change.

To regenerate the main figure preview from its PDF:

```bash
python -m pip install -e '.[paper]'
python tools/render_readme_figure.py
```

Inspect the preview and stage the PDF, PNG, and release inventory together. New
media requires an intentional allowlist change and content review.
