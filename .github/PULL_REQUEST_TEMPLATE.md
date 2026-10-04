## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## How it was tested

<!-- The tests you added. For a change to numbers, the script you reran and the reports it rewrote. -->

## Checklist

- [ ] `pytest -m "not network and not slow"`, `ruff check src tests scripts` and `ruff format --check src tests scripts` pass locally.
- [ ] New logic has a test that runs offline.
- [ ] No `.joblib` files, imagery, credentials or data under restrictive terms in the diff.
- [ ] Changed numbers come from a script in `scripts/`, with the report in `docs/` regenerated and the README, `docs/RESULTS.md` and the site updated to match.
- [ ] Thresholds are chosen on the validation split, or the text says they are not.
- [ ] `CHANGELOG.md` has an entry under Unreleased if a user would notice the change.
