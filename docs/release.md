# Release Checklist

Cerememory is still pre-release, so breaking changes are allowed when they reduce
security risk or simplify the long-term contract. Every release candidate should
still prove the full operational story before tagging.

## Preflight

1. Confirm `CHANGELOG.md` has an `[Unreleased]` entry for every user-visible or
   compatibility-affecting change.
2. Confirm new security advisory ignores in `deny.toml` include an owner,
   review-after date, and exit plan. Prefer removing the dependency instead.
3. Confirm storage/archive format changes include fixtures or fixture manifest
   updates under `tests/fixtures/`.
4. Confirm any new optional dependency is covered by `.github/workflows/ci.yml`
   feature matrix.

## Verification

Run:

```sh
scripts/release-check.sh
```

The script checks formatting, all CLI feature combinations, clippy, workspace
tests, benchmark compilation, `cargo audit`, `cargo deny`, root dependency
freshness, and whitespace errors.

## Tagging

1. Move `[Unreleased]` notes into a dated version section.
2. Run `scripts/release-check.sh` again after the changelog edit.
3. Commit the release prep change.
4. Create an annotated tag:

```sh
git tag -a vX.Y.Z -m "Release vX.Y.Z"
```

5. Push the commit and tag:

```sh
git push origin main
git push origin vX.Y.Z
```
