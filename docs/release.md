# Release Checklist

Cerememory is still pre-release, so breaking changes are allowed when they reduce
security risk or simplify the long-term contract. Every release candidate should
still prove the full operational story before tagging.

## Preflight

1. Confirm `CHANGELOG.md` has an `[Unreleased]` entry for every user-visible or
   compatibility-affecting change.
2. Confirm the default operating path still requires no external LLM API keys:
   `[llm].provider = "none"` must remain the documented default, and MCP/HTTP
   store/recall/stats flows must work without OpenAI, Anthropic, or Gemini
   secrets.
3. Confirm new security advisory ignores in `deny.toml` include an owner,
   review-after date, and exit plan. Prefer removing the dependency instead.
4. Confirm storage/archive format changes include fixtures or fixture manifest
   updates under `tests/fixtures/`.
5. Confirm any new optional dependency is covered by `.github/workflows/ci.yml`
   feature matrix.
6. Treat external LLM adapter E2E as an explicit opt-in check, not a release
   gate for the standard no-external-API distribution. Run it only when provider
   secrets are intentionally available and the adapter surface changed.
7. Confirm the default `cerememory-cli` feature set remains adapter-free; LLM
   provider builds must stay explicit feature opt-ins.

## Verification

Run:

```sh
scripts/release-check.sh
```

The script checks formatting, all CLI feature combinations, clippy, workspace
tests, benchmark compilation, `cargo audit`, `cargo deny`, root dependency
freshness, and whitespace errors.

External LLM adapter E2E is intentionally excluded from `scripts/release-check.sh`.
To run it manually, dispatch the CI workflow with `run_llm_e2e = true` after
confirming provider API secrets are intentionally present.

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
