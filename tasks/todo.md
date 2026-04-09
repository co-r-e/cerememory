# Cerememory Phase 1 Implementation

## Phase 1a: Foundation
- [x] Step 0: cerememory-core — CMP protocol types + async trait conversion (17 tests)
- [x] Step 1: cerememory-store-working — InMemory Working Memory (20 tests)
- [x] Step 2: cerememory-store-episodic — redb Episodic Store (11 tests)
- [x] Step 3: cerememory-store-semantic — redb Graph Store (18 tests)

## Phase 1b: Dynamics
- [x] Step 4: cerememory-decay — Power-law Decay Engine (34 tests)
- [x] Step 5: cerememory-association — Spreading Activation (10 tests)
- [x] Step 6: cerememory-index — Hippocampal Coordinator (8 tests)
- [x] Step 7: Stub crates (procedural, emotional, evolution) (5 tests)

## Phase 1c: Integration
- [x] Step 8: cerememory-engine — Orchestrator (10 tests)
- [x] Step 9: cerememory-transport-http — Axum REST API (6 tests)
- [x] Step 10: LLM Adapter stubs (claude, openai, gemini) (15 tests)
- [x] Step 11: Integration tests (8 tests)

## Verification
- [x] `cargo test --workspace` — 162 tests, 0 failures
- [x] `cargo clippy --workspace` — 0 warnings

## 2026-03-27 GHCR Package Access Investigation
- [x] Confirm current `gh` auth identity and token scopes relevant to Packages
- [x] Inspect release workflows / repository-side package permissions
- [x] Probe GitHub Packages API for `ghcr.io/co-r-e/cerememory` access from CLI
- [ ] Apply package-side configuration via `gh` if current permissions allow it
- [x] Record verification results and any remaining blocker

## Review
- `gh repo view` shows `viewerPermission: ADMIN` on `co-r-e/cerememory`, but package administration is separate from repository administration.
- `gh auth status` shows the active session is `mokuwaki (GH_TOKEN)`. `gh api '/orgs/co-r-e/packages/container/cerememory'` and `gh api '/orgs/co-r-e/packages?package_type=container'` both returned `403 Resource not accessible by personal access token`.
- `.github/workflows/release-github.yml` already grants `packages: write`, so the workflow-side token permission is configured.
- GitHub's documented Packages REST API exposes package metadata/delete/restore/version endpoints, but not a documented endpoint for `Manage Actions access` or `Connect repository`, so there is no supported `gh api` call here to flip those settings directly.
- Added `org.opencontainers.image.source=https://github.com/co-r-e/cerememory` to the runtime image so a future privileged push can establish repository linkage from the image metadata side.
- Verified the Dockerfile syntax with `docker build --check .` (passed; no warnings).
- Remaining blocker: package-side admin action is still required via the GitHub UI, or via a successful push authenticated with credentials that actually have package write/admin access for `ghcr.io/co-r-e/cerememory`.
