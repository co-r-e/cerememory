#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Run the full pre-release verification suite.

Usage:
  scripts/release-check.sh

Requires:
  rustc 1.95.0
  cargo-audit
  cargo-deny
  cargo-outdated
  protoc
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 127
  fi
}

require_cargo_subcommand() {
  if ! cargo "$1" --version >/dev/null 2>&1; then
    echo "missing required cargo subcommand: cargo $1" >&2
    exit 127
  fi
}

require_cmd cargo
require_cmd rustc
require_cmd protoc
require_cmd git
require_cargo_subcommand audit
require_cargo_subcommand deny
require_cargo_subcommand outdated

rustc_version="$(rustc --version)"
if [[ "${rustc_version}" != rustc\ 1.95.0\ * ]]; then
  echo "release checks require rustc 1.95.0; found: ${rustc_version}" >&2
  exit 1
fi

cargo fmt --all --check
cargo check -p cerememory-cli --locked
cargo check -p cerememory-cli --no-default-features --locked
cargo check -p cerememory-cli --no-default-features --features llm-openai --locked
cargo check -p cerememory-cli --no-default-features --features llm-claude --locked
cargo check -p cerememory-cli --no-default-features --features llm-gemini --locked
cargo check -p cerememory-cli --all-features --locked
cargo clippy -p cerememory-cli --all-targets --all-features --locked -- -D warnings
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo bench --workspace --no-run --locked
cargo audit
cargo deny check advisories licenses sources bans
cargo outdated --workspace --root-deps-only
git diff --check
