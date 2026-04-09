#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Run a targeted Rust test suite with CPU-friendly defaults.

Usage:
  scripts/test-rust-cool.sh <cargo test args> [-- <libtest args>]

Examples:
  scripts/test-rust-cool.sh -p cerememory-engine --lib
  scripts/test-rust-cool.sh -p cerememory-integration-tests --test phase3
  scripts/test-rust-cool.sh -p cerememory-decay --lib tests::batch_processing_10000_records -- --exact

Defaults:
  CARGO_BUILD_JOBS=1
  RUST_TEST_THREADS=1
  RAYON_NUM_THREADS=1

Override them by exporting the environment variables before running the script.
EOF
}

if [ "$#" -eq 0 ]; then
  usage
  exit 1
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

build_jobs="${CARGO_BUILD_JOBS:-1}"
test_threads="${RUST_TEST_THREADS:-1}"
rayon_threads="${RAYON_NUM_THREADS:-1}"

cargo_args=()
test_args=()
seen_separator=0
has_jobs_flag=0
has_test_threads_flag=0
uses_workspace=0

for arg in "$@"; do
  if [ "$arg" = "--" ] && [ "$seen_separator" -eq 0 ]; then
    seen_separator=1
    continue
  fi

  if [ "$seen_separator" -eq 0 ]; then
    cargo_args+=("$arg")
    case "$arg" in
      -j|--jobs|--jobs=*)
        has_jobs_flag=1
        ;;
      --workspace)
        uses_workspace=1
        ;;
    esac
  else
    test_args+=("$arg")
    case "$arg" in
      --test-threads|--test-threads=*)
        has_test_threads_flag=1
        ;;
    esac
  fi
done

if [ "${#cargo_args[@]}" -eq 0 ]; then
  usage
  exit 1
fi

if [ "$uses_workspace" -eq 1 ]; then
  cat >&2 <<'EOF'
warning: --workspace is expensive. Prefer a targeted package or integration test during routine development.
If you really want a full sweep, use scripts/test-rust-workspace-cool.sh instead.
EOF
fi

if [ "$has_jobs_flag" -eq 0 ]; then
  cargo_args=(-j "$build_jobs" "${cargo_args[@]}")
fi

if [ "$has_test_threads_flag" -eq 0 ]; then
  test_args+=(--test-threads="$test_threads")
fi

echo "Running cargo test with CARGO_BUILD_JOBS=${build_jobs}, RUST_TEST_THREADS=${test_threads}, RAYON_NUM_THREADS=${rayon_threads}" >&2

exec env \
  CARGO_BUILD_JOBS="$build_jobs" \
  RUST_TEST_THREADS="$test_threads" \
  RAYON_NUM_THREADS="$rayon_threads" \
  cargo test "${cargo_args[@]}" -- "${test_args[@]}"
