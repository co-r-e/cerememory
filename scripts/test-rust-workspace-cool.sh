#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Run the full Rust workspace test sweep with CPU-friendly defaults.

Usage:
  scripts/test-rust-workspace-cool.sh [cargo test args] [-- <libtest args>]

Examples:
  scripts/test-rust-workspace-cool.sh
  scripts/test-rust-workspace-cool.sh --exclude cerememory-python --exclude cerememory-napi
  scripts/test-rust-workspace-cool.sh -- --nocapture

Defaults:
  CARGO_BUILD_JOBS=2
  RUST_TEST_THREADS=1
  RAYON_NUM_THREADS=1

Prefer scripts/test-rust-cool.sh for everyday development and reserve this command
for pre-PR or release sweeps.
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

build_jobs="${CARGO_BUILD_JOBS:-2}"
test_threads="${RUST_TEST_THREADS:-1}"
rayon_threads="${RAYON_NUM_THREADS:-1}"

cargo_args=(--workspace)
test_args=()
seen_separator=0
has_jobs_flag=0
has_test_threads_flag=0

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

if [ "$has_jobs_flag" -eq 0 ]; then
  cargo_args=(-j "$build_jobs" "${cargo_args[@]}")
fi

if [ "$has_test_threads_flag" -eq 0 ]; then
  test_args+=(--test-threads="$test_threads")
fi

echo "Running workspace cargo test with CARGO_BUILD_JOBS=${build_jobs}, RUST_TEST_THREADS=${test_threads}, RAYON_NUM_THREADS=${rayon_threads}" >&2

exec env \
  CARGO_BUILD_JOBS="$build_jobs" \
  RUST_TEST_THREADS="$test_threads" \
  RAYON_NUM_THREADS="$rayon_threads" \
  cargo test "${cargo_args[@]}" -- "${test_args[@]}"
