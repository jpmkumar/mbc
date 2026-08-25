#!/usr/bin/env bash
# Resolve docker/requirements-paperc.lock: the exact transitive delta between the
# digest-pinned NGC base image and the packages Paper C needs.
#
# Run this ONCE on the server. It writes the lock and prints its SHA-256, which
# then gets pinned into build_paperc_image.sh so later builds cannot drift.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

BASE="nvcr.io/nvidia/pytorch:26.06-py3@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1"
WANT=(timm huggingface_hub transformers safetensors datasets scikit-learn scipy seaborn)
OUT="docker/requirements-paperc.lock"

echo "Resolving Paper C dependency delta against the pinned NGC base..."
docker run --rm "$BASE" bash -lc "
  set -e
  python -m pip freeze --disable-pip-version-check > /tmp/before.txt 2>/dev/null
  python -m pip install --quiet --no-cache-dir ${WANT[*]} >/dev/null 2>&1
  python -m pip freeze --disable-pip-version-check > /tmp/after.txt 2>/dev/null
  comm -13 <(sort /tmp/before.txt) <(sort /tmp/after.txt)
" | grep -E '^[A-Za-z0-9._-]+==' | sort > "$OUT"

if [[ ! -s "$OUT" ]]; then
  echo "Resolution produced an empty lock - inspect manually." >&2
  exit 1
fi

SHA="$(sha256sum "$OUT" | awk '{print $1}')"
echo
echo "Wrote $OUT ($(wc -l < "$OUT") packages)"
echo "Lock SHA-256: $SHA"
echo
echo "Pin this into papers/paper_c/scripts/build_paperc_image.sh as"
echo "EXPECTED_LOCK_SHA, then commit the lock."
