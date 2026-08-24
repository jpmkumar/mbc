#!/usr/bin/env bash
# Build the immutable Paper C experiment image. Mirrors
# scripts/build_histopath_server_image.sh, but with the Paper C dependency lock
# and no PennyLane qualification step.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

LOCK="docker/requirements-paperc.lock"
# Set after the first resolve_paperc_lock.sh run. Empty means "not yet
# qualified" and the build will record, rather than enforce, the hash.
EXPECTED_LOCK_SHA="${MBC_PAPERC_LOCK_SHA:-}"

if [[ ! -f "$LOCK" ]]; then
  echo "Missing $LOCK - run papers/paper_c/scripts/resolve_paperc_lock.sh first." >&2
  exit 1
fi
LOCK_SHA="$(sha256sum "$LOCK" | awk '{print $1}')"
if [[ -n "$EXPECTED_LOCK_SHA" && "$LOCK_SHA" != "$EXPECTED_LOCK_SHA" ]]; then
  echo "Dependency lock hash mismatch: $LOCK_SHA" >&2
  echo "Expected qualified lock: $EXPECTED_LOCK_SHA" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Refusing to build an experiment image from a dirty tracked tree." >&2
  exit 1
fi

COMMIT="$(git rev-parse HEAD)"
ENV_TAG="mbc-paperc:env-${LOCK_SHA:0:12}"
CODE_TAG="mbc-paperc:${COMMIT:0:12}"
CONTEXT="$(mktemp -d)"
trap 'rm -rf "$CONTEXT"' EXIT

docker build \
  --file docker/Dockerfile.paperc \
  --tag "$ENV_TAG" \
  docker

git archive HEAD | tar -x -C "$CONTEXT"
docker build \
  --file "$CONTEXT/docker/Dockerfile.experiment" \
  --build-arg "BASE_IMAGE=$ENV_TAG" \
  --build-arg "MBC_GIT_COMMIT=$COMMIT" \
  --tag "$CODE_TAG" \
  "$CONTEXT"

IMAGE_ID="$(docker image inspect "$CODE_TAG" --format '{{.Id}}')"
mkdir -p results/server_setup
cat > results/server_setup/current_paperc_image.env <<EOF
MBC_IMAGE=$CODE_TAG
MBC_IMAGE_ID=$IMAGE_ID
MBC_GIT_COMMIT=$COMMIT
MBC_ENV_LOCK_SHA=$LOCK_SHA
EOF

docker run --rm -i --gpus all "$CODE_TAG" python - <<'PY'
import json
import timm
import torch

print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "timm": timm.__version__,
}, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA qualification failed.")
PY

echo "Built Paper C experiment image: $CODE_TAG"
echo "Image ID: $IMAGE_ID"
echo "Lock SHA: $LOCK_SHA"
echo "Metadata: results/server_setup/current_paperc_image.env"
