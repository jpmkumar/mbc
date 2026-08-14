#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

LOCK="docker/requirements-server.lock"
EXPECTED_LOCK_SHA="55ae29e55a5e3643fb59be8e3aaa2c1466e63efcd3559bc22b4addab4e7c829a"

if [[ ! -f "$LOCK" ]]; then
  echo "Missing $LOCK" >&2
  exit 1
fi
LOCK_SHA="$(sha256sum "$LOCK" | awk '{print $1}')"
if [[ "$LOCK_SHA" != "$EXPECTED_LOCK_SHA" ]]; then
  echo "Dependency lock hash mismatch: $LOCK_SHA" >&2
  echo "Expected qualified lock: $EXPECTED_LOCK_SHA" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Refusing to build an experiment image from a dirty tracked tree." >&2
  exit 1
fi

COMMIT="$(git rev-parse HEAD)"
ENV_TAG="mbc-gpu:env-${LOCK_SHA:0:12}"
CODE_TAG="mbc-gpu:${COMMIT:0:12}"
CONTEXT="$(mktemp -d)"
trap 'rm -rf "$CONTEXT"' EXIT

docker build \
  --file docker/Dockerfile.gpu \
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
cat > results/server_setup/current_server_image.env <<EOF
MBC_IMAGE=$CODE_TAG
MBC_IMAGE_ID=$IMAGE_ID
MBC_GIT_COMMIT=$COMMIT
MBC_ENV_LOCK_SHA=$LOCK_SHA
EOF

docker run --rm -i --gpus all "$CODE_TAG" python - <<'PY'
import json
import pennylane as qml
import torch

print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "pennylane": qml.__version__,
}, indent=2))
if not torch.cuda.is_available():
    raise SystemExit("CUDA qualification failed.")
PY

echo "Built immutable experiment image: $CODE_TAG"
echo "Image ID: $IMAGE_ID"
echo "Metadata: results/server_setup/current_server_image.env"

