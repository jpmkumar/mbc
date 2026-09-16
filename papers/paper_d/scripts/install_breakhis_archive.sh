#!/usr/bin/env bash
# Copy BreakHis archive into papers/paper_d/data/ (local only; gitignored).
# Run from macOS Terminal if Cursor lacks Downloads privacy access.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DEST_DIR="$ROOT/papers/paper_d/data"
DEST="$DEST_DIR/BreaKHis_v1.tar.gz"
FORCE=0
SRC=""

usage() {
  echo "Usage: $0 [--force] [path-to-BreaKHis_v1.tar.gz|.zip]"
  echo "  With no path, searches ~/Downloads for BreaKHis_v1.tar.gz / BreakHis*."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) SRC="$1"; shift ;;
  esac
done

if [[ -z "$SRC" ]]; then
  for cand in \
    "$HOME/Downloads/BreaKHis_v1.tar.gz" \
    "$HOME/Downloads/BreakHis_v1.tar.gz" \
    "$HOME/Downloads/BreaKHis_v1.tar.gz.crdownload" \
    "$HOME/Downloads/BreaKHis_v1.tar.gz.part"
  do
    if [[ -f "$cand" ]]; then SRC="$cand"; break; fi
  done
  if [[ -z "$SRC" ]]; then
    # Prefer complete archives over partials.
    shopt -s nullglob
    matches=(
      "$HOME"/Downloads/*[Bb]rea[Kk][Hh]is*.tar.gz
      "$HOME"/Downloads/*[Bb]reak[Hh]is*.tar.gz
      "$HOME"/Downloads/*[Bb]rea[Kk][Hh]is*.zip
    )
    shopt -u nullglob
    if [[ ${#matches[@]} -gt 0 ]]; then
      SRC="${matches[0]}"
    fi
  fi
fi

if [[ -z "$SRC" || ! -e "$SRC" ]]; then
  echo "No BreakHis archive found." >&2
  echo "Pass the path explicitly, e.g.:" >&2
  echo "  $0 ~/Downloads/BreaKHis_v1.tar.gz" >&2
  exit 1
fi

# Wait out an in-progress browser download (.crdownload / growing size).
wait_stable() {
  local path="$1"
  local prev=-1
  local size
  echo "Waiting for download to finish: $path"
  while true; do
    case "$path" in
      *.crdownload|*.part|*.download)
        if [[ ! -f "$path" ]]; then
          # Browser may have renamed to final name.
          local base="${path%.crdownload}"
          base="${base%.part}"
          base="${base%.download}"
          if [[ -f "$base" ]]; then path="$base"; break; fi
        fi
        ;;
    esac
    size=$(stat -f%z "$path" 2>/dev/null || echo 0)
    if [[ "$size" -gt 0 && "$size" -eq "$prev" ]]; then
      # Stable for one more check.
      sleep 3
      size=$(stat -f%z "$path" 2>/dev/null || echo 0)
      if [[ "$size" -eq "$prev" ]]; then
        echo "$path"
        return 0
      fi
    fi
    prev=$size
    printf '  size=%s bytes; waiting...\n' "$size"
    sleep 5
  done
  echo "$path"
}

SRC="$(wait_stable "$SRC")"

# If we were watching a partial, prefer the final .tar.gz next to it.
if [[ "$SRC" == *.crdownload || "$SRC" == *.part || "$SRC" == *.download ]]; then
  final="${SRC%.crdownload}"
  final="${final%.part}"
  final="${final%.download}"
  if [[ -f "$final" ]]; then SRC="$final"; fi
fi

mkdir -p "$DEST_DIR"

if [[ -e "$DEST" && "$FORCE" -ne 1 ]]; then
  echo "Destination exists: $DEST" >&2
  echo "Pass --force to overwrite." >&2
  exit 1
fi

echo "Copying:"
echo "  from: $SRC"
echo "  to:   $DEST"
cp -f "$SRC" "$DEST"

# Keep original extension if user supplied a zip; still hash the dest bytes.
SHA=$(shasum -a 256 "$DEST" | awk '{print $1}')
BYTES=$(stat -f%z "$DEST")
echo
echo "Installed."
echo "  bytes:  $BYTES"
echo "  sha256: $SHA"
echo
echo "Next: tell the agent the archive is ready, or run:"
echo "  python3 papers/paper_d/scripts/write_paperd_lock.py --archive-sha256 $SHA"
echo "(after case-ID splits exist; Phase-0 stub still needs --allow-empty-splits only before splits.)"
