#!/usr/bin/env bash
# Raystrack macOS Installer (Rhino 8)
# - Force-(re)installs this repo into Rhino 8 CPython
# - Copies all *.ghuser (recursive) into GH UserObjects/raystrack
# - Deletes any existing destination folder first

set -uo pipefail
LOG="[raystrack]"

die() { echo "$LOG ERROR: $*" >&2; echo; read -n 1 -s -r -p "$LOG Press any key to close..."; echo; exit 1; }

echo "$LOG Starting installation..."
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "$LOG Repo directory: $REPO_DIR"

# 1) Verify pyproject.toml exists
[[ -f "$REPO_DIR/pyproject.toml" ]] || die "pyproject.toml not found in $REPO_DIR"

# 2) Locate Rhino 8 CPython interpreter
PY="${RAYSTRACK_RHINO_PY:-}"
if [[ -n "${PY}" && -x "${PY}" ]]; then
  echo "$LOG Using Rhino Python from RAYSTRACK_RHINO_PY: $PY"
else
  PY=""
  shopt -s nullglob
  # Common Rhino 8 mac layouts: ~/.rhinocode/py39-rh8/python3.9 (COMPAS docs),
  # or ~/.rhinocode/py*-rh8/bin/python3*
  CAND=( "$HOME/.rhinocode"/py*-rh8/python3.* "$HOME/.rhinocode"/py*-rh8/bin/python3* )
  for p in "${CAND[@]}"; do
    if [[ -x "$p" ]]; then PY="$p"; break; fi
  done
  shopt -u nullglob
  [[ -n "$PY" ]] || die "Could not find Rhino 8 CPython under ~/.rhinocode (set RAYSTRACK_RHINO_PY to override)."
  echo "$LOG Rhino Python: $PY"
fi

# 3) Ensure pip; best-effort upgrade of build tooling
"$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade || die "ensurepip failed"
"$PY" -m pip install --upgrade pip setuptools wheel build >/dev/null 2>&1 || true

# 4) Force-(re)install this repo into Rhino 8 Python
echo "$LOG Installing/overwriting raystrack into Rhino 8 Python..."
if ! "$PY" -m pip install --no-cache-dir --upgrade --force-reinstall --no-deps "$REPO_DIR"; then
  die "pip install failed"
fi
echo "$LOG Package installed (force-reinstalled) successfully."

# 5) Resolve Grasshopper UserObjects destination (macOS)
# Canonical for Rhino 8:
DEST_PRIMARY="$HOME/Library/Application Support/McNeel/Rhinoceros/8.0/Plug-ins/Grasshopper/UserObjects"
# Fallbacks for older layouts:
DEST_ALT1="$HOME/Library/Application Support/McNeel/Rhinoceros/8.0/MacPlugIns/Grasshopper/UserObjects"
DEST_ALT2="$HOME/Library/Application Support/McNeel/Rhinoceros/7.0/Plug-ins/Grasshopper/UserObjects"

DST_BASE=""
for d in "$DEST_PRIMARY" "$DEST_ALT1" "$DEST_ALT2"; do
  # pick the first whose parent exists; otherwise create the canonical later
  parent="$(dirname "$d")"
  if [[ -d "$parent" ]]; then DST_BASE="$d"; break; fi
done
[[ -n "$DST_BASE" ]] || DST_BASE="$DEST_PRIMARY"

echo "$LOG Preparing destination: $DST_BASE/raystrack"
mkdir -p "$DST_BASE" || die "Could not create $DST_BASE"
rm -rf "$DST_BASE/raystrack"
mkdir -p "$DST_BASE/raystrack" || die "Could not create $DST_BASE/raystrack"

# 6) Find .ghuser files (prefer repo/rhino/components, then components, then whole repo)
if [[ -d "$REPO_DIR/rhino/components" ]]; then
  SEARCH_ROOT="$REPO_DIR/rhino/components"
elif [[ -d "$REPO_DIR/components" ]]; then
  SEARCH_ROOT="$REPO_DIR/components"
else
  SEARCH_ROOT="$REPO_DIR"
fi
echo "$LOG Searching for .ghuser files under: $SEARCH_ROOT"

FOUND=0
COPIED=0
# Copy flattened into raystrack
while IFS= read -r -d '' f; do
  ((FOUND++))
  echo "$LOG Copying: $f"
  if cp -f "$f" "$DST_BASE/raystrack/"; then
    ((COPIED++))
  else
    echo "$LOG WARNING: Failed to copy: $f" >&2
  fi
done < <(find "$SEARCH_ROOT" -type f -name '*.ghuser' -print0 2>/dev/null)

if [[ "$FOUND" -eq 0 ]]; then
  echo "$LOG NOTE: No .ghuser files found under $SEARCH_ROOT"
else
  echo "$LOG Found $FOUND .ghuser file(s); copied $COPIED to $DST_BASE/raystrack"
fi

echo
echo "$LOG ==================== DONE ===================="
echo "$LOG Installation and UserObjects copy steps completed."
read -n 1 -s -r -p "$LOG Press any key to close..."; echo
exit 0
