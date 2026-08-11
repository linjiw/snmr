#!/usr/bin/env bash
# Clone the pinned external dependencies (GMR teacher + holosoma) into the sibling layout that
# snmr/paths.py expects, and print the install commands. See THIRD_PARTY.md for licenses.
set -euo pipefail

GMR_SHA="bb1bbe40774794fceb2a7c579a3464a28e68c844"
HOLO_SHA="20699ffa20f494b9563aa68601940c53397bf088"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # snmr repo root
DEST="${SNMR_EXTERNALS_DIR:-$(dirname "$HERE")}"          # default: sibling of the repo

clone_pin () {
  local url="$1" dir="$2" sha="$3"
  if [ -d "$dir/.git" ]; then
    local current
    current="$(git -C "$dir" rev-parse HEAD)"
    if [[ "$current" != "$sha" ]]; then
      printf '[fetch_externals] ERROR: %s exists at %s, expected %s.\n' \
        "$dir" "$current" "$sha" >&2
      printf '[fetch_externals] Use a different SNMR_EXTERNALS_DIR or reconcile that clone explicitly; it was not modified.\n' >&2
      return 2
    fi
    echo "[fetch_externals] $dir already pinned at $sha"
  else
    git clone "$url" "$dir"
    git -C "$dir" checkout --quiet "$sha"
    echo "[fetch_externals] cloned $url @ $sha"
  fi
}

clone_pin https://github.com/YanjieZe/GMR.git       "$DEST/GMR"      "$GMR_SHA"
clone_pin https://github.com/amazon-far/holosoma.git "$DEST/holosoma" "$HOLO_SHA"

cat <<EOF

Next steps:
  # teacher package (needed only for data generation):
  uv pip install --python "$HERE/.venv/bin/python" -e "$DEST/GMR" --no-deps
  uv pip install --python "$HERE/.venv/bin/python" mink "qpsolvers[daqp]" loop_rate_limiters rich tqdm natsort psutil imageio opencv-python-headless

  # regenerate the paired dataset (see docs/DATA.md):
  python scripts/make_pairs_lafan1.py --robots unitree_g1 booster_t1_29dof fourier_n1 engineai_pm01 stanford_toddy
EOF
