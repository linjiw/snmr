#!/usr/bin/env bash
# Clone the pinned external dependencies (GMR teacher + holosoma) into the sibling layout that
# snmr/paths.py expects, and print the install commands. See THIRD_PARTY.md for licenses.
set -euo pipefail

GMR_SHA="bb1bbe40774794fceb2a7c579a3464a28e68c844"
HOLO_SHA="38009aad61851d59277fa4ebaf4f54c44ec483f7"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # snmr repo root
DEST="${SNMR_EXTERNALS_DIR:-$(dirname "$HERE")}"          # default: sibling of the repo

clone_pin () {
  local url="$1" dir="$2" sha="$3"
  if [ -d "$dir/.git" ]; then
    echo "[fetch_externals] $dir exists; leaving as-is (HEAD $(git -C "$dir" rev-parse --short HEAD))"
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
  pip install -e "$DEST/GMR" --no-deps
  pip install mink "qpsolvers[daqp]" loop_rate_limiters rich tqdm natsort psutil imageio opencv-python-headless

  # regenerate the paired dataset (see docs/DATA.md):
  python scripts/make_pairs_lafan1.py --robots unitree_g1 booster_t1_29dof fourier_n1 engineai_pm01 stanford_toddy
EOF
