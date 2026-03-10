#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   bash scripts/colab_benchmark.sh --ticks 5000 --seeds "42 123 7" --ants 200 --out-dir /tmp/run --skip-install

REPO_URL=""
BRANCH=""
WORKDIR="/content/work"
TICKS=5000
SEEDS="42 123 7"
ANTS=200
OUT_DIR=""
CONFIG="colony_config.yaml"
MAX_FOOD_DROP="0.20"
MAX_DEATH_INCREASE="0.20"
INSTALL_DEPS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)
      REPO_URL="$2"; shift 2 ;;
    --branch)
      BRANCH="$2"; shift 2 ;;
    --workdir)
      WORKDIR="$2"; shift 2 ;;
    --ticks)
      TICKS="$2"; shift 2 ;;
    --seeds)
      SEEDS="$2"; shift 2 ;;
    --ants)
      ANTS="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --config)
      CONFIG="$2"; shift 2 ;;
    --max-food-drop)
      MAX_FOOD_DROP="$2"; shift 2 ;;
    --max-death-increase)
      MAX_DEATH_INCREASE="$2"; shift 2 ;;
    --skip-install)
      INSTALL_DEPS=0; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

if [[ -n "$REPO_URL" ]]; then
  mkdir -p "$WORKDIR"
  REPO_DIR="$WORKDIR/repo"
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone "$REPO_URL" "$REPO_DIR"
  fi
  cd "$REPO_DIR"
  if [[ -n "$BRANCH" ]]; then
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH"
  fi
fi

if [[ -z "$OUT_DIR" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="reports/colab_runs/$TS"
fi

mkdir -p "$OUT_DIR"

# Install dependencies if running in a fresh environment.
if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  python3 -m pip install -q -r requirements.txt
fi

python3 compare_brains.py \
  --brains nn torch_nn transformer torch_transformer \
  --ticks "$TICKS" \
  --seeds $SEEDS \
  --ants "$ANTS" \
  --config "$CONFIG" \
  --out "$OUT_DIR/migration.json"

python3 scripts/check_torch_migration.py \
  --input "$OUT_DIR/migration.json" \
  --max-food-drop "$MAX_FOOD_DROP" \
  --max-death-increase "$MAX_DEATH_INCREASE" \
  --output "$OUT_DIR/migration_check.json"

python3 - "$OUT_DIR" "$TICKS" "$SEEDS" "$ANTS" "$MAX_FOOD_DROP" "$MAX_DEATH_INCREASE" <<'PY'
import json
import pathlib
import subprocess
import sys

out_dir = pathlib.Path(sys.argv[1])
meta = {
    "ticks": int(sys.argv[2]),
    "seeds": [int(x) for x in sys.argv[3].split()],
    "ants": int(sys.argv[4]),
    "thresholds": {
        "max_food_drop": float(sys.argv[5]),
        "max_death_increase": float(sys.argv[6]),
    },
}

try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    meta["git_commit"] = sha
except Exception:
    meta["git_commit"] = None

try:
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    meta["git_branch"] = branch
except Exception:
    meta["git_branch"] = None

try:
    import torch
    meta["torch"] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "device": "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"),
    }
except Exception as e:
    meta["torch"] = {"error": str(e)}

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
PY

echo
echo "Artifacts saved to: $OUT_DIR"
echo "  - $OUT_DIR/migration.json"
echo "  - $OUT_DIR/migration_check.json"
echo "  - $OUT_DIR/run_meta.json"
