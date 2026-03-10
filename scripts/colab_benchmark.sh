#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   bash scripts/colab_benchmark.sh --mode torch_only --ticks 5000 --seeds "42 123 7" --ants 200 --out-dir /tmp/run --skip-install

REPO_URL=""
BRANCH=""
WORKDIR="/content/work"
MODE="full"
TICKS=5000
WARMUP_TICKS=0
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
    --mode)
      MODE="$2"; shift 2 ;;
    --ticks)
      TICKS="$2"; shift 2 ;;
    --warmup-ticks)
      WARMUP_TICKS="$2"; shift 2 ;;
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

if [[ "$MODE" != "full" && "$MODE" != "torch_only" ]]; then
  echo "--mode must be one of: full, torch_only" >&2
  exit 2
fi

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

if [[ "$MODE" == "torch_only" ]]; then
  BRAINS="torch_nn torch_transformer"
else
  BRAINS="nn torch_nn transformer torch_transformer"
fi

if [[ "$WARMUP_TICKS" -gt 0 ]]; then
  FIRST_SEED="$(echo "$SEEDS" | awk '{print $1}')"
  python3 compare_brains.py \
    --brains $BRAINS \
    --torch-batch-policy on \
    --ticks "$WARMUP_TICKS" \
    --seeds "$FIRST_SEED" \
    --ants "$ANTS" \
    --config "$CONFIG" \
    --quiet \
    --out "$OUT_DIR/warmup.json" >/dev/null
fi

python3 compare_brains.py \
  --brains $BRAINS \
  --torch-batch-policy on \
  --ticks "$TICKS" \
  --seeds $SEEDS \
  --ants "$ANTS" \
  --config "$CONFIG" \
  --out "$OUT_DIR/migration.json"

CHECK_ARGS=(
  --input "$OUT_DIR/migration.json"
  --max-food-drop "$MAX_FOOD_DROP"
  --max-death-increase "$MAX_DEATH_INCREASE"
  --output "$OUT_DIR/migration_check.json"
)
if [[ "$MODE" == "torch_only" ]]; then
  CHECK_ARGS+=(--allow-missing-pairs)
fi
python3 scripts/check_torch_migration.py "${CHECK_ARGS[@]}"

python3 - "$OUT_DIR" "$TICKS" "$SEEDS" "$ANTS" "$MAX_FOOD_DROP" "$MAX_DEATH_INCREASE" "$MODE" "$WARMUP_TICKS" <<'PY'
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
    "mode": sys.argv[7],
    "warmup_ticks": int(sys.argv[8]),
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

migration = json.loads((out_dir / "migration.json").read_text())
perf_summary = {
    "mode": meta["mode"],
    "ticks": meta["ticks"],
    "seeds": meta["seeds"],
    "brains": {},
}
for brain, reports in migration.items():
    if not reports:
        continue
    avg_tps = sum(float(r["performance"]["ticks_per_second"]) for r in reports) / len(reports)
    avg_wall = sum(float(r["performance"]["wall_time_seconds"]) for r in reports) / len(reports)
    perf_summary["brains"][brain] = {
        "avg_ticks_per_second": round(avg_tps, 3),
        "avg_wall_time_seconds": round(avg_wall, 3),
    }
(out_dir / "perf_summary.json").write_text(json.dumps(perf_summary, indent=2))
PY

echo
echo "Artifacts saved to: $OUT_DIR"
echo "  - $OUT_DIR/migration.json"
echo "  - $OUT_DIR/migration_check.json"
echo "  - $OUT_DIR/run_meta.json"
echo "  - $OUT_DIR/perf_summary.json"
