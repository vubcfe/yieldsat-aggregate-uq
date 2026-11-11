#!/bin/bash
# Full pipeline, from the raw YieldSAT folders to the per-seed summaries in results/.
#   RAW=/path/to/yieldsat/raw bash run_all.sh
# Optional: RUNS (output directory, default ./runs), PY (python), NTHREADS (LightGBM threads, default 16),
# NWORKERS (feature extraction processes, default 16), SEEDS (default 0,1,2).
# Every step skips (pair, seed) outputs that already exist, so the script can be re-run after an interruption.
set -euo pipefail
: "${RAW:?set RAW to the YieldSAT raw directory (<country>/<record>/...)}"
RUNS=${RUNS:-$PWD/runs}; PY=${PY:-python}; SEEDS=${SEEDS:-0,1,2}
export NTHREADS=${NTHREADS:-16}
S=$(cd "$(dirname "$0")" && pwd)/src
mkdir -p "$RUNS"
REC=$RUNS/e0/records.csv

# E0: physical fields, blocks of farms, pixel features, pseudo-region bags, reported yields
[ -f "$REC" ] || $PY -I "$S/e0_footprints.py" "$RAW" "$RUNS/e0"
$PY -I "$S/e0_features.py" "$RAW" "$RUNS/features" "${NWORKERS:-16}"
[ -f "$RUNS/e0/bags_l3.csv" ] || $PY -I "$S/e2_bags.py" "$RAW" "$REC" "$RUNS/e0/bags_l3.csv"
[ -f "$RUNS/e0/fields.csv" ] || $PY -I "$S/e6_fields.py" "$RAW" "$RUNS/e0/fields.csv"

# E1: dense-label oracle under three split protocols
for split in block physical_field field; do
  SPLIT=$split $PY -W ignore -I "$S/e1_oracle.py" "$RUNS/features" "$REC" "$RUNS/e1/$split" "$SEEDS"
done

# E2: learning from aggregate labels (needs the E1 block folds)
$PY -W ignore -I "$S/e2_aggregate.py" "$RUNS/features" "$REC" "$RUNS/e0/bags_l3.csv" "$RUNS/e1/block" "$RUNS/e2" "$SEEDS"

# E3: label-value curve and transfer between countries
$PY -W ignore -I "$S/e3_dense_k.py" "$RUNS/features" "$REC" "$RUNS/e3" "$SEEDS"

# E4/E5: multi-scale uncertainty, for the dense-label oracle and for four aggregate learners
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
$PY -W ignore -I "$S/e45_uq.py" "$RUNS/e1/block" "$REC" "$RUNS/e45/B0" pred "$SEEDS"
for col in L1_B11 L1_B8 L1_B2pv L2_B11; do
  E1_DIR=$RUNS/e1/block $PY -W ignore -I "$S/e45_uq.py" "$RUNS/e2" "$REC" "$RUNS/e45/$col" "$col" "$SEEDS"
done

# E6: farmer-reported yields as the aggregate
mkdir -p "$RUNS/e6"
$PY -W ignore -I "$S/e6_reported.py" "$RUNS/e1/block" "$REC" "$RUNS/e0/fields.csv" "$RUNS/e6/B0.json" pred

# Summaries, tables and figures
RUNS=$RUNS $PY "$S/../analysis/export_results.py"
$PY "$S/../analysis/make_tables.py"
$PY "$S/../analysis/make_figures.py"
