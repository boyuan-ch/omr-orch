#!/usr/bin/env bash
# ============================================================================
# End-to-end demo on the 10 sample pages (Beethoven Symphony No. 1, pages 1-10).
#
#   bash scripts/run_sample.sh                 # full pipeline, Qwen3-VL-8B for the MLLM steps
#   bash scripts/run_sample.sh --skip-stage1   # start from ground-truth metadata you supply
#   bash scripts/run_sample.sh --skip-stage1 --skip-stage2   # only re-run evaluation
#                                              # (no GPU / no MLLM weights needed)
#
# The three stages need different Python environments (see README "Requirements"); point these at
# them, e.g.  PY_STAGE1=~/miniconda3/envs/metadata/bin/python bash scripts/run_sample.sh
#   PY_STAGE1  OCR + detection + MLLM        PY_STAGE2  OMR backbone        PY_EVAL  evaluation
# ============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_STAGE1="${PY_STAGE1:-python}"; PY_STAGE2="${PY_STAGE2:-python}"; PY_EVAL="${PY_EVAL:-python}"
SHEET="Bee_1_challenge"; PAGES="1-10"; BACKEND="qwen3"; SKIP1=0; SKIP2=0
WORK="$DIR/work"

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-stage1) SKIP1=1; shift;;
    --skip-stage2) SKIP2=1; shift;;
    --backend) BACKEND="$2"; shift 2;;
    --pages) PAGES="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 1;;
  esac
done
mkdir -p "$WORK"

if [ ! -f "$DIR/stage2_omr/omr/checkpoints/unet_big/model.onnx" ]; then
  echo "!! segmentation checkpoint missing -- run: bash scripts/fetch_weights.sh"; exit 1
fi

IMG_DIR="$DIR/data/sample/images"
if [ -z "$(ls "$IMG_DIR"/*.png 2>/dev/null)" ]; then
  echo "!! no page images in data/sample/images/"
  echo "   This repository ships no score images and no ground truth -- they come from"
  echo "   third-party editions we cannot redistribute. See data/README.md for how to"
  echo "   obtain them and the layout they go in."
  exit 1
fi

GT_META="$DIR/data/sample/gt/metadata"
if [ "$SKIP1" -eq 1 ] && [ "$SKIP2" -eq 0 ] && [ -z "$(ls "$GT_META"/*.csv 2>/dev/null)" ]; then
  echo "!! --skip-stage1 needs ground-truth metadata in data/sample/gt/metadata/ to feed stage 2."
  echo "   See data/README.md section 3. Drop --skip-stage1 to generate it with the MLLM instead."
  exit 1
fi

if [ "$SKIP1" -eq 0 ]; then
  echo "##### STAGE 1 — layout metadata (backend: $BACKEND) #####"
  PY_STAGE1="$PY_STAGE1" bash "$DIR/stage1_layout/run_stage1.sh" \
      --images "$DIR/data/sample/images" --sheet "$SHEET" --work "$WORK" --backend "$BACKEND"
  META="$WORK/metadata"
else
  if [ "$SKIP2" -eq 0 ]; then
    echo "##### STAGE 1 skipped — using the ground-truth metadata in data/sample/gt/metadata/ #####"
  else
    echo "##### STAGE 1 skipped #####"
  fi
  META="$DIR/data/sample/gt/metadata"
fi

if [ "$SKIP2" -eq 0 ]; then
  echo "##### BRIDGE — metadata + images -> the layout stage 2 expects #####"
  "$PY_STAGE2" "$DIR/scripts/bridge_metadata_to_omr.py" \
      --metadata-dir "$META" --images-dir "$DIR/data/sample/images" \
      --sheet "$SHEET" --out "$WORK/orch_dataset"
fi

if [ "$SKIP2" -eq 0 ]; then
  echo "##### STAGE 2 — OMR: page image + metadata -> MusicXML #####"
  ( cd "$DIR/stage2_omr" && "$PY_STAGE2" run_omr.py \
      --sheet "$SHEET" --pages "$PAGES" --data-root "$WORK/orch_dataset" )
else
  echo "##### STAGE 2 skipped — reusing the MusicXML already in work/ #####"
fi
PRED="$WORK/orch_dataset/$SHEET/xmls"
# `ls` on a missing directory fails, and `set -o pipefail` would take the whole script down
N_PRED=0
[ -d "$PRED" ] && N_PRED=$(find "$PRED" -maxdepth 1 -name '*.musicxml' | wc -l)
echo "  predictions: $N_PRED MusicXML files"
if [ "$N_PRED" -eq 0 ]; then
  echo "!! nothing to score. Re-run without --skip-stage2 to produce the MusicXML first."
  exit 1
fi

echo "##### STAGE 3 — evaluation #####"
if [ "$SKIP1" -eq 0 ]; then
  if [ -n "$(ls "$DIR/data/sample/gt/metadata"/*.csv 2>/dev/null)" ]; then
  echo "--- metadata metrics (Table 1 style) ---"
  "$PY_EVAL" "$DIR/stage3_eval/metadata_metrics.py" \
      --pred-dir "$WORK/orch_dataset/$SHEET/csv" --gt-dir "$DIR/data/sample/gt/metadata" \
      --sheet "$SHEET" --out "$WORK/metadata_metrics.json" || true
  else
    echo "--- metadata metrics skipped: no reference CSVs in data/sample/gt/metadata/ ---"
  fi
else
  echo "--- metadata metrics skipped: stage 1 was not run, so the metadata IS the ground truth ---"
fi

GT_XML="$DIR/data/sample/gt/musicxml"
if [ -z "$(ls "$GT_XML"/*.musicxml 2>/dev/null)" ]; then
  echo "--- TEDn and OMR-NED skipped: no reference MusicXML in data/sample/gt/musicxml/ ---"
  echo "    The MusicXML predictions are in $PRED and can be scored later;"
  echo "    see data/README.md section 2 for how to obtain the reference."
else
  echo "--- TEDn ---"
  ( cd "$DIR/stage3_eval/tedn" && \
    "$PY_EVAL" make_manifest.py --gt-dir "$GT_XML" --pred-dir "$PRED" \
          --sheet "$SHEET" --out "$WORK/tedn_manifest.json" && \
    "$PY_EVAL" run_all.py --manifest "$WORK/tedn_manifest.json" --results "$WORK/tedn_results" \
          --summary-dir "$WORK/tedn_summary" --workers 4 )

  echo "--- OMR-NED ---"
  "$PY_EVAL" "$DIR/stage3_eval/omrned/run_omrned.py" \
      --gt-dir "$GT_XML" --pred-dir "$PRED" --out "$WORK/omrned.json"
fi

echo "##### DONE — results under $WORK #####"
