#!/usr/bin/env bash
# ============================================================================
# Stage 1 — page layout + instrument metadata for a folder of score pages.
#
#   bash run_stage1.sh --images ../data/sample/images --sheet Bee_1_challenge \
#                      --work ../work [--backend qwen3]
#
# The numbered scripts (from the original pipeline) each process ONE page folder whose image is
# named after the folder, so pages are fanned out into work/pages/<page>/. The two MLLM steps are
# the expensive ones, so they run ONCE over all pages (the model is loaded a single time) and their
# outputs are fanned back in under the exact filenames the GPT versions produced, which lets the
# downstream steps run unmodified.
#
#   steps 1..5  per page   OCR, YOLO + oemer staff detection, OCR filtering
#   step 6      batched    OCR token classification      (MLLM)
#   step 8      batched    staff group / ensemble info   (MLLM)
#   steps b,7,f per page   CSV assembly, matching, post-processing
# ============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY_STAGE1:-python}"
IMAGES=""; SHEET="Bee_1_challenge"; WORK=""; BACKEND="qwen3"; RES="api"

while [ $# -gt 0 ]; do
  case "$1" in
    --images) IMAGES="$2"; shift 2;;
    --sheet) SHEET="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --backend) BACKEND="$2"; shift 2;;
    --res) RES="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 1;;
  esac
done
[ -n "$IMAGES" ] && [ -n "$WORK" ] || { echo "usage: $0 --images DIR --work DIR [--sheet NAME] [--backend qwen3|internvl3_5|gpt]"; exit 1; }
mkdir -p "$WORK/pages" "$WORK/metadata"

echo "=== stage 1: fan out pages ==="
STEMS=()
for img in "$IMAGES"/*.png "$IMAGES"/*.jpg "$IMAGES"/*.jpeg; do
  [ -e "$img" ] || continue
  stem="$(basename "${img%.*}")"; ext="${img##*.}"
  mkdir -p "$WORK/pages/$stem"
  [ -e "$WORK/pages/$stem/$stem.$ext" ] || cp "$img" "$WORK/pages/$stem/$stem.$ext"
  STEMS+=("$stem")
done
echo "  ${#STEMS[@]} pages"

for stem in "${STEMS[@]}"; do
  f="$WORK/pages/$stem"
  echo "--- $stem: OCR + staff detection (steps 1-5) ---"
  $PY "$DIR/1_doctr_full_page_process.py" "$f"
  $PY "$DIR/2_staff_box.py"               "$f"
  $PY "$DIR/2_1_staff_box_plot.py"        "$f"
  $PY "$DIR/2_cnt_staff_omr.py"           "$f"
  $PY "$DIR/3_count_staff2json.py"        "$f"
  $PY "$DIR/3_check_staff_cnt_pk.py"      "$f"
  $PY "$DIR/4_ocr_filter.py"              "$f"
  $PY "$DIR/5_ocr_filtered_plot.py"       "$f"
done

echo "=== steps 6 + 8: MLLM ($BACKEND) ==="
if [ "$BACKEND" = "gpt" ]; then
  for stem in "${STEMS[@]}"; do
    $PY "$DIR/6_1_gpt_classify_reformat.py" "$WORK/pages/$stem"
    $PY "$DIR/8_gpt_staffinfo.py"           "$WORK/pages/$stem"
  done
else
  IN="$WORK/mllm_in"; OUT="$WORK/mllm_out"
  mkdir -p "$IN/$SHEET" "$IN/${SHEET}_ocr_filtered"
  for stem in "${STEMS[@]}"; do
    cp -n "$WORK/pages/$stem/$stem".png "$IN/$SHEET/" 2>/dev/null || true
    cp -n "$WORK/pages/$stem/${stem}_ocr_filtered.json" "$IN/${SHEET}_ocr_filtered/" 2>/dev/null || true
  done
  $PY "$DIR/mllm/classify_tokens_vlm.py" --task "$SHEET" --model "$BACKEND" --res "$RES" \
      --dataset-root "$IN" --out-root "$OUT" --resume
  $PY "$DIR/mllm/run_staffinfo_vlm.py"   --task "$SHEET" --model "$BACKEND" --res "$RES" \
      --dataset-root "$IN" --out-root "$OUT" --resume
  echo "--- fan MLLM output back into the page folders ---"
  for stem in "${STEMS[@]}"; do
    find "$OUT" -name "${stem}_ocr_filtered_classified_normalized.json" -exec cp {} "$WORK/pages/$stem/" \;
    find "$OUT" -name "${stem}_staffgroup.json"                         -exec cp {} "$WORK/pages/$stem/" \;
  done
fi

echo "=== steps b, 7, f: assemble the per-page CSV ==="
for stem in "${STEMS[@]}"; do
  f="$WORK/pages/$stem"
  $PY "$DIR/b_trans_gpt.py"                "$f"
  $PY "$DIR/7_matching_reformat_standard.py" "$f"
  $PY "$DIR/f.py"                          "$f"
  cp "$f/${stem}_trans_gpt_final.csv" "$WORK/metadata/" 2>/dev/null || echo "  WARNING: no final CSV for $stem"
done

echo "=== stage 1 done: $(ls "$WORK/metadata" | wc -l) CSVs in $WORK/metadata ==="
