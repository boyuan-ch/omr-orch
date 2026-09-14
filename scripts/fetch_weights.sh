#!/usr/bin/env bash
# Download the one model file too large to ship in git.
#
# Everything else is already in the repo:
#   stage1_layout/weights/ola-layout-analysis-2.0-2025-03-09.pt   39 MB  staff/layout YOLO
#   stage2_omr/weights/best_time_signature.pt                     20 MB  time-signature YOLO
#   stage2_omr/training/*.pth                                     29 MB  clef/accidental/rest/stem CNNs
#
# This script fetches the oemer segmentation network (68 MB) from the upstream oemer release and
# installs it for both stages. Qwen weights are pulled from the Hugging Face hub on first use.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="https://github.com/BreezeWhite/oemer/releases/download/checkpoints"
# sha256 of both files verified identical to the checkpoints used for the paper
UNET_SHA="37512e858731096439746f60b377c049f07055b4a23ec6eb9a178ce92cfba174"
SEG_SHA="ed2e1a86ea75712ee6cdc740e96f7a36753543cf9bb980227c071c9256d9d82e"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

fetch () {   # fetch <remote name> <sha256> <local file>
  echo "downloading $1 ..."
  curl -fL --retry 3 -o "$TMP/$3" "$BASE/$1"
  echo "$2  $TMP/$3" | sha256sum -c - >/dev/null || { echo "checksum mismatch for $1"; exit 1; }
}
fetch 1st_model.onnx "$UNET_SHA" unet_big.onnx
fetch 2nd_model.onnx "$SEG_SHA"  seg_net.onnx

# stage 1 only counts staves and needs unet_big; stage 2 runs both segmentation passes
mkdir -p "$DIR/stage1_layout/omr/checkpoints/unet_big"
cp "$TMP/unet_big.onnx" "$DIR/stage1_layout/omr/checkpoints/unet_big/model.onnx"
echo "  installed -> stage1_layout/omr/checkpoints/unet_big/model.onnx"
for pair in "unet_big:unet_big.onnx" "seg_net:seg_net.onnx"; do
  name="${pair%%:*}"; file="${pair##*:}"
  mkdir -p "$DIR/stage2_omr/omr/checkpoints/$name"
  cp "$TMP/$file" "$DIR/stage2_omr/omr/checkpoints/$name/model.onnx"
  echo "  installed -> stage2_omr/omr/checkpoints/$name/model.onnx"
done
echo "done."
