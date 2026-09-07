#!/bin/bash
# fix-glm53-kda-bf16: build the KDA fused input projection (in_proj_qkvgfab)
# unquantized. Weight-only NVFP4 checkpoints ship the KDA projections as raw
# BF16. See README.md in this directory.

set -euo pipefail

MOD_DIR="$(dirname "$0")"
TARGET_DIR="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn"
TARGET="$TARGET_DIR/kimi_gdn_linear_attn.py"
MARKER="fix-glm53-kda-bf16"

echo "[fix-glm53-kda-bf16] Patching $TARGET..."
if [ ! -f "$TARGET" ]; then
    echo "[fix-glm53-kda-bf16] ERROR: target not found: $TARGET"
    exit 1
fi

if grep -q "$MARKER" "$TARGET"; then
    echo "[fix-glm53-kda-bf16] Already applied; nothing to do."
    exit 0
fi

tr -d '\r' < "$MOD_DIR/fix-glm53-kda-inproj-unquantized.patch" > /tmp/fix-kda-inproj.patch

if patch --forward --batch -p0 -d "$TARGET_DIR" < /tmp/fix-kda-inproj.patch; then
    if grep -q "$MARKER" "$TARGET"; then
        echo "[fix-glm53-kda-bf16] Done."
    else
        echo "[fix-glm53-kda-bf16] ERROR: patch applied but marker missing; refusing to continue."
        exit 1
    fi
else
    echo "[fix-glm53-kda-bf16] ERROR: patch did not apply (model revision drift?)."
    rm -f /tmp/fix-kda-inproj.patch
    exit 1
fi
rm -f /tmp/fix-kda-inproj.patch
