# fix-glm53-kda-bf16

Builds the KDA fused input projection (`in_proj_qkvgfab`) unquantized in
`vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`.

`LibertAIDAI/GLM-5.3-Flash-NVFP4` is a weight-only NVFP4 checkpoint:
routed experts are quantized, the KDA projections ship as raw BF16. Its
`quantization_config.ignore` globs name the fused projection by its
checkpoint-side name (`in_proj_qkvbfg_a`), which never matches the vLLM
module (`in_proj_qkvgfab`). Without this mod the projection is created
with FP4-packed slots and the raw BF16 load asserts in
`parameter.py:load_merged_column_weight`.

The patch is anchored on the b12x image source at commit `2a979314d`
(`v0.1.dev20596+g2a979314d.d20260907`). The runner is idempotent (marker
check) and fails loudly if the target drifts from that revision —
re-derive the patch rather than force it. If a weight load still fails,
re-run with `mods/diag-glm53-kda-load`; its log line identifies the
failing tensor.
