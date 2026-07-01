"""Shared fixtures for the Qwen3-30B-A3B bf16 decoder-layer HIR description.

Pins the value oracle and the model contract that every component test in
this package shares:

- the Hugging Face reference (``transformers`` ``Qwen3MoeDecoderLayer`` built
  from a ``Qwen3MoeConfig`` at the Qwen3-30B-A3B dimensions, random weights at
  a fixed seed),
- the model dimensions (GQA 32 query / 4 key-value heads, 128-expert top-8
  MoE),
- the dynamic sequence / context symbols, and
- the component -> HF-submodule map.

Component HIR ``@func`` constructors and the HF weight / KV-cache mapping
helpers also live here so each component test file composes them rather than
duplicating the description.
"""
from __future__ import annotations

from tilefoundry import func
from tilefoundry.dsl import DimVar, ReduceKind, Tensor, tf

# ── Qwen3-30B-A3B dimensions (config.json) ──────────────────────────────
HIDDEN = 2048
HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS    # 8 query heads share one kv head
Q_PROJ = NUM_Q_HEADS * HEAD_DIM            # 4096
KV_PROJ = NUM_KV_HEADS * HEAD_DIM          # 512
QKV_PROJ = Q_PROJ + 2 * KV_PROJ            # 5120
NUM_EXPERTS = 128
TOP_K = 8
MOE_INTERMEDIATE = 768
INTERMEDIATE = 6144
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0
NORM_TOPK_PROB = True
ATTENTION_BIAS = False
VOCAB = 151936
MAX_POS = 40960

# Decode-step static contract: a fixed seq tile and a fixed cache capacity. The
# production capacity is MAX_POS; the decode oracle uses a small static capacity
# for test speed (the op semantics are capacity-agnostic).
S_CAP = 4
CACHE_CAP = 16

# ── Dynamic symbols ─────────────────────────────────────────────────────
# ``S`` (seq_len) and ``C`` (context_len) are dynamic dims. ``cur_pos`` is a
# runtime scalar int — the RoPE position and the KV-cache slice write/read
# offset — passed at the eval boundary as a shape-(1,) i32 tensor, not a dim.
# The runtime relation is ``C = cur_pos + S``.
SEQ_LEN = DimVar(name="seq_len", lo=1, hi=MAX_POS)
CTX_LEN = DimVar(name="ctx_len", lo=1, hi=MAX_POS)
# Aliases used by the component @func annotations below: S = new tokens,
# P = prior cache length, C = full context length, ROPE_CACHE = cos/sin rows.
S = SEQ_LEN
P = CTX_LEN
C = DimVar(name="ctx_total", lo=1, hi=MAX_POS)
ROPE_CACHE = DimVar(name="rope_cache", lo=1, hi=MAX_POS)

# Token-count symbol for the MoE / residual @funcs. Defaults to the dynamic
# ``SEQ_LEN`` (prefill / component oracle). ``build_decode_layer`` rebinds it to
# the static ``S_CAP`` before building so the composed decode-step graph carries
# no ``DimVar`` (the parser resolves this annotation name from module globals).
SEQ_SYM = SEQ_LEN

# Reset before a build to retarget the component @func tensor dtypes; the
# parser resolves this annotation name from this module's globals.
DT = "bf16"

# ── Component -> HF submodule map ───────────────────────────────────────
# Each component's HIR is validated against these submodules of a single
# ``Qwen3MoeDecoderLayer``.
COMPONENT_HF_SUBMODULES = {
    "attention": ("input_layernorm", "self_attn"),
    "moe": ("post_attention_layernorm", "mlp"),
    "layer": (".",),
}


def build_hf_config():
    """Build the Qwen3-30B-A3B ``Qwen3MoeConfig`` (one decoder layer).

    ``decoder_sparse_step=1`` with empty ``mlp_only_layers`` makes layer 0 a
    sparse (MoE) layer, matching the deployed model.
    """
    from transformers import Qwen3MoeConfig  # noqa: PLC0415

    return Qwen3MoeConfig(
        hidden_size=HIDDEN,
        head_dim=HEAD_DIM,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_INTERMEDIATE,
        intermediate_size=INTERMEDIATE,
        norm_topk_prob=NORM_TOPK_PROB,
        attention_bias=ATTENTION_BIAS,
        rms_norm_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
        num_hidden_layers=1,
        vocab_size=VOCAB,
        max_position_embeddings=MAX_POS,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


def build_hf_layer(seed=0, device="cuda", dtype=None):
    """Build a ``Qwen3MoeDecoderLayer`` with random weights at a fixed seed."""
    import torch  # noqa: PLC0415
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # noqa: PLC0415
        Qwen3MoeDecoderLayer,
    )

    cfg = build_hf_config()
    torch.manual_seed(seed)
    layer = Qwen3MoeDecoderLayer(cfg, layer_idx=0).to(device).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0.0, 0.05)
    if dtype is not None:
        layer = layer.to(dtype)
    return layer


def rope_caches(cfg, max_pos, device="cuda", dtype=None):
    """Full cos / sin caches [max_pos, head_dim] from the HF rotary embedding.

    Row ``p`` is the rotary embedding for absolute position ``p``, so gathering
    by ``pos_ids`` reproduces the cos / sin the HF attention applies.
    """
    import torch  # noqa: PLC0415
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # noqa: PLC0415
        Qwen3MoeRotaryEmbedding,
    )

    rotary = Qwen3MoeRotaryEmbedding(cfg).to(device)
    position_ids = torch.arange(max_pos, device=device).unsqueeze(0)
    ref = torch.zeros(1, max_pos, cfg.hidden_size, device=device)
    cos, sin = rotary(ref, position_ids)
    cos, sin = cos[0], sin[0]
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def additive_causal_mask(seq, cur_pos, total_ctx, device="cuda", dtype=None):
    """Additive attention mask [1, 1, seq, total_ctx]: 0 where a query at
    absolute position ``cur_pos + i`` may attend key ``j`` (``j <= cur_pos+i``),
    ``-inf`` otherwise."""
    import torch  # noqa: PLC0415

    q_pos = torch.arange(cur_pos, cur_pos + seq, device=device).unsqueeze(1)
    k_pos = torch.arange(total_ctx, device=device).unsqueeze(0)
    mask = torch.where(k_pos <= q_pos, 0.0, float("-inf"))
    if dtype is not None:
        mask = mask.to(dtype)
    return mask.view(1, 1, seq, total_ctx)


# ── Component HIR @func constructors ────────────────────────────────────
# Built at the current ``DT``. The attention math is two @funcs (a compound
# ``DimVar`` axis from a ``concat`` cannot feed ``matmul``), so the KV append
# and the score computation are validated separately and chained in Python.


def build_kv_update():
    """Full cache after key RMSNorm + RoPE + append: ``(k_full, v_full)`` of
    shape ``[1, prior+new, kv_heads, head_dim]`` (returned, not matmul'd)."""

    @func
    def attn_kv_update(
        hidden: Tensor[(1, S, HIDDEN), DT],
        gamma_in: Tensor[(HIDDEN,), DT],
        w_k: Tensor[(1, HIDDEN, KV_PROJ), DT],
        w_v: Tensor[(1, HIDDEN, KV_PROJ), DT],
        gamma_k: Tensor[(HEAD_DIM,), DT],
        cos_cache: Tensor[(ROPE_CACHE, HEAD_DIM), DT],
        sin_cache: Tensor[(ROPE_CACHE, HEAD_DIM), DT],
        pos_ids: Tensor[(S,), "i32"],
        k_cache_prev: Tensor[(1, P, NUM_KV_HEADS, HEAD_DIM), DT],
        v_cache_prev: Tensor[(1, P, NUM_KV_HEADS, HEAD_DIM), DT],
    ):
        hidden_norm = tf.rms_norm(hidden, gamma_in)
        k = tf.reshape(tf.matmul(hidden_norm, w_k), new_shape=(1, S, NUM_KV_HEADS, HEAD_DIM))
        v = tf.reshape(tf.matmul(hidden_norm, w_v), new_shape=(1, S, NUM_KV_HEADS, HEAD_DIM))
        k_norm = tf.rms_norm(k, gamma_k)
        _, k_rope = tf.rope(k_norm, k_norm, cos_cache, sin_cache, pos_ids)
        k_full = tf.concat(k_cache_prev, k_rope, axis=1)
        v_full = tf.concat(v_cache_prev, v, axis=1)
        return (k_full, v_full)

    return attn_kv_update


def build_scores():
    """GQA attention over a full cache of a single ``DimVar`` context length."""

    @func
    def attn_scores(
        hidden: Tensor[(1, S, HIDDEN), DT],
        gamma_in: Tensor[(HIDDEN,), DT],
        w_q: Tensor[(1, HIDDEN, Q_PROJ), DT],
        gamma_q: Tensor[(HEAD_DIM,), DT],
        cos_cache: Tensor[(ROPE_CACHE, HEAD_DIM), DT],
        sin_cache: Tensor[(ROPE_CACHE, HEAD_DIM), DT],
        pos_ids: Tensor[(S,), "i32"],
        k_full: Tensor[(1, C, NUM_KV_HEADS, HEAD_DIM), DT],
        v_full: Tensor[(1, C, NUM_KV_HEADS, HEAD_DIM), DT],
        attn_mask: Tensor[(1, 1, S, C), DT],
        scale: Tensor[(1, 1, 1, 1), DT],
        w_o: Tensor[(1, Q_PROJ, HIDDEN), DT],
    ) -> Tensor[(1, S, HIDDEN), DT]:
        hidden_norm = tf.rms_norm(hidden, gamma_in)
        q = tf.reshape(tf.matmul(hidden_norm, w_q), new_shape=(1, S, NUM_Q_HEADS, HEAD_DIM))
        q_norm = tf.rms_norm(q, gamma_q)
        q_rope, _ = tf.rope(q_norm, q_norm, cos_cache, sin_cache, pos_ids)

        k_b = tf.repeat_interleave(k_full, repeats=GQA_GROUP, axis=2)
        v_b = tf.repeat_interleave(v_full, repeats=GQA_GROUP, axis=2)
        q_h = tf.transpose(q_rope, perm=(0, 2, 1, 3))
        k_h = tf.transpose(k_b, perm=(0, 2, 1, 3))
        v_h = tf.transpose(v_b, perm=(0, 2, 1, 3))
        q_s = tf.mul(q_h, scale)
        k_t = tf.transpose(k_h, perm=(0, 1, 3, 2))
        scores = tf.add(tf.matmul(q_s, k_t), attn_mask)
        probs = tf.softmax(scores, axis=-1)
        ctx = tf.matmul(probs, v_h)
        attn_out = tf.transpose(ctx, perm=(0, 2, 1, 3))
        return tf.matmul(tf.reshape(attn_out, new_shape=(1, S, Q_PROJ)), w_o)

    return attn_scores


def build_moe():
    """Top-8 MoE over a full token batch with runtime expert routing."""

    @func
    def moe_component(
        hidden: Tensor[(1, SEQ_SYM, HIDDEN), DT],
        gamma_post: Tensor[(HIDDEN,), DT],
        w_router: Tensor[(HIDDEN, NUM_EXPERTS), DT],
        w_gate: Tensor[(NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN), DT],
        w_up: Tensor[(NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN), DT],
        w_down: Tensor[(NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE), DT],
    ) -> Tensor[(1, SEQ_SYM, HIDDEN), DT]:
        hidden_norm = tf.rms_norm(hidden, gamma_post)
        tokens = tf.reshape(hidden_norm, new_shape=(SEQ_SYM, HIDDEN))

        # Router in f32 to match the HF softmax/topk dtype (selection must agree).
        logits = tf.cast(tf.matmul(tokens, w_router), dtype="f32")
        probs = tf.softmax(logits, axis=-1)
        top_vals, indices = tf.topk(probs, k=TOP_K, axis=-1)
        denom = tf.reduce(top_vals, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
        weights = tf.cast(tf.div(top_vals, denom), dtype=DT)

        # Runtime gather of the selected experts; batched matmul over [tokens, top_k].
        gu_g = tf.gather(w_gate, indices, axis=0)
        gu_u = tf.gather(w_up, indices, axis=0)
        dn = tf.gather(w_down, indices, axis=0)
        tok4 = tf.reshape(tokens, new_shape=(SEQ_SYM, 1, HIDDEN, 1))
        gate = tf.reshape(tf.matmul(gu_g, tok4), new_shape=(SEQ_SYM, TOP_K, MOE_INTERMEDIATE))
        up = tf.reshape(tf.matmul(gu_u, tok4), new_shape=(SEQ_SYM, TOP_K, MOE_INTERMEDIATE))
        act = tf.mul(gate, tf.sigmoid(gate))
        h = tf.mul(act, up)
        h4 = tf.reshape(h, new_shape=(SEQ_SYM, TOP_K, MOE_INTERMEDIATE, 1))
        down = tf.reshape(tf.matmul(dn, h4), new_shape=(SEQ_SYM, TOP_K, HIDDEN))
        weighted = tf.mul(down, tf.reshape(weights, new_shape=(SEQ_SYM, TOP_K, 1)))
        out = tf.reduce(weighted, axes=(1,), keepdim=False, kind=ReduceKind.SUM)
        return tf.reshape(out, new_shape=(1, SEQ_SYM, HIDDEN))

    return moe_component


def build_residual():
    """Layer-level residual add ``a + b`` as an HIR op."""

    @func
    def residual_add(
        a: Tensor[(1, SEQ_SYM, HIDDEN), DT],
        b: Tensor[(1, SEQ_SYM, HIDDEN), DT],
    ) -> Tensor[(1, SEQ_SYM, HIDDEN), DT]:
        return tf.add(a, b)

    return residual_add


def decode_attn_mask(cur_pos, s, device="cuda", dtype=None):
    """Additive decode mask [1, 1, S_CAP, CACHE_CAP].

    Active row ``i < s`` (token at absolute position ``cur_pos + i``) may attend
    keys ``0 .. cur_pos + i`` (causal over the filled context). Inactive padding
    rows ``i >= s`` keep a single safe visible key (column 0) so ``softmax``
    never sees an all-``-inf`` row; those rows are discarded downstream.
    """
    import torch  # noqa: PLC0415

    mask = torch.full((S_CAP, CACHE_CAP), float("-inf"), device=device)
    for i in range(S_CAP):
        if i < s:
            mask[i, : cur_pos + i + 1] = 0.0
        else:
            mask[i, 0] = 0.0
    if dtype is not None:
        mask = mask.to(dtype)
    return mask.view(1, 1, S_CAP, CACHE_CAP)


def build_decode_attention():
    """Single-Function decode-step attention over a fixed-capacity KV cache.

    Writes the new K/V via ``cache_update`` and reads the full ``CACHE_CAP``
    cache + mask; returns ``(out, k_cache1, v_cache1)``. All shapes static.
    """

    @func
    def decode_attention(
        hidden: Tensor[(1, S_CAP, HIDDEN), DT],
        gamma_in: Tensor[(HIDDEN,), DT],
        w_q: Tensor[(1, HIDDEN, Q_PROJ), DT],
        w_k: Tensor[(1, HIDDEN, KV_PROJ), DT],
        w_v: Tensor[(1, HIDDEN, KV_PROJ), DT],
        gamma_q: Tensor[(HEAD_DIM,), DT],
        gamma_k: Tensor[(HEAD_DIM,), DT],
        cos_cache: Tensor[(CACHE_CAP, HEAD_DIM), DT],
        sin_cache: Tensor[(CACHE_CAP, HEAD_DIM), DT],
        pos_ids: Tensor[(S_CAP,), "i32"],
        k_cache0: Tensor[(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM), DT],
        v_cache0: Tensor[(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM), DT],
        cur_pos: Tensor[(1,), "i32"],
        s: Tensor[(1,), "i32"],
        attn_mask: Tensor[(1, 1, S_CAP, CACHE_CAP), DT],
        scale: Tensor[(1, 1, 1, 1), DT],
        w_o: Tensor[(1, Q_PROJ, HIDDEN), DT],
    ):
        hidden_norm = tf.rms_norm(hidden, gamma_in)
        q = tf.rms_norm(
            tf.reshape(tf.matmul(hidden_norm, w_q), new_shape=(1, S_CAP, NUM_Q_HEADS, HEAD_DIM)),
            gamma_q,
        )
        k = tf.rms_norm(
            tf.reshape(tf.matmul(hidden_norm, w_k), new_shape=(1, S_CAP, NUM_KV_HEADS, HEAD_DIM)),
            gamma_k,
        )
        v = tf.reshape(tf.matmul(hidden_norm, w_v), new_shape=(1, S_CAP, NUM_KV_HEADS, HEAD_DIM))
        q_rope, _ = tf.rope(q, q, cos_cache, sin_cache, pos_ids)
        _, k_rope = tf.rope(k, k, cos_cache, sin_cache, pos_ids)
        k_cache1 = tf.cache_update(k_cache0, cur_pos, s, k_rope)
        v_cache1 = tf.cache_update(v_cache0, cur_pos, s, v)

        k_b = tf.repeat_interleave(k_cache1, repeats=GQA_GROUP, axis=2)
        v_b = tf.repeat_interleave(v_cache1, repeats=GQA_GROUP, axis=2)
        q_h = tf.transpose(q_rope, perm=(0, 2, 1, 3))
        k_h = tf.transpose(k_b, perm=(0, 2, 1, 3))
        v_h = tf.transpose(v_b, perm=(0, 2, 1, 3))
        q_s = tf.mul(q_h, scale)
        k_t = tf.transpose(k_h, perm=(0, 1, 3, 2))
        scores = tf.add(tf.matmul(q_s, k_t), attn_mask)
        probs = tf.softmax(scores, axis=-1)
        ctx = tf.matmul(probs, v_h)
        attn_out = tf.transpose(ctx, perm=(0, 2, 1, 3))
        out = tf.matmul(tf.reshape(attn_out, new_shape=(1, S_CAP, Q_PROJ)), w_o)
        return (out, k_cache1, v_cache1)

    return decode_attention


def build_decode_layer():
    """Single-Function full decode-step layer over a fixed-capacity KV cache.

    Composes the whole layer in one HIR ``@func``: ``decode_attention`` (RMSNorm
    + QKV + RoPE + ``cache_update`` write + full-cache read), the attention
    residual, the static-``S_CAP`` MoE (post-attention norm + top-8 experts), and
    the MoE residual. Returns ``(out, k_cache1, v_cache1)`` — the layer output
    plus the updated KV cache — so the decode state transition is one Function.

    The two component @funcs are nested call targets (closure-captured
    ``hir.Function`` values); the MoE is built with ``SEQ_SYM`` rebound to the
    static ``S_CAP`` so the composed graph carries no ``DimVar``.
    """
    global SEQ_SYM

    decode_attention = build_decode_attention()
    SEQ_SYM = S_CAP
    try:
        moe_component = build_moe()
    finally:
        SEQ_SYM = SEQ_LEN

    @func
    def decode_layer(
        hidden: Tensor[(1, S_CAP, HIDDEN), DT],
        gamma_in: Tensor[(HIDDEN,), DT],
        w_q: Tensor[(1, HIDDEN, Q_PROJ), DT],
        w_k: Tensor[(1, HIDDEN, KV_PROJ), DT],
        w_v: Tensor[(1, HIDDEN, KV_PROJ), DT],
        gamma_q: Tensor[(HEAD_DIM,), DT],
        gamma_k: Tensor[(HEAD_DIM,), DT],
        cos_cache: Tensor[(CACHE_CAP, HEAD_DIM), DT],
        sin_cache: Tensor[(CACHE_CAP, HEAD_DIM), DT],
        pos_ids: Tensor[(S_CAP,), "i32"],
        k_cache0: Tensor[(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM), DT],
        v_cache0: Tensor[(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM), DT],
        cur_pos: Tensor[(1,), "i32"],
        s: Tensor[(1,), "i32"],
        attn_mask: Tensor[(1, 1, S_CAP, CACHE_CAP), DT],
        scale: Tensor[(1, 1, 1, 1), DT],
        w_o: Tensor[(1, Q_PROJ, HIDDEN), DT],
        gamma_post: Tensor[(HIDDEN,), DT],
        w_router: Tensor[(HIDDEN, NUM_EXPERTS), DT],
        w_gate: Tensor[(NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN), DT],
        w_up: Tensor[(NUM_EXPERTS, MOE_INTERMEDIATE, HIDDEN), DT],
        w_down: Tensor[(NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE), DT],
    ):
        attn_out, k_cache1, v_cache1 = decode_attention(
            hidden, gamma_in, w_q, w_k, w_v, gamma_q, gamma_k,
            cos_cache, sin_cache, pos_ids, k_cache0, v_cache0,
            cur_pos, s, attn_mask, scale, w_o,
        )
        h1 = tf.add(hidden, attn_out)
        moe_out = moe_component(h1, gamma_post, w_router, w_gate, w_up, w_down)
        out = tf.add(h1, moe_out)
        return (out, k_cache1, v_cache1)

    return decode_layer
