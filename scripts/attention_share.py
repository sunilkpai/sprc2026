"""Per-token MACs per layer for real models: MLP (dense or MoE active experts),
attention projections, and the attention core, as a function of context length.

The dense idealisation "MLP = 8 d^2" is wrong for MoE models: the active MLP is
(k + shared) x 3 x d x h_expert, and MLA's absorbed score dimension is
(kv_lora + rope) per head rather than the head dim.  Configs are from the
published model cards; assumptions are flagged in ASSUME.

Run:  python3 scripts/attention_share.py
"""

M = 1e6
CTX = {"8k": 8192, "128k": 131072, "1M": 1048576}


def swiglu(d, h, n_active):
    """Three matrices (up, gate, down) per active expert."""
    return n_active * 3 * d * h


def mla_proj(d, heads, q_lora, kv_lora, rope, nope, vdim):
    """MLA projections per token: q_a, q_b, kv_a, kv_b, o."""
    return (d * q_lora + q_lora * heads * (nope + rope) + d * (kv_lora + rope)
            + kv_lora * heads * (nope + vdim) + heads * vdim * d)


def mla_core_per_key(heads, kv_lora, rope):
    """Absorbed-form MLA: scores against the (kv_lora + rope) latent, values from kv_lora."""
    return heads * (kv_lora + rope) + heads * kv_lora


def gqa_proj(d, heads_q, heads_kv, hd):
    return d * heads_q * hd + 2 * d * heads_kv * hd + heads_q * hd * d


def gqa_core_per_key(heads_q, hd):
    return 2 * heads_q * hd


MODELS = {}

# ---- Llama 3 70B, dense GQA, reference
d, hq, hkv, hd, h = 8192, 64, 8, 128, 28672
MODELS["Llama 3 70B (dense, GQA)"] = dict(
    mlp=swiglu(d, h, 1), proj=gqa_proj(d, hq, hkv, hd),
    core=lambda L: 0.5 * L * gqa_core_per_key(hq, hd),           # causal: L/2 keys on average
    note="dense causal attention; 0.5 L keys on average")

# ---- DeepSeek-V3 / V3.2 (d=7168, 58 of 61 layers MoE: 256 routed + 1 shared, k=8, h=2048; MLA 128 heads)
d, heads, q_lora, kv_lora, rope, nope, vdim = 7168, 128, 1536, 512, 64, 128, 128
mlp_v3 = swiglu(d, 2048, 9)
proj_v3 = mla_proj(d, heads, q_lora, kv_lora, rope, nope, vdim)
core_key = mla_core_per_key(heads, kv_lora, rope)                 # 139k MACs per key
MODELS["DeepSeek-V3 (MoE, MLA dense)"] = dict(
    mlp=mlp_v3, proj=proj_v3, core=lambda L: 0.5 * L * core_key,
    note="9 active experts x 3 x 7168 x 2048; MLA absorbed: 128 heads x 576 scores + 128 x 512 values per key")
# V3.2: DSA = lightning indexer (64 heads x 128, over all L) + sparse core over top-2048 keys
IDX = 64 * 128
MODELS["DeepSeek-V3.2 (MoE, DSA sparse)"] = dict(
    mlp=mlp_v3, proj=proj_v3,
    core=lambda L: 0.5 * L * IDX + min(2048, 0.5 * L) * core_key,
    note="indexer 64x128 per key over the context, plus MLA core over 2048 selected keys")

# ---- Kimi K3 (d=7168, 93 layers: 69 KDA + 24 gated MLA; 16 of 896 experts + 2 shared, h=3072; 96 heads; 1M)
d, heads = 7168, 96
mlp_k3 = swiglu(d, 3072, 18)
proj_k3 = mla_proj(d, heads, q_lora, kv_lora, rope, nope, vdim)   # ASSUME: DeepSeek MLA dims
core_key_k3 = mla_core_per_key(heads, kv_lora, rope)
KDA_PER_TOKEN = heads * 128 * 128 * 4                             # ASSUME: linear-attention state update, O(heads x hd^2)
f_mla = 24 / 93
MODELS["Kimi K3 (MoE, 69 KDA + 24 MLA, MLA sparse)"] = dict(
    mlp=mlp_k3, proj=(1 - f_mla) * 3 * d * heads * 128 + f_mla * proj_k3,
    core=lambda L: (1 - f_mla) * KDA_PER_TOKEN + f_mla * (0.5 * L * IDX + min(2048, 0.5 * L) * core_key_k3),
    note="ASSUME MLA layers use DSA-style selection and DeepSeek MLA dims; KDA layers cost O(1) in L")
MODELS["Kimi K3 (same, MLA dense)"] = dict(
    mlp=mlp_k3, proj=(1 - f_mla) * 3 * d * heads * 128 + f_mla * proj_k3,
    core=lambda L: (1 - f_mla) * KDA_PER_TOKEN + f_mla * 0.5 * L * core_key_k3,
    note="ASSUME the 24 MLA layers attend densely over the full context")

# ---- DeepSeek-V4 Pro: only the ratio to V3.2 at 1M is published (27% of single-token FLOPs, 10% KV cache)
v32 = MODELS["DeepSeek-V3.2 (MoE, DSA sparse)"]
tot_v32_1m = v32["mlp"] + v32["proj"] + v32["core"](CTX["1M"])
mlp_v4 = mlp_v3 * 49 / 37                                          # ASSUME active MLP scales with active params
MODELS["DeepSeek-V4 Pro (CSA + HCA hybrid, 1M)"] = dict(
    mlp=mlp_v4, proj=proj_v3,
    core=lambda L: (0.27 * tot_v32_1m - mlp_v4 - proj_v3) if L >= CTX["1M"] else float("nan"),
    note="ASSUME from the published 27%-of-V3.2 single-token FLOPs at 1M; attention = remainder after MLP and projections")


def main():
    print("Attention share of per-token MACs per layer (MLP share in parentheses)\n")
    print(f"{'model':46s}" + "".join(f"{c:>18s}" for c in CTX) + "   MLP MMACs")
    for name, m in MODELS.items():
        row = f"{name:46s}"
        for c, L in CTX.items():
            core = m["core"](L)
            if core != core:  # nan
                row += f"{'–':>18s}"; continue
            tot = m["mlp"] + m["proj"] + core
            row += f"{100 * core / tot:7.0f}% ({100 * m['mlp'] / tot:3.0f}%)   "
        print(row + f"{m['mlp'] / M:8.0f}")
    print("\nNotes:")
    for name, m in MODELS.items():
        print(f"  {name}: {m['note']}")
    print("\nRow form, model at context: attention / MLP / projections share of MACs per token per layer")
    for name, L in (("Llama 3 70B (dense, GQA)", "128k"), ("DeepSeek-V3.2 (MoE, DSA sparse)", "128k"),
                    ("Kimi K3 (MoE, 69 KDA + 24 MLA, MLA sparse)", "128k"), ("Kimi K3 (MoE, 69 KDA + 24 MLA, MLA sparse)", "1M"),
                    ("DeepSeek-V4 Pro (CSA + HCA hybrid, 1M)", "1M")):
        m = MODELS[name]; core = m["core"](CTX[L]); tot = m["mlp"] + m["proj"] + core
        print(f"  {name:46s} {L:>5s}  attn {100 * core / tot:3.0f}%  MLP {100 * m['mlp'] / tot:3.0f}%  proj {100 * m['proj'] / tot:3.0f}%")
    print(f"\nFor comparison, 8 d^2 at d=7168 is {8 * 7168**2 / M:.0f} MMACs; DeepSeek's active MLP happens to match it, Kimi K3's is 2.9x it.")


if __name__ == "__main__":
    main()
