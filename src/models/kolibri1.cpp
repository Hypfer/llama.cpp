// Source: https://github.com/Aleph-Alpha/aleph-alpha-inference/blob/main/aleph_alpha_inference/kolibri1.py
// Uses the AFMoE graph primitives with Kolibri routing and sandwich norms.
#include "models.h"

void llama_model_kolibri1::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    if (hparams.n_swa == 0 || hparams.n_expert_shared != 1 || hparams.n_ff_shexp == 0) {
        throw std::runtime_error("Kolibri1: invalid sliding window or shared expert configuration");
    }
    hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    load_swa_pattern(ml, 5);
    hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_kolibri1::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);
    const int64_t n_ff_exp = hparams.n_ff_exp();
    const int64_t n_ff_shared = hparams.n_ff_shexp;
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
        layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shared, n_embd}, 0);
        layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), {n_embd, n_ff_shared}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_kolibri1::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_kolibri1::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t head_dim = hparams.n_embd_head_k();
    GGML_ASSERT(head_dim == hparams.n_embd_head_v());
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    const float kq_scale = 1.0f / sqrtf(float(head_dim));
    ggml_tensor * cur = nullptr;
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * residual = inpL;
        cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        auto [q, k, v] = build_qkv(layer, cur, head_dim, n_head, n_head_kv, il);
        q = build_norm(q, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        k = build_norm(k, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        if (hparams.is_swa(il)) {
            const float base = model.get_rope_freq_base(cparams, il);
            const float scale = model.get_rope_freq_scale(cparams, il);
            q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, base, scale, ext_factor, attn_factor, beta_fast, beta_slow);
            k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, base, scale, ext_factor, attn_factor, beta_fast, beta_slow);
        }
        cur = build_attn(inp_attn, layer.wo, nullptr, layer.wo_s, q, k, v, nullptr, nullptr, nullptr, kq_scale, il);
        cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, residual);
        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

        // Selection uses biased logits; weights use sigmoid of unbiased logits.
        ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        ggml_prec_set_src(logits, GGML_PREC_F32, 1);
        ggml_tensor * selection = ggml_add(ctx0, logits, layer.ffn_exp_probs_b);
        ggml_tensor * selected = ggml_argsort_top_k(ctx0, selection, n_expert_used);
        ggml_tensor * moe = build_moe_ffn(cur, layer.ffn_gate_inp,
                layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, nullptr,
                n_expert, n_expert_used, LLM_FFN_SILU, hparams.expert_weights_norm, 1.0f,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il, logits,
                nullptr, nullptr, nullptr, nullptr, selected);
        ggml_tensor * shared = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr, nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = ggml_add(ctx0, moe, shared);
        cur = build_norm(cur, layer.ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cur = ggml_add(ctx0, cur, ffn_inp);
        inpL = build_cvec(cur, il);
        cb(inpL, "l_out", il);
    }
    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur, model.output_s);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
