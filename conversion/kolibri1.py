from __future__ import annotations
from .afmoe import AfmoeModel
from .base import ModelBase, gguf


@ModelBase.register("Kolibri1ForCausalLM")
@ModelBase.example("Aleph-Alpha/Kolibri-1-BF16")
class Kolibri1Model(AfmoeModel):
    model_arch = gguf.MODEL_ARCH.KOLIBRI1

    def get_vocab_base_pre(self, tokenizer):
        # Same pre-tokenizer regex as Qwen2; the vocabulary remains Kolibri's.
        return "qwen2"

    def set_gguf_parameters(self):
        self.hparams["num_shared_experts"] = 1
        self.hparams["intermediate_size"] = self.hparams["moe_intermediate_size"]
        self.hparams["route_norm"] = self.hparams["norm_topk_prob"]
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_shared_feed_forward_length(self.hparams["shared_expert_intermediate_size"])
        self.gguf_writer.add_sliding_window_pattern([x == "sliding_attention" for x in self.hparams["layer_types"]])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        name = name.replace(".moe.router.expert_bias", ".mlp.expert_bias")
        name = name.replace(".post_attn_norm", ".post_self_attn_layernorm")
        name = name.replace(".post_ffn_norm", ".post_feedforward_layernorm")
        return super().filter_tensors((name, gen))
