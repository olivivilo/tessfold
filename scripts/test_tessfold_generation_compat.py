"""Reproduce UND's None-ID failure on tiny Qwen; independently gate the fix."""
import json
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from vqa.tessfold_generation_compat import preserve_prefill_ids


class GuardedQwen(Qwen3ForCausalLM):
    # Same relevant behavior as official UND: require IDs, consume embeddings.
    def forward(self, input_ids, inputs_embeds=None, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        assert len(input_ids) >= 1
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        return super().forward(inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, **kwargs)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261115)
    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, pad_token_id=0, bos_token_id=1, eos_token_id=127)
    model = GuardedQwen(config).eval()
    ids = torch.tensor([[1, 9, 10, 11, 12, 13]])
    embeds = model.get_input_embeddings()(ids)
    settings = dict(attention_mask=torch.ones_like(ids), do_sample=False,
                    max_new_tokens=6, min_new_tokens=6, use_cache=True)
    native = model.generate(input_ids=ids, **settings)
    try:
        model.generate(input_ids=ids, inputs_embeds=embeds, **settings)
    except TypeError as exc:
        assert 'NoneType' in str(exc)
    else:
        raise AssertionError('Original failure was not reproduced')
    original = model.prepare_inputs_for_generation
    with preserve_prefill_ids(model):
        fixed = model.generate(input_ids=ids, inputs_embeds=embeds, **settings)
        assert torch.equal(native, fixed)
        # Native IDs-only behavior and nested scopes must remain unchanged.
        with preserve_prefill_ids(model):
            assert torch.equal(native, model.generate(input_ids=ids, **settings))
    assert model.prepare_inputs_for_generation == original
    try:
        with preserve_prefill_ids(model):
            raise RuntimeError('restore on exceptions')
    except RuntimeError:
        pass
    assert model.prepare_inputs_for_generation == original
    # Independent question suffixes clone the same prefix; no answer KV leaks.
    import copy
    prefix = ids[:, :3]
    cache = model(input_ids=prefix, use_cache=True).past_key_values
    for suffix in [torch.tensor([[11, 12, 13]]), torch.tensor([[21, 22]])]:
        full = torch.cat([prefix, suffix], dim=1)
        kw = dict(attention_mask=torch.ones_like(full), use_cache=True,
                  max_new_tokens=6, min_new_tokens=6, do_sample=False)
        full_out = model.generate(input_ids=full, **kw)
        cached_out = model.generate(input_ids=full, past_key_values=copy.deepcopy(cache), **kw)
        assert torch.equal(full_out, cached_out)
        assert cache.get_seq_length() == prefix.shape[1]
    print(json.dumps(dict(original_failure_reproduced=True,
        embeddings_generation_equal=True, native_generation_equal=True,
        scoped_restoration=True, independent_prefix_questions=2)))


if __name__ == '__main__':
    main()
