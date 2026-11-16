"""Scoped UND/HF embeddings-generation compatibility; no weight/source edits.

HF 4.57 deliberately sets input_ids=None for embeddings prefill. UND needs the
real IDs for its input guard, although its forward consumes only inputs_embeds.
Restore those real IDs AFTER upstream preparation, without changing positions,
masks, cache slicing, logits processing, or any decoder arithmetic.
"""
from contextlib import contextmanager
from types import MethodType
import torch


@contextmanager
def preserve_prefill_ids(model):
    original = model.prepare_inputs_for_generation
    had_override = 'prepare_inputs_for_generation' in model.__dict__
    previous_override = model.__dict__.get('prepare_inputs_for_generation')

    def prepare(self, input_ids, past_key_values=None, inputs_embeds=None,
                attention_mask=None, cache_position=None, **kwargs):
        prepared = original(
            input_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            cache_position=cache_position, **kwargs)
        if prepared.get('input_ids') is None and prepared.get('inputs_embeds') is not None:
            length = prepared['inputs_embeds'].shape[1]
            if cache_position is None or len(cache_position) != length:
                raise ValueError('Cannot recover real IDs for embeddings prefill')
            if input_ids.shape[1] == length:
                actual_ids = input_ids
            else:
                actual_ids = input_ids.index_select(1, cache_position.to(input_ids.device))
            assert actual_ids.shape == prepared['inputs_embeds'].shape[:2]
            prepared['input_ids'] = actual_ids.clone(memory_format=torch.contiguous_format)
        return prepared

    model.prepare_inputs_for_generation = MethodType(prepare, model)
    try:
        yield
    finally:
        if had_override:
            model.prepare_inputs_for_generation = previous_override
        else:
            delattr(model, 'prepare_inputs_for_generation')
