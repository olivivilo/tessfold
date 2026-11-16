"""MetaCompress section 4 adaptation: common-prefix prediction KL and row entropy.

The prediction term uses the full vocabulary at temperature one. The teacher's
greedy response supplies the SAME prefix to both models; gold answers are unused.
"""
import math
import torch


def row_entropy(matrix):
    """Exact entropy of sparse support + dense uniform floor, in float64 nats."""
    g, w = matrix.group.cpu(), matrix.weight.cpu().double()
    base = matrix.floor / matrix.n
    p = (1-matrix.floor)*w + base
    h = torch.zeros(matrix.m, dtype=torch.float64).index_add_(0, g, -torch.special.xlogy(p, p))
    sizes = torch.bincount(g, minlength=matrix.m)
    if base:
        h -= (matrix.n-sizes)*base*math.log(base)
    return float(h.mean())


def common_prefix_logits(model, prompt_ids, projected, response_ids):
    """One or more equal-length visual trials, no cache/history, causal shift exact.

    Forward input is prompt + teacher response[:-1]. Last T logits predict the
    T response tokens, including EOS if generated. No answer labels are passed.
    """
    b = len(projected)
    device = prompt_ids.device
    response = torch.tensor(response_ids, device=device, dtype=prompt_ids.dtype)
    assert len(response) > 0
    ids = torch.cat((prompt_ids, response[:-1][None]), -1).expand(b, -1)
    embeds = model.get_input_embeddings()(ids).clone()
    mask = ids == model.config.image_token_id
    assert projected.shape[1] == int(mask[0].sum())
    embeds[mask] = projected.reshape(-1, projected.shape[-1])
    return model(input_ids=ids, inputs_embeds=embeds,
                 attention_mask=torch.ones_like(ids), use_cache=False,
                 logits_to_keep=len(response)).logits.float()


def prediction_kl(teacher_logp, student_logits):
    logq = student_logits.log_softmax(-1)
    per_token = (teacher_logp.exp()*(teacher_logp-logq)).sum(-1)
    # Rounding can introduce tiny negative values; preserve them for audit.
    return per_token
