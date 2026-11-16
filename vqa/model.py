import json
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor

ROOT = Path(__file__).resolve().parents[1]

def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path

class Projector(nn.Module):
    """Official Und PatchMerger uses spatial_merge_size=1: 768 -> 768 -> 2048."""
    def __init__(self, vision_dim, language_dim):
        super().__init__()
        self.ln_q = nn.LayerNorm(vision_dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(vision_dim, vision_dim), nn.GELU(), nn.Linear(vision_dim, language_dim))
        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        return self.mlp(self.ln_q(features))

def load_vision(config):
    path = resolve(config['vision_model'])
    vc = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    # Official sdpa implementation forces CUDA efficient attention; eager also works on CPU.
    vc.vision_attn_implementation = config.get('vision_attention', 'eager')
    dtype = getattr(torch, config.get('vision_dtype', config['dtype']))
    vision = AutoModel.from_pretrained(path, config=vc, trust_remote_code=True,
                                       local_files_only=True, dtype=dtype).to(config['device']).eval()
    vision.requires_grad_(False)
    kwargs = json.loads((path / 'preprocessor_config.json').read_text())
    for key in ('auto_map', 'processor_class', 'image_processor_type'):
        kwargs.pop(key, None)
    kwargs.update(max_pixels=config['max_pixels'], min_pixels=config['min_pixels'])
    return vision, Qwen2VLImageProcessor(**kwargs)

class VLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.settings = config
        self.device_name = config['device']
        self.dtype = getattr(torch, config['dtype'])
        path = resolve(config['language_model'])
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.lm = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                                     dtype=self.dtype, attn_implementation='sdpa').to(self.device_name)
        vc = json.loads((resolve(config['vision_model']) / 'config.json').read_text())
        self.projector = Projector(vc['embed_dim'], self.lm.config.hidden_size).to(self.device_name, dtype=self.dtype)
        self.lm.requires_grad_(False)
        self.lm.eval()
        self.features = OrderedDict()

    def feature(self, sample):
        doc = sample['document_id']
        if doc not in self.features:
            path = resolve(self.settings['output_dir']) / 'features' / f'{doc}.pt'
            self.features[doc] = torch.load(path, map_location='cpu', weights_only=True)['features'].to(self.device_name, dtype=self.dtype)
            limit = self.settings.get('feature_cache_limit', 64)
            while len(self.features) > limit:
                self.features.popitem(last=False)
        if isinstance(self.features, OrderedDict):
            self.features.move_to_end(doc)
        return self.features[doc]

    def ids(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(self.device_name)

    def prompt(self, question):
        if self.settings.get('prompt_style') == 'official_monkey_und':
            # The official ms-swift Und template is plain ChatML, with no thinking prefill.
            return ('<|im_start|>user\n<|vision_start|>',
                    '<|vision_end|>' + question + '<|im_end|>\n<|im_start|>assistant\n')
        # Qwen3 original non-thinking template. The visual block occupies the <image> location.
        content = '<image>\n' + question
        text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
        left, right = text.split('<image>', 1)
        return left + '<|vision_start|>', '<|vision_end|>' + right

    def embeds(self, sample, answer=None, blank_image=False):
        left, right = self.prompt(sample['question'])
        features = self.feature(sample)
        if blank_image:
            features = torch.zeros_like(features)
        vision = self.projector(features).unsqueeze(0)
        embedding = self.lm.get_input_embeddings()
        prefix = torch.cat([embedding(self.ids(left)), vision, embedding(self.ids(right))], dim=1)
        if answer is None:
            return prefix
        ids = self.ids(answer + self.tokenizer.eos_token)
        if ids.shape[1] > self.settings['max_answer_tokens']:
            raise ValueError(f"Answer exceeds limit for {sample['id']}; raise max_answer_tokens, never silently truncate.")
        return torch.cat([prefix, embedding(ids)], dim=1), ids, prefix.shape[1]

    def loss(self, sample):
        embeds, targets, prefix_length = self.embeds(sample, sample['answers'][0])
        # Compute vocabulary logits only on answer positions; identical causal CE, much less RAM.
        states = self.lm.model(inputs_embeds=embeds, use_cache=False, return_dict=True).last_hidden_state
        logits = self.lm.lm_head(states[:, prefix_length-1:-1]).float()
        return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)), targets.numel()

    @torch.no_grad()
    def predict(self, sample, blank_image=False):
        embeds = self.embeds(sample, blank_image=blank_image)
        # Passing only embeddings makes HF return generated ids, without a fake prompt to decode.
        ids = self.lm.generate(inputs_embeds=embeds, attention_mask=torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long),
                               max_new_tokens=self.settings['max_new_tokens'], do_sample=False, use_cache=True,
                               pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
