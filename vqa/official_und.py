"""Author's full model/processor path, with an explicit Ascend eager-vision choice."""
import os
import sys
from pathlib import Path

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get('TESSFOLD_UND_MODEL',ROOT / 'models/MonkeyOCRv2-B-Und'))


def use_eager_vision(model_path=MODEL):
    """Select the author's eager attention; leave Qwen SDPA and all weights intact."""
    cls = get_class_from_dynamic_module('modeling_monkeyocrv2_vision.MonkeyOCRv2VisionTransformer', str(model_path))
    module = sys.modules[cls.__module__]
    for key in tuple(module.VISION_ATTENTION_CLASSES):
        module.VISION_ATTENTION_CLASSES[key] = module.VisionAttentionV2


def set_device():
    import torch_npu
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.npu.set_device(local)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(4)
    torch.manual_seed(42)
    return f'npu:{local}'


class OfficialUND:
    def __init__(self, model_path=MODEL, adapter=None, device=None):
        self.device = device or set_device()
        use_eager_vision(model_path)
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        config.vision_config.attn_implementation = 'eager'
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), config=config, dtype=torch.bfloat16,
            trust_remote_code=True, local_files_only=True, attn_implementation='sdpa').to(self.device).eval()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False).eval()

    def inputs(self, row):
        from qwen_vl_utils import process_vision_info
        path = Path(row['image'])
        if not path.is_absolute():
            path = ROOT / path
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': str(path), 'max_pixels': 1003520},
            {'type': 'text', 'text': row['question']}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        batch = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        assert int((batch['input_ids'] == self.model.config.image_token_id).sum()) == int(batch['image_grid_thw'].prod())
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.no_grad()
    def predict(self, row, max_new_tokens=256):
        inputs = self.inputs(row)
        outputs = self.model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=max_new_tokens)
        generated = outputs[0, inputs['input_ids'].shape[1]:]
        text = self.processor.decode(generated, skip_special_tokens=True)
        return text, {'input_tokens': inputs['input_ids'].shape[1], 'new_tokens': generated.numel(),
                      'visual_tokens': int(inputs['image_grid_thw'].prod()),
                      'hit_token_limit': generated.numel() == max_new_tokens}
