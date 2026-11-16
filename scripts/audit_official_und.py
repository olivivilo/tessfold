"""Compare official Swift training prefixes with native inference, without loading weights."""
import json
import os
from pathlib import Path

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
os.environ.setdefault('MAX_PIXELS', '1003520')
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from swift.llm import get_model_tokenizer, get_template

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/MonkeyOCRv2-B-Und'


def main():
    torch.set_num_threads(2)
    _, processor = get_model_tokenizer(str(MODEL), load_model=False, model_type='monkeyocrv2', device_map='cpu')
    template = get_template('monkeyocrv2_und', processor, max_length=8196, max_pixels=1003520)
    template.set_mode('train')
    native = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    source = [json.loads(line) for line in (ROOT / 'data/official_und/validation_index.jsonl').read_text().splitlines()]
    results = []
    for row in source[::64]:
        path = str(ROOT / row['image'])
        encoded = template.encode({'messages': [{'role': 'user', 'content': '<image>' + row['question']},
                                                {'role': 'assistant', 'content': row['answers'][0]}], 'images': [path]})
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': path, 'max_pixels': 1003520},
                                               {'type': 'text', 'text': row['question']}]}]
        text = native.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        batch = native(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        labels = encoded['labels']
        prefix_length = next(i for i, value in enumerate(labels) if value != -100)
        train_prefix, infer_prefix = encoded['input_ids'][:prefix_length], batch['input_ids'][0].tolist()
        text_train = [t for t in train_prefix if t != 151655]
        text_infer = [t for t in infer_prefix if t != 151655]
        train_pixels, infer_pixels = encoded['pixel_values'], batch['pixel_values']
        result = {'id': row['id'], 'task': row['task'], 'text_prefix_equal': text_train == text_infer,
                  'full_prefix_equal': train_prefix == infer_prefix,
                  'training_grid': encoded['image_grid_thw'].tolist(), 'native_grid': batch['image_grid_thw'].tolist(),
                  'training_text_prefix': native.decode(text_train), 'native_text_prefix': native.decode(text_infer),
                  'pixel_shape_equal': train_pixels.shape == infer_pixels.shape,
                  'training_visual_tokens': train_prefix.count(151655), 'native_visual_tokens': infer_prefix.count(151655)}
        if train_pixels.shape == infer_pixels.shape:
            result['pixel_max_absolute_difference'] = float((train_pixels - infer_pixels).abs().max())
            result['pixel_mean_absolute_difference'] = float((train_pixels - infer_pixels).abs().mean())
        assert result['text_prefix_equal'], result
        assert result['training_visual_tokens'] == int(encoded['image_grid_thw'].prod()), result
        assert all(label == -100 for token, label in zip(encoded['input_ids'], labels) if token == 151655)
        results.append(result)
    destination = ROOT / 'artifacts/official_und/input_pipeline_audit.json'
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
