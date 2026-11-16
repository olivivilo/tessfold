"""Single-question native Und inference on an allocated Ascend NPU."""
import argparse
import json
from pathlib import Path

from vqa.official_und import ROOT, OfficialUND


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--adapter', help='Adapter directory, or "selected" for validation-selected model')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()
    adapter = args.adapter
    if adapter == 'selected':
        selection = json.loads((ROOT / 'artifacts/official_und/model_selection.json').read_text())
        adapter = selection.get('adapter')
    image = Path(args.image).resolve()
    assert image.exists(), image
    engine = OfficialUND(adapter=adapter)
    prediction, runtime = engine.predict({'image': str(image), 'question': args.question}, args.max_new_tokens)
    print(json.dumps({'question': args.question, 'answer': prediction, 'adapter': adapter,
                      'image': str(image), 'runtime': runtime}, ensure_ascii=False))


if __name__ == '__main__':
    main()
