"""Swift external plugin selecting the author's eager vision path on Ascend."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vqa.official_und import use_eager_vision
use_eager_vision()

from swift.llm.template.template.monkeyocrv2 import MonkeyOCRv2UndTemplate
_encode = MonkeyOCRv2UndTemplate._encode


def checked_encode(self, inputs):
    result = _encode(self, inputs)
    if inputs.images:
        count = sum(int(token) == self.image_token_id for token in result['input_ids'])
        assert count == int(result['image_grid_thw'].prod(dim=-1).sum()), 'Und image tokens must expand exactly once'
        assert len(result['input_ids']) <= 8196, 'Never silently truncate official SFT samples'
        if result.get('labels') is not None:
            assert all(label == -100 for token, label in zip(result['input_ids'], result['labels']) if token == self.image_token_id)
    return result


MonkeyOCRv2UndTemplate._encode = checked_encode
