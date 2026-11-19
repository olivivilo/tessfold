"""Select an SFT checkpoint by heldout loss, then select deployment by heldout generation."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/official_und'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    if args.checkpoint:
        states = list((OUT / 'sft').glob('checkpoint-*/trainer_state.json'))
        assert states, 'No completed SFT checkpoint'
        latest = max((json.loads(p.read_text()) for p in states), key=lambda r: r['global_step'])
        best = latest['best_model_checkpoint']
        assert best and Path(best, 'adapter_config.json').exists()
        result = {'adapter': str(Path(best).resolve()), 'criterion': 'Lowest heldout teacher-forced validation loss',
                  'global_step': latest['global_step'], 'best_metric': latest['best_metric'], 'test_used': False}
        (OUT / 'sft_selection.json').write_text(json.dumps(result, indent=2))
        print(result['adapter'])
    else:
        values = {label: json.loads((OUT / f'{label}_validation_metrics.json').read_text())
                  for label in ('baseline', 'finetuned')}
        selected = 'finetuned' if values['finetuned']['task_macro_mean_percent'] > values['baseline']['task_macro_mean_percent'] else 'baseline'
        result = {'selected': selected, 'criterion': 'Mean native generation scores on the SAME fixed 256 validation QA',
                  'baseline_validation_mean': values['baseline']['task_macro_mean_percent'],
                  'finetuned_validation_mean': values['finetuned']['task_macro_mean_percent'],
                  'reason': 'Select higher four-task mean on fixed validation; ties retain original model', 'test_used': False}
        if selected == 'finetuned':
            result['adapter'] = json.loads((OUT / 'sft_selection.json').read_text())['adapter']
        (OUT / 'model_selection.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
