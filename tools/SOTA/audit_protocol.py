"""Preview actual aligned settings without images, GPUs or weight downloads."""
import argparse
import json
from pathlib import Path

from protocol import PROJECT, REFERENCE_ROOT, alignment_report, configure_method, reference_protocol, override_learning_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', default=str(REFERENCE_ROOT))
    parser.add_argument('--dataset', choices=['FFA', 'Endo', 'ODIR'], nargs='+', default=['FFA', 'Endo', 'ODIR'])
    parser.add_argument('--protocol', choices=['aligned', 'native'], default='aligned')
    parser.add_argument('--output')
    parser.add_argument('--lr', type=float, default=.005, help='Match the Bede submission base LR')
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    methods = json.loads((PROJECT/'configs/SOTA/methods.json').read_text())
    reports = {}
    for dataset in args.dataset:
        reference = reference_protocol(dataset, args.reference_root)
        batch_size = args.batch_size if args.batch_size is not None else reference['batch_size']
        if batch_size < 2:
            parser.error('Require batch-size >= 2')
        reports[dataset] = {}
        for method in [*methods, 'BankCausal']:
            settings = configure_method(method, methods.get(method, dict(epochs=100)), reference, args.protocol)
            if method != 'CIPL':
                settings = override_learning_rate(settings, args.lr)
            if args.protocol == 'aligned' and method in ('MICCAI2024', 'MLC-PAT', 'BankCausal'):
                if args.lr != reference['optimizer']['lr']:
                    parser.error('Controlled recipe requires the reference LR')
                settings.update(lr=reference['optimizer']['lr'],
                                weight_decay=reference['optimizer'].get('weight_decay', 0.),
                                optimization_recipe='q2l_sgd_v1')
                if method == 'MLC-PAT':
                    settings['activation_checkpointing'] = True
            if method == 'CIPL':
                settings['cipl_recipe'] = 'official_mechanisms_v2'
            reports[dataset][method] = alignment_report(method, args.protocol, reference, settings,
                                                       batch_size, 448)
            reports[dataset][method]['resolved_method_settings'] = settings
            report = reports[dataset][method]
            print(f'{dataset:5} {method:11} batch={batch_size} lr={settings.get("lr")} '
                  f'optimization={report["optimization_recipe"]} '
                  f'passes={report["total_optimizer_dataset_passes"]} '
                  f'loss={report["classification_loss"]} '
                  f'exceptions={report["exceptions"]}')
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
