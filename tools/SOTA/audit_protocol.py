"""Preview actual aligned settings without images, GPUs or weight downloads."""
import argparse
import json
from pathlib import Path

from protocol import PROJECT, REFERENCE_ROOT, alignment_report, configure_method, reference_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', default=str(REFERENCE_ROOT))
    parser.add_argument('--dataset', choices=['FFA', 'Endo', 'ODIR'], nargs='+', default=['FFA', 'Endo', 'ODIR'])
    parser.add_argument('--protocol', choices=['aligned', 'native'], default='aligned')
    parser.add_argument('--output')
    args = parser.parse_args()
    methods = json.loads((PROJECT/'configs/SOTA/methods.json').read_text())
    reports = {}
    for dataset in args.dataset:
        reference = reference_protocol(dataset, args.reference_root)
        reports[dataset] = {}
        for method in [*methods, 'BankCausal']:
            settings = configure_method(method, methods.get(method, dict(epochs=100)), reference, args.protocol)
            reports[dataset][method] = alignment_report(method, args.protocol, reference, settings,
                                                       reference['batch_size'], 448)
            report = reports[dataset][method]
            print(f'{dataset:5} {method:11} batch={reference["batch_size"]} '
                  f'passes={report["total_optimizer_dataset_passes"]} '
                  f'loss={report["classification_loss"]} '
                  f'exceptions={report["exceptions"]}')
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
