"""Train controlled SOTA adaptations using the project's existing annotations."""
import argparse
import copy
import importlib.util
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from adapters import make_model
from common import metrics
from data import ANNOTATIONS, Images
from fetch_sources import verify
from protocol import REFERENCE_ROOT, ReferenceLoss, alignment_report, configure_method, reference_protocol

PROJECT = Path(__file__).resolve().parents[2]
SWAV = ('https://download.openmmlab.com/mmselfsup/1.x/swav/'
        'swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/'
        'swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220825-5b3fc7fc.pth')


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def move(data, device):
    return [x.to(device) for x in data] if isinstance(data, (list, tuple)) else data.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels, indices = [], [], []
    for image, label, index in loader:
        scores.append(model(move(image, device)).cpu().numpy())
        labels.append(label.numpy())
        indices.extend(index.tolist())
    return np.concatenate(labels), np.concatenate(scores), indices


@torch.no_grad()
def update_ema(ema, model, momentum):
    for dest, source in zip(ema.parameters(), model.parameters()):
        dest.lerp_(source, 1-momentum)
    for dest, source in zip(ema.buffers(), model.buffers()):
        # Upstream PAT averages floating BN statistics, copies counters.
        if dest.is_floating_point():
            dest.lerp_(source, 1-momentum)
        else:
            dest.copy_(source)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['MLC-PAT', 'CIPL', 'ProCI', 'MICCAI2024'], required=True)
    parser.add_argument('--dataset', choices=list(ANNOTATIONS), required=True)
    parser.add_argument('--image-root', required=True, help='Directory containing FFA, Endo, OIA-ODIR')
    parser.add_argument('--annotation-root', default=str(PROJECT/'data'))
    parser.add_argument('--train-ann')
    parser.add_argument('--val-ann')
    parser.add_argument('--test-ann')
    parser.add_argument('--source-root', default=str(PROJECT/'third_party/sota'))
    parser.add_argument('--settings', default=str(PROJECT/'configs/SOTA/methods.json'))
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, help='Default: reference batch in aligned mode, 8 in native mode')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--checkpoint', help='Default: reference backbone checkpoint')
    parser.add_argument('--protocol', choices=['aligned', 'native'], default='aligned')
    parser.add_argument('--reference-root', default=str(REFERENCE_ROOT))
    parser.add_argument('--scratch', action='store_true', help='Debug only; explicitly disable backbone pretraining')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--eval', metavar='CHECKPOINT', help='Evaluate a trained adapter checkpoint only')
    parser.add_argument('--max-train-batches', type=int, help='Debug only; recorded in run.json')
    return parser.parse_args()


def main():
    args = parse_args()
    reference = reference_protocol(args.dataset, args.reference_root)
    if args.batch_size is None:
        args.batch_size = reference['batch_size'] if args.protocol == 'aligned' else 8
    if args.checkpoint is None:
        args.checkpoint = reference['backbone']['init_cfg']['checkpoint'] if args.protocol == 'aligned' else SWAV
    if args.batch_size < 2 or args.workers < 0 or args.image_size < 32:
        raise ValueError('Require batch-size >=2, workers >=0 and image-size >=32')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    verify(args.source_root, args.method)
    settings = configure_method(args.method, json.loads(Path(args.settings).read_text())[args.method], reference, args.protocol)
    epochs = args.epochs if args.epochs is not None else settings['epochs']
    if epochs < 1:
        raise ValueError('epochs must be positive')
    settings['epochs'] = epochs
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if not args.eval and ((work/'best.pth').exists() or (work/'history.jsonl').exists()):
        raise FileExistsError(f'Refusing to overwrite a previous run: {work}')
    annotation = [Path(getattr(args, name+'_ann') or Path(args.annotation_root)/relative).resolve()
                  for name, relative in zip(('train', 'val', 'test'), ANNOTATIONS[args.dataset])]
    datasets = [Images(ann, args.image_root, args.image_size, train=(i == 0),
                       method=args.method, grid=settings.get('grid', 2),
                       reference=reference if args.protocol == 'aligned' else None) for i, ann in enumerate(annotation)]
    classes = datasets[0].classes
    if any(ds.classes != classes for ds in datasets):
        raise ValueError('Class order differs between annotation splits')
    for i, ds in enumerate(datasets):
        for other in datasets[i+1:]:
            if set(ds.paths) & set(other.paths):
                raise ValueError('Image overlap between train/val/test annotations')
            groups = {r['group_id'] for r in ds.rows if 'group_id' in r}
            other_groups = {r['group_id'] for r in other.rows if 'group_id' in r}
            if groups & other_groups:
                raise ValueError('Patient/study overlap between annotation splits')
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, worker_init_fn=seed_worker,
                              generator=generator, drop_last=True)
    if not len(train_loader):
        raise ValueError('Training data smaller than batch size')
    val_loader, test_loader = [DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
                               for ds in datasets[1:]]
    model = make_model(args.method, classes, args.source_root, settings, args.image_size,
                       None if args.scratch or args.eval else args.checkpoint,
                       classification_loss=(ReferenceLoss(reference) if args.protocol == 'aligned' and args.method != 'CIPL'
                                            else None),
                       backbone_config=reference['backbone'] if args.protocol == 'aligned' else None).to(device)
    ema = copy.deepcopy(model).requires_grad_(False) if settings['ema'] else None
    if args.method == 'CIPL':
        core = model.core
        optimizer = torch.optim.Adam([
            dict(params=core.features.parameters(), lr=settings['backbone_lr']),
            dict(params=core.add_on_layers.parameters()), dict(params=[core.prototype_vectors])], lr=settings['lr'])
    elif args.method == 'MICCAI2024':
        optimizer = torch.optim.SGD(model.parameters(), lr=settings['lr'], momentum=.9, weight_decay=settings['weight_decay'])
    elif args.method == 'ProCI':
        source = Path(args.source_root)/'ProCI/utilis/sam.py'
        spec = importlib.util.spec_from_file_location('_official_sam', source)
        sam = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sam)
        optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, rho=settings['sam_rho'],
                            lr=settings['lr'], weight_decay=settings['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler = None
    if args.method == 'MLC-PAT':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=settings['lr'],
                                                        epochs=epochs, steps_per_epoch=len(train_loader), pct_start=.2)
    record = dict(vars(args), classes=classes, settings=settings, annotations=[str(p) for p in annotation],
                  annotation_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in annotation],
                  alignment=alignment_report(args.method, args.protocol, reference, settings,
                                             args.batch_size, args.image_size, args.scratch),
                  initialization_audit=(model.core.features.initialization_audit if args.method == 'CIPL'
                                        else model.core.backbone.initialization_audit if args.method == 'ProCI'
                                        else model.backbone.initialization_audit),
                  backbone='MMPretrain ResNet-50', protocol='controlled-adaptation',
                  selection='validation mAUC only; student/EMA selected on validation',
                  sources=json.loads(Path(__file__).with_name('sources.json').read_text()))
    (work/('evaluation_run.json' if args.eval else 'run.json')).write_text(json.dumps(record, indent=2))
    best = -float('inf')

    def consider(epoch, stage):
        nonlocal best
        for name, candidate in [('student', model)] + ([('ema', ema)] if ema is not None else []):
            target, score, _ = evaluate(candidate, val_loader, device)
            result = dict(epoch=epoch, stage=stage, weights=name, **metrics(target, score))
            with (work/'history.jsonl').open('a') as handle:
                handle.write(json.dumps(result)+'\n')
            print(json.dumps(result), flush=True)
            if result['mAUC'] > best:
                best = result['mAUC']
                torch.save(dict(model=candidate.state_dict(), epoch=epoch, stage=stage, weights=name,
                                validation=result, classes=classes, method=args.method,
                                config=record), work/'best.pth')

    if not args.eval:
        for epoch in range(epochs):
            model.train()
            if args.method == 'CIPL' and epoch in (10, 20, 25, 30, 35):
                for group in optimizer.param_groups:
                    group['lr'] *= .5
            elif args.method == 'ProCI':
                lr = settings['lr'] * min(1., (epoch+1)/5) * (.001+np.cos(.5*np.pi*epoch/epochs))/1.001
                for group in optimizer.param_groups:
                    group['lr'] = float(lr)
            elif args.method == 'MICCAI2024':
                lr = settings['lr'] * min(1., (epoch+1)/5) * .5*(1+np.cos(np.pi*epoch/epochs))
                for group in optimizer.param_groups:
                    group['lr'] = float(lr)
            for step, (images, labels, _) in enumerate(train_loader):
                if args.max_train_batches is not None and step >= args.max_train_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                images, labels = move(images, device), labels.to(device)
                snapshot = None
                if args.method == 'ProCI':
                    snapshot = model.core.memory.protos.detach().clone()
                    with torch.no_grad():
                        confidence = model.core(images)[0].sigmoid()
                        if (epoch*len(train_loader)+step) % 10 == 0:
                            ema.eval()
                            count = min(8, len(images))
                            confidence[:count] = ema(images[:count])
                        model.confidence_override = confidence
                loss = model.loss(images, labels, epoch, step)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at epoch={epoch}, step={step}')
                loss.backward()
                if args.method == 'ProCI':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if args.method == 'ProCI':
                    optimizer.first_step(zero_grad=True)
                    smooth = model.loss(images, labels, epoch, step, write_memory=False, memory_snapshot=snapshot)
                    if not torch.isfinite(smooth):
                        raise FloatingPointError('Non-finite SAM second loss')
                    smooth.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.second_step(zero_grad=True)
                    model.confidence_override = None
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    update_ema(ema, model, settings['ema'])
                if step % 20 == 0:
                    print(f'epoch={epoch} step={step}/{len(train_loader)} loss={loss.item():.6f}', flush=True)
            consider(epoch, 'joint')
            if args.method == 'CIPL' and epoch >= settings['push_start'] and epoch % settings['push_interval'] == 0:
                push_data = Images(annotation[0], args.image_root, args.image_size, method='ProCI',
                                   reference=reference if args.protocol == 'aligned' else None)
                model.push(DataLoader(push_data, batch_size=args.batch_size, num_workers=args.workers), device)
                consider(epoch, 'push')
                model.core.features.requires_grad_(False)
                model.core.add_on_layers.requires_grad_(False)
                model.core.prototype_vectors.requires_grad_(False)
                model.core.last_layer.requires_grad_(True)
                last_optimizer = torch.optim.Adam(model.core.last_layer.parameters(), lr=1e-4)
                for last in range(settings['last_epochs']):
                    model.train()
                    for images, labels, _ in train_loader:
                        last_optimizer.zero_grad(set_to_none=True)
                        loss = model.loss(move(images, device), labels.to(device), epoch, last_only=True)
                        loss.backward()
                        last_optimizer.step()
                        model.constrain_last_layer()
                    consider(epoch, f'last-{last}')
                model.core.features.requires_grad_(True)
                model.core.add_on_layers.requires_grad_(True)
                model.core.prototype_vectors.requires_grad_(True)
                model.core.last_layer.requires_grad_(False)
    state = torch.load(args.eval or work/'best.pth', map_location=device)
    trained_mode = state.get('config', {}).get('alignment', {}).get('mode', 'native')
    if trained_mode != args.protocol:
        raise ValueError(f'Checkpoint protocol is {trained_mode}; evaluate with --protocol {trained_mode}')
    if state['classes'] != classes or state['method'] != args.method:
        raise ValueError('Checkpoint method/class order does not match this evaluation')
    model.load_state_dict(state['model'], strict=True)
    target, score, indices = evaluate(model, test_loader, device)
    result = dict(metrics(target, score), selected_validation=state['validation'])
    (work/'test_metrics.json').write_text(json.dumps(result, indent=2))
    predictions = [dict(img_path=datasets[2].paths[index], gt_score=y, pred_score=p,
                        group_id=datasets[2].rows[index].get('group_id'))
                   for index, y, p in zip(indices, target, score)]
    with (work/'pred_results.pkl').open('wb') as handle:
        pickle.dump(predictions, handle)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
