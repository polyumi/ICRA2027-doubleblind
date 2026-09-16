#!/usr/bin/env python3
"""
Train and compare the four water-level classifiers: CNN encoders + MLP head.

    vision + tactile + audio      vision + tactile      vision + audio      vision only

Each is the same WaterClassifier with different encoders attached, trained on identical splits
with identical hyperparameters, so a gap between them is about the sensor.

HOW IT IS SCORED, and why not just training accuracy. At ~60 trials against 0.3-0.8 M parameters
every variant reaches 100% on its training set within a few epochs, including on a modality that
carries no information at all. So the reported number is held-out accuracy under stratified
K-fold: every trial is predicted exactly once, by a model that never saw it. Training accuracy is
printed too, but only to show that the fit succeeded -- it cannot rank the variants.

The shuffled-label control is not optional here. With 60 trials, a model this size scores well
above the nominal 1/3 on shuffled labels often enough that an unremarkable real result can look
like a finding. The run ends by measuring that floor on the richest modality set and printing the
threshold a result has to clear.

Usage (on polyumi-server, ROS sourced for torch):
    python3 train_water_cnn.py --tensors water_tensors.npz --epochs 40
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from water_model import WaterClassifier  # noqa: E402

COMBOS = {
    'vision+tactile+audio': ('vision', 'tactile', 'audio'),
    'vision+tactile': ('vision', 'tactile'),
    'vision+audio': ('vision', 'audio'),
    'vision': ('vision',),
    'tactile+audio': ('tactile', 'audio'),
    'tactile': ('tactile',),
    'audio': ('audio',),
}


def load(path: pathlib.Path, device: str):
    """Load the npz into normalised float tensors on `device`."""
    d = np.load(path, allow_pickle=True)
    out = {
        'vision': torch.from_numpy(d['vision']).float().div_(255.0),
        'tactile': torch.from_numpy(d['tactile']).float().div_(255.0),
        'audio': torch.from_numpy(d['audio']).float(),
    }
    # Per-dataset standardisation of the log-mel: it is log-power, so its offset is arbitrary and
    # leaving it un-centred just makes the first conv layer spend capacity on a constant.
    a = out['audio']
    out['audio'] = (a - a.mean()) / (a.std() + 1e-6)
    y = torch.from_numpy(d['label']).long()
    return {k: v.to(device) for k, v in out.items()}, y.to(device), [str(s) for s in d['levels']]


def train_one(combo, X, y, tr_idx, va_idx, te_idx, args, device, early_stop=True):
    """
    Train on `tr_idx` for up to `args.epochs` epochs; select and score on `va_idx` and `te_idx`.

    This is where validation actually does the textbook job: at every epoch the loss on `va_idx`
    is measured, and whichever epoch had the LOWEST validation loss is the checkpoint restored
    before anything is scored. Previously training ran a fixed 120 epochs regardless of `va_idx`
    and every combo reached 100% train accuracy by the end -- val was scored only after the fact,
    so it could not prevent the overfitting it was supposed to catch. Now it does: `va_idx` picks
    the model, exactly once, and `te_idx` is scored only against that already-chosen checkpoint.

    `early_stop=False` skips all of that and just trains the full `args.epochs`, unconditionally.
    Use it whenever `va_idx` is not really a validation set for THIS call -- a cross-validation
    fold's own held-out slice is fine to select on (that IS its validation set), but a final
    refit whose accuracy on some set will be reported as "test" must not use that set to choose
    its own checkpoint, or the number stops being held-out.

    Returns (val_acc, train_acc, val_pred, test_acc, test_pred, best_epoch, best_val_loss).
    `best_epoch` is -1 when early stopping was off or `va_idx` was empty (every epoch was used,
    and `best_val_loss` is nan).

    Mini-batched, not full-batch: at 224x224 with K=8 frames a single step over 75 trials pushes
    600 frames through the stem at once, and the first layer's activations alone run to ~1 GB.
    Batching by trial keeps it bounded and, at this size, costs nothing in wall time.
    """
    torch.manual_seed(args.seed)
    model = WaterClassifier(combo).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss()

    def batches(idx, shuffle):
        order = torch.randperm(len(idx), device=device) if shuffle else torch.arange(len(idx), device=device)
        for i in range(0, len(idx), args.batch_size):
            sel = idx[order[i : i + args.batch_size]]
            yield {m: X[m][sel] for m in combo}, y[sel]

    def mean_loss(idx):
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in batches(idx, shuffle=False):
                total += lossf(model(xb), yb).item() * len(yb)
                n += len(yb)
        return total / max(n, 1)

    best_state, best_epoch, best_val_loss = None, -1, float('nan')
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in batches(tr_idx, shuffle=True):
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
        if early_stop and len(va_idx):
            vl = mean_loss(va_idx)
            if best_epoch < 0 or vl < best_val_loss:
                best_val_loss, best_epoch = vl, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():

        def acc_and_pred(idx):
            if len(idx) == 0:
                return float('nan'), np.empty(0, dtype=int)
            preds = []
            for xb, _ in batches(idx, shuffle=False):
                preds.append(model(xb).argmax(1))
            pred = torch.cat(preds)
            return (pred == y[idx]).float().mean().item(), pred.cpu().numpy()

        train_acc, _ = acc_and_pred(tr_idx)
        val_acc, val_pred = acc_and_pred(va_idx)
        if len(te_idx):
            test_acc, test_pred = acc_and_pred(te_idx)
        else:
            test_acc, test_pred = float('nan'), np.empty(0, dtype=int)
    return val_acc, train_acc, val_pred, test_acc, test_pred, best_epoch, best_val_loss


def main_random(args, X, y, levels) -> int:
    """
    Random per-class holdout, stratified K-fold on the rest, one final score on the holdout.

    Per class, `--test-per-class` trials are drawn at random (seeded by `--split-seed`) and set
    aside. Everything else is the development pool: each variant is cross-validated on it with
    `--folds` stratified folds, variants are ranked on the CV mean, and then every variant is
    retrained on the whole pool and scored once on the held-out test trials.

    Caveat this mode cannot remove: collection is interleaved in blocks, so a random draw puts
    trials from the SAME block into both the pool and the test set. Anything that drifts from
    block to block rather than with the object -- the scene behind the GoPro, the finger camera's
    level and colour balance -- is then shared between train and test and can be learnt as a
    shortcut. `--split fixed` holds out whole blocks instead.
    """
    from sklearn.metrics import confusion_matrix

    y_np = y.cpu().numpy()
    k = len(levels)
    rng = np.random.default_rng(args.split_seed)
    test, pool = [], []
    for c in range(k):
        members = rng.permutation(np.flatnonzero(y_np == c))
        if len(members) <= args.test_per_class:
            print(f'class {levels[c]} has only {len(members)} trials', file=sys.stderr)
            return 1
        test.extend(members[: args.test_per_class])
        pool.extend(members[args.test_per_class :])
    test, pool = np.sort(test), np.array(pool)

    # stratified folds over the pool: deal each class's (shuffled) trials round-robin
    fold_of = np.empty(len(y_np), dtype=int)
    for c in range(k):
        members = rng.permutation(pool[y_np[pool] == c])
        fold_of[members] = np.arange(len(members)) % args.folds

    dev = args.device
    empty = torch.as_tensor([], dtype=torch.long, device=dev)
    as_idx = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=dev)  # noqa: E731

    print(f'trials: {len(y_np)}   per class: {dict(zip(levels, np.bincount(y_np, minlength=k).tolist()))}')
    print(
        f'split:  RANDOM (seed {args.split_seed}) -- {len(pool)} development trials, '
        f'{len(test)} test ({args.test_per_class}/class); {args.folds}-fold CV on development'
    )
    print(f'device: {dev}   {args.epochs} epochs, lr {args.lr}, batch {args.batch_size}\n')

    results = {}
    for name in args.combos:
        combo = COMBOS[name]
        fold_acc, cv_pred = [], np.empty(len(y_np), dtype=int)
        for f in range(args.folds):
            tr = pool[fold_of[pool] != f]
            ho = pool[fold_of[pool] == f]
            acc, _, pred, _, _, _, _ = train_one(combo, X, y, as_idx(tr), as_idx(ho), empty, args, dev)
            fold_acc.append(acc)
            cv_pred[ho] = pred
        # early_stop=False here: `test` is passed in the va_idx slot only to get train_one to
        # score it, but it is the actual held-out test set -- selecting a checkpoint by its loss
        # would leak it into model choice. Train the plain, fixed-epoch model instead.
        test_acc, train_acc, test_pred, _, _, _, _ = train_one(
            combo, X, y, as_idx(pool), as_idx(test), empty, args, dev, early_stop=False
        )
        results[name] = dict(
            cv=float(np.mean(fold_acc)),
            cv_sd=float(np.std(fold_acc)),
            folds=fold_acc,
            test=test_acc,
            train=train_acc,
            cv_cm=confusion_matrix(y_np[pool], cv_pred[pool], labels=range(k)),
            test_cm=confusion_matrix(y_np[test], test_pred, labels=range(k)),
        )
        r = results[name]
        print(
            f'{name:<24} cv={r["cv"]:.3f} +- {r["cv_sd"]:.3f}  '
            f'(folds {" ".join(f"{a:.2f}" for a in fold_acc)})  test={test_acc:.3f}  train={train_acc:.3f}'
        )

    def show(cm, title):
        print(f'\n  {title}')
        print('         ' + ''.join(f'{lv:>18}' for lv in levels))
        for i, lv in enumerate(levels):
            print(f'  {lv:>16} ' + ''.join(f'{v:>18}' for v in cm[i]))

    print('\nconfusion matrices (rows = true, cols = predicted)')
    for name, r in results.items():
        show(r['cv_cm'], f'{name} -- cross-validation, development trials')
        show(r['test_cm'], f'{name} -- held-out test')

    if args.skip_null:
        null = np.array([1.0 / k])
        print('\n(shuffled-label floor skipped)')
    else:
        print(
            f'\nshuffled-label floor ({args.n_shuffles} runs, all three modalities, '
            'trained on development, scored on test):'
        )
        null = []
        for _ in range(args.n_shuffles):
            y_shuf = y.clone()
            y_shuf[as_idx(pool)] = torch.as_tensor(rng.permutation(y_np[pool]), device=dev)
            # Same reason as the final refit above: early_stop=False so this never checkpoints
            # on the test set it is about to be scored against.
            acc, _, _, _, _, _, _ = train_one(
                ('vision', 'tactile', 'audio'),
                X,
                y_shuf,
                as_idx(pool),
                as_idx(test),
                empty,
                args,
                dev,
                early_stop=False,
            )
            null.append(acc)
        null = np.asarray(null)
        print(f'  mean {null.mean():.3f}  max {null.max():.3f}')

    print('\nranking (cross-validation mean; test shown alongside, it did not pick the order):')
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]['cv']):
        mark = '' if r['test'] > null.max() else '   (test not above the shuffled floor)'
        print(f'  cv {r["cv"]:.3f} +- {r["cv_sd"]:.3f}   test {r["test"]:.3f}   {name}{mark}')
    return 0


def main() -> int:
    """Evaluate every modality combination on the held-out trials, then the shuffled-label floor."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tensors', required=True, type=pathlib.Path)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-shuffles', type=int, default=3)
    ap.add_argument(
        '--combos',
        nargs='+',
        default=list(COMBOS),
        choices=list(COMBOS),
        help='which modality combinations to train; default is all four',
    )
    ap.add_argument(
        '--skip-null',
        action='store_true',
        help='skip the shuffled-label floor (it is the slowest part and only '
        'meaningful once, so skip it when re-running a single variant)',
    )
    ap.add_argument(
        '--split',
        choices=['fixed', 'random'],
        default='fixed',
        help='fixed: use the train/val/test labels written at extraction (whole blocks held out). '
        'random: draw --test-per-class test trials per class at random, K-fold CV the rest',
    )
    ap.add_argument('--test-per-class', type=int, default=10, help='random split: test trials per class')
    ap.add_argument('--folds', type=int, default=5, help='random split: cross-validation folds')
    ap.add_argument('--split-seed', type=int, default=0, help='random split: seed for the draw and the folds')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    from sklearn.metrics import confusion_matrix

    X, y, levels = load(args.tensors, args.device)
    if args.split == 'random':
        return main_random(args, X, y, levels)
    d = np.load(args.tensors, allow_pickle=True)
    split = np.asarray([str(s) for s in d['split']])
    tr_idx = torch.as_tensor(np.flatnonzero(split == 'train'), device=args.device)
    va_idx = torch.as_tensor(np.flatnonzero(split == 'val'), device=args.device)
    te_idx = torch.as_tensor(np.flatnonzero(split == 'test'), device=args.device)
    y_va = y[va_idx].cpu().numpy()
    y_te = y[te_idx].cpu().numpy()

    per_class = {lv: int((y == i).sum()) for i, lv in enumerate(levels)}
    print(f'trials: {len(y)}   per class: {per_class}   nominal chance {1 / len(levels):.3f}')
    print(f'split:  {len(tr_idx)} train / {len(va_idx)} val / {len(te_idx)} test')
    if len(te_idx) == 0:
        print('        (no test split in this npz -- pass --test-from at extraction to make one)')
    print(f'device: {args.device}   {args.epochs} epochs, lr {args.lr}, batch {args.batch_size}\n')
    if len(va_idx) == 0:
        print('no validation trials -- was --val-from set too high at extraction?', file=sys.stderr)
        return 1

    results = {}
    selected = {k: COMBOS[k] for k in args.combos}
    for name, combo in selected.items():
        # early_stop=True (the default): va_idx genuinely is this run's validation set, so it
        # gets to pick the checkpoint. te_idx is scored only against whatever va_idx already chose.
        val_acc, train_acc, pred, test_acc, test_pred, best_epoch, best_val_loss = train_one(
            combo, X, y, tr_idx, va_idx, te_idx, args, args.device
        )
        results[name] = (
            val_acc,
            train_acc,
            confusion_matrix(y_va, pred, labels=range(len(levels))),
            test_acc,
            confusion_matrix(y_te, test_pred, labels=range(len(levels))) if len(te_idx) else None,
            best_epoch,
            best_val_loss,
        )
        params = WaterClassifier(combo).n_parameters()
        te_s = f'  test={test_acc:.3f}' if len(te_idx) else ''
        print(
            f'{name:<24} params={params / 1e6:.2f}M  train={train_acc:.3f}  val={val_acc:.3f}{te_s}  '
            f'best_epoch={best_epoch}/{args.epochs}  val_loss={best_val_loss:.4f}'
        )

    print('\nconfusion matrices (rows = true, cols = predicted; validation trials only)')
    for name, (_, _, cm, _, _, _, _) in results.items():
        print(f'\n  {name}')
        print('         ' + ''.join(f'{lv:>8}' for lv in levels))
        for i, lv in enumerate(levels):
            print(f'  {lv:>6} ' + ''.join(f'{v:>8}' for v in cm[i]))

    # With only a handful of validation trials per class, a model that knows nothing still lands
    # well above 1/3 some of the time. This measures how high, using the richest modality set --
    # the one most able to memorise a shuffle -- so a real result has a bar to clear.
    if args.skip_null:
        print('\n(shuffled-label floor skipped; re-run without --skip-null for the real chance threshold)')
        null = np.array([1.0 / len(levels)])
    else:
        print(f'\nshuffled-label floor ({args.n_shuffles} runs, all three modalities):')
        rng = np.random.default_rng(args.seed)
        null = []
        for _ in range(args.n_shuffles):
            y_shuf = y.clone()
            y_shuf[tr_idx] = torch.as_tensor(rng.permutation(y[tr_idx].cpu().numpy()), device=args.device)
            acc, _, _, _, _, _, _ = train_one(
                ('vision', 'tactile', 'audio'), X, y_shuf, tr_idx, va_idx, te_idx, args, args.device
            )
            null.append(acc)
        null = np.asarray(null)
        print(f'  mean {null.mean():.3f}  max {null.max():.3f}')
        print(f'  => treat val <= {null.max():.3f} as indistinguishable from no signal')

    print('\nranking (validation):')
    for name, (acc, tr, _, _, _, _, _) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        mark = '' if acc > null.max() else '   (not above the shuffled floor)'
        print(f'  {acc:.3f}  {name:<24} (train {tr:.3f}){mark}')
    base = results['vision'][0] if 'vision' in results else float('nan')
    # Chosen on VALIDATION. The test number is then reported for that one variant and is the
    # only figure to quote, because it is the only one no decision was made against.
    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f'\nvision-only baseline: {base:.3f}')
    print(f'best on val: {best[0]} at {best[1][0]:.3f}  -> extra sensors worth {best[1][0] - base:+.3f}')

    if len(te_idx):
        print('\n--- held-out TEST, reported once, for the variant validation selected ---')
        print(f'  {best[0]}: test {best[1][3]:.3f}  (val {best[1][0]:.3f})')
        print('  confusion (rows = true, cols = predicted):')
        print('         ' + ''.join(f'{lv:>8}' for lv in levels))
        for i, lv in enumerate(levels):
            print(f'  {lv:>6} ' + ''.join(f'{v:>8}' for v in best[1][4][i]))
        print('\n  every variant on test, for the record -- these did NOT pick the winner:')
        for name, (_, _, _, te, _, _, _) in sorted(results.items(), key=lambda kv: -kv[1][0]):
            print(f'    {te:.3f}  {name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
