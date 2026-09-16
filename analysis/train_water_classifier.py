#!/usr/bin/env python3
"""
Train and compare four water-level classifiers, one per modality combination.

    vision + tactile + audio
    vision + tactile
    vision + audio
    vision only

The four differ ONLY in which feature blocks from ``water_dataset.py`` they receive, so a gap
between them is attributable to the sensor rather than to preprocessing or architecture.

ON "TRAINED TO OVERFIT". With ~20 trials per class against a few hundred features, every one of
these fits the training set perfectly -- that is reported below as `train`, and it is not evidence
of anything. The number that answers "does this sensor know the water level" is `cv`, the
leave-one-out cross-validated accuracy, because it is the only one that ever sees a trial the
model was not fitted on. A configuration that scores 1.00 train and 0.33 cv has learned the trial
identities and nothing about water. Chance is 1/3.

Leave-one-out rather than a single split: with 60 trials a held-out set large enough to separate
70% from 85% would leave too little to train on, and LOO uses every trial as a test case exactly
once. It is affordable at this size and has no split-seed luck to argue about.

Usage (on polyumi-server, with ROS sourced so sklearn is importable):
    python3 train_water_classifier.py --features water_features.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

COMBOS = {
    'vision+tactile+audio': ('vision', 'tactile', 'audio'),
    'vision+tactile': ('vision', 'tactile'),
    'vision+audio': ('vision', 'audio'),
    'vision': ('vision',),
}


def build_model(kind: str, seed: int = 0):
    """
    Build one classifier pipeline: standardise, then fit.

    Two kinds, and the choice is not cosmetic. `logreg` is linear, so it can only find a water
    level that is linearly separable in these features. `mlp` can fit a curved boundary, which
    matters here because a bottle's resonance does not move monotonically with fill -- "half" can
    sit spectrally between "empty" and "some" rather than beyond them. Running only the linear
    model risks reporting "this sensor knows nothing" when it knows something nonlinear, which is
    a false negative and the most expensive mistake this experiment can make.

    Both are regularised hard, because at up to 336 features and 60 trials an unconstrained fit of
    either is memorisation and the comparison would measure capacity rather than information.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == 'logreg':
        head = LogisticRegression(max_iter=5000, C=0.1)
    elif kind == 'mlp':
        # One small hidden layer and a strong weight penalty: enough to bend the boundary,
        # not enough to trace 60 points. early_stopping is off on purpose -- at n=60 the
        # validation split it carves out is too small to stop on meaningfully.
        head = MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, max_iter=5000, random_state=seed)
    else:
        raise ValueError(f'unknown model kind: {kind}')
    return make_pipeline(StandardScaler(), head)


def main() -> int:
    """Fit each combination, report train and leave-one-out accuracy, and a confusion matrix."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', required=True, type=pathlib.Path)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument(
        '--n-permutations', type=int, default=30, help='shuffled-label runs used to find the real chance floor'
    )
    ap.add_argument(
        '--models',
        nargs='+',
        default=['logreg', 'mlp'],
        choices=['logreg', 'mlp'],
        help='run each model kind over every modality combination',
    )
    args = ap.parse_args()

    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    d = np.load(args.features, allow_pickle=True)
    y = d['label']
    levels = [str(s) for s in d['levels']]
    counts = {lv: int((y == i).sum()) for i, lv in enumerate(levels)}
    print(f'trials: {len(y)}   per class: {counts}   chance = {1 / len(levels):.3f}')
    if len(y) < 2 * len(levels):
        print('not enough trials to cross-validate', file=sys.stderr)
        return 1
    print()

    results = {}
    for kind in args.models:
        print(f'--- model: {kind} ---')
        for name, blocks in COMBOS.items():
            X = np.concatenate([d[b] for b in blocks], axis=1)
            model = build_model(kind, args.seed)
            model.fit(X, y)
            train_acc = float((model.predict(X) == y).mean())
            pred = cross_val_predict(build_model(kind, args.seed), X, y, cv=LeaveOneOut())
            cv_acc = float((pred == y).mean())
            results[(kind, name)] = (cv_acc, train_acc, X.shape[1], confusion_matrix(y, pred))
            print(f'  {name:<24} dims={X.shape[1]:<5} train={train_acc:.3f}   cv={cv_acc:.3f}')
        print()

    print('confusion matrices (rows = true, cols = predicted; leave-one-out)')
    for (kind, name), (_, _, _, cm) in results.items():
        print(f'\n  {kind} / {name}')
        print('         ' + ''.join(f'{lv:>8}' for lv in levels))
        for i, lv in enumerate(levels):
            print(f'  {lv:>6} ' + ''.join(f'{v:>8}' for v in cm[i]))

    print('\nranking by cross-validated accuracy (all model kinds):')
    for (kind, name), (cv, tr, _, _) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f'  {cv:.3f}  {kind:<7} {name:<24} (train {tr:.3f})')

    # Per modality combination, the better of the model kinds. The linear model failing where the
    # MLP succeeds is evidence of a NONLINEAR boundary, not of a dead sensor -- which is the
    # false negative this whole comparison has to avoid.
    print('\nbest model per modality combination:')
    for name in COMBOS:
        per = {k: results[(k, name)][0] for k in args.models}
        bk = max(per, key=per.get)
        spread = max(per.values()) - min(per.values())
        note = '   <- nonlinear: MLP beats linear' if bk == 'mlp' and spread >= 0.1 else ''
        detail = ', '.join(f'{k}={v:.3f}' for k, v in per.items())
        print(f'  {name:<24} {bk:<7} {per[bk]:.3f}   ({detail}){note}')

    # The chance floor is NOT 1/3. With 60 trials, leave-one-out accuracy on features that carry
    # no label at all still lands well above 1/3 a good fraction of the time -- in testing, pure
    # Gaussian noise scored 0.43. So a real result has to clear the SHUFFLED-LABEL distribution,
    # not the nominal chance line, before it means anything. This measures that floor directly on
    # the richest feature set, which is the one most able to memorise a shuffle.
    print('\nnull distribution (labels shuffled, same features and model):')
    X_all = np.concatenate([d[b] for b in ('vision', 'tactile', 'audio')], axis=1)
    rng = np.random.default_rng(args.seed)
    null = []
    for _ in range(args.n_permutations):
        y_shuf = rng.permutation(y)
        pred = cross_val_predict(build_model(args.models[0], args.seed), X_all, y_shuf, cv=LeaveOneOut())
        null.append(float((pred == y_shuf).mean()))
    null = np.asarray(null)
    thresh = float(np.percentile(null, 95))
    print(f'  {args.n_permutations} shuffles: mean {null.mean():.3f}  p95 {thresh:.3f}  max {null.max():.3f}')
    print(f'  => treat cv <= {thresh:.3f} as indistinguishable from no signal')

    best = max(results.items(), key=lambda kv: kv[1][0])
    vision_only = max(results[(k, 'vision')][0] for k in args.models)
    print(f'\nbest overall: {best[0][0]} / {best[0][1]} at {best[1][0]:.3f}')
    print(
        f'vision-only baseline (best model): {vision_only:.3f}  '
        f'-> the extra sensors are worth {best[1][0] - vision_only:+.3f}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
