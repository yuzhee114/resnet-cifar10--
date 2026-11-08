#!/usr/bin/env python
"""Aggregate per-run results into results/summary.csv.

Reads the one-row-per-run CSV written by `trainer.py --run-csv logs/runs.csv`
and produces the "Paper vs Ours" table used in the report:

    python scripts/summarize.py
    python scripts/summarize.py --input logs/runs.csv --output results/summary.csv

Columns of the output:
    model, params_M, paper_err, n_seeds, our_err_mean, our_err_std,
    delta_abs, delta_rel_pct

`paper_err` is the test error of Table 6 of He et al. (2016); `delta_abs` is
ours - paper in percentage points, `delta_rel_pct` is the relative difference.
"""
import argparse
import csv
import os
import statistics

# Table 6 of the paper: model -> (parameters in millions, test error %)
PAPER = {
    "resnet20":   (0.27, 8.75),
    "resnet32":   (0.46, 7.51),
    "resnet44":   (0.66, 7.17),
    "resnet56":   (0.85, 6.97),
    "resnet110":  (1.7,  6.43),
    "resnet1202": (19.4, 7.93),
}


def model_order(name):
    """Sort resnet20 < resnet32 < ... numerically instead of alphabetically."""
    digits = ''.join(ch for ch in name if ch.isdigit())
    return (int(digits) if digits else 10 ** 6, name)


def load_runs(path):
    """Return {arch: [test_err, ...]} from the run CSV, ignoring empty rows."""
    runs = {}
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            arch = (row.get('arch') or '').strip()
            value = (row.get('test_err') or '').strip()
            if not arch or value == '':
                continue            # runs without a test evaluation (--val-split 0)
            runs.setdefault(arch, []).append(float(value))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default='logs/runs.csv',
                        help='run CSV written by trainer.py (default: logs/runs.csv)')
    parser.add_argument('--output', default='results/summary.csv',
                        help='where to write the aggregate table')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print('missing {} - run at least one training job with --run-csv'.format(args.input))
        return 1

    runs = load_runs(args.input)
    if not runs:
        print('no completed runs with a test evaluation found in {}'.format(args.input))
        return 1

    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rows = []
    print('{:<12} {:>6} {:>9} {:>8} {:>14} {:>10} {:>10}'.format(
        'model', 'n', 'paper', 'ours', 'ours (mean+-sd)', 'delta', 'delta %'))
    for arch in sorted(runs, key=model_order):
        errors = runs[arch]
        mean = statistics.fmean(errors)
        std = statistics.stdev(errors) if len(errors) > 1 else 0.0
        params, paper = PAPER.get(arch, ('', ''))
        delta = mean - paper if paper != '' else ''
        delta_rel = (delta / paper * 100.0) if paper not in ('', 0) else ''
        rows.append({
            'model': arch,
            'params_M': params,
            'paper_err': paper,
            'n_seeds': len(errors),
            'our_err_mean': '{:.3f}'.format(mean),
            'our_err_std': '{:.3f}'.format(std),
            'delta_abs': '' if delta == '' else '{:+.3f}'.format(delta),
            'delta_rel_pct': '' if delta_rel == '' else '{:+.1f}'.format(delta_rel),
        })
        print('{:<12} {:>6} {:>9} {:>8.3f} {:>14} {:>10} {:>10}'.format(
            arch, len(errors), paper, mean,
            '{:.2f} +- {:.2f}'.format(mean, std),
            '' if delta == '' else '{:+.2f}'.format(delta),
            '' if delta_rel == '' else '{:+.1f}%'.format(delta_rel)))

    with open(args.output, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'model', 'params_M', 'paper_err', 'n_seeds',
            'our_err_mean', 'our_err_std', 'delta_abs', 'delta_rel_pct'])
        writer.writeheader()
        writer.writerows(rows)

    print('\nwritten {} ({} models)'.format(args.output, len(rows)))
    incomplete = [a for a in runs if len(runs[a]) < 3]
    if incomplete:
        print('warning: fewer than 3 seeds for {}'.format(', '.join(incomplete)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
