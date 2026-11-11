"""Run EGWC experiments using the bundled dataset manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs', nargs='+', help='Dataset names; defaults to all datasets.')
    p.add_argument('--data-root', type=Path, default=Path.cwd(),
                   help='Directory against which manifest paths are resolved.')
    p.add_argument('--out', type=Path, required=True, help='New JSONL results file.')
    p.add_argument('--resume', action='store_true', help='Skip completed, matching runs.')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--eps', type=float, nargs='+', choices=[0.2, 0.1, 0.05], default=[0.2, 0.1, 0.05])
    p.add_argument('--seeds', type=int, nargs='+', help='Override the manifest repetition plan.')
    p.add_argument('--exact', choices=['auto', 'yes', 'no'], default='auto')
    p.add_argument('--threads', type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument('--q', type=int, default=64)
    p.add_argument('--init-q', type=int, default=64)
    p.add_argument('--terminals', type=int, help='Default: ceil(sqrt(n)).')
    p.add_argument('--smooth', type=int, default=4)
    p.add_argument('--refine', type=int, default=64)
    p.add_argument('--delta', type=float, default=0.05)
    p.add_argument('--rtol', type=float, default=1e-8)
    a = p.parse_args()
    if min(a.k, a.threads, a.q, a.init_q, a.refine) < 1 or a.smooth < 0:
        p.error('Budgets, widths, threads, and refinement size must be positive; smooth must be nonnegative.')
    if a.terminals is not None and a.terminals < 1:
        p.error('--terminals must be positive.')
    if not 0 < a.delta < 1 or not 0 < a.rtol < 1 or any(not 0 < e < 1 for e in a.eps):
        p.error('delta, rtol, and eps must lie strictly between zero and one.')
    if len(set(a.eps)) != len(a.eps) or (a.seeds and len(set(a.seeds)) != len(a.seeds)):
        p.error('Accuracy levels and seeds must be unique.')
    if a.seeds and any(seed < 0 for seed in a.seeds):
        p.error('Seeds must be nonnegative.')
    return a


def main():
    a = parse_args()
    # Set native-library limits before importing NumPy, SciPy, or Numba.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = str(a.threads)
    import numpy as np
    import scipy
    import numba
    from . import core as E
    numba.set_num_threads(a.threads)

    datasets = json.loads((ROOT / 'datasets.json').read_text(encoding='utf-8'))
    names = [d['name'] for d in datasets]
    if a.graphs and (len(set(a.graphs)) != len(a.graphs) or set(a.graphs) - set(names)):
        raise SystemExit('Unknown or repeated dataset name. Available: ' + ', '.join(names))
    if a.graphs:
        datasets = [d for d in datasets if d['name'] in a.graphs]
    if a.out.exists() and not a.resume:
        raise SystemExit('Output already exists. Choose a new path or use --resume.')

    config = dict(k=a.k, q=a.q, init_q=a.init_q, terminals=a.terminals,
                  smooth=a.smooth, refine=a.refine, delta=a.delta, rtol=a.rtol,
                  threads=a.threads, init_seed=12345, batch=32, r_max=4096)
    code_sha = hashlib.sha256((ROOT / 'core.py').read_bytes()).hexdigest()
    existing = {}
    if a.out.exists():
        for number, line in enumerate(a.out.read_text(encoding='utf-8').splitlines(), 1):
            try:
                r = json.loads(line)
            except ValueError:
                raise SystemExit('Invalid JSON on output line %d; repair the interrupted record first.' % number)
            if r.get('config') != config or r.get('code_sha') != code_sha:
                raise SystemExit('Existing output uses a different configuration or implementation.')
            key = (r['graph'], r['method'], r.get('eps_s'), r.get('seed'))
            if key in existing:
                raise SystemExit('Duplicate completed run in existing output: ' + str(key))
            existing[key] = r
    a.out.parent.mkdir(parents=True, exist_ok=True)
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, numba=numba.__version__)

    for item in datasets:
        name = item['name']
        seeds = a.seeds if a.seeds is not None else item['seeds']
        run_exact = a.exact == 'yes' or (a.exact == 'auto' and item['exact'])
        plan = ([('Exact', None, None)] if run_exact else []) + [
            ('Schur', eps, seed) for eps in a.eps for seed in seeds]
        pending = [(method, eps, seed) for method, eps, seed in plan
                   if (name, method, eps, seed) not in existing]
        if not pending:
            print(name + ': all requested runs already complete.', flush=True)
            continue
        path = a.data_root / item['path']
        print('Loading ' + name, flush=True)
        start = time.perf_counter()
        g, info = E.load(path)
        info['load_seconds'] = time.perf_counter() - start
        if info['sha256'] != item['sha256'] or (g.n, g.m) != (item['n'], item['m']):
            raise SystemExit('Dataset checksum or processed size mismatch: ' + name)
        if a.k > g.m:
            raise SystemExit('k exceeds the number of processed edges: ' + name)
        old = [r for key, r in existing.items() if key[0] == name]
        if old:
            # Reuse the common initialization and its timing on a resumed graph.
            r = old[0]
            if any(x['sha256'] != info['sha256'] or x['e1'] != r['e1'] for x in old):
                raise SystemExit('Inconsistent resumed graph records: ' + name)
            e1 = r['e1']
            ist = {k: v for k, v in r.items() if k.startswith('init_')}
            h1 = r['H_e1']
        else:
            e1, ist, _ = E.approx_singleton(g, q=a.init_q, seed=12345, rtol=a.rtol)
            h1 = E.egwc(g, np.arange(g.m) == e1)
        base = dict(graph=name, **info, **ist, k=a.k, e1=e1, H_e1=h1,
                    config=config, code_sha=code_sha, environment=versions)
        for method, eps, seed in pending:
            if method == 'Exact':
                selected, seconds = E.exact_greedy(g, a.k, e1)
                extra = {}
            else:
                selected, seconds, rounds = E.schur_greedy(
                    g, a.k, e1, eps_s=eps, delta=a.delta, q=a.q, seed=seed,
                    threads=a.threads, b=a.terminals, smooth=a.smooth,
                    refine=a.refine, rtol=a.rtol)
                extra = dict(eps_s=eps, q=a.q, rounds=rounds)
            mask = np.zeros(g.m, dtype=bool)
            mask[selected] = True
            record = dict(base, method=method, seed=seed, selected=selected,
                          select_seconds=seconds, total_seconds=ist['init_seconds'] + seconds,
                          objective=E.egwc(g, mask), **extra)
            with a.out.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            print('%s %s eps=%s seed=%s: H=%.8g, time=%.3fs' %
                  (name, method, eps, seed, record['objective'], record['total_seconds']), flush=True)


if __name__ == '__main__':
    main()
