"""Aggregate recorded EGWC runs into CSV, Markdown, and LaTeX tables."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
EPS = [0.2, 0.1, 0.05]


def aggregate(path):
    manifest = json.loads((ROOT / 'datasets.json').read_text(encoding='utf-8'))
    records = [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
    seen = set()
    groups = {}
    for r in records:
        key = (r['graph'], r['method'], r.get('eps_s'), r.get('seed'))
        if key in seen:
            raise ValueError('Duplicate run: ' + str(key))
        seen.add(key)
        groups.setdefault(r['graph'], []).append(r)
    if len({r['k'] for r in records}) > 1:
        raise ValueError('Cannot aggregate different edge budgets into one table.')
    rows = []
    for item in manifest:
        rr = groups.get(item['name'], [])
        if not rr:
            continue
        if any((r['n'], r['m'], r['sha256']) != (item['n'], item['m'], item['sha256']) for r in rr):
            raise ValueError('Dataset metadata mismatch: ' + item['name'])
        if len({r['e1'] for r in rr}) != 1:
            raise ValueError('Methods do not share their initial edge: ' + item['name'])
        if len({json.dumps(r.get('config'), sort_keys=True) for r in rr}) != 1:
            raise ValueError('Mixed configurations for ' + item['name'])
        exact = [r for r in rr if r['method'] == 'Exact']
        row = dict(network=item['name'], n=item['n'], m=item['m'], k=rr[0]['k'],
                   init_seconds=rr[0]['init_seconds'],
                   exact_seconds=exact[0]['total_seconds'] if exact else None,
                   exact_objective=exact[0]['objective'] if exact else None)
        terminals = [rd['terminals'] for r in rr for rd in r.get('rounds', [])]
        row['b'] = terminals[0] if terminals else None
        for eps in EPS:
            ss = [r for r in rr if r['method'] == 'Schur' and r['eps_s'] == eps]
            tag = str(eps)
            row['runs_' + tag] = len(ss)
            row['seconds_' + tag] = mean(r['total_seconds'] for r in ss) if ss else None
            row['objective_' + tag] = mean(r['objective'] for r in ss) if ss else None
            forests = [rd['forests'] for r in ss for rd in r['rounds']]
            row['forests_' + tag] = mean(forests) if forests else None
            row['seeds_' + tag] = ','.join(str(r['seed']) for r in sorted(ss, key=lambda r: r['seed']))
        rows.append(row)
    if not rows:
        raise ValueError('No recognized experiment records found.')
    return rows


def format_row(r):
    def value(x, time=False, integer=False):
        if x is None:
            return '--'
        return ('{:,d}'.format(int(x)) if integer else
                format(x, ',.1f' if time and x >= 100 else ',.2f'))
    return [r['network'], value(r['n'], integer=True), value(r['m'], integer=True), value(r['b'], integer=True),
            value(r['init_seconds'], time=True), value(r['exact_seconds'], time=True)] + [
                value(r['seconds_' + str(e)], time=True) for e in EPS] + [
                value(r['exact_objective'])] + [value(r['objective_' + str(e)]) for e in EPS] + [
                '--' if r['forests_' + str(e)] is None else format(r['forests_' + str(e)], ',.0f') for e in EPS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    rows = aggregate(a.input)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.output_dir / 'table1.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = ['Network', 'n', 'm', 'b', 'Init (s)', 'Exact (s)', '0.2 (s)', '0.1 (s)', '0.05 (s)',
              'Exact H', 'H 0.2', 'H 0.1', 'H 0.05', 'Forests 0.2', 'Forests 0.1', 'Forests 0.05']
    lines = ['# EGWC results', '', 'Times include shared initialization; forests are averaged over rounds and runs.', '',
             '| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    for r in rows:
        lines.append('| ' + ' | '.join(format_row(r)) + ' |')
    lines += ['', '## Independent runs', '', '| Network | 0.2 | 0.1 | 0.05 |', '| --- | --- | --- | --- |']
    for r in rows:
        lines.append('| ' + r['network'] + ' | ' + ' | '.join(str(r['runs_' + str(e)]) for e in EPS) + ' |')
    (a.output_dir / 'table1.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    tex = [r'\begin{tabular}{lrrrrrrrrrrrrrrr}', r'\toprule',
           ' & '.join(x.replace(' (s)', '').replace(' H', '') for x in header) + r' \\', r'\midrule']
    tex += [' & '.join(format_row(r)) + r' \\' for r in rows]
    tex += [r'\bottomrule', r'\end{tabular}']
    (a.output_dir / 'table1.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    print('Wrote table1.csv, table1.md, and table1.tex for %d networks.' % len(rows))


if __name__ == '__main__':
    main()
