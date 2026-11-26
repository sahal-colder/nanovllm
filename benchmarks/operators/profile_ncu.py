#!/usr/bin/env python3
"""Collect real NCU reports; preserve permission failures without fake metrics."""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys

METRICS = (
    'dram__throughput.avg.pct_of_peak_sustained_elapsed,'
    'dram__bytes.sum.per_second,'
    'sm__warps_active.avg.pct_of_peak_sustained_active'
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'results/ncu')
    parser.add_argument('--dtype', choices=['all', 'f32', 'f16', 'bf16'], default='all')
    parser.add_argument('--ncu', help='Explicit NCU executable, useful with sudo secure_path')
    parser.add_argument('--import-existing', action='store_true', help='Export previously captured reports without rerunning GPU work')
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error('benchmark binary does not exist')
    ncu = args.ncu or shutil.which('ncu')
    if not ncu:
        parser.error('ncu is not installed or not on PATH')
    args.out.mkdir(parents=True, exist_ok=True)
    dtypes = ['f32', 'f16', 'bf16'] if args.dtype == 'all' else [args.dtype]
    status = []
    for op, kernel in [('gemv', 'mi_gemv_vec4_kernel'), ('rms', 'mi_rms_vec4_kernel')]:
        for dtype in dtypes:
            stem = args.out.resolve() / f'{op}-{dtype}'
            # Refuse stale reports, so an earlier successful capture cannot mask
            # a failed current attempt. Select a fresh output directory to rerun.
            existing = [stem.with_suffix(s) for s in ['.ncu-rep', '.ncu-repz'] if stem.with_suffix(s).is_file()]
            if existing and not args.import_existing:
                parser.error(f'existing report: {stem}.ncu-rep; choose a fresh --out')
            if args.import_existing and len(existing) != 1:
                parser.error(f'expected exactly one .ncu-rep/.ncu-repz for {stem}')
            command = [ncu, '--metrics', METRICS, '--kernel-name', f'regex:{kernel}',
                       '--launch-count', '1', '--clock-control', 'none',
                       '--cache-control', 'all', '--export', str(stem),
                       str(binary), '--profile', op, '--dtype', dtype]
            if args.import_existing:
                exit_code = 0
                log = stem.with_suffix('.log').read_text()
            else:
                result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                exit_code = result.returncode
                log = result.stdout
                stem.with_suffix('.log').write_text('$ ' + ' '.join(command) + '\n' + log)
            reports = [stem.with_suffix(s) for s in ['.ncu-rep', '.ncu-repz'] if stem.with_suffix(s).is_file()]
            report = reports[0] if len(reports) == 1 else stem.with_suffix('.ncu-rep')
            reason = ''
            if 'ERR_NVGPUCTRPERM' in log:
                reason = 'ERR_NVGPUCTRPERM'
            elif exit_code or not report.is_file():
                reason = 'capture_failed_or_no_report'
            if not reason:
                exported = subprocess.run([ncu, '--import', str(report), '--csv', '--page', 'raw'],
                                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stem.with_suffix('.export.log').write_text(exported.stderr)
                if exported.returncode or any(metric not in exported.stdout for metric in METRICS.split(',')):
                    reason = 'export_failed_or_missing_metrics'
                else:
                    stem.with_suffix('.csv').write_text(exported.stdout)
            status.append(dict(operator=op, dtype=dtype, status='blocked' if reason else 'captured',
                               exit_code=exit_code, reason=reason,
                               report=report.name if report.is_file() else '',
                               csv=stem.with_suffix('.csv').name if not reason else ''))
            print(f'{op}/{dtype}: {reason or "captured"}', flush=True)
    with (args.out / 'status.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(status[0]))
        writer.writeheader()
        writer.writerows(status)
    return 1 if any(s['status'] != 'captured' for s in status) else 0

if __name__ == '__main__':
    sys.exit(main())
