"""Release this job's private tmpfs cache after its storage guard exits."""
import argparse,json,shutil
from pathlib import Path
import psutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--guard-pid',type=int,required=True)
p.add_argument('--cache',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();cache=a.cache.resolve()
if cache.parent!=Path('/dev/shm') or not cache.name.startswith('danny_ostrack_'):
    p.error('Only this task private tmpfs directories may be cleaned')
try:
    process=psutil.Process(a.guard_pid)
    command=process.cmdline()
    if not any(Path(v).name=='run_storage_guard.py' for v in command) or str(a.cache) not in command:
        p.error('PID is not this cache storage guard')
    while process.is_running():
        try: process.wait(timeout=5)
        except psutil.TimeoutExpired: pass
except psutil.NoSuchProcess: pass
report_path=a.output/'quant_compile_probe.json'
if report_path.exists():
    report=json.loads(report_path.read_text())
    running=[k for k in ('calibration','quantization','compilation') if report.get(k)=='running']
    if running:
        for stage in running:report[stage]='failed'
        report['error']='Optimizer process ended without a final stage report; inspect resource_guard.json and job log'
        report_path.write_text(json.dumps(report,indent=2))
shutil.rmtree(cache,ignore_errors=False)
(a.output/'cache_cleanup.json').write_text(json.dumps(dict(private_tmpfs_cache_removed=True)))
print('Private optimizer cache cleaned',flush=True)
