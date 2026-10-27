"""Run one owned DFC child job; stop it if disk/RAM approaches exhaustion."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import psutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--cache',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();command=a.command
if command and command[0]=='--':command=command[1:]
if not command:p.error('command required')
a.output.mkdir(parents=True,exist_ok=True);a.cache.mkdir(parents=True,exist_ok=True)
child=subprocess.Popen(command,start_new_session=True)
reason=None
while child.poll() is None:
    disk=psutil.disk_usage(a.output).free
    cache=psutil.disk_usage(a.cache).free
    ram=psutil.virtual_memory().available
    if min(disk,cache)<10*2**30 or ram<40*2**30:
        reason=dict(error='Resource guard stopped this job',disk_free_gb=disk/2**30,cache_free_gb=cache/2**30,ram_available_gb=ram/2**30)
        (a.output/'resource_guard.json').write_text(json.dumps(reason,indent=2))
        try:
            proc=psutil.Process(child.pid)
            descendants=proc.children(recursive=True)
            for item in descendants:item.terminate()
            os.killpg(child.pid,signal.SIGTERM)
            _,alive=psutil.wait_procs(descendants,timeout=5)
            for item in alive:item.kill()
        except (psutil.Error,ProcessLookupError):pass
        break
    time.sleep(1)
code=child.wait()
print(json.dumps(dict(child_exit_code=code,guard_reason=reason)),flush=True)
raise SystemExit(code if code>=0 else 3)
