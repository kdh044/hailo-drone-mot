"""Compile an existing >=1024-image level-4 quantized HAR with layout overrides."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--har',type=Path,required=True)
    p.add_argument('--quant-report',type=Path,required=True)
    p.add_argument('--alls',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    source=json.loads(a.quant_report.read_text())
    if source.get('quantization')!='passed' or source.get('samples',0)<1024 or source.get('optimization_level')!=4:
        p.error('A passed >=1024-image level-4 quantization report is required')
    from hailo_sdk_client import ClientRunner
    a.output_dir.mkdir(parents=True,exist_ok=True)
    commands=a.alls.read_text()
    report=dict(model=source.get('model'),quantization_samples=source['samples'],
        optimization_level=source['optimization_level'],calibration_changed=False,
        compiler_script=commands,compile='running')
    report_path=a.output_dir/'report.json'
    report_path.write_text(json.dumps(report,indent=2))
    started=time.perf_counter()
    try:
        runner=ClientRunner(har=str(a.har))
        runner.load_model_script(commands)
        hef=runner.compile()
        output=a.output_dir/'ostrack384.hef'
        output.write_bytes(hef)
        report.update(compile='passed',hef_bytes=len(hef),hef_sha256=hashlib.sha256(hef).hexdigest())
    except Exception as error:
        report.update(compile='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    report['seconds']=time.perf_counter()-started
    report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='traceback'},indent=2),flush=True)
    raise SystemExit(0 if report['compile']=='passed' else 2)


if __name__=='__main__':main()
