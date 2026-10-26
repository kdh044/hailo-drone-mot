"""Run only the Hailo-8 parser; does not quantize or claim HEF compatibility."""
import argparse
import json
from pathlib import Path
import platform
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    import hailo_sdk_client
    from hailo_sdk_client import ClientRunner
    report=dict(model=a.model.name,architecture=platform.machine(),dfc=hailo_sdk_client.__version__,
        target='hailo8',stage='parse_only',calibration='not_run',quantization='not_run',compilation='not_run')
    start=time.perf_counter()
    try:
        runner=ClientRunner(hw_arch='hailo8')
        runner.translate_onnx_model(str(a.model),net_name='ostrack384')
        har=a.model.with_suffix('.har')
        runner.save_har(str(har))
        report.update(parse='passed',har=har.name)
    except Exception as e:
        report.update(parse='failed',error_type=type(e).__name__,error=str(e).replace(str(a.model),a.model.name))
        report['traceback']=traceback.format_exc().replace(str(a.model),a.model.name)
    report['elapsed_seconds']=time.perf_counter()-start
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0 if report['parse']=='passed' else 2)


if __name__=='__main__':
    main()
