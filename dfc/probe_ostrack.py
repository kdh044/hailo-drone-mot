"""Export the official OSTrack checkpoint and verify FP32 ONNX on CPU.

This checks export/runtime correctness, NOT Hailo support or tracking accuracy.
The trained model and candidate elimination remain intact. Box decoding is CPU
postprocessing; the exported outputs are score/size/offset maps.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.home()/'ext/OSTrack')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--disable-ce',action='store_true',help='Explicit compatibility variant: disable candidate elimination; accuracy is no longer the original configuration')
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    sys.path.insert(0,str(a.repo))
    from lib.config.ostrack.config import cfg,update_config_from_file
    from lib.models.ostrack import build_ostrack
    from lib.utils.ce_utils import generate_mask_cond
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    state=checkpoint.get('net',checkpoint.get('state_dict',checkpoint))
    n=int(state['backbone.pos_embed_x'].shape[1])
    size=int(n**.5)*16
    if size not in (256,384):
        raise ValueError(f'Unrecognized checkpoint search size {size}')
    config=a.repo/f'experiments/ostrack/vitb_{size}_mae_ce_32x4_ep300.yaml'
    update_config_from_file(str(config))
    model=build_ostrack(cfg,training=False)
    model.load_state_dict(state,strict=True)
    model.eval()
    del checkpoint,state
    mask=generate_mask_cond(cfg,1,torch.device('cpu'),None)

    class Maps(torch.nn.Module):
        def __init__(self,net):
            super().__init__()
            self.net=net
            self.register_buffer('mask',mask)
        def forward(self,template,search):
            out=self.net(template=template,search=search,ce_template_mask=self.mask,
                         ce_keep_rate=1.0 if a.disable_ce else None)
            return out['score_map'],out['size_map'],out['offset_map']

    wrapper=Maps(model).eval()
    tsize=cfg.TEST.TEMPLATE_SIZE
    template=torch.randn(1,3,tsize,tsize)
    search=torch.randn(1,3,size,size)
    with torch.inference_mode():
        wrapper(template,search)
        start=time.perf_counter()
        expected=wrapper(template,search)
        cpu_ms=(time.perf_counter()-start)*1000
        expected=[x.numpy() for x in expected]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    torch.onnx.export(wrapper,(template,search),str(a.output),opset_version=17,
        input_names=['template','search'],output_names=['score_map','size_map','offset_map'],
        dynamic_axes=None,do_constant_folding=True,dynamo=False)
    export_seconds=time.perf_counter()-start
    graph=onnx.load(str(a.output))
    onnx.checker.check_model(graph)
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    session=ort.InferenceSession(str(a.output),sess_options=options,providers=['CPUExecutionProvider'])
    actual=session.run(None,dict(template=template.numpy(),search=search.numpy()))
    errors=[float(np.max(np.abs(x-y))) for x,y in zip(expected,actual)]
    for x,y in zip(expected,actual):
        np.testing.assert_allclose(x,y,rtol=2e-3,atol=2e-4)
    counts=dict(Counter(n.op_type for n in graph.graph.node))
    review=['MatMul','Softmax','LayerNormalization','TopK','GatherElements','ScatterElements','NonZero','GatherND']
    report=dict(model=f'OSTrack-{size}',checkpoint_source='official OSTrack README-linked ModelScope damo/cv_vitb_video-single-object-tracking_ostrack',
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        upstream_commit=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),
        architecture=platform.machine(),torch=torch.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__,
        pretrained_weights_loaded_strictly=True,candidate_elimination_preserved=not a.disable_ce,
        template_shape=list(template.shape),search_shape=list(search.shape),
        output_shapes=[list(x.shape) for x in actual],cpu_single_forward_ms=cpu_ms,
        export_seconds=export_seconds,onnx_checker='passed',onnxruntime_comparison='passed',max_abs_errors=errors,
        operator_counts=counts,operators_to_review_for_hailo={k:counts[k] for k in review if k in counts},
        hailo_dfc_available=importlib.util.find_spec('hailo_sdk_client') is not None,
        hailo_parse='not_run',quantization_calibration='not_run',hef_compilation='not_run',
        accuracy_evaluation='not_run; random tensors used only for numeric export verification',
        note='CPU graph validity does not establish Hailo-8 compatibility. Parse on the DFC host before preparing calibration. Calibration needs representative paired template/search inputs and matching preprocessing.')
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
