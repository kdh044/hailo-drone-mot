# 1단계: RemDet(PyTorch) -> ONNX. 헤드의 마지막 conv 6개까지만 내보낸다 (DFL decode / sigmoid / NMS 제외).
# usage: python export_onnx.py <config.py> <checkpoint.pth> <out.onnx>
#   예: python export_onnx.py config_remdet/remdet/remdet_s-300e_visdrone.py remdet_s.pth remdet_s.onnx
import sys, torch
from mmdet.apis import init_detector

cfg, ckpt, out = sys.argv[1:4]
model = init_detector(cfg, ckpt, device="cpu").eval()


class RawHead(torch.nn.Module):
    """backbone + neck + 스케일별 (cls conv, reg conv). 출력 6개: cls/reg x stride 8, 16, 32"""
    def __init__(self, m):
        super().__init__()
        self.m, self.h = m, m.bbox_head.head_module

    def forward(self, x):
        feats = self.m.neck(self.m.backbone(x))
        outs = []
        for f, cls, reg in zip(feats, self.h.cls_preds, self.h.reg_preds):
            outs += [cls(f), reg(f)]
        return tuple(outs)


names = ["cls_p3", "reg_p3", "cls_p4", "reg_p4", "cls_p5", "reg_p5"]
torch.onnx.export(RawHead(model), torch.zeros(1, 3, 640, 640), out, opset_version=11,
                  input_names=["images"], output_names=names)
print("saved", out)
