"""Single selected-target UI adapter; full detection is never passed into MOT.

KF modes still require detections each frame. This is a gated detector/KF
baseline, not the detector-off KCF/CSRT SOT protocol. Only one candidate enters
association per frame; identity after loss remains unverified.
"""
import numpy as np
from benchmark_system import Engine, best_match


class SelectedTargetEngine(Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gated_detections = 0
        self.available_detections = 0

    def update(self, image, detections):
        self.available_detections = len(detections)
        if self.is_mot:
            reference = self.box if self.box is not None else self.selection
            candidates = detections
            if self.selected_class is not None:
                candidates = candidates[candidates[:,5] == self.selected_class]
            index = best_match(candidates[:,:4], reference, minimum=.1) if reference is not None else None
            detections = candidates[index:index+1] if index is not None else np.empty((0,6))
        self.gated_detections = len(detections)
        return super().update(image, detections)
