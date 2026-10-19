"""CPU adapters for UAV-specific MOT trackers, run on cached detections.

Adapters: SFTrackAdapter, DetectTrackUAVAdapter, SMOTOCSortAdapter (YOLOv8-SMOT OC-SORT),
HoldfastAdapter (C++20 via ctypes), MosaicAdapter (MOSAIC-Tracker S_Ioutrack),
CSFTSTAdapter (CSF-TST CFS_TST_Tracker).

Common interface
----------------
    trk = SFTrackAdapter(**kw)            # or any adapter below
    out = trk.update(dets, frame)         # call once per frame, in order, starting at frame 1

    dets  : float32 (N, 6) [x1, y1, x2, y2, score, cls] in original image pixels
    frame : full-resolution BGR uint8 image (CMC + SFTrack colour/template cues)
    out   : float32 (M, 6) [x1, y1, x2, y2, track_id, score]  -- exactly the boxes the repo's
            own result writer would put into the MOT txt for this frame (incl. its box filters).

One instance = one sequence & one class group. Track-ID counters are kept per instance
(the repos use a global ``BaseTrack._count`` / ``KalmanBoxTracker.count``; we swap it in/out around
every update so several instances can be interleaved in one process without ID collisions).
IDs start at 1.

``update`` is called for every frame even when ``dets`` is empty (the repo scripts skip the
tracker only when the detector returned nothing at all; with per-class-group splitting an empty
group is common, and skipping would desynchronise the lost-track age / CMC previous frame).

CMC state: every tracker instance owns its own CMC object (no class-level/shared CMC state in any
copied file). Additionally every Python CMC object is wrapped by ``_guard_cmc``: if the incoming
frame shape differs from the previous frame's, the CMC is reset (identity warp for that frame)
instead of running LK / descriptor matching across resolutions. Needed because VisDrone val has
stray odd-sized frames *inside* sequences: uav0000117_02622_v/0000349.jpg is 1904x1071 (rest
2720x1530) and uav0000182_00000_v/0000363.jpg is 2720x1530 (rest 1344x756) -> previously
cv2.calcOpticalFlowPyrLK "prevPyr[...].size() == nextPyr[...].size()" on the last frame.
Holdfast's C++ estimator gets the same full-resolution-shape reset inside the C wrapper.

No adapter sets torch threads; none of the copied tracker files imports torch.
"""
import os
import sys
import types
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DTU_ROOT = os.path.expanduser('~/ext/DetectTrack-UAV')


def _as_dets(dets):
    dets = np.asarray(dets, dtype=np.float32)
    if dets.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return dets.reshape(-1, 6)


def _pack(rows):
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _guard_cmc(gmc):
    """Wrap ``gmc.apply`` (BoT-SORT-style GMC/CMC classes) so a change of frame shape resets the
    CMC to its first-frame state (identity warp) instead of matching across resolutions."""
    if gmc is None or getattr(gmc, '_hailo_guarded', False):
        return gmc
    orig = gmc.apply
    last = {'shape': None}

    def apply(raw_frame, *a, **k):
        shape = None if raw_frame is None else tuple(raw_frame.shape)
        if last['shape'] is not None and shape != last['shape']:
            gmc.initializedFirstFrame = False
            gmc.prevFrame = None
            gmc.prevKeyPoints = None
            gmc.prevDescriptors = None
        last['shape'] = shape
        return orig(raw_frame, *a, **k)
    gmc.apply = apply
    gmc._hailo_guarded = True
    return gmc


# --------------------------------------------------------------------------------------------
# SFTrack (IROS 2024)
# --------------------------------------------------------------------------------------------
class SFTrackAdapter:
    """SFTrack (Song et al., IROS 2024) -- ``yolox/tracker/sf_tracker.py::SFTracker``.

    Code: minimal copy of the 5 tracker files in ``ext/sftrack/`` (importing ``yolox`` pulls in the
    YOLOX model package), patched only for: relative imports, ``np.float`` -> ``np.float64``
    (numpy 2), ``cython_bbox`` -> numpy port with identical +1-pixel IoU (``ext/sftrack/_bbox.py``),
    unused torch imports dropped, GMC first-frame return made a tuple, RANSAC ``None`` -> identity.

    Kept (part of the method): BoT/Byte-style two-stage association; high/low split at
    ``track_thresh``; low-score (0.1 < s < track_thresh) association using IoU x colour-histogram
    (8x8x8 BGR, Bhattacharyya) x template-MSE (64x64) with top-3 pruning on previous/current frame
    crops; unconfirmed matching with the same fused cue; new-track init from unmatched high dets
    (score >= track_thresh + 0.1) **and** from unmatched low dets; uniform-scale sparse-optical-flow
    CMC; output = activated tracks only.

    Removed: fast-reid embeddings (``id_feature``). Every detection gets the same constant dummy
    feature, so cosine distance == 0 everywhere. Effects: (a) first association
    ``fuse_score_three`` = IoU similarity only; (b) ``fuse_score_five`` never used ReID anyway;
    (c) the ReID gate for low-score track initiation (mean cosine distance to remaining high dets
    <= 0.25) always passes when >=1 unmatched high det exists in the frame (if none, the repo's NaN
    check still skips init). (A per-det feature trick to disable that gate would leak into the
    first-stage cost via the track's stored feature, so no such option is offered.)

    Hyperparameters (VisDrone evaluation path = ``tools/track.py`` + ``MOTEvaluator.evaluate_sftrack``):
        track_thresh = 0.6       (tools/track.py default; det_thresh for new tracks = 0.7)
        track_buffer = 30        (evaluator forces 30), frame_rate = 30 -> max_time_lost = 30
        match thresholds 0.7 / 0.5 / 0.7 (hard-coded in sf_tracker; ``--match_thresh`` unused)
        low-score floor 0.1      (hard-coded)
        min_box_area = 100       (tools/track.py ``--min-box-area``; demo_track.py uses 10)
        no aspect-ratio filter   (commented out in the evaluator)
        CMC: GMC('sparseOptFlow', downscale=2): goodFeaturesToTrack(1000 corners) + pyramidal LK
             + estimateAffinePartial2D(RANSAC); only the larger scale factor is applied.
    Scores: detector score is used directly (repo uses obj*cls_conf from YOLOX).
    """

    def __init__(self, track_thresh=0.6, track_buffer=30, frame_rate=30, min_box_area=100.0,
                 name='seq'):
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from ext.sftrack import sf_tracker, basetrack
        self._bt = basetrack.BaseTrack
        self._sf = sf_tracker
        args = types.SimpleNamespace(track_thresh=float(track_thresh), track_buffer=int(track_buffer),
                                     name=name, mot20=False)
        self.tracker = sf_tracker.SFTracker(args, frame_rate=frame_rate)
        _guard_cmc(self.tracker.gmc)
        self.min_box_area = float(min_box_area)
        self._count = 0

    def update(self, dets, frame):
        dets = _as_dets(dets)
        out5 = np.ascontiguousarray(dets[:, :5], dtype=np.float64)  # repo divides bboxes in place
        feats = np.ones((len(out5), 8), dtype=np.float64)  # constant dummy 'ReID' feature
        h, w = frame.shape[:2]
        saved = self._bt._count
        self._bt._count = self._count
        try:
            with warnings.catch_warnings():  # repo's ReID gate does np.mean([]) -> NaN (expected)
                warnings.simplefilter('ignore', RuntimeWarning)
                targets = self.tracker.update(out5, [h, w], feats, [h, w], frame)
        finally:
            self._count = self._bt._count
            self._bt._count = saved
        rows = []
        for t in targets:
            tlwh = t.tlwh
            if tlwh[2] * tlwh[3] > self.min_box_area:
                rows.append([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3], t.track_id, t.score])
        return _pack(rows)


# --------------------------------------------------------------------------------------------
# DetectTrack-UAV (FusionSORT-UAV)
# --------------------------------------------------------------------------------------------
def _import_detecttrack():
    """Import ~/ext/DetectTrack-UAV/tracker/fusion_sort_uav.py without the OSNet ReID module."""
    if 'reid.osnet_ibn_x1_0_interface' not in sys.modules:
        # stub: fusion_sort_uav imports OSNetReIDInterface at module level (only used with_appearance)
        pkg = sys.modules.get('reid') or types.ModuleType('reid')
        pkg.__path__ = getattr(pkg, '__path__', [])
        stub = types.ModuleType('reid.osnet_ibn_x1_0_interface')

        class OSNetReIDInterface:  # noqa: D401
            def __init__(self, *a, **k):
                raise RuntimeError('ReID disabled in DetectTrackUAVAdapter')
        stub.OSNetReIDInterface = OSNetReIDInterface
        sys.modules.setdefault('reid', pkg)
        sys.modules['reid.osnet_ibn_x1_0_interface'] = stub
    if _DTU_ROOT not in sys.path:
        sys.path.insert(0, _DTU_ROOT)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):  # silence "cython_bbox not installed" print
        from tracker import fusion_sort_uav, matching, basetrack
    # numpy port of cython_bbox (+1 px convention) so iou_distance takes the repo's preferred path
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from ext.sftrack._bbox import bbox_overlaps
    matching.bbox_ious = bbox_overlaps
    matching.is_cython_bbox = True
    return fusion_sort_uav, basetrack


class DetectTrackUAVAdapter:
    """DetectTrack-UAV (``tracker/fusion_sort_uav.py::FusionSORTUAV``), imported in place from
    ~/ext/DetectTrack-UAV via sys.path (repo is already numpy-2 clean). Patches at import time
    only: ``reid.osnet_ibn_x1_0_interface`` stubbed; ``matching.bbox_ious`` set to a numpy port of
    cython_bbox (repo comment: cython_bbox IoU "gives a better result" than its iou_batch fallback).

    This is the README's "motion only" VisDrone configuration (``--default-parameters``, no
    ``--with_appearance`` / ``--with-hiou`` / ``--with-confidence``): ByteTrack-style two-stage
    association, 10-D Kalman filter on (x, y, w, h, score) + velocities (``KalmanFilterScore``),
    first-stage cost = 1 - IoU*det_score, second stage plain IoU (thresh 0.5), unconfirmed 0.7.
    Output = all tracked (not lost) tracks *including unconfirmed* ones (repo comments out the
    ``is_activated`` filter).

    Removed: OSNet-IBN ReID (``with_appearance=False``; it is opt-in in the repo anyway).
    There is no non-CNN appearance cue in this method.

    Hyperparameters (run_tracker_uav.py with --default-parameters, benchmark VisDrone):
        track_high_thresh = 0.6, track_low_thresh = 0.1, new_track_thresh = 0.7 (= high + 0.1)
        track_buffer = 30, fps = 30 -> max_time_lost = 30
        match_thresh = 0.8, filter_type = 'KF', with_nsa = False, second_matching_distance = 'iou'
        iou_thresh = 0.5, appearance_thresh = 0.25, lambda1..4 = 1.0/0.15/0.1/0.1 (unused w/o cues)
        writer filters: min_box_area = 10 and drop boxes with w/h > aspect_ratio_thresh = 1.6
            (applied to every class in the repo; pass aspect_ratio_thresh=None to disable)
        CMC: run script default ``--cmc-method none`` and README commands don't override it ->
            disabled by default. ``cmc_method='sparseOptFlow'|'orb'|'ecc'|'sift'`` enables the repo's
            CPU CMC (cmc_uav.CMC, downscale 2).
    Detections are fed as 7 columns [x1,y1,x2,y2,score,1.0,cls] so score = score*1 and the class
    is read from the last column (the repo's 5-column branch would read the class from the score).
    """

    def __init__(self, track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=None,
                 track_buffer=30, match_thresh=0.8, frame_rate=30, min_box_area=10.0,
                 aspect_ratio_thresh=1.6, cmc_method='none', with_hiou=False, with_confidence=False,
                 name='seq'):
        fsu, basetrack = _import_detecttrack()
        self._bt = basetrack.BaseTrack
        if new_track_thresh is None:
            new_track_thresh = track_high_thresh + 0.1
        args = types.SimpleNamespace(
            track_high_thresh=float(track_high_thresh), track_low_thresh=float(track_low_thresh),
            new_track_thresh=float(new_track_thresh), track_buffer=int(track_buffer),
            match_thresh=float(match_thresh), filter_type='KF', with_nsa=False,
            iou_thresh=0.5, appearance_thresh=0.25, with_appearance=False,
            with_hiou=bool(with_hiou), with_confidence=bool(with_confidence),
            lambda1=1.0, lambda2=0.15, lambda3=0.1, lambda4=0.1, second_matching_distance='iou',
            cmc_method=cmc_method, benchmark='VisDrone', name=name, ablation=False,
            device='cpu', fast_reid_weights=None, fps=frame_rate)
        saved = self._bt._count
        self.tracker = fsu.FusionSORTUAV(args, frame_rate=frame_rate)  # calls BaseTrack.clear_count()
        _guard_cmc(self.tracker.cmc)
        self._bt._count = saved
        self._count = 0
        self.min_box_area = float(min_box_area)
        self.aspect_ratio_thresh = aspect_ratio_thresh

    def update(self, dets, frame):
        dets = _as_dets(dets).astype(np.float64)
        d7 = np.zeros((len(dets), 7), dtype=np.float64)
        d7[:, :5] = dets[:, :5]
        d7[:, 5] = 1.0
        d7[:, 6] = dets[:, 5]
        saved = self._bt._count
        self._bt._count = self._count
        try:
            targets = self.tracker.step(d7, frame)
        finally:
            self._count = self._bt._count
            self._bt._count = saved
        rows = []
        for t in targets:
            tlwh = t.tlwh
            if tlwh[3] <= 0:
                continue
            vertical = (self.aspect_ratio_thresh is not None and
                        tlwh[2] / tlwh[3] > self.aspect_ratio_thresh)
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                rows.append([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3], t.track_id, t.score])
        return _pack(rows)


# --------------------------------------------------------------------------------------------
# YOLOv8-SMOT (MVA 2025 SMOT4SB challenge) -- OC-SORT variant
# --------------------------------------------------------------------------------------------
class SMOTOCSortAdapter:
    """YOLOv8-SMOT (``trackers/ocsort_tracker/ocsort.py::OCSort``), minimal copy in ``ext/smot/``
    (ocsort.py, association.py, kalmanfilter.py; importing the repo pulls mmyolo/mmengine/torch).
    Patches: ``import torch`` dropped (only an isinstance check), the two "track_thresh decrease"
    prints silenced, and ``update`` additionally returns the score of each track's last observation
    (column 5; for a never-updated frame-1 track, its birth detection score). Repo is numpy-2 clean.

    Kept (the whole method -- it is motion-only): OC-SORT 7-D (x, y, s, r) Kalman filter
    (repo's ``KalmanFilterNew``) with observation-centric re-update (OOS freeze/unfreeze), velocity
    direction consistency (inertia) cost, EMA-smoothed direction (alpha 0.8), ``diou_expand``
    association (DIoU after enlarging both boxes by 0.5 x w/h per side; arXiv 2507.12087), BYTE
    low-score round, OCR round on last observations, and the SMOT-specific *adaptive threshold
    decay*: while fewer than ~11 high-score dets have been seen, if >10 frames (or >15 capped dets)
    with any det above the low threshold occur, (high, low) drop 0.25/0.1 -> 0.2/0.05, then after
    >20 frames / >30 dets -> 0.1/0.03 (designed for tiny, low-confidence birds; with normal
    VisDrone dets the counter saturates on the first frames and thresholds stay 0.25/0.1).
    Output = tracks updated this frame with hit_streak >= min_hits (or frame <= min_hits); box =
    last *observation* (detection box), not the KF state.

    Removed: nothing algorithmic (no ReID, no CMC in this method -> ``frame`` is unused).

    Hyperparameters (README example command == utils/args.py defaults, used by tools/predict.py):
        track_thresh (det_thresh) = 0.25, track_thresh_low = 0.1, iou_thresh = 0.25,
        iou_thresh_decrease = 0.08 (2nd/3rd rounds use 0.17), min_hits = 1, deltat = 1,
        asso = 'diou_expand', use_byte = True, inertia = 0.2 (OCSort default; predict.py does not
        pass --inertia), max_age = 30 (OCSort default; --track_buffer is not passed),
        writer: min_box_area = 10 (w*h > 10).
    Note: the repo's predict.py feeds 5 columns (score used as-is) and skips frames without any
    detection; we call update every frame (empty array) for the reason in the module docstring.
    """

    def __init__(self, track_thresh=0.25, track_thresh_low=0.1, iou_thresh=0.25,
                 iou_thresh_decrease=0.08, min_hits=1, deltat=1, asso='diou_expand', use_byte=True,
                 inertia=0.2, max_age=30, min_box_area=10.0):
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from ext.smot import ocsort
        self._kbt = ocsort.KalmanBoxTracker
        saved = self._kbt.count
        self.tracker = ocsort.OCSort(det_thresh=float(track_thresh), max_age=int(max_age),
                                     min_hits=int(min_hits), iou_threshold=float(iou_thresh),
                                     delta_t=int(deltat), asso_func=asso, inertia=float(inertia),
                                     use_byte=bool(use_byte), iou_thresh_decrease=float(iou_thresh_decrease),
                                     track_thresh_low=float(track_thresh_low))  # resets class count
        self._kbt.count = saved
        self._count = 0
        self.min_box_area = float(min_box_area)

    def update(self, dets, frame=None):
        dets = _as_dets(dets)
        d5 = np.ascontiguousarray(dets[:, :5], dtype=np.float64)
        saved = self._kbt.count
        self._kbt.count = self._count
        try:
            res = self.tracker.update(d5)
        finally:
            self._count = self._kbt.count
            self._kbt.count = saved
        rows = []
        for t in res:  # [x1, y1, x2, y2, id, score]
            if (t[2] - t[0]) * (t[3] - t[1]) > self.min_box_area:
                rows.append([t[0], t[1], t[2], t[3], t[4], t[5]])
        return _pack(rows)


# --------------------------------------------------------------------------------------------
# Holdfast (C++20) via ctypes
# --------------------------------------------------------------------------------------------
_HF_LIB = None


def _load_holdfast():
    global _HF_LIB
    if _HF_LIB is None:
        import ctypes
        path = os.path.join(_HERE, 'ext', 'holdfast', 'libholdfast_c.so')
        if not os.path.exists(path):
            raise RuntimeError(f'{path} missing; run ext/holdfast/build.sh')
        lib = ctypes.CDLL(path)
        lib.hf_create.restype = ctypes.c_void_p
        lib.hf_create.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        lib.hf_destroy.restype = None
        lib.hf_destroy.argtypes = [ctypes.c_void_p]
        lib.hf_update.restype = ctypes.c_int
        lib.hf_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_double, ctypes.c_void_p]
        lib.hf_fetch.restype = ctypes.c_int
        lib.hf_fetch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        _HF_LIB = lib
    return _HF_LIB


class HoldfastAdapter:
    """Holdfast (github.com/Amsozzer1/holdfast) -- C++20 ``holdfast::Tracker`` +
    ``CameraMotionEstimator``, compiled from the unmodified repo sources (src/bbox.cpp, kalman.cpp,
    hungarian.cpp, tracker.cpp, app/camera_motion.cpp) plus our 90-line C-ABI shim
    ``ext/holdfast/hf_capi.cpp`` into ``ext/holdfast/libholdfast_c.so`` (``ext/holdfast/build.sh``;
    g++ 14 -O3, Eigen 3.4.0 headers from ~/ext/eigen-3.4.0 because libeigen3-dev is not installed,
    system OpenCV 4.10). Loaded with ctypes. The shim reproduces src/app/pipeline.cpp per frame:
    ``A = cmc.estimate(img); stale = cmc.last_frame_difference() < 0.5;
    out = tracker.update({t, dets, A, stale})`` with t = (frame_idx - 1) / fps.

    Kept: the "full" config = ByteTrack-style 2-stage IoU association (Hungarian), time-based
    constant-velocity 8-D KF (cx, cy, w, h + velocities, DeepSORT size-proportional noise),
    recovery stage for lost tracks (position-only Mahalanobis gate + size-ratio gate),
    OC-SORT-style re-update over gaps, stale (frozen) frame guard, and CMC: goodFeaturesToTrack on a
    640-px-wide grayscale downscale, pyramidal LK (21x21, 3 levels), RANSAC similarity
    (estimateAffinePartial2D), warp applied to KF state/covariance and last observation.
    Output = Confirmed tracks matched this frame (all Confirmed on stale frames), KF box, last det
    score. Removed: ONNX detector, degradation simulator, renderer, MOT writer (no box filters in
    the repo's writer). Added in shim only: CMC estimator re-created if full-res frame shape changes
    (repo resets only on a change of the *downscaled* size; all 16:9 inputs downscale to 640x360).
    Detection class is ignored by the tracker (one instance per class group anyway).

    Hyperparameters (TrackerConfig::full() / holdfast_run defaults):
        high_score 0.5, low_score 0.1, new_track_score 0.6, stage1_max_cost 0.8 (IoU >= 0.2),
        stage2_max_cost 0.5, tentative_max_cost 0.7, n_init 3, gate_chi2 0 (off),
        variable_dt on, fps 30 (nominal_dt 1/30), max_age 1.5 s, recovery on (chi2 9.21, size ratio 2),
        oc_reupdate on (max 60 virtual steps), stale_guard on, cmc on;
        CMC params: work_width 640, max_corners 400, quality 0.01, min_distance 8, ransac_px 1.5,
        min_inliers 25; cv::setNumThreads(1) (repo --threads default 1; affects only the system
        OpenCV used by the .so, not the pip cv2).
    ``config`` accepts TrackerConfig::by_name specs, e.g. 'full', 'baseline', 'full+recovery=0'.
    """

    def __init__(self, config='full', fps=30.0, cv_threads=1, work_width=640, max_corners=400,
                 quality=0.01, min_distance=8.0, ransac_px=1.5, min_inliers=25, use_frame=True):
        import ctypes
        self._ct = ctypes
        self._lib = _load_holdfast()
        self._h = self._lib.hf_create(config.encode(), float(fps), int(cv_threads), int(work_width),
                                      int(max_corners), float(quality), float(min_distance),
                                      float(ransac_px), int(min_inliers))
        if not self._h:
            raise ValueError(f'unknown holdfast config {config!r}')
        self.fps = float(fps)
        self.use_frame = bool(use_frame)
        self._frame_idx = 0
        self._info = np.zeros(3, dtype=np.float64)  # [cmc_failed, inliers, frame_diff] of last frame

    def __del__(self):
        h, lib = getattr(self, '_h', None), getattr(self, '_lib', None)
        if h and lib is not None:
            lib.hf_destroy(h)
            self._h = None

    def update(self, dets, frame=None):
        dets = _as_dets(dets)
        d5 = np.ascontiguousarray(dets[:, :5], dtype=np.float32)
        t = self._frame_idx / self.fps
        self._frame_idx += 1
        if self.use_frame and frame is not None:
            img = np.ascontiguousarray(frame, dtype=np.uint8)
            ptr, H, W = img.ctypes.data, img.shape[0], img.shape[1]
        else:
            img, ptr, H, W = None, None, 0, 0
        n = self._lib.hf_update(self._h, ptr, H, W, d5.ctypes.data if len(d5) else None, len(d5), t,
                                self._info.ctypes.data)
        out = np.zeros((n, 6), dtype=np.float32)
        if n:
            self._lib.hf_fetch(self._h, out.ctypes.data, n)
        return out


# --------------------------------------------------------------------------------------------
# MOSAIC-Tracker (S_Ioutrack, Yolov7-tracker framework)
# --------------------------------------------------------------------------------------------
class MosaicAdapter:
    """MOSAIC-Tracker (``tracker/trackers/s_iou_track.py::S_Ioutrack``), minimal copy in
    ``ext/mosaic/`` (s_iou_track, basetrack, tracklet, matching, camera_motion_compensation,
    kalman_filters/{base,botsort}_kalman). Patches: torch/torchvision/ReID imports dropped
    (``discard_reid=True``); tracklet.py imports only the BoT KF; ``cython_bbox`` -> numpy port
    with identical +1-px IoU; ``np.float`` -> float64; unused ``numpy.core`` import dropped;
    an unused ``scores_keep / scores_keep.sum()`` line dropped; ORB GMC made robust (no
    descriptors / <2 knn neighbours / no surviving matches / RANSAC failure -> identity) and its
    "not enough matching points" print silenced; matplotlib import stubbed (debug only).

    Kept: BoT-SORT-style 8-D xywh Kalman ('bot' kalman_format), ORB GMC (FAST(20) keypoints +
    ORB descriptors on a 2x downscale, BFMatcher-Hamming kNN ratio 0.9, spatial-distance outlier
    rejection, estimateAffinePartial2D RANSAC) applied to tracked+lost and unconfirmed tracks;
    stage 1 high dets vs tracked+lost, IoU cost, thresh 0.9; stage 2 *as in the repo*: remaining
    tracked tracks vs a fresh copy of the **high**-score dets (the repo builds detections_second
    from ``dets``, not ``dets_second`` -- low-score dets are never used), IoU, thresh 0.5;
    unconfirmed vs unmatched high dets with the repo's expanded-IoU ``eiou_distance(.., 0.5)``,
    thresh 0.7; new tracks for score >= conf_thresh + 0.1; duplicate removal (IoU > 0.85).
    Output = activated tracked tracks, KF box, writer filter w*h > min_area.
    The repo passes detections in **tlwh** to the tracker and to GMC, so the GMC detection mask
    (meant for tlbr) is computed on tlwh values exactly as in the repo (effectively mostly unmasked).

    Removed: OSNet ReID (``--discard_reid``), the detector's two-frame fusion input (detector side).

    Hyperparameters (README VisDrone command: --tracker s_iou_track --kalman_format bot
    --conf_thresh 0.4; rest from track.py defaults):
        conf_thresh = 0.4 (high/low split; low floor 0.1 hard-coded), det_thresh = 0.5 (= +0.1),
        track_buffer = 30, frame_rate = 30 -> max_time_lost = 30, min_area = 150,
        GMC('orb', downscale=2). (The detector runs at conf 0.1; dets below 0.1 are ignored here too.)
    Scores: detector score used directly; detection class goes to ``Tracklet.category`` only.
    """

    def __init__(self, conf_thresh=0.4, track_buffer=30, frame_rate=30, min_area=150.0,
                 kalman_format='bot', cmc_method='orb'):
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from ext.mosaic import s_iou_track, basetrack
        self._bt = basetrack.BaseTrack
        args = types.SimpleNamespace(conf_thresh=float(conf_thresh), track_buffer=int(track_buffer),
                                     kalman_format=kalman_format, discard_reid=True,
                                     reid_model=None, reid_model_path=None)
        self.tracker = s_iou_track.S_Ioutrack(args, frame_rate=frame_rate)
        if cmc_method != 'orb':
            from ext.mosaic.camera_motion_compensation import GMC
            self.tracker.gmc = GMC(method=cmc_method, downscale=2)
        _guard_cmc(self.tracker.gmc)
        self.min_area = float(min_area)
        self._count = 0

    def update(self, dets, frame):
        dets = _as_dets(dets).astype(np.float64)
        out = dets.copy()  # repo: [x1,y1,x2,y2,conf,cls] -> tlwh in place before tracker.update
        out[:, 2] -= out[:, 0]
        out[:, 3] -= out[:, 1]
        saved = self._bt._count
        self._bt._count = self._count
        try:
            tracks = self.tracker.update(out, frame, frame)
        finally:
            self._count = self._bt._count
            self._bt._count = saved
        rows = []
        for t in tracks:
            tlwh = t.tlwh
            if tlwh[2] * tlwh[3] > self.min_area:
                rows.append([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3], t.track_id, t.score])
        return _pack(rows)


# --------------------------------------------------------------------------------------------
# CSF-TST (CFS_TST_Tracker)
# --------------------------------------------------------------------------------------------
class CSFTSTAdapter:
    """CSF-TST (``yolox/tracker/CFS_TST_tracker.py::CFS_TST_Tracker``), minimal copy in
    ``ext/csftst/`` (tracker, basetrack, kalman_filter, matching, gmc). The shipped x86
    ``yolox/_C*.so`` is never imported (the copy does not touch the ``yolox`` package).
    Patches: relative imports; ``cython_bbox`` -> numpy port (+1 px IoU); ``np.float`` -> float64;
    gmc.py: torch / SuperGlue / LightGlue / XFeat imports removed (those methods now raise),
    the ORB branch no longer writes ``GMC_files/<dir>/GMC-<seq>.txt``, RANSAC ``None`` -> identity,
    print silenced; ``associate`` honours the repo's (otherwise dead) ``self.iou_only`` flag.

    Appearance OFF: the adapter sets ``tracker.iou_only = True`` so every association uses the
    IoU (score-fused in stage 1 / tentative stage) cost only, and passes constant dummy embeddings
    (required by ``update``; they only fill STrack features). Constant embeddings alone would *not*
    disable appearance here: cosine distance 0 everywhere, masked to 1 only where IoU-dist > 0.5,
    then min(iou, emb) -> cost 0 for every pair with IoU >= 0.5.

    Kept: ByteTrack-style two stages with 8-D xywh KF; stage 1 confirmed (Tracked) tracks vs high
    dets (score > track_thresh), IoU x score fusion, thresh match_thresh 0.8; stage 2 vs low dets
    (0.1 < s <= track_thresh), IoU, thresh 0.5; tentative-track "birth" stage (tracks need
    birth_threshold=3 hits to become Tracked; tentative tracks vs unmatched high dets, thresh 0.7;
    unmatched tentative -> removed); new tracks from high dets with score >= track_thresh + 0.1;
    CMC on confirmed tracks only (repo behaviour) using the repo's CPU 'orb' GMC (FAST+ORB,
    downscale 2, ratio-test 0.9, RANSAC partial affine; detections masked); output = Tracked **and**
    Tentative tracks (as the repo returns). Repo quirk kept: lost tracks are never re-matched
    (stage 1 only uses state == Tracked), they just age out after track_buffer frames.
    Removed: ReID head embeddings (see above); LightGlue/SuperGlue/XFeat CMC; precomputed GMC files.

    Hyperparameters (tools/track.py defaults; the repo ships no VisDrone run script):
        track_thresh 0.6, low_thresh 0.1, new_track_thresh 0.7, match_thresh 0.8, track_buffer 30,
        fps 30, birth_threshold 3, mot20 False (fuse_score on), mask_emb_with_iou True,
        fuse_emb_and_iou 'min' (both moot with iou_only), cmc_method 'orb' (task: CPU method; repo
        default 'none'), writer: w*h > min_box_area 100, w/h < aspect_ratio_thresh 1.6 and h/w < 4
        (applied to every class as in the repo; aspect_ratio_thresh=None disables both ratio checks).
    Detections are fed as 7 columns [x1,y1,x2,y2,score,1.0,cls] (repo reads score from col 4 and
    class from col 6).
    """

    def __init__(self, track_thresh=0.6, low_thresh=0.1, match_thresh=0.8, track_buffer=30,
                 frame_rate=30, birth_threshold=3, cmc_method='orb', min_box_area=100.0,
                 aspect_ratio_thresh=1.6, iou_only=True):
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        from ext.csftst import CFS_TST_tracker as cst, basetrack
        self._bt = basetrack.BaseTrack
        args = types.SimpleNamespace(track_thresh=float(track_thresh), low_thresh=float(low_thresh),
                                     track_buffer=int(track_buffer), match_thresh=float(match_thresh),
                                     birth_threshold=int(birth_threshold), mot20=False,
                                     mask_emb_with_iou=True, fuse_emb_and_iou='min',
                                     cmc_method=cmc_method, cmc_seq_name='', cmc_file_dir='')
        saved = self._bt._count
        self.tracker = cst.CFS_TST_Tracker(args, frame_rate=frame_rate)  # calls BaseTrack.clear_count()
        self._bt._count = saved
        self.tracker.iou_only = bool(iou_only)
        _guard_cmc(self.tracker.gmc)
        self._count = 0
        self.min_box_area = float(min_box_area)
        self.aspect_ratio_thresh = aspect_ratio_thresh

    def update(self, dets, frame):
        dets = _as_dets(dets).astype(np.float64)
        d7 = np.zeros((len(dets), 7), dtype=np.float64)
        d7[:, :5] = dets[:, :5]
        d7[:, 5] = 1.0
        d7[:, 6] = dets[:, 5]
        emb = np.ones((len(d7), 8), dtype=np.float64)  # constant dummy embedding
        h, w = frame.shape[:2]
        saved = self._bt._count
        self._bt._count = self._count
        try:
            tracks = self.tracker.update(d7, [h, w], [h, w], embeddings=emb, img=frame)
        finally:
            self._count = self._bt._count
            self._bt._count = saved
        rows = []
        for t in tracks:
            tlwh = t.tlwh
            if tlwh[2] <= 0 or tlwh[3] <= 0:
                continue
            ok = tlwh[2] * tlwh[3] > self.min_box_area
            if ok and self.aspect_ratio_thresh is not None:
                ok = tlwh[2] / tlwh[3] < self.aspect_ratio_thresh and tlwh[3] / tlwh[2] < 4.
            if ok:
                rows.append([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3], t.track_id, t.score])
        return _pack(rows)
