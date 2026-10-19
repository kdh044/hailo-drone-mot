// Minimal C ABI around Holdfast's Tracker + CameraMotionEstimator (hailo-test, ctypes).
// Mirrors src/app/pipeline.cpp per delivered frame: A = cmc.estimate(img); stale = diff < 0.5;
// out = tracker.update({t, dets, A, stale}). Built from ~/ext/holdfast sources by build.sh.
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

#include "holdfast/app/camera_motion.hpp"
#include "holdfast/tracker.hpp"

using namespace holdfast;

struct HfHandle {
    std::unique_ptr<Tracker> tracker;
    CameraMotionEstimator::Params cmc_params;
    std::unique_ptr<CameraMotionEstimator> cmc;
    int prev_w = -1, prev_h = -1;
    std::vector<TrackOutput> last;
};

extern "C" {

// spec: TrackerConfig::by_name string ("full", "baseline", "full+recovery=0", ...). Returns NULL if unknown.
void* hf_create(const char* spec, double fps, int cv_threads, int work_width, int max_corners, double quality,
                double min_distance, double ransac_px, int min_inliers) {
    auto cfg = TrackerConfig::by_name(spec);
    if (!cfg) return nullptr;
    cfg->nominal_dt = 1.0 / fps;  // pipeline.cpp does the same
    if (cv_threads >= 0) cv::setNumThreads(cv_threads);
    auto* h = new HfHandle;
    h->tracker = std::make_unique<Tracker>(*cfg);
    h->cmc_params = {work_width, max_corners, quality, min_distance, ransac_px, min_inliers};
    h->cmc = std::make_unique<CameraMotionEstimator>(h->cmc_params);
    return h;
}

void hf_destroy(void* p) { delete static_cast<HfHandle*>(p); }

// bgr: H x W x 3 uint8 contiguous (may be NULL -> identity motion, never stale).
// dets: n x 5 float [x1, y1, x2, y2, score]. info (optional, 3 doubles): [cmc_failed, inliers,
// frame_diff]. Returns the number of output tracks; fetch them with hf_fetch.
int hf_update(void* p, const unsigned char* bgr, int H, int W, const float* dets, int n, double t,
              double* info) {
    auto* h = static_cast<HfHandle*>(p);
    Affine2 A = Affine2::identity();
    bool stale = false;
    if (bgr != nullptr) {
        if (W != h->prev_w || H != h->prev_h) {
            // [hailo-test] resolution change (e.g. a stray odd-sized frame): fresh estimator,
            // which returns identity for this frame (the repo only resets when the *downscaled*
            // size differs, and all 16:9 inputs downscale to the same 640x360).
            h->cmc = std::make_unique<CameraMotionEstimator>(h->cmc_params);
            h->prev_w = W;
            h->prev_h = H;
        }
        cv::Mat img(H, W, CV_8UC3, const_cast<unsigned char*>(bgr));
        A = h->cmc->estimate(img);
        stale = h->cmc->last_frame_difference() < 0.5;
        if (info) {
            info[0] = h->cmc->last_fit_failed();
            info[1] = h->cmc->last_inliers();
            info[2] = h->cmc->last_frame_difference();
        }
    }
    FrameInput in;
    in.t = t;
    in.camera_motion = A;
    in.stale = stale;
    in.detections.reserve(n);
    for (int i = 0; i < n; ++i) {
        const float* d = dets + 5 * i;
        Detection det;
        det.box = {d[0], d[1], d[2] - d[0], d[3] - d[1]};
        det.score = d[4];
        in.detections.push_back(det);
    }
    h->last = h->tracker->update(in);
    return static_cast<int>(h->last.size());
}

// out: rows x 6 float [x1, y1, x2, y2, id, score] for the last hf_update; returns rows written.
int hf_fetch(void* p, float* out, int max_out) {
    auto* h = static_cast<HfHandle*>(p);
    int k = 0;
    for (const auto& o : h->last) {
        if (k >= max_out) break;
        float* r = out + 6 * k++;
        r[0] = o.box.x;
        r[1] = o.box.y;
        r[2] = o.box.x + o.box.w;
        r[3] = o.box.y + o.box.h;
        r[4] = static_cast<float>(o.id);
        r[5] = o.score;
    }
    return k;
}

}  // extern "C"
