#include "../include/image_enhance.hpp"
#include "../include/log.hpp"
#include "../include/utils.hpp"
#include <assert.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <vector>

#ifndef YOLO_INPUT_FORMAT
#define YOLO_INPUT_FORMAT SSNE_RGB
#endif

#ifndef YOLO_INPUT_FORMAT_NAME
#define YOLO_INPUT_FORMAT_NAME "RGB"
#endif

#ifndef YOLO_OUTPUT_LAYOUT_NCHW
#define YOLO_OUTPUT_LAYOUT_NCHW 0
#endif

static ssne_tensor_t g_last_img;
static ssne_tensor_t g_last_pipe_input;
static bool g_has_frame = false;
static bool g_pose_debug_dumped = false;
static bool g_output_layout_logged = false;
// AI 前增强模块：
static image_enhance::AdaptiveImageEnhancer g_image_enhancer;
static image_enhance::SceneType g_last_scene = image_enhance::SceneType::kNormal;

struct PredictPerfStats {
    uint64_t frames = 0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double getoutput_ms = 0.0;
    double decode_ms = 0.0;
    std::chrono::steady_clock::time_point window_begin = std::chrono::steady_clock::now();
};

static PredictPerfStats g_predict_perf;

static const int YOLO_NUM_CLASSES = 1;
static const int YOLO_REG_MAX = 16;
static const int YOLO_BOX_CHANNELS = 4 * YOLO_REG_MAX;   // 64
static const int YOLO_KPT_NUM = 17;
static const int YOLO_KPT_DIMS = 3;
static const int YOLO_KPT_CHANNELS = YOLO_KPT_NUM * YOLO_KPT_DIMS;   // 51
static const int YOLO_BRANCH_COUNT = 3;
static const int YOLO_OUTPUT_COUNT = 9;

bool IsValidTensor(ssne_tensor_t tensor) {
    return get_data(tensor) != nullptr &&
           get_width(tensor) > 0 &&
           get_height(tensor) > 0 &&
           get_mem_size(tensor) > 0;
}

void ReleaseOutputTensors(ssne_tensor_t* outputs, int count) {
    if (outputs == nullptr || count <= 0) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        release_tensor(outputs[i]);
        outputs[i] = ssne_tensor_t{};
    }
}

namespace {

struct PoseOutputBranch {
    const float* box_ptr = nullptr;
    const float* cls_ptr = nullptr;
    const float* kpt_ptr = nullptr;
    int feat_h = 0;
    int feat_w = 0;
    int stride = 0;

    bool IsComplete() const {
        return box_ptr != nullptr && cls_ptr != nullptr && kpt_ptr != nullptr &&
               feat_h > 0 && feat_w > 0 && stride > 0;
    }
};

struct PoseInputRoi {
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;

    int Width() const { return right - left; }
    int Height() const { return bottom - top; }
    bool IsValid() const { return Width() > 0 && Height() > 0; }
};

void EnsurePoseRoiMinSize(PoseInputRoi* roi,
                          const std::array<int, 2>& img_shape,
                          int min_width,
                          int min_height) {
    if (roi == nullptr || !roi->IsValid()) {
        return;
    }

    min_width = std::max(1, std::min(min_width, img_shape[0]));
    min_height = std::max(1, std::min(min_height, img_shape[1]));

    int width = roi->Width();
    int height = roi->Height();
    if (width >= min_width && height >= min_height) {
        return;
    }

    const float cx = 0.5f * static_cast<float>(roi->left + roi->right);
    const float cy = 0.5f * static_cast<float>(roi->top + roi->bottom);
    width = std::max(width, min_width);
    height = std::max(height, min_height);

    int left = static_cast<int>(std::floor(cx - 0.5f * static_cast<float>(width)));
    int top = static_cast<int>(std::floor(cy - 0.5f * static_cast<float>(height)));
    int right = left + width;
    int bottom = top + height;

    if (left < 0) {
        right -= left;
        left = 0;
    }
    if (top < 0) {
        bottom -= top;
        top = 0;
    }
    if (right > img_shape[0]) {
        const int shift = right - img_shape[0];
        left = std::max(0, left - shift);
        right = img_shape[0];
    }
    if (bottom > img_shape[1]) {
        const int shift = bottom - img_shape[1];
        top = std::max(0, top - shift);
        bottom = img_shape[1];
    }

    roi->left = left;
    roi->top = top;
    roi->right = right;
    roi->bottom = bottom;
}

void FitPoseRoiToAspect(PoseInputRoi* roi,
                        const std::array<int, 2>& img_shape,
                        const std::array<int, 2>& det_shape) {
    if (roi == nullptr || !roi->IsValid() || det_shape[0] <= 0 || det_shape[1] <= 0) {
        return;
    }

    const float target_aspect = static_cast<float>(det_shape[0]) / static_cast<float>(det_shape[1]);
    int width = roi->Width();
    int height = roi->Height();
    if (width <= 0 || height <= 0) {
        return;
    }

    if (static_cast<float>(width) / static_cast<float>(height) < target_aspect) {
        width = static_cast<int>(std::ceil(static_cast<float>(height) * target_aspect));
    } else {
        height = static_cast<int>(std::ceil(static_cast<float>(width) / target_aspect));
    }

    width = std::min(width, img_shape[0]);
    height = std::min(height, img_shape[1]);
    const float cx = 0.5f * static_cast<float>(roi->left + roi->right);
    const float cy = 0.5f * static_cast<float>(roi->top + roi->bottom);

    int left = static_cast<int>(std::floor(cx - 0.5f * static_cast<float>(width)));
    int top = static_cast<int>(std::floor(cy - 0.5f * static_cast<float>(height)));
    int right = left + width;
    int bottom = top + height;

    if (left < 0) {
        right -= left;
        left = 0;
    }
    if (top < 0) {
        bottom -= top;
        top = 0;
    }
    if (right > img_shape[0]) {
        const int shift = right - img_shape[0];
        left = std::max(0, left - shift);
        right = img_shape[0];
    }
    if (bottom > img_shape[1]) {
        const int shift = bottom - img_shape[1];
        top = std::max(0, top - shift);
        bottom = img_shape[1];
    }

    roi->left = left;
    roi->top = top;
    roi->right = right;
    roi->bottom = bottom;
}

PoseInputRoi BuildPoseInputRoi(const std::array<float, 4>& focus_box,
                               const std::array<int, 2>& img_shape) {
    PoseInputRoi roi;
    const float box_w = std::max(1.0f, focus_box[2] - focus_box[0]);
    const float box_h = std::max(1.0f, focus_box[3] - focus_box[1]);
    const float box_cx = 0.5f * (focus_box[0] + focus_box[2]);
    const float box_cy = 0.5f * (focus_box[1] + focus_box[3]);
    const float roi_size = std::max(box_w, box_h) * 1.8f;
    const float half = 0.5f * roi_size;

    float left = box_cx - half;
    float top = box_cy - half;
    float right = box_cx + half;
    float bottom = box_cy + half;

    if (left < 0.0f) {
        right -= left;
        left = 0.0f;
    }
    if (top < 0.0f) {
        bottom -= top;
        top = 0.0f;
    }
    if (right > static_cast<float>(img_shape[0])) {
        const float overflow = right - static_cast<float>(img_shape[0]);
        left = std::max(0.0f, left - overflow);
        right = static_cast<float>(img_shape[0]);
    }
    if (bottom > static_cast<float>(img_shape[1])) {
        const float overflow = bottom - static_cast<float>(img_shape[1]);
        top = std::max(0.0f, top - overflow);
        bottom = static_cast<float>(img_shape[1]);
    }

    roi.left = std::max(0, static_cast<int>(std::floor(left)));
    roi.top = std::max(0, static_cast<int>(std::floor(top)));
    roi.right = std::min(img_shape[0], static_cast<int>(std::ceil(right)));
    roi.bottom = std::min(img_shape[1], static_cast<int>(std::ceil(bottom)));

    roi.left &= ~1;
    roi.right = std::max(roi.left, std::min(img_shape[0], roi.right));

    const int roi_width = roi.right - roi.left;
    if (roi_width >= 8) {
        const int aligned_width = roi_width & ~7;
        if (aligned_width != roi_width) {
            const int trim = roi_width - aligned_width;
            const int trim_left = trim / 2;
            const int trim_right = trim - trim_left;
            roi.left += trim_left;
            roi.right -= trim_right;
            roi.left &= ~1;
            roi.right = roi.left + aligned_width;
        }
    }

    return roi;
}

void NormalizePoseInputRoi(PoseInputRoi* roi, const std::array<int, 2>& img_shape) {
    if (roi == nullptr) {
        return;
    }

    roi->left = std::max(0, std::min(roi->left, img_shape[0]));
    roi->top = std::max(0, std::min(roi->top, img_shape[1]));
    roi->right = std::max(roi->left, std::min(roi->right, img_shape[0]));
    roi->bottom = std::max(roi->top, std::min(roi->bottom, img_shape[1]));

    roi->left &= ~1;
    roi->right = std::max(roi->left, roi->right & ~1);

    int width = roi->right - roi->left;
    if (width < 8) {
        roi->right = roi->left;
        return;
    }

    const int aligned_width = width & ~7;
    if (aligned_width <= 0) {
        roi->right = roi->left;
        return;
    }

    if (aligned_width != width) {
        const int trim = width - aligned_width;
        const int trim_left = trim / 2;
        roi->left += trim_left;
        roi->left &= ~1;
        roi->right = roi->left + aligned_width;
        if (roi->right > img_shape[0]) {
            roi->right = img_shape[0] & ~1;
            roi->left = std::max(0, roi->right - aligned_width);
            roi->left &= ~1;
            roi->right = roi->left + aligned_width;
        }
    }
}

bool ShouldUsePoseRoiCrop(const PoseInputRoi& roi, const std::array<int, 2>& img_shape) {
    if (!roi.IsValid() || img_shape[0] <= 0 || img_shape[1] <= 0) {
        return false;
    }

    const int roi_w = roi.Width();
    const int roi_h = roi.Height();
    const int frame_w = img_shape[0];
    const int frame_h = img_shape[1];
    const int64_t roi_area = static_cast<int64_t>(roi_w) * static_cast<int64_t>(roi_h);
    const int64_t frame_area = static_cast<int64_t>(frame_w) * static_cast<int64_t>(frame_h);

    // SSNE 离线预处理要求 YUV422 ROI 宽度 8 对齐，否则会直接返回 ret=503/543。
    if (roi_w < 8 || (roi_w & 7) != 0) {
        return false;
    }

    // ROI 太接近全图时裁剪收益很小，直接走全图 pose，避免额外 tensor 生命周期风险。
    if (roi_w >= frame_w - 8 || roi_h >= frame_h - 8) {
        return false;
    }
    if (roi_area * 100 >= frame_area * 60) {
        return false;
    }
    return true;
}
bool CropYuv422Tensor(ssne_tensor_t input, const PoseInputRoi& roi, ssne_tensor_t* cropped) {
    if (cropped == nullptr || !roi.IsValid()) {
        return false;
    }
    if ((roi.left & 1) != 0 || (roi.Width() & 7) != 0) {
        LOG_WARN("pose roi width is not 8-aligned [%d x %d], skip roi crop\n", roi.Width(), roi.Height());
        return false;
    }
    *cropped = input;
    if (get_data(input) == nullptr || get_data_format(input) != SSNE_YUV422_16) {
        return false;
    }

    const uint32_t input_width = get_width(input);
    const uint32_t input_height = get_height(input);
    const size_t input_mem_size = get_mem_size(input);
    if (input_width == 0 || input_height == 0 || input_mem_size == 0) {
        return false;
    }

    const size_t src_stride = input_mem_size / static_cast<size_t>(input_height);
    if (src_stride < static_cast<size_t>(input_width) * 2U) {
        return false;
    }

    ssne_tensor_t roi_tensor = create_tensor(static_cast<uint32_t>(roi.Width()),
                                             static_cast<uint32_t>(roi.Height()),
                                             SSNE_YUV422_16,
                                             SSNE_BUF_AI);
    if (!IsValidTensor(roi_tensor)) {
        LOG_WARN("pose roi tensor allocation failed for [%d x %d], skip roi crop\n",
                 roi.Width(), roi.Height());
        release_tensor(roi_tensor);
        return false;
    }
    uint8_t* dst = static_cast<uint8_t*>(get_data(roi_tensor));
    const uint8_t* src = static_cast<const uint8_t*>(get_data(input));
    if (dst == nullptr || src == nullptr) {
        release_tensor(roi_tensor);
        return false;
    }

    const size_t dst_stride = get_mem_size(roi_tensor) / static_cast<size_t>(roi.Height());
    const size_t copy_bytes = static_cast<size_t>(roi.Width()) * 2U;
    for (int y = 0; y < roi.Height(); ++y) {
        const uint8_t* src_row =
            src + static_cast<size_t>(roi.top + y) * src_stride + static_cast<size_t>(roi.left) * 2U;
        uint8_t* dst_row = dst + static_cast<size_t>(y) * dst_stride;
        std::memcpy(dst_row, src_row, copy_bytes);
    }

    *cropped = roi_tensor;
    return true;
}

void LogEnhanceScene(const image_enhance::SceneStats& stats) {
    if (stats.scene == g_last_scene) {
        return;
    }

    LOG_INFO("enhance scene=%s mean=%.1f center=%.1f edge=%.1f dark=%.3f bright=%.3f p05=%u p95=%u focus_roi=%d\n",
             image_enhance::AdaptiveImageEnhancer::SceneTypeName(stats.scene),
             stats.mean_luma,
             stats.center_luma,
             stats.edge_luma,
             stats.dark_ratio,
             stats.bright_ratio,
             static_cast<unsigned>(stats.p05),
             static_cast<unsigned>(stats.p95),
             stats.used_focus_roi ? 1 : 0);
    g_last_scene = stats.scene;
}

void FlushPredictPerfIfNeeded() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    const double elapsed_ms =
        static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(now - g_predict_perf.window_begin).count());
    if (elapsed_ms < 1000.0 || g_predict_perf.frames == 0) {
        return;
    }

    const double fps = static_cast<double>(g_predict_perf.frames) * 1000.0 / elapsed_ms;
    const double inv = 1.0 / static_cast<double>(g_predict_perf.frames);
    LOG_INFO("predict fps=%.2f preprocess=%.2fms inference=%.2fms getoutput=%.2fms decode=%.2fms\n",
             fps,
             g_predict_perf.preprocess_ms * inv,
             g_predict_perf.inference_ms * inv,
             g_predict_perf.getoutput_ms * inv,
             g_predict_perf.decode_ms * inv);

    g_predict_perf.frames = 0;
    g_predict_perf.preprocess_ms = 0.0;
    g_predict_perf.inference_ms = 0.0;
    g_predict_perf.getoutput_ms = 0.0;
    g_predict_perf.decode_ms = 0.0;
    g_predict_perf.window_begin = now;
}

inline float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float DecodePoseCoord(float raw, int grid, int stride) {
    // YOLOv8-Pose 的 x/y 是线性偏移，不能做 sigmoid；只有关键点置信度 raw_v 需要 sigmoid。
    return (raw * 2.0f - 0.5f + static_cast<float>(grid)) * static_cast<float>(stride);
}

inline int OffsetOutput(int y, int x, int c, int feat_h, int feat_w, int channels) {
#if YOLO_OUTPUT_LAYOUT_NCHW
    (void)channels;
    return (c * feat_h + y) * feat_w + x;
#else
    (void)feat_h;
    return (y * feat_w + x) * channels + c;
#endif
}

float DFLIntegral(const float* logits, int reg_max) {
    float max_logit = logits[0];
    for (int i = 1; i < reg_max; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float exp_sum = 0.0f;
    float weighted_sum = 0.0f;
    for (int i = 0; i < reg_max; ++i) {
        const float e = std::exp(logits[i] - max_logit);
        exp_sum += e;
        weighted_sum += e * static_cast<float>(i);
    }

    if (exp_sum <= 1e-12f) {
        return 0.0f;
    }
    return weighted_sum / exp_sum;
}

float IoUBox(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);

    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;

    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    const float uni = area_a + area_b - inter;

    if (uni <= 1e-6f) {
        return 0.0f;
    }
    return inter / uni;
}

inline size_t TensorTypeBytes(uint8_t dtype) {
    switch (dtype) {
        case SSNE_UINT8:
        case SSNE_INT8:
            return 1;
        case SSNE_FLOAT32:
            return sizeof(float);
        default:
            return 0;
    }
}

const char* TensorTypeName(uint8_t dtype) {
    switch (dtype) {
        case SSNE_UINT8:
            return "uint8";
        case SSNE_INT8:
            return "int8";
        case SSNE_FLOAT32:
            return "float32";
        default:
            return "unknown";
    }
}

int GetTensorChannels(ssne_tensor_t tensor) {
    const uint32_t width = get_width(tensor);
    const uint32_t height = get_height(tensor);
    if (width == 0 || height == 0) {
        return -1;
    }

    const uint64_t hw = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    const uint64_t elem_count = static_cast<uint64_t>(get_total_size(tensor));
    if (elem_count == 0 || elem_count % hw != 0) {
        return -1;
    }

    return static_cast<int>(elem_count / hw);
}

int InferStrideFromTensor(ssne_tensor_t tensor, const std::array<int, 2>& det_shape) {
    const uint32_t width = get_width(tensor);
    const uint32_t height = get_height(tensor);
    if (width == 0 || height == 0) {
        return -1;
    }
    if ((det_shape[0] % static_cast<int>(width)) != 0 ||
        (det_shape[1] % static_cast<int>(height)) != 0) {
        return -1;
    }

    const int stride_x = det_shape[0] / static_cast<int>(width);
    const int stride_y = det_shape[1] / static_cast<int>(height);
    if (stride_x != stride_y) {
        return -1;
    }
    return stride_x;
}

int BranchIndexFromStride(int stride) {
    switch (stride) {
        case 8:
            return 0;
        case 16:
            return 1;
        case 32:
            return 2;
        default:
            return -1;
    }
}

void NMSPose(std::vector<PoseDetection>& dets, float iou_thres, int top_k) {
    std::sort(dets.begin(), dets.end(), [](const PoseDetection& a, const PoseDetection& b) {
        return a.score > b.score;
    });

    std::vector<int> suppressed(dets.size(), 0);
    std::vector<PoseDetection> keep;
    keep.reserve(top_k > 0 ? std::min(static_cast<int>(dets.size()), top_k)
                           : static_cast<int>(dets.size()));

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        keep.push_back(dets[i]);
        if (top_k > 0 && static_cast<int>(keep.size()) >= top_k) {
            break;
        }
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            if (dets[i].class_id != dets[j].class_id) {
                continue;
            }

            const float iou = IoUBox(dets[i].box, dets[j].box);
            if (iou > iou_thres) {
                suppressed[j] = 1;
            }
        }
    }

    dets.swap(keep);
}

void DecodePoseBranch(const float* box_ptr,
                           const float* cls_ptr,
                           const float* kpt_ptr,
                           int feat_h,
                           int feat_w,
                           int stride,
                           float conf_threshold,
                           float w_scale,
                           float h_scale,
                           float roi_offset_x,
                           float roi_offset_y,
                           const std::array<int, 2>& output_shape,
                           const std::array<int, 2>& det_shape,
                           std::vector<PoseDetection>& dets) {
    if (box_ptr == nullptr || cls_ptr == nullptr || kpt_ptr == nullptr) {
        LOG_WARN("null output pointer found at stride %d, skip branch\n", stride);
        return;
    }

    for (int gy = 0; gy < feat_h; ++gy) {
        for (int gx = 0; gx < feat_w; ++gx) {
            const float cls_logit = cls_ptr[OffsetOutput(gy, gx, 0, feat_h, feat_w, YOLO_NUM_CLASSES)];
            const float score = Sigmoid(cls_logit);
            if (score < conf_threshold) {
                continue;
            }

            float side_logits[YOLO_REG_MAX];
            float ltrb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int side = 0; side < 4; ++side) {
                for (int bin = 0; bin < YOLO_REG_MAX; ++bin) {
                    const int c = side * YOLO_REG_MAX + bin;
                    side_logits[bin] = box_ptr[OffsetOutput(gy, gx, c, feat_h, feat_w, YOLO_BOX_CHANNELS)];
                }
                ltrb[side] = DFLIntegral(side_logits, YOLO_REG_MAX);
            }

            const float anchor_x = static_cast<float>(gx) + 0.5f;
            const float anchor_y = static_cast<float>(gy) + 0.5f;

            float x1 = (anchor_x - ltrb[0]) * static_cast<float>(stride);
            float y1 = (anchor_y - ltrb[1]) * static_cast<float>(stride);
            float x2 = (anchor_x + ltrb[2]) * static_cast<float>(stride);
            float y2 = (anchor_y + ltrb[3]) * static_cast<float>(stride);

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(det_shape[0])));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(det_shape[1])));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(det_shape[0])));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(det_shape[1])));

            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            PoseDetection det;
            det.score = score;
            det.class_id = 0;

            // 解码出的坐标先从模型输入尺度映射回 ROI，再叠加 ROI 在原图中的偏移。
            det.box = {
                std::max(0.0f, std::min(roi_offset_x + x1 * w_scale, static_cast<float>(output_shape[0]))),
                std::max(0.0f, std::min(roi_offset_y + y1 * h_scale, static_cast<float>(output_shape[1]))),
                std::max(0.0f, std::min(roi_offset_x + x2 * w_scale, static_cast<float>(output_shape[0]))),
                std::max(0.0f, std::min(roi_offset_y + y2 * h_scale, static_cast<float>(output_shape[1])))
            };

            if (det.box[2] <= det.box[0] || det.box[3] <= det.box[1]) {
                continue;
            }


            float raw_x_min = 0.0f;
            float raw_x_max = 0.0f;
            float raw_y_min = 0.0f;
            float raw_y_max = 0.0f;
            bool capture_raw_range = !g_pose_debug_dumped;
            for (int k = 0; k < YOLO_KPT_NUM; ++k) {
                const int base_c = k * YOLO_KPT_DIMS;
                const float raw_x = kpt_ptr[OffsetOutput(gy, gx, base_c + 0, feat_h, feat_w, YOLO_KPT_CHANNELS)];
                const float raw_y = kpt_ptr[OffsetOutput(gy, gx, base_c + 1, feat_h, feat_w, YOLO_KPT_CHANNELS)];
                const float raw_v = kpt_ptr[OffsetOutput(gy, gx, base_c + 2, feat_h, feat_w, YOLO_KPT_CHANNELS)];
                if (capture_raw_range) {
                    if (k == 0) {
                        raw_x_min = raw_x_max = raw_x;
                        raw_y_min = raw_y_max = raw_y;
                    } else {
                        raw_x_min = std::min(raw_x_min, raw_x);
                        raw_x_max = std::max(raw_x_max, raw_x);
                        raw_y_min = std::min(raw_y_min, raw_y);
                        raw_y_max = std::max(raw_y_max, raw_y);
                    }
                }

                float kp_x = DecodePoseCoord(raw_x, gx, stride);
                float kp_y = DecodePoseCoord(raw_y, gy, stride);
                kp_x = std::max(0.0f, std::min(kp_x, static_cast<float>(det_shape[0])));
                kp_y = std::max(0.0f, std::min(kp_y, static_cast<float>(det_shape[1])));

                det.keypoints[k].x = std::max(0.0f, std::min(roi_offset_x + kp_x * w_scale, static_cast<float>(output_shape[0])));
                det.keypoints[k].y = std::max(0.0f, std::min(roi_offset_y + kp_y * h_scale, static_cast<float>(output_shape[1])));
                det.keypoints[k].conf = Sigmoid(raw_v);
            }

            if (capture_raw_range) {
                LOG_DEBUG("stride=%d cell=(%d,%d) kpt raw range: x=[%.4f, %.4f] y=[%.4f, %.4f]\n",
                          stride, gx, gy, raw_x_min, raw_x_max, raw_y_min, raw_y_max);
                LOG_DEBUG("YOLOv8-Pose expects linear x/y. If ranges stay near [0, 1], activation or output layout may be wrong.\n");
                g_pose_debug_dumped = true;
            }

            dets.push_back(det);
        }
    }
}

void PrintOutputInfo(int output_idx, ssne_tensor_t tensor, const std::array<int, 2>& det_shape) {
    const uint8_t dtype = get_data_type(tensor);
    const int channels = GetTensorChannels(tensor);
    const int stride = InferStrideFromTensor(tensor, det_shape);
    printf("  output[%d] : dtype=%s total=%u mem=%zu shape=[1 x %d x %u x %u] stride=%d\n",
           output_idx,
           TensorTypeName(dtype),
           get_total_size(tensor),
           get_mem_size(tensor),
           channels,
           get_height(tensor),
           get_width(tensor),
           stride);
}

bool AssignOutputToBranch(ssne_tensor_t tensor,
                          int output_idx,
                          const std::array<int, 2>& det_shape,
                          std::array<PoseOutputBranch, YOLO_BRANCH_COUNT>& branches) {
    const uint8_t dtype = get_data_type(tensor);
    if (dtype != SSNE_FLOAT32) {
        LOG_WARN("output[%d] dtype=%s is not float32, skip this tensor\n",
                 output_idx, TensorTypeName(dtype));
        return false;
    }

    const int channels = GetTensorChannels(tensor);
    const int stride = InferStrideFromTensor(tensor, det_shape);
    const int branch_idx = BranchIndexFromStride(stride);
    if (channels <= 0 || branch_idx < 0) {
        LOG_WARN("output[%d] has invalid channels=%d or stride=%d, skip this tensor\n",
                 output_idx, channels, stride);
        return false;
    }

    const float* data_ptr = reinterpret_cast<const float*>(get_data(tensor));
    if (data_ptr == nullptr) {
        LOG_WARN("output[%d] data pointer is null, skip this tensor\n", output_idx);
        return false;
    }

    PoseOutputBranch& branch = branches[branch_idx];
    branch.feat_h = static_cast<int>(get_height(tensor));
    branch.feat_w = static_cast<int>(get_width(tensor));
    branch.stride = stride;

    if (channels == YOLO_BOX_CHANNELS) {
        branch.box_ptr = data_ptr;
        LOG_DEBUG("output[%d] -> stride=%d box branch\n", output_idx, stride);
        return true;
    }
    if (channels == YOLO_NUM_CLASSES) {
        branch.cls_ptr = data_ptr;
        LOG_DEBUG("output[%d] -> stride=%d cls branch\n", output_idx, stride);
        return true;
    }
    if (channels == YOLO_KPT_CHANNELS) {
        branch.kpt_ptr = data_ptr;
        LOG_DEBUG("output[%d] -> stride=%d kpt branch\n", output_idx, stride);
        return true;
    }

    LOG_WARN("output[%d] has unsupported channels=%d, skip this tensor\n",
             output_idx, channels);
    return false;
}

}  // namespace

void YUNET::Initialize(std::string& model_path,
                       std::array<int, 2>* in_img_shape,
                       std::array<int, 2>* in_det_shape,
                       bool in_use_kps,
                       int in_box_len) {
    nms_threshold = 0.45f;
    keep_top_k = 100;
    top_k = 300;

    img_shape = *in_img_shape;
    det_shape = *in_det_shape;
    use_kps = in_use_kps;
    box_len = in_box_len;

    w_scale = static_cast<float>(img_shape[0]) / static_cast<float>(det_shape[0]);
    h_scale = static_cast<float>(img_shape[1]) / static_cast<float>(det_shape[1]);

    pipe_offline = GetAIPreprocessPipe();

    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);
    const int move_to_sram_ret = ssne_movemodeltosram(model_id);
    if (move_to_sram_ret == 0) {
        LOG_INFO("model moved to SRAM for faster access\n");
    } else {
        LOG_WARN("ssne_movemodeltosram ret=%d, continue with default placement\n", move_to_sram_ret);
    }
    SetNormalize(pipe_offline, model_id);

    const uint32_t det_width = static_cast<uint32_t>(det_shape[0]);
    const uint32_t det_height = static_cast<uint32_t>(det_shape[1]);

    inputs[0] = create_tensor(det_width, det_height, YOLO_INPUT_FORMAT, SSNE_BUF_AI);
    if (!IsValidTensor(inputs[0])) {
        LOG_ERROR("pose input tensor allocation failed for [%u x %u], mem=%zu\n",
                  det_width, det_height, get_mem_size(inputs[0]));
        return;
    }

    int dtype = -1;
    ssne_get_model_input_dtype(model_id, &dtype);
    set_data_type(inputs[0], dtype);

    LOG_INFO("YOLOv8-Pose initialized\n");
LOG_INFO("model=%s crop=[%d,%d] det=[%d,%d] input=[%u,%u] format=%s scale=(%.6f,%.6f)\n",
             model_path.c_str(), img_shape[0], img_shape[1], det_shape[0], det_shape[1],
             det_width, det_height, YOLO_INPUT_FORMAT_NAME, w_scale, h_scale);
}

void YUNET::SetEnhanceFocusBox(const std::array<float, 4>* focus_box) {
    if (focus_box == nullptr) {
        enhance_focus_valid = false;
        enhance_focus_box = {0.0f, 0.0f, 0.0f, 0.0f};
        return;
    }

    enhance_focus_box = *focus_box;
    enhance_focus_box[0] = std::max(0.0f, std::min(enhance_focus_box[0], static_cast<float>(img_shape[0])));
    enhance_focus_box[1] = std::max(0.0f, std::min(enhance_focus_box[1], static_cast<float>(img_shape[1])));
    enhance_focus_box[2] = std::max(0.0f, std::min(enhance_focus_box[2], static_cast<float>(img_shape[0])));
    enhance_focus_box[3] = std::max(0.0f, std::min(enhance_focus_box[3], static_cast<float>(img_shape[1])));
    enhance_focus_valid =
        enhance_focus_box[2] > enhance_focus_box[0] && enhance_focus_box[3] > enhance_focus_box[1];
}

void YUNET::Predict(ssne_tensor_t* img, FaceDetectionResult* result, float conf_threshold) {
    using clock = std::chrono::steady_clock;
    if (result == nullptr) {
        LOG_ERROR("pose predict got null result pointer\n");
        return;
    }
    result->Clear();
    if (img == nullptr || !IsValidTensor(*img)) {
        LOG_ERROR("pose predict got invalid input tensor\n");
        return;
    }
    if (!IsValidTensor(inputs[0])) {
        LOG_ERROR("pose input tensor is invalid, skip this frame\n");
        return;
    }
    image_enhance::SceneStats scene_stats;
    ssne_tensor_t roi_img = *img;
    ssne_tensor_t preprocess_img = *img;
    bool use_roi_input = false;
    bool use_enhanced_input = false;
    PoseInputRoi pose_roi = {};
    float decode_w_scale = w_scale;
    float decode_h_scale = h_scale;
    float decode_offset_x = 0.0f;
    float decode_offset_y = 0.0f;

    if (enhance_focus_valid) {
        pose_roi = BuildPoseInputRoi(enhance_focus_box, img_shape);
        EnsurePoseRoiMinSize(&pose_roi, img_shape, det_shape[0], det_shape[1]);
        FitPoseRoiToAspect(&pose_roi, img_shape, det_shape);
        NormalizePoseInputRoi(&pose_roi, img_shape);
        if (ShouldUsePoseRoiCrop(pose_roi, img_shape) && CropYuv422Tensor(*img, pose_roi, &roi_img)) {
            use_roi_input = true;
            decode_w_scale = static_cast<float>(pose_roi.Width()) / static_cast<float>(det_shape[0]);
            decode_h_scale = static_cast<float>(pose_roi.Height()) / static_cast<float>(det_shape[1]);
            decode_offset_x = static_cast<float>(pose_roi.left);
            decode_offset_y = static_cast<float>(pose_roi.top);

            const std::array<float, 4> local_focus_box = {
                0.0f,
                0.0f,
                static_cast<float>(pose_roi.Width()),
                static_cast<float>(pose_roi.Height())
            };
            use_enhanced_input =
                g_image_enhancer.PrepareForInference(roi_img, &preprocess_img, &scene_stats, &local_focus_box);
        }
    }

    if (!use_roi_input) {
        preprocess_img = *img;
        scene_stats = image_enhance::SceneStats();
    }
    LogEnhanceScene(scene_stats);

    if (!IsValidTensor(preprocess_img)) {
        LOG_ERROR("pose preprocess tensor is invalid before preprocess pipe\n");
        if (use_enhanced_input) {
            release_tensor(preprocess_img);
        }
        if (use_roi_input) {
            release_tensor(roi_img);
        }
        return;
    }

    const auto preprocess_begin = clock::now();
    int ret = RunAiPreprocessPipe(pipe_offline, preprocess_img, inputs[0]);
    if (use_enhanced_input) {
        release_tensor(preprocess_img);
    }
    if (use_roi_input) {
        release_tensor(roi_img);
    }
    const auto preprocess_end = clock::now();

    if (ret != 0) {
        LOG_ERROR("Failed to run AI preprocess pipe, ret=%d\n", ret);
        return;
    }

    g_last_img = *img;
    g_last_pipe_input = inputs[0];
    g_has_frame = true;

    const auto inference_begin = clock::now();
    ret = ssne_inference(model_id, 1, inputs);
    const auto inference_end = clock::now();
    if (ret != 0) {
        LOG_ERROR("ssne inference fail, ret=%d\n", ret);
        return;
    }

    const auto getoutput_begin = clock::now();
    ret = ssne_getoutput(model_id, YOLO_OUTPUT_COUNT, outputs);
    const auto getoutput_end = clock::now();
    if (ret != 0) {
        LOG_ERROR("ssne getoutput fail, ret=%d\n", ret);
        return;
    }

    // [box_s8, box_s16, box_s32, cls_s8, cls_s16, cls_s32, kpt_s8, kpt_s16, kpt_s32]
    if (!g_output_layout_logged) {
        LOG_INFO("YOLOv8-Pose raw outputs (layout=%s):\n",
#if YOLO_OUTPUT_LAYOUT_NCHW
                 "NCHW");
#else
                 "NHWC");
#endif
        for (int i = 0; i < YOLO_OUTPUT_COUNT; ++i) {
            PrintOutputInfo(i, outputs[i], det_shape);
        }
        g_output_layout_logged = true;
    }

    const auto decode_begin = clock::now();
    std::array<PoseOutputBranch, YOLO_BRANCH_COUNT> branches = {};
    for (int i = 0; i < YOLO_OUTPUT_COUNT; ++i) {
        AssignOutputToBranch(outputs[i], i, det_shape, branches);
    }

    std::vector<PoseDetection> dets;
    dets.reserve(2000);

    for (int i = 0; i < YOLO_BRANCH_COUNT; ++i) {
        const PoseOutputBranch& branch = branches[i];
        if (!branch.IsComplete()) {
            const int expected_stride = (i == 0) ? 8 : ((i == 1) ? 16 : 32);
            LOG_WARN("stride=%d branch is incomplete (box=%d cls=%d kpt=%d), skip decode\n",
                     branch.stride > 0 ? branch.stride : expected_stride,
                     branch.box_ptr != nullptr,
                     branch.cls_ptr != nullptr,
                     branch.kpt_ptr != nullptr);
            continue;
        }

        DecodePoseBranch(branch.box_ptr, branch.cls_ptr, branch.kpt_ptr,
                             branch.feat_h, branch.feat_w, branch.stride,
                             conf_threshold, decode_w_scale, decode_h_scale,
                             decode_offset_x, decode_offset_y, img_shape, det_shape, dets);
    }

    NMSPose(dets, nms_threshold, top_k);
    const auto decode_end = clock::now();

    result->Reserve(std::min(static_cast<int>(dets.size()), keep_top_k));

    const int final_count = std::min(static_cast<int>(dets.size()), keep_top_k);
    for (int i = 0; i < final_count; ++i) {
        result->detections.push_back(dets[i]);
        result->boxes.push_back(dets[i].box);
        result->scores.push_back(dets[i].score);
        result->class_ids.push_back(dets[i].class_id);
        result->keypoints.push_back(dets[i].keypoints);
    }

    result->Resize(final_count);

    g_predict_perf.frames += 1;
    g_predict_perf.preprocess_ms +=
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_begin).count()) / 1000.0;
    g_predict_perf.inference_ms +=
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_begin).count()) / 1000.0;
    g_predict_perf.getoutput_ms +=
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(getoutput_end - getoutput_begin).count()) / 1000.0;
    g_predict_perf.decode_ms +=
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_begin).count()) / 1000.0;
    FlushPredictPerfIfNeeded();
}

void YUNET::Release() {
    g_has_frame = false;
    g_last_img = ssne_tensor_t{};
    g_last_pipe_input = ssne_tensor_t{};

    release_tensor(inputs[0]);
    inputs[0] = ssne_tensor_t{};

    ReleaseOutputTensors(outputs, YOLO_OUTPUT_COUNT);

    ReleaseAIPreprocessPipe(pipe_offline);
}

void YUNET::saveImageBin(const void* data, int w, int h, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file != nullptr) {
        fwrite(&w, sizeof(int), 1, file);
        fwrite(&h, sizeof(int), 1, file);
        fwrite(data, sizeof(char), w * h * 3, file);
        fclose(file);
        std::cout << "write file " << filename << " successfully!" << std::endl;
    } else {
        std::cerr << "failed to write " << filename << std::endl;
    }
}

void YUNET::saveFloatBin(const float* data, int length, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file != nullptr) {
        fwrite(&length, sizeof(int), 1, file);
        fwrite(data, sizeof(float), length, file);
        fclose(file);
        std::cout << "write file " << filename << " successfully!" << std::endl;
    } else {
        std::cerr << "failed to write " << filename << std::endl;
    }
}


