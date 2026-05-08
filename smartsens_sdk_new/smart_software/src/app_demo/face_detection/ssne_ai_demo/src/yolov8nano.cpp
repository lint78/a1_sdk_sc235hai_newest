/*
 * @Filename: yolov8nano.cpp
 * @Description: YOLOv8 nano detect implementation for SmartSens pipeline
 */

#include "../include/common.hpp"
#include "../include/log.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
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

namespace {

struct DetectOutputBranch {
    const float* box_ptr = nullptr;
    const float* cls_ptr = nullptr;
    int feat_h = 0;
    int feat_w = 0;
    int stride = 0;

    bool IsComplete() const {
        return box_ptr != nullptr && cls_ptr != nullptr &&
               feat_h > 0 && feat_w > 0 && stride > 0;
    }
};

struct DetectPerfStats {
    uint64_t frames = 0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double getoutput_ms = 0.0;
    double decode_ms = 0.0;
    int last_det_count = 0;
    int last_best_class = -1;
    float last_best_score = 0.0f;
    float last_best_person_cls_score = 0.0f;
    int last_person_cls_hits = 0;
    std::chrono::steady_clock::time_point window_begin = std::chrono::steady_clock::now();
};

static DetectPerfStats g_detect_perf;
static bool g_detect_output_layout_logged = false;
static int g_detect_empty_frames = 0;
static const int kYoloBranchCount = 3;
static const int kYoloOutputCount = 6;
static const int kYoloRegMax = 16;
static const int kYoloBoxChannels = 4 * kYoloRegMax;
static const int kPersonClassId = 4;
static const int kDarkPersonClassId = 6;
static const std::array<const char*, 7> kDetectClassNames = {
    "cat", "dog", "snake", "mouse", "person", "fire", "person"
};
static constexpr float kPersonRescueConfThreshold = 0.18f;
static constexpr float kPersonTrackAssistConfThreshold = 0.24f;
static constexpr float kPersonRescueMinAspect = 1.15f;
static constexpr float kPersonRescueMinAreaRatio = 0.012f;
static constexpr float kPersonRescueMinLongSideRatio = 0.18f;

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

inline float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
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

int NormalizeModelClassId(int class_id) {
    return class_id == kDarkPersonClassId ? kPersonClassId : class_id;
}

const char* DetectClassName(int class_id) {
    if (class_id < 0 || class_id >= static_cast<int>(kDetectClassNames.size())) {
        return "unknown";
    }
    return kDetectClassNames[static_cast<size_t>(class_id)];
}

std::string FormatClassLabel(int class_id) {
    std::ostringstream oss;
    oss << "class=" << class_id << "(" << DetectClassName(class_id) << ")";
    return oss.str();
}

bool IsPersonLikeClass(int class_id) {
    return class_id == kPersonClassId || class_id == kDarkPersonClassId;
}

float PersonLikeScoreAt(const float* cls_ptr,
                        int y,
                        int x,
                        int feat_h,
                        int feat_w,
                        int num_classes) {
    if (cls_ptr == nullptr || num_classes <= kPersonClassId) {
        return 0.0f;
    }

    float score = Sigmoid(cls_ptr[OffsetOutput(y, x, kPersonClassId, feat_h, feat_w, num_classes)]);
    if (num_classes > kDarkPersonClassId) {
        score = std::max(score,
                         Sigmoid(cls_ptr[OffsetOutput(y, x, kDarkPersonClassId, feat_h, feat_w, num_classes)]));
    }
    return score;
}

float DFLIntegral(const float* logits, int reg_max) {
    float max_logit = logits[0];
    for (int i = 1; i < reg_max; ++i) {
        max_logit = std::max(max_logit, logits[i]);
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

bool HasClassDetection(const std::vector<ObjectDetection>& dets, int class_id) {
    for (const auto& det : dets) {
        if (det.class_id == class_id) {
            return true;
        }
    }
    return false;
}

bool HasPersonLikeDetection(const std::vector<ObjectDetection>& dets) {
    for (const auto& det : dets) {
        if (NormalizeModelClassId(det.class_id) == kPersonClassId) {
            return true;
        }
    }
    return false;
}

void CollectPersonLikeScoreStats(const float* cls_ptr,
                                 int feat_h,
                                 int feat_w,
                                 int num_classes,
                                 float hit_threshold,
                                 float* best_score,
                                 int* hit_count) {
    if (cls_ptr == nullptr || num_classes <= kPersonClassId) {
        return;
    }

    float local_best = 0.0f;
    int local_hits = 0;
    for (int gy = 0; gy < feat_h; ++gy) {
        for (int gx = 0; gx < feat_w; ++gx) {
            const float score = PersonLikeScoreAt(cls_ptr, gy, gx, feat_h, feat_w, num_classes);
            local_best = std::max(local_best, score);
            if (score >= hit_threshold) {
                ++local_hits;
            }
        }
    }

    if (best_score != nullptr) {
        *best_score = std::max(*best_score, local_best);
    }
    if (hit_count != nullptr) {
        *hit_count += local_hits;
    }
}

bool IsHorizontalPersonRescueCandidate(const ObjectDetection& det,
                                       const std::array<int, 2>& crop_shape) {
    if (det.class_id != kPersonClassId || det.score < kPersonRescueConfThreshold) {
        return false;
    }

    const float w = std::max(0.0f, det.box[2] - det.box[0]);
    const float h = std::max(0.0f, det.box[3] - det.box[1]);
    if (w < 40.0f || h < 20.0f) {
        return false;
    }

    const float area_ratio = (w * h) /
        static_cast<float>(std::max(1, crop_shape[0] * crop_shape[1]));
    const float long_side_ratio =
        std::max(w / static_cast<float>(std::max(1, crop_shape[0])),
                 h / static_cast<float>(std::max(1, crop_shape[1])));
    const float aspect = w / std::max(h, 1.0f);

    return aspect >= kPersonRescueMinAspect &&
           (area_ratio >= kPersonRescueMinAreaRatio ||
            long_side_ratio >= kPersonRescueMinLongSideRatio);
}

bool IsPersonTrackAssistCandidate(const ObjectDetection& det,
                                  const std::array<int, 2>& crop_shape) {
    if (det.class_id != kPersonClassId || det.score < kPersonTrackAssistConfThreshold) {
        return false;
    }

    const float w = std::max(0.0f, det.box[2] - det.box[0]);
    const float h = std::max(0.0f, det.box[3] - det.box[1]);
    if (w < 32.0f || h < 48.0f) {
        return false;
    }

    const float area_ratio = (w * h) /
        static_cast<float>(std::max(1, crop_shape[0] * crop_shape[1]));
    const float long_side_ratio =
        std::max(w / static_cast<float>(std::max(1, crop_shape[0])),
                 h / static_cast<float>(std::max(1, crop_shape[1])));

    return area_ratio >= 0.010f || long_side_ratio >= 0.16f;
}

bool IsLikelyLowLightFrame(ssne_tensor_t tensor) {
    if (get_data(tensor) == nullptr || get_data_format(tensor) != SSNE_YUV422_16) {
        return false;
    }

    const uint32_t width = get_width(tensor);
    const uint32_t height = get_height(tensor);
    const size_t mem_size = get_mem_size(tensor);
    if (width == 0 || height == 0 || mem_size == 0) {
        return false;
    }

    const size_t row_stride = mem_size / static_cast<size_t>(height);
    if (row_stride < static_cast<size_t>(width) * 2U) {
        return false;
    }

    const uint8_t* data = static_cast<const uint8_t*>(get_data(tensor));
    if (data == nullptr) {
        return false;
    }

    const uint32_t sample_step_y = std::max<uint32_t>(1, height / 48U);
    const uint32_t sample_step_x = std::max<uint32_t>(1, width / 64U);
    uint64_t sample_sum = 0;
    uint64_t dark_count = 0;
    uint64_t sample_count = 0;

    for (uint32_t y = 0; y < height; y += sample_step_y) {
        const uint8_t* row = data + static_cast<size_t>(y) * row_stride;
        for (uint32_t x = 0; x < width; x += sample_step_x) {
            const uint8_t luma0 = row[static_cast<size_t>(x) * 2U];
            const uint8_t luma1 = row[static_cast<size_t>(x) * 2U + 1U];
            const uint8_t luma = (std::abs(static_cast<int>(luma0) - 128) >
                                  std::abs(static_cast<int>(luma1) - 128))
                ? luma0
                : luma1;
            sample_sum += luma;
            dark_count += (luma < 48U) ? 1U : 0U;
            sample_count += 1U;
        }
    }

    if (sample_count == 0) {
        return false;
    }

    const float mean_luma = static_cast<float>(sample_sum) / static_cast<float>(sample_count);
    const float dark_ratio = static_cast<float>(dark_count) / static_cast<float>(sample_count);
    return mean_luma < 60.0f || dark_ratio > 0.55f;
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
    if (elem_count == 0 || (elem_count % hw) != 0) {
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

void NMSDetect(std::vector<ObjectDetection>& dets, float iou_thres, int top_k) {
    std::sort(dets.begin(), dets.end(), [](const ObjectDetection& a, const ObjectDetection& b) {
        return a.score > b.score;
    });

    std::vector<int> suppressed(dets.size(), 0);
    std::vector<ObjectDetection> keep;
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
            if (suppressed[j] ||
                NormalizeModelClassId(dets[i].class_id) !=
                    NormalizeModelClassId(dets[j].class_id)) {
                continue;
            }

            if (IoUBox(dets[i].box, dets[j].box) > iou_thres) {
                suppressed[j] = 1;
            }
        }
    }

    dets.swap(keep);
}

void PrintOutputInfo(int output_idx, ssne_tensor_t tensor, const std::array<int, 2>& det_shape) {
    const uint8_t dtype = get_data_type(tensor);
    const int channels = GetTensorChannels(tensor);
    const int stride = InferStrideFromTensor(tensor, det_shape);
    std::printf("  detect_output[%d] : dtype=%s total=%u mem=%zu shape=[1 x %d x %u x %u] stride=%d\n",
                output_idx, TensorTypeName(dtype), get_total_size(tensor), get_mem_size(tensor),
                channels, get_height(tensor), get_width(tensor), stride);
}

bool AssignOutputToBranch(ssne_tensor_t tensor,
                          int output_idx,
                          const std::array<int, 2>& det_shape,
                          int num_classes,
                          std::array<DetectOutputBranch, kYoloBranchCount>& branches) {
    const uint8_t dtype = get_data_type(tensor);
    if (dtype != SSNE_FLOAT32) {
        LOG_WARN("detect output[%d] dtype=%s is not float32, skip this tensor\n",
                 output_idx, TensorTypeName(dtype));
        return false;
    }

    const int channels = GetTensorChannels(tensor);
    const int stride = InferStrideFromTensor(tensor, det_shape);
    const int branch_idx = BranchIndexFromStride(stride);
    if (channels <= 0 || branch_idx < 0) {
        LOG_WARN("detect output[%d] has invalid channels=%d or stride=%d, skip this tensor\n",
                 output_idx, channels, stride);
        return false;
    }

    const float* data_ptr = reinterpret_cast<const float*>(get_data(tensor));
    if (data_ptr == nullptr) {
        LOG_WARN("detect output[%d] data pointer is null, skip this tensor\n", output_idx);
        return false;
    }

    // 裁剪后的 nano 模型输出为 3 个 box 分支和 3 个 cls 分支，按 stride 自动放回对应尺度。
    DetectOutputBranch& branch = branches[branch_idx];
    branch.feat_h = static_cast<int>(get_height(tensor));
    branch.feat_w = static_cast<int>(get_width(tensor));
    branch.stride = stride;

    if (channels == kYoloBoxChannels) {
        branch.box_ptr = data_ptr;
        return true;
    }
    if (channels == num_classes) {
        branch.cls_ptr = data_ptr;
        return true;
    }

    LOG_WARN("detect output[%d] has unsupported channels=%d, skip this tensor\n",
             output_idx, channels);
    return false;
}

void DecodeDetectBranch(const float* box_ptr,
                        const float* cls_ptr,
                        int feat_h,
                        int feat_w,
                        int stride,
                        int num_classes,
                        float conf_threshold,
                        float person_conf_threshold,
                        float w_scale,
                        float h_scale,
                        const std::array<int, 2>& crop_shape,
                        const std::array<int, 2>& det_shape,
                        std::vector<ObjectDetection>& dets) {
    if (box_ptr == nullptr || cls_ptr == nullptr) {
        LOG_WARN("detect branch pointer is null at stride=%d\n", stride);
        return;
    }

    for (int gy = 0; gy < feat_h; ++gy) {
        for (int gx = 0; gx < feat_w; ++gx) {
            int best_class = -1;
            float best_score = 0.0f;
            for (int cls = 0; cls < num_classes; ++cls) {
                const float score = Sigmoid(cls_ptr[OffsetOutput(gy, gx, cls, feat_h, feat_w, num_classes)]);
                if (score > best_score) {
                    best_score = score;
                    best_class = cls;
                }
            }

            if (best_class < 0) {
                continue;
            }

            const float person_like_score =
                PersonLikeScoreAt(cls_ptr, gy, gx, feat_h, feat_w, num_classes);
            const bool keep_best =
                best_score >= (IsPersonLikeClass(best_class) ? person_conf_threshold : conf_threshold);
            const bool keep_person =
                !IsPersonLikeClass(best_class) && person_like_score >= person_conf_threshold;
            if (!keep_best && !keep_person) {
                continue;
            }

            float side_logits[kYoloRegMax];
            float ltrb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int side = 0; side < 4; ++side) {
                for (int bin = 0; bin < kYoloRegMax; ++bin) {
                    const int c = side * kYoloRegMax + bin;
                    side_logits[bin] = box_ptr[OffsetOutput(gy, gx, c, feat_h, feat_w, kYoloBoxChannels)];
                }
                ltrb[side] = DFLIntegral(side_logits, kYoloRegMax);
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

            const std::array<float, 4> mapped_box = {
                std::max(0.0f, std::min(x1 * w_scale, static_cast<float>(crop_shape[0]))),
                std::max(0.0f, std::min(y1 * h_scale, static_cast<float>(crop_shape[1]))),
                std::max(0.0f, std::min(x2 * w_scale, static_cast<float>(crop_shape[0]))),
                std::max(0.0f, std::min(y2 * h_scale, static_cast<float>(crop_shape[1])))
            };

            if (mapped_box[2] <= mapped_box[0] || mapped_box[3] <= mapped_box[1]) {
                continue;
            }

            if (keep_best) {
                ObjectDetection det;
                det.class_id = best_class;
                det.score = best_score;
                det.box = mapped_box;
                dets.push_back(det);
            }

            if (keep_person) {
                ObjectDetection det;
                det.class_id = kPersonClassId;
                det.score = person_like_score;
                det.box = mapped_box;
                dets.push_back(det);
            }
        }
    }
}

void DecodeHorizontalPersonRescueBranch(const float* box_ptr,
                                        const float* cls_ptr,
                                        int feat_h,
                                        int feat_w,
                                        int stride,
                                        int num_classes,
                                        float conf_threshold,
                                        bool horizontal_only,
                                        float w_scale,
                                        float h_scale,
                                        const std::array<int, 2>& crop_shape,
                                        const std::array<int, 2>& det_shape,
                                        std::vector<ObjectDetection>& dets) {
    if (num_classes <= kPersonClassId) {
        return;
    }

    for (int gy = 0; gy < feat_h; ++gy) {
        for (int gx = 0; gx < feat_w; ++gx) {
            const float person_score = PersonLikeScoreAt(cls_ptr, gy, gx, feat_h, feat_w, num_classes);
            if (person_score < conf_threshold) {
                continue;
            }

            float side_logits[kYoloRegMax];
            float ltrb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int side = 0; side < 4; ++side) {
                for (int bin = 0; bin < kYoloRegMax; ++bin) {
                    const int c = side * kYoloRegMax + bin;
                    side_logits[bin] = box_ptr[OffsetOutput(gy, gx, c, feat_h, feat_w, kYoloBoxChannels)];
                }
                ltrb[side] = DFLIntegral(side_logits, kYoloRegMax);
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

            ObjectDetection det;
            det.class_id = kPersonClassId;
            det.score = person_score;
            det.box = {
                std::max(0.0f, std::min(x1 * w_scale, static_cast<float>(crop_shape[0]))),
                std::max(0.0f, std::min(y1 * h_scale, static_cast<float>(crop_shape[1]))),
                std::max(0.0f, std::min(x2 * w_scale, static_cast<float>(crop_shape[0]))),
                std::max(0.0f, std::min(y2 * h_scale, static_cast<float>(crop_shape[1])))
            };

            const bool keep_candidate = horizontal_only ?
                IsHorizontalPersonRescueCandidate(det, crop_shape) :
                IsPersonTrackAssistCandidate(det, crop_shape);
            if (!keep_candidate) {
                continue;
            }

            dets.push_back(det);
        }
    }
}

void FlushDetectPerfIfNeeded() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    const double elapsed_ms =
        static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(now - g_detect_perf.window_begin).count());
    if (elapsed_ms < 1000.0 || g_detect_perf.frames == 0) {
        return;
    }

    const double fps = static_cast<double>(g_detect_perf.frames) * 1000.0 / elapsed_ms;
    const double inv = 1.0 / static_cast<double>(g_detect_perf.frames);
    const std::string best_class_label = FormatClassLabel(g_detect_perf.last_best_class);
    LOG_INFO("detect fps=%.2f preprocess=%.2fms inference=%.2fms getoutput=%.2fms decode=%.2fms det_count=%d best=%s best_score=%.3f\n",
             fps,
             g_detect_perf.preprocess_ms * inv,
             g_detect_perf.inference_ms * inv,
             g_detect_perf.getoutput_ms * inv,
             g_detect_perf.decode_ms * inv,
             g_detect_perf.last_det_count,
             best_class_label.c_str(),
             g_detect_perf.last_best_score);
    g_detect_perf.frames = 0;
    g_detect_perf.preprocess_ms = 0.0;
    g_detect_perf.inference_ms = 0.0;
    g_detect_perf.getoutput_ms = 0.0;
    g_detect_perf.decode_ms = 0.0;
    g_detect_perf.last_det_count = 0;
    g_detect_perf.last_best_class = -1;
    g_detect_perf.last_best_score = 0.0f;
    g_detect_perf.last_best_person_cls_score = 0.0f;
    g_detect_perf.last_person_cls_hits = 0;
    g_detect_perf.window_begin = now;
}

}  // namespace

void YOLOV8NANO::Initialize(std::string& model_path,
                            std::array<int, 2>* in_img_shape,
                            std::array<int, 2>* in_det_shape,
                            int in_box_len,
                            int in_num_classes) {
    img_shape = *in_img_shape;
    det_shape = *in_det_shape;
    box_len = in_box_len;
    num_classes = in_num_classes;

    w_scale = static_cast<float>(img_shape[0]) / static_cast<float>(det_shape[0]);
    h_scale = static_cast<float>(img_shape[1]) / static_cast<float>(det_shape[1]);

    pipe_offline = GetAIPreprocessPipe();

    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);
    const int move_to_sram_ret = ssne_movemodeltosram(model_id);
    if (move_to_sram_ret == 0) {
        LOG_INFO("detect model moved to SRAM for faster access\n");
    } else {
        LOG_WARN("detect ssne_movemodeltosram ret=%d, continue with default placement\n", move_to_sram_ret);
    }
    SetNormalize(pipe_offline, model_id);

    const uint32_t det_width = static_cast<uint32_t>(det_shape[0]);
    const uint32_t det_height = static_cast<uint32_t>(det_shape[1]);
    inputs[0] = create_tensor(det_width, det_height, YOLO_INPUT_FORMAT, SSNE_BUF_AI);
    if (!IsValidTensor(inputs[0])) {
        LOG_ERROR("detect input tensor allocation failed for [%u x %u], mem=%zu\n",
                  det_width, det_height, get_mem_size(inputs[0]));
        return;
    }

    int dtype = -1;
    ssne_get_model_input_dtype(model_id, &dtype);
    set_data_type(inputs[0], dtype);

    LOG_INFO("YOLOv8 detect initialized\n");
    LOG_INFO("model=%s crop=[%d,%d] det=[%d,%d] input=[%u,%u] format=%s classes=%d scale=(%.6f,%.6f)\n",
             model_path.c_str(), img_shape[0], img_shape[1], det_shape[0], det_shape[1],
             det_width, det_height, YOLO_INPUT_FORMAT_NAME, num_classes, w_scale, h_scale);
}

void YOLOV8NANO::Predict(ssne_tensor_t* img,
                         ObjectDetectionResult* result,
                         float conf_threshold,
                         float person_conf_threshold) {
    using clock = std::chrono::steady_clock;
    if (result == nullptr) {
        LOG_ERROR("detect predict got null result pointer\n");
        return;
    }
    result->Clear();

    if (img == nullptr || !IsValidTensor(*img)) {
        LOG_ERROR("detect predict got invalid input tensor\n");
        return;
    }
    if (!IsValidTensor(inputs[0])) {
        LOG_ERROR("detect input tensor is invalid, skip this frame\n");
        return;
    }
    if (person_conf_threshold < 0.0f) {
        person_conf_threshold = conf_threshold;
    }

    const auto preprocess_begin = clock::now();
    int ret = RunAiPreprocessPipe(pipe_offline, *img, inputs[0]);
    const auto preprocess_end = clock::now();
    if (ret != 0) {
        LOG_ERROR("detect preprocess failed, ret=%d\n", ret);
        return;
    }

    const auto inference_begin = clock::now();
    ret = ssne_inference(model_id, 1, inputs);
    const auto inference_end = clock::now();
    if (ret != 0) {
        LOG_ERROR("detect inference failed, ret=%d\n", ret);
        return;
    }

    const auto getoutput_begin = clock::now();
    ret = ssne_getoutput(model_id, kYoloOutputCount, outputs);
    const auto getoutput_end = clock::now();
    if (ret != 0) {
        LOG_ERROR("detect getoutput failed, ret=%d\n", ret);
        return;
    }

    if (!g_detect_output_layout_logged) {
        LOG_INFO("YOLOv8 detect raw outputs (layout=%s):\n",
#if YOLO_OUTPUT_LAYOUT_NCHW
                 "NCHW");
#else
                 "NHWC");
#endif
        for (int i = 0; i < kYoloOutputCount; ++i) {
            PrintOutputInfo(i, outputs[i], det_shape);
        }
        g_detect_output_layout_logged = true;
    }

    const auto decode_begin = clock::now();
    std::array<DetectOutputBranch, kYoloBranchCount> branches = {};
    for (int i = 0; i < kYoloOutputCount; ++i) {
        AssignOutputToBranch(outputs[i], i, det_shape, num_classes, branches);
    }

    // 这里仅统计 person/dark_person 的弱响应，用于后面的低阈值人体兜底，不影响其它类别阈值。
    float best_person_cls_score = 0.0f;
    int person_cls_hits = 0;
    for (int i = 0; i < kYoloBranchCount; ++i) {
        const DetectOutputBranch& branch = branches[i];
        if (!branch.IsComplete()) {
            continue;
        }
        CollectPersonLikeScoreStats(branch.cls_ptr,
                                    branch.feat_h,
                                    branch.feat_w,
                                    num_classes,
                                    std::max(0.20f, person_conf_threshold - 0.10f),
                                    &best_person_cls_score,
                                    &person_cls_hits);
    }

    std::vector<ObjectDetection> dets;
    dets.reserve(2000);
    float decode_conf_threshold = conf_threshold;
    for (int i = 0; i < kYoloBranchCount; ++i) {
        const DetectOutputBranch& branch = branches[i];
        if (!branch.IsComplete()) {
            const int expected_stride = (i == 0) ? 8 : ((i == 1) ? 16 : 32);
            LOG_WARN("detect stride=%d branch is incomplete (box=%d cls=%d), skip decode\n",
                     branch.stride > 0 ? branch.stride : expected_stride,
                     branch.box_ptr != nullptr,
                     branch.cls_ptr != nullptr);
            continue;
        }

        DecodeDetectBranch(branch.box_ptr, branch.cls_ptr,
                           branch.feat_h, branch.feat_w, branch.stride, num_classes,
                           decode_conf_threshold, person_conf_threshold,
                           w_scale, h_scale, img_shape, det_shape, dets);
    }
    const bool likely_low_light = IsLikelyLowLightFrame(*img);
    // 二次宽松解码只在低照或连续空检时触发，避免正常光照下把背景纹理误当成目标。
    const bool allow_relaxed_decode =
        dets.empty() && (likely_low_light || g_detect_empty_frames >= 2);
    if (allow_relaxed_decode) {
        const float relaxed_person_conf_threshold =
            std::max(0.20f, person_conf_threshold - 0.08f);
        if (relaxed_person_conf_threshold < person_conf_threshold) {
            for (int i = 0; i < kYoloBranchCount; ++i) {
                const DetectOutputBranch& branch = branches[i];
                if (!branch.IsComplete()) {
                    continue;
                }

                DecodeDetectBranch(branch.box_ptr, branch.cls_ptr,
                                   branch.feat_h, branch.feat_w, branch.stride, num_classes,
                                   conf_threshold, relaxed_person_conf_threshold,
                                   w_scale, h_scale, img_shape, det_shape, dets);
            }
        }
    }
    // 极暗环境下，人躺倒或侧身时框会变扁；此兜底只补 person-like，不放宽动物和火焰。
    const bool allow_person_rescue =
        !HasPersonLikeDetection(dets) &&
        likely_low_light &&
        g_detect_empty_frames >= 2;
    if (allow_person_rescue) {
        const float person_rescue_conf_threshold =
            std::max(kPersonRescueConfThreshold, person_conf_threshold - 0.05f);
        for (int i = 0; i < kYoloBranchCount; ++i) {
            const DetectOutputBranch& branch = branches[i];
            if (!branch.IsComplete()) {
                continue;
            }

            DecodeHorizontalPersonRescueBranch(branch.box_ptr, branch.cls_ptr,
                                               branch.feat_h, branch.feat_w, branch.stride, num_classes,
                                               person_rescue_conf_threshold, true, w_scale, h_scale,
                                               img_shape, det_shape, dets);
        }
    }
    // pose/track 已给出人体先验时，允许保留更弱的人框，帮助主目标持续跟踪。
    const bool allow_person_track_assist =
        !HasPersonLikeDetection(dets) &&
        person_cls_hits > 0;
    if (allow_person_track_assist) {
        const float person_assist_conf_threshold =
            std::max(kPersonTrackAssistConfThreshold, person_conf_threshold - 0.06f);
        for (int i = 0; i < kYoloBranchCount; ++i) {
            const DetectOutputBranch& branch = branches[i];
            if (!branch.IsComplete()) {
                continue;
            }

            DecodeHorizontalPersonRescueBranch(branch.box_ptr, branch.cls_ptr,
                                               branch.feat_h, branch.feat_w, branch.stride, num_classes,
                                               person_assist_conf_threshold, false, w_scale, h_scale,
                                               img_shape, det_shape, dets);
        }
    }
    NMSDetect(dets, nms_threshold, top_k);
    const auto decode_end = clock::now();

    result->Reserve(std::min(static_cast<int>(dets.size()), keep_top_k));

    const int final_count = std::min(static_cast<int>(dets.size()), keep_top_k);
    for (int i = 0; i < final_count; ++i) {
        result->detections.push_back(dets[i]);
        result->boxes.push_back(dets[i].box);
        result->scores.push_back(dets[i].score);
        result->class_ids.push_back(dets[i].class_id);
    }
    result->Resize(final_count);

    g_detect_perf.last_det_count = final_count;
    if (final_count > 0) {
        g_detect_perf.last_best_class = result->class_ids[0];
        g_detect_perf.last_best_score = result->scores[0];
        g_detect_empty_frames = 0;
    } else {
        g_detect_perf.last_best_class = -1;
        g_detect_perf.last_best_score = 0.0f;
        g_detect_empty_frames += 1;
    }
    g_detect_perf.last_best_person_cls_score = best_person_cls_score;
    g_detect_perf.last_person_cls_hits = person_cls_hits;

    g_detect_perf.frames += 1;
    g_detect_perf.preprocess_ms +=
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_begin).count()) / 1000.0;
    g_detect_perf.inference_ms +=
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_begin).count()) / 1000.0;
    g_detect_perf.getoutput_ms +=
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(getoutput_end - getoutput_begin).count()) / 1000.0;
    g_detect_perf.decode_ms +=
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_begin).count()) / 1000.0;
    FlushDetectPerfIfNeeded();

}

void YOLOV8NANO::Release() {
    release_tensor(inputs[0]);
    inputs[0] = ssne_tensor_t{};
    ReleaseOutputTensors(outputs, kYoloOutputCount);
    ReleaseAIPreprocessPipe(pipe_offline);
}
