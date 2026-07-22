#include "stranger_mode.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>
#include <sys/stat.h>

#include "log.hpp"

namespace {

#ifndef STRANGER_OUTPUT_LAYOUT_NCHW
#define STRANGER_OUTPUT_LAYOUT_NCHW 0
#endif

#ifndef STRANGER_SWAP_CLS_OBJ_OUTPUTS
#define STRANGER_SWAP_CLS_OBJ_OUTPUTS 0
#endif

#ifndef STRANGER_USE_FULL_FRAME_ROI
#define STRANGER_USE_FULL_FRAME_ROI 1
#endif

#ifndef STRANGER_USE_MODEL_NORMALIZE
#define STRANGER_USE_MODEL_NORMALIZE 0
#endif

constexpr const char* kDefaultGenericFaceModelPath =
    "/app_demo/app_assets/models/yunet_160x120.m1model";
constexpr float kKnownScoreThreshold = 0.50f;
constexpr float kKnownAssociationIouThreshold = 0.20f;
constexpr float kSecondStageDecodeThreshold = 0.15f;
constexpr int kStrangerDecodeLevelBegin = 1;
constexpr int kStrangerDecodeLevelEnd = 2;
constexpr int kSecondStageFrameInterval = 5;
constexpr float kRoiExpandScale = 1.20f;
constexpr float kRoiMinSideScale = 1.25f;
constexpr int kRoiMinSidePx = 224;
constexpr int kRoiMaxSidePx = 384;
constexpr int kRoiAlign = 8;

bool IsValidTensor(ssne_tensor_t tensor) {
    return get_data(tensor) != nullptr &&
           get_width(tensor) > 0 &&
           get_height(tensor) > 0 &&
           get_mem_size(tensor) > 0;
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

inline size_t OffsetOutput(int y, int x, int c, int feat_h, int feat_w, int channels) {
#if STRANGER_OUTPUT_LAYOUT_NCHW
    (void)channels;
    return static_cast<size_t>((c * feat_h + y) * feat_w + x);
#else
    (void)feat_h;
    return static_cast<size_t>((y * feat_w + x) * channels + c);
#endif
}

void RepackOutputToChw(ssne_tensor_t tensor, std::vector<float>* dst) {
    if (dst == nullptr) {
        return;
    }

    const int width = static_cast<int>(get_width(tensor));
    const int height = static_cast<int>(get_height(tensor));
    const int channels = GetTensorChannels(tensor);
    const float* src = reinterpret_cast<const float*>(get_data(tensor));
    if (src == nullptr || width <= 0 || height <= 0 || channels <= 0) {
        dst->clear();
        return;
    }

    dst->assign(static_cast<size_t>(channels) * static_cast<size_t>(height) *
                    static_cast<size_t>(width),
                0.0f);
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t src_idx = OffsetOutput(y, x, c, height, width, channels);
                const size_t dst_idx =
                    (static_cast<size_t>(c) * static_cast<size_t>(height) +
                     static_cast<size_t>(y)) *
                        static_cast<size_t>(width) +
                    static_cast<size_t>(x);
                (*dst)[dst_idx] = src[src_idx];
            }
        }
    }
}

void LogTensorInfo(const char* prefix, int index, ssne_tensor_t tensor) {
    LOG_INFO("%s[%d]: dtype=%s total=%u mem=%zu shape=[1 x %d x %u x %u] layout=%s\n",
             prefix,
             index,
             TensorTypeName(get_data_type(tensor)),
             get_total_size(tensor),
             get_mem_size(tensor),
             GetTensorChannels(tensor),
             get_height(tensor),
             get_width(tensor),
#if STRANGER_OUTPUT_LAYOUT_NCHW
             "NCHW");
#else
             "NHWC->CHW");
#endif
}

void LogInputTensorPreview(const char* prefix, ssne_tensor_t tensor) {
    if (get_data(tensor) == nullptr) {
        LOG_WARN("%s tensor is null\n", prefix);
        return;
    }

    const uint8_t dtype = get_data_type(tensor);
    const uint32_t total = get_total_size(tensor);
    float minv = 0.0f;
    float maxv = 0.0f;
    float sample0 = 0.0f;
    float sample1 = 0.0f;
    float sample2 = 0.0f;
    bool initialized = false;

    if (dtype == SSNE_INT8) {
        const int8_t* ptr = reinterpret_cast<const int8_t*>(get_data(tensor));
        for (uint32_t i = 0; i < total; ++i) {
            const float v = static_cast<float>(ptr[i]);
            if (!initialized) {
                minv = maxv = v;
                initialized = true;
            } else {
                minv = std::min(minv, v);
                maxv = std::max(maxv, v);
            }
        }
        if (total > 0) sample0 = static_cast<float>(ptr[0]);
        if (total > 1) sample1 = static_cast<float>(ptr[1]);
        if (total > 2) sample2 = static_cast<float>(ptr[2]);
    } else if (dtype == SSNE_UINT8) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(get_data(tensor));
        for (uint32_t i = 0; i < total; ++i) {
            const float v = static_cast<float>(ptr[i]);
            if (!initialized) {
                minv = maxv = v;
                initialized = true;
            } else {
                minv = std::min(minv, v);
                maxv = std::max(maxv, v);
            }
        }
        if (total > 0) sample0 = static_cast<float>(ptr[0]);
        if (total > 1) sample1 = static_cast<float>(ptr[1]);
        if (total > 2) sample2 = static_cast<float>(ptr[2]);
    } else if (dtype == SSNE_FLOAT32) {
        const float* ptr = reinterpret_cast<const float*>(get_data(tensor));
        for (uint32_t i = 0; i < total; ++i) {
            const float v = ptr[i];
            if (!initialized) {
                minv = maxv = v;
                initialized = true;
            } else {
                minv = std::min(minv, v);
                maxv = std::max(maxv, v);
            }
        }
        if (total > 0) sample0 = ptr[0];
        if (total > 1) sample1 = ptr[1];
        if (total > 2) sample2 = ptr[2];
    }

    LOG_INFO("%s: dtype=%s total=%u min=%.3f max=%.3f sample=[%.3f %.3f %.3f]\n",
             prefix,
             TensorTypeName(dtype),
             total,
             minv,
             maxv,
             sample0,
             sample1,
             sample2);
}

float ClampFloat(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

float Sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

struct FaceRoi {
    int left = 0;
    int top = 0;
    int size = 0;
};

bool CropYuv422SquareTensor(ssne_tensor_t input, const FaceRoi& roi, ssne_tensor_t* cropped) {
    if (cropped == nullptr || roi.size <= 0) {
        return false;
    }
    if (get_data(input) == nullptr || get_data_format(input) != SSNE_YUV422_16) {
        return false;
    }

    const uint32_t src_w = get_width(input);
    const uint32_t src_h = get_height(input);
    if (src_w == 0 || src_h == 0) {
        return false;
    }
    if (roi.left < 0 || roi.top < 0 ||
        roi.left + roi.size > static_cast<int>(src_w) ||
        roi.top + roi.size > static_cast<int>(src_h)) {
        return false;
    }

    ssne_tensor_t roi_tensor = create_tensor(static_cast<uint32_t>(roi.size),
                                             static_cast<uint32_t>(roi.size),
                                             SSNE_YUV422_16,
                                             SSNE_BUF_AI);
    if (!IsValidTensor(roi_tensor)) {
        return false;
    }

    const uint8_t* src = static_cast<const uint8_t*>(get_data(input));
    uint8_t* dst = static_cast<uint8_t*>(get_data(roi_tensor));
    if (src == nullptr || dst == nullptr) {
        release_tensor(roi_tensor);
        return false;
    }

    const size_t src_stride = get_mem_size(input) / static_cast<size_t>(src_h);
    const size_t dst_stride = get_mem_size(roi_tensor) / static_cast<size_t>(roi.size);
    const size_t row_bytes = static_cast<size_t>(roi.size) * 2U;

    for (int y = 0; y < roi.size; ++y) {
        const uint8_t* src_row =
            src + static_cast<size_t>(roi.top + y) * src_stride + static_cast<size_t>(roi.left) * 2U;
        uint8_t* dst_row = dst + static_cast<size_t>(y) * dst_stride;
        std::memcpy(dst_row, src_row, row_bytes);
    }

    *cropped = roi_tensor;
    return true;
}

FaceTensorType ToFaceTensorType(uint8_t dtype) {
    return (dtype == SSNE_FLOAT32) ? FaceTensorType::kFloat32 : FaceTensorType::kInt8;
}

FaceRoi BuildFaceRoi(const FaceDetection& face, int image_width, int image_height) {
    const float center_x = face.x + face.w * 0.5f;
    const float center_y = face.y + face.h * 0.5f;
    const float base_side = std::max(face.w, face.h);
    const float desired_side =
        std::max(base_side * kRoiExpandScale,
                 std::max(base_side * kRoiMinSideScale,
                          static_cast<float>(kRoiMinSidePx)));

    const int max_square = std::max(
        kRoiAlign,
        (std::min(image_width, image_height) / kRoiAlign) * kRoiAlign);

    int size = static_cast<int>(std::ceil(desired_side));
    size = std::max(size, kRoiMinSidePx);
    size = std::min(size, kRoiMaxSidePx);
    size = std::min(size, max_square);
    size = (size / kRoiAlign) * kRoiAlign;
    if (size < kRoiAlign) {
        size = kRoiAlign;
    }

    int left = static_cast<int>(std::lround(center_x - static_cast<float>(size) * 0.5f));
    int top = static_cast<int>(std::lround(center_y - static_cast<float>(size) * 0.5f));

    left = std::max(0, std::min(left, image_width - size));
    top = std::max(0, std::min(top, image_height - size));

    FaceRoi roi;
    roi.left = left;
    roi.top = top;
    roi.size = size;
    return roi;
}

FaceRoi BuildFullFrameRoi(int image_width, int image_height) {
    int size = std::min(image_width, image_height);
    size = (size / kRoiAlign) * kRoiAlign;
    size = std::max(size, kRoiAlign);

    FaceRoi roi;
    roi.left = std::max(0, (image_width - size) / 2);
    roi.top = std::max(0, (image_height - size) / 2);
    roi.size = size;
    return roi;
}

float ReadFaceTensorValue(const FaceRawTensor& t, int c, int y, int x) {
    const size_t index =
        (static_cast<size_t>(c) * static_cast<size_t>(t.height) +
         static_cast<size_t>(y)) *
            static_cast<size_t>(t.width) +
        static_cast<size_t>(x);
    if (t.type == FaceTensorType::kInt8) {
        return static_cast<float>(static_cast<const int8_t*>(t.data)[index]) * t.scale;
    }
    return static_cast<const float*>(t.data)[index];
}

struct ScoreSummary {
    float best_cls = 0.0f;
    float best_obj = 0.0f;
    float best_fused = 0.0f;
};

float DetectionIou(const FaceDetection& a, const FaceDetection& b) {
    const float left = std::max(a.x, b.x);
    const float top = std::max(a.y, b.y);
    const float right = std::min(a.x + a.w, b.x + b.w);
    const float bottom = std::min(a.y + a.h, b.y + b.h);
    const float inter =
        std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    const float area_a = std::max(0.0f, a.w) * std::max(0.0f, a.h);
    const float area_b = std::max(0.0f, b.w) * std::max(0.0f, b.h);
    return inter / std::max(1e-6f, area_a + area_b - inter);
}

struct AssociationSummary {
    float best_score = 0.0f;
    float best_iou = 0.0f;
    float best_global_score = 0.0f;
};

AssociationSummary FindAssociatedKnown(const FaceResult& known_result,
                                       const FaceRoi& roi,
                                       const FaceDetection& crop_det) {
    AssociationSummary summary;
    float best_associated_score = 0.0f;
    for (int i = 0; i < known_result.count; ++i) {
        FaceDetection known = known_result.faces[i];
        known.x += static_cast<float>(roi.left);
        known.y += static_cast<float>(roi.top);
        summary.best_global_score = std::max(summary.best_global_score, known.score);
        const float iou = DetectionIou(known, crop_det);
        summary.best_iou = std::max(summary.best_iou, iou);
        if (iou >= kKnownAssociationIouThreshold) {
            best_associated_score = std::max(best_associated_score, known.score);
        }
    }

    summary.best_score = best_associated_score;
    return summary;
}

struct OutputBinding {
#if STRANGER_SWAP_CLS_OBJ_OUTPUTS
    int cls[3] = {6, 7, 8};
    int bbox[3] = {3, 4, 5};
    int obj[3] = {0, 1, 2};
    int kps[3] = {9, 10, 11};
#else
    int cls[3] = {0, 1, 2};
    int bbox[3] = {3, 4, 5};
    int obj[3] = {6, 7, 8};
    int kps[3] = {9, 10, 11};
#endif
};

struct TensorRange {
    float min_value = 0.0f;
    float max_value = 0.0f;
    float sigmoid_max = 0.0f;
    float hinge_prob_max = 0.0f;
};

TensorRange GetTensorRange(const FaceRawTensor& tensor) {
    TensorRange range;
    if (tensor.data == nullptr || tensor.width <= 0 || tensor.height <= 0 ||
        tensor.channels <= 0) {
        return range;
    }

    bool initialized = false;
    for (int c = 0; c < tensor.channels; ++c) {
        for (int y = 0; y < tensor.height; ++y) {
            for (int x = 0; x < tensor.width; ++x) {
                const float value = ReadFaceTensorValue(tensor, c, y, x);
                if (!initialized) {
                    range.min_value = value;
                    range.max_value = value;
                    initialized = true;
                } else {
                    range.min_value = std::min(range.min_value, value);
                    range.max_value = std::max(range.max_value, value);
                }
            }
        }
    }
    range.sigmoid_max = Sigmoid(range.max_value);
    range.hinge_prob_max = ClampFloat(range.max_value / 6.0f, 0.0f, 1.0f);
    return range;
}

void BuildBoundOutputs(const FaceRawTensor raw[FACE_OUTPUT_COUNT],
                       const OutputBinding& binding,
                       FaceRawTensor bound[FACE_OUTPUT_COUNT]) {
    for (int level = 0; level < 3; ++level) {
        bound[level] = raw[binding.cls[level]];
        bound[3 + level] = raw[binding.bbox[level]];
        bound[6 + level] = raw[binding.obj[level]];
        bound[9 + level] = raw[binding.kps[level]];
    }
}

void LogOneChannelOutputCandidates(const FaceRawTensor raw[FACE_OUTPUT_COUNT],
                                   const OutputBinding& binding) {
    LOG_INFO("stranger target output binding: cls=[%d,%d,%d] bbox=[%d,%d,%d] obj=[%d,%d,%d] kps=[%d,%d,%d] decode_levels=[%d,%d]\n",
             binding.cls[0], binding.cls[1], binding.cls[2],
             binding.bbox[0], binding.bbox[1], binding.bbox[2],
             binding.obj[0], binding.obj[1], binding.obj[2],
             binding.kps[0], binding.kps[1], binding.kps[2],
             kStrangerDecodeLevelBegin,
             kStrangerDecodeLevelEnd);

    for (int i = 0; i < FACE_OUTPUT_COUNT; ++i) {
        if (raw[i].channels != 1) {
            continue;
        }
        const TensorRange range = GetTensorRange(raw[i]);
        LOG_INFO("stranger target one-channel output[%d]: shape=[1x1x%dx%d] raw=[%.3f, %.3f] sigmoid_max=%.3f hinge_prob_max=%.3f\n",
                 i,
                 raw[i].height,
                 raw[i].width,
                 range.min_value,
                 range.max_value,
                 range.sigmoid_max,
                 range.hinge_prob_max);
    }
}

ScoreSummary ComputeScoreSummary(const FaceRawTensor raw[FACE_OUTPUT_COUNT],
                                 int level_begin,
                                 int level_end) {
    ScoreSummary summary;
    level_begin = std::max(0, std::min(level_begin, 2));
    level_end = std::max(level_begin, std::min(level_end, 2));
    for (int level = level_begin; level <= level_end; ++level) {
        const FaceRawTensor& cls = raw[level];
        const FaceRawTensor& obj = raw[6 + level];
        if (cls.data == nullptr || obj.data == nullptr ||
            cls.channels != 1 || obj.channels != 1 ||
            cls.width != obj.width || cls.height != obj.height) {
            continue;
        }
        for (int y = 0; y < cls.height; ++y) {
            for (int x = 0; x < cls.width; ++x) {
                const float cls_score =
                    Sigmoid(ReadFaceTensorValue(cls, 0, y, x));
                const float obj_score =
                    ClampFloat(ReadFaceTensorValue(obj, 0, y, x) / 6.0f, 0.0f, 1.0f);
                const float fused_score =
                    cls_score * obj_score;
                summary.best_cls = std::max(summary.best_cls, cls_score);
                summary.best_obj = std::max(summary.best_obj, obj_score);
                summary.best_fused = std::max(summary.best_fused, fused_score);
            }
        }
    }
    return summary;
}

void LogScoreHeadRange(const FaceRawTensor& cls,
                       const FaceRawTensor& obj,
                       int level) {
    if (cls.data == nullptr || obj.data == nullptr ||
        cls.channels != 1 || obj.channels != 1 ||
        cls.width != obj.width || cls.height != obj.height) {
        return;
    }

    float cls_min = 0.0f;
    float cls_max = 0.0f;
    float obj_min = 0.0f;
    float obj_max = 0.0f;
    bool initialized = false;
    for (int y = 0; y < cls.height; ++y) {
        for (int x = 0; x < cls.width; ++x) {
            const float cls_v = ReadFaceTensorValue(cls, 0, y, x);
            const float obj_v = ReadFaceTensorValue(obj, 0, y, x);
            if (!initialized) {
                cls_min = cls_max = cls_v;
                obj_min = obj_max = obj_v;
                initialized = true;
            } else {
                cls_min = std::min(cls_min, cls_v);
                cls_max = std::max(cls_max, cls_v);
                obj_min = std::min(obj_min, obj_v);
                obj_max = std::max(obj_max, obj_v);
            }
        }
    }

    LOG_INFO("stranger target score head[%d]: cls_raw=[%.3f, %.3f] obj_hinge=[%.3f, %.3f]\n",
             level,
             cls_min,
             cls_max,
             obj_min,
             obj_max);
}

}  // namespace

void StrangerModeRunner::Initialize(const std::string& model_path) {
    model_path_ = model_path;
    generic_face_model_path_ = kDefaultGenericFaceModelPath;
    warned_process_placeholder_ = false;
    last_error_.clear();
    generic_face_ready_ = false;

#if STRANGER_USE_MODEL_NORMALIZE
    const char* normalize_mode = "on";
#else
    const char* normalize_mode = "off";
#endif
    LOG_INFO("stranger mode setup begin target=%s generic=%s input=%dx%d obj=hinge normalize=%s\n",
             model_path_.c_str(),
             generic_face_model_path_.c_str(),
             FACE_NET_W,
             FACE_NET_H,
             normalize_mode);

    adapter_.Initialize(model_path_);
    LOG_INFO("stranger mode target adapter initialized=%d\n", adapter_.IsInitialized() ? 1 : 0);

    initialized_ = true;
    LOG_INFO("stranger mode initialized with deferred generic detector\n");
}

void StrangerModeRunner::Release() {
    initialized_ = false;
    generic_face_ready_ = false;
    adapter_.Release();
    generic_face_detector_.Release();
    warned_process_placeholder_ = false;
    last_error_.clear();
    has_last_second_stage_frame_ = false;
    last_second_stage_ok_ = false;
    last_second_stage_frame_id_ = 0;
    last_second_stage_result_ = FaceResult{};
    last_second_stage_cls_best_ = 0.0f;
    last_second_stage_obj_best_ = 0.0f;
    last_second_stage_fused_best_ = 0.0f;
}

bool StrangerModeRunner::IsInitialized() const {
    return initialized_;
}

const std::string& StrangerModeRunner::ModelPath() const {
    return model_path_;
}

const std::string& StrangerModeRunner::LastError() const {
    return last_error_;
}

int StrangerModeRunner::ProcessFrame(ssne_tensor_t* img,
                                     int crop_offset_x,
                                     int img_width,
                                     int img_height,
                                     uint16_t frame_id,
                                     FaceResult* result) {
    if (result != nullptr) {
        result->frame_id = frame_id;
        result->count = 0;
    }
    if (!initialized_) {
        last_error_ = "stranger mode runtime not initialized";
        return -1;
    }
    if (img == nullptr || !IsValidTensor(*img)) {
        last_error_ = "stranger mode got invalid image tensor";
        return -2;
    }
    if (!generic_face_ready_) {
        const std::array<int, 2> crop_shape = {1080, 1080};
        const std::array<int, 2> generic_det_shape = {160, 120};
        LOG_INFO("stranger mode lazy init generic detector model=%s\n",
                 generic_face_model_path_.c_str());
        generic_face_detector_.Initialize(generic_face_model_path_, crop_shape, generic_det_shape);
        generic_face_ready_ = generic_face_detector_.IsInitialized();
        if (!generic_face_ready_) {
            last_error_ = "generic yunet face detector init failed";
            LOG_ERROR("%s\n", last_error_.c_str());
            return -3;
        }
        LOG_INFO("stranger mode generic detector ready\n");
    }
    if (result == nullptr) {
        last_error_ = "stranger mode got null result";
        return -4;
    }

    std::vector<OfficialYuNet160Detection> generic_faces;
    if (!generic_face_detector_.Predict(img, &generic_faces, &last_error_)) {
        return -5;
    }

    const int crop_width = static_cast<int>(get_width(*img));
    const int crop_height = static_cast<int>(get_height(*img));
    const int final_count = std::min(static_cast<int>(generic_faces.size()), FACE_MAX_DETECTIONS);
    result->count = 0;
    for (int i = 0; i < final_count; ++i) {
        const OfficialYuNet160Detection& face = generic_faces[static_cast<size_t>(i)];
        FaceDetection det;
        det.x = ClampFloat(face.box[0] + static_cast<float>(crop_offset_x), 0.0f,
                           static_cast<float>(img_width));
        det.y = ClampFloat(face.box[1], 0.0f, static_cast<float>(img_height));
        const float x2 = ClampFloat(face.box[2] + static_cast<float>(crop_offset_x), 0.0f,
                                    static_cast<float>(img_width));
        const float y2 = ClampFloat(face.box[3], 0.0f, static_cast<float>(img_height));
        det.w = std::max(0.0f, x2 - det.x);
        det.h = std::max(0.0f, y2 - det.y);
        det.score = face.score;
        det.identity = FaceIdentity::kStranger;
        for (int p = 0; p < 5; ++p) {
            det.landmarks[p].x = ClampFloat(face.landmarks[static_cast<size_t>(p) * 2U] +
                                                static_cast<float>(crop_offset_x),
                                            0.0f,
                                            static_cast<float>(img_width));
            det.landmarks[p].y = ClampFloat(face.landmarks[static_cast<size_t>(p) * 2U + 1U],
                                            0.0f,
                                            static_cast<float>(img_height));
        }

        const FaceRoi roi = BuildFaceRoi(det, img_width, img_height);
        (void)roi;
        result->faces[result->count++] = det;
    }

    if (!adapter_.IsInitialized()) {
        last_error_ = "generic face detector active; target stranger model is not integrated yet";
        return 1;
    }

    bool any_known = false;
#if STRANGER_USE_FULL_FRAME_ROI
    const FaceRoi shared_roi = BuildFullFrameRoi(crop_width, crop_height);
    const bool should_run_second_stage =
        !has_last_second_stage_frame_ ||
        static_cast<uint16_t>(frame_id - last_second_stage_frame_id_) >=
            static_cast<uint16_t>(kSecondStageFrameInterval);
    bool second_stage_run_this_frame = false;
#endif
    for (int i = 0; i < result->count; ++i) {
        FaceDetection& det = result->faces[i];
        FaceDetection crop_det = det;
        crop_det.x = ClampFloat(det.x - static_cast<float>(crop_offset_x), 0.0f,
                                static_cast<float>(crop_width));
        crop_det.y = ClampFloat(det.y, 0.0f, static_cast<float>(crop_height));
        crop_det.w = ClampFloat(det.w, 0.0f,
                                static_cast<float>(crop_width) - crop_det.x);
        crop_det.h = ClampFloat(det.h, 0.0f,
                                static_cast<float>(crop_height) - crop_det.y);
        for (int p = 0; p < 5; ++p) {
            crop_det.landmarks[p].x =
                ClampFloat(det.landmarks[p].x - static_cast<float>(crop_offset_x),
                           0.0f,
                           static_cast<float>(crop_width));
            crop_det.landmarks[p].y =
                ClampFloat(det.landmarks[p].y, 0.0f, static_cast<float>(crop_height));
        }

#if STRANGER_USE_FULL_FRAME_ROI
        const FaceRoi roi = shared_roi;
#else
        const FaceRoi roi = BuildFaceRoi(crop_det, crop_width, crop_height);
#endif

        FaceResult known_result = {};
        ScoreSummary score_summary = {};

#if STRANGER_USE_FULL_FRAME_ROI
        if (should_run_second_stage && !second_stage_run_this_frame) {
#endif
            FaceRawTensor raw_outputs[FACE_OUTPUT_COUNT] = {};
            const int infer_ret =
                adapter_.infer_yuv_roi(img, roi.left, roi.top, roi.size, raw_outputs);
            if (infer_ret != 0) {
                LOG_WARN("stranger target face[%d] infer failed roi=[l=%d t=%d s=%d] crop_det=[x=%.1f y=%.1f w=%.1f h=%.1f] ret=%d\n",
                         i,
                         roi.left,
                         roi.top,
                         roi.size,
                         crop_det.x,
                         crop_det.y,
                         crop_det.w,
                         crop_det.h,
                         infer_ret);
#if STRANGER_USE_FULL_FRAME_ROI
                has_last_second_stage_frame_ = true;
                last_second_stage_frame_id_ = frame_id;
                last_second_stage_ok_ = false;
                second_stage_run_this_frame = true;
#endif
                continue;
            }

            const float roi_scale =
                static_cast<float>(FACE_NET_W) / static_cast<float>(std::max(roi.size, 1));
            FaceConfig cfg;
            cfg.score_threshold = kSecondStageDecodeThreshold;
            cfg.decode_level_begin = kStrangerDecodeLevelBegin;
            cfg.decode_level_end = kStrangerDecodeLevelEnd;
            OutputBinding binding;
            FaceRawTensor bound_outputs[FACE_OUTPUT_COUNT] = {};
            BuildBoundOutputs(raw_outputs, binding, bound_outputs);
            const int decode_ret =
                face_decode(bound_outputs, roi.size, roi.size, roi_scale, cfg, &known_result);
            score_summary =
                ComputeScoreSummary(bound_outputs, kStrangerDecodeLevelBegin, kStrangerDecodeLevelEnd);
            if (decode_ret != 0) {
                LOG_WARN("stranger target face[%d] decode error roi=[l=%d t=%d s=%d] decode_ret=%d cls_best=%.3f obj_hinge_prob_best=%.3f fused_best=%.3f\n",
                         i,
                         roi.left,
                         roi.top,
                         roi.size,
                         decode_ret,
                         score_summary.best_cls,
                         score_summary.best_obj,
                         score_summary.best_fused);
#if STRANGER_USE_FULL_FRAME_ROI
                has_last_second_stage_frame_ = true;
                last_second_stage_frame_id_ = frame_id;
                last_second_stage_ok_ = false;
                second_stage_run_this_frame = true;
#endif
                continue;
            }
#if STRANGER_USE_FULL_FRAME_ROI
            has_last_second_stage_frame_ = true;
            last_second_stage_frame_id_ = frame_id;
            last_second_stage_result_ = known_result;
            last_second_stage_cls_best_ = score_summary.best_cls;
            last_second_stage_obj_best_ = score_summary.best_obj;
            last_second_stage_fused_best_ = score_summary.best_fused;
            last_second_stage_ok_ = true;
            second_stage_run_this_frame = true;
        } else {
            if (!last_second_stage_ok_) {
                continue;
            }
            known_result = last_second_stage_result_;
            score_summary.best_cls = last_second_stage_cls_best_;
            score_summary.best_obj = last_second_stage_obj_best_;
            score_summary.best_fused = last_second_stage_fused_best_;
        }
#endif

        const AssociationSummary association =
            FindAssociatedKnown(known_result, roi, crop_det);
        LOG_INFO("stranger target face[%d] roi_mode=%s infer=%s roi=[l=%d t=%d s=%d] decode_levels=16,32 cls_best=%.3f obj_hinge_prob_best=%.3f fused_best=%.3f decode_best=%.3f global_best=%.3f assoc_iou=%.3f known_count=%d threshold=%.2f iou_threshold=%.2f\n",
                 i,
#if STRANGER_USE_FULL_FRAME_ROI
                 "full_frame",
#else
                 "face",
#endif
#if STRANGER_USE_FULL_FRAME_ROI
                 should_run_second_stage ? "run" : "cached",
#else
                 "run",
#endif
                 roi.left,
                 roi.top,
                 roi.size,
                 score_summary.best_cls,
                 score_summary.best_obj,
                 score_summary.best_fused,
                 association.best_score,
                 association.best_global_score,
                 association.best_iou,
                 known_result.count,
                 kKnownScoreThreshold,
                 kKnownAssociationIouThreshold);
        if (association.best_score >= kKnownScoreThreshold) {
            det.identity = FaceIdentity::kKnown;
            det.score = std::max(det.score, association.best_score);
            any_known = true;
        }
    }

    last_error_ = any_known
        ? "generic face detector active; 640x640 target model matched enrolled face"
        : "generic face detector active; 640x640 target model did not match enrolled face";
    return 1;
}

void StrangerFaceAdapterStub::Initialize(const std::string& model_path) {
    Release();
    model_path_ = model_path;
    if (model_path_.empty()) {
        return;
    }

    struct stat st = {};
    if (stat(model_path_.c_str(), &st) == 0) {
        LOG_INFO("stranger target model file exists path=%s size=%lld\n",
                 model_path_.c_str(),
                 static_cast<long long>(st.st_size));
    } else {
        LOG_ERROR("stranger target model stat failed path=%s errno=%d\n",
                  model_path_.c_str(),
                  errno);
        Release();
        return;
    }

    pipe_offline_ = GetAIPreprocessPipe();
    pipe_initialized_ = true;

    char* model_path_char = const_cast<char*>(model_path_.c_str());
    model_alloc_flag_ = SSNE_STATIC_ALLOC;
    model_id_ = ssne_loadmodel(model_path_char, model_alloc_flag_);
    if (model_id_ == 0) {
        LOG_WARN("stranger target static load failed, retry with dynamic alloc: %s (size=%lld)\n",
                 model_path_.c_str(),
                 static_cast<long long>(st.st_size));
        model_alloc_flag_ = SSNE_DYNAMIC_ALLOC;
        model_id_ = ssne_loadmodel(model_path_char, model_alloc_flag_);
        if (model_id_ == 0) {
            LOG_ERROR("stranger target model load failed in both static/dynamic modes: %s (size=%lld)\n",
                      model_path_.c_str(),
                      static_cast<long long>(st.st_size));
            Release();
            return;
        }
    }

    LOG_INFO("stranger target model loaded id=%u alloc=%s\n",
             static_cast<unsigned int>(model_id_),
             model_alloc_flag_ == SSNE_STATIC_ALLOC ? "static" : "dynamic");

    const int move_to_sram_ret = ssne_movemodeltosram(model_id_);
    if (move_to_sram_ret == 0) {
        LOG_INFO("stranger target model moved to SRAM for faster access\n");
    } else {
        LOG_WARN("stranger target ssne_movemodeltosram ret=%d, continue with current placement\n",
                 move_to_sram_ret);
    }

    const int input_num = ssne_get_model_input_num(model_id_);
    int dtype = -1;
    int dtype_ret = ssne_get_model_input_dtype(model_id_, &dtype);
    int mean[3] = {0, 0, 0};
    int std[3] = {0, 0, 0};
    int is_uint8 = 0;
    const int norm_ret =
        ssne_get_model_normalize_params(model_id_, mean, std, &is_uint8);
    model_is_uint8_ = is_uint8;
    LOG_INFO("stranger target model meta: input_num=%d input_dtype=%s(%d) dtype_ret=%d norm_ret=%d is_uint8=%d mean=[%d,%d,%d] std=[%d,%d,%d]\n",
             input_num,
             TensorTypeName(static_cast<uint8_t>(dtype)),
             dtype,
             dtype_ret,
             norm_ret,
             is_uint8,
             mean[0], mean[1], mean[2],
             std[0], std[1], std[2]);

#if STRANGER_USE_MODEL_NORMALIZE
    SetNormalize(pipe_offline_, model_id_);
#else
    LOG_WARN("stranger target normalize disabled: model expects uint8 RGB input scale=1.0\n");
#endif
    input_tensor_ = create_tensor(FACE_NET_W, FACE_NET_H, SSNE_RGB, SSNE_BUF_AI);
    input_created_ = true;
    if (!IsValidTensor(input_tensor_)) {
        LOG_ERROR("stranger target input tensor allocation failed\n");
        Release();
        return;
    }

    dtype = -1;
    ssne_get_model_input_dtype(model_id_, &dtype);
    model_input_dtype_ = dtype;
    set_data_type(input_tensor_, dtype);
    LOG_INFO("stranger target using runtime input tensor dtype=%s(%d); input_size=%dx%d normalize=%s normalize params is_uint8=%d is kept for diagnostics only\n",
             TensorTypeName(static_cast<uint8_t>(dtype)),
             dtype,
             FACE_NET_W,
             FACE_NET_H,
#if STRANGER_USE_MODEL_NORMALIZE
             "on",
#else
             "off",
#endif
             model_is_uint8_);

    initialized_ = true;
    LOG_INFO("stranger target adapter initialized model=%s input_dtype=%d\n",
             model_path_.c_str(),
             dtype);
}

void StrangerFaceAdapterStub::Release() {
    initialized_ = false;
    if (input_created_) {
        release_tensor(input_tensor_);
    }
    input_tensor_ = ssne_tensor_t{};
    input_created_ = false;

    for (int i = 0; i < FACE_OUTPUT_COUNT; ++i) {
        if (output_ready_[i]) {
            release_tensor(output_tensors_[i]);
        }
        output_tensors_[i] = ssne_tensor_t{};
        output_ready_[i] = false;
    }

    if (pipe_initialized_) {
        ReleaseAIPreprocessPipe(pipe_offline_);
    }
    pipe_offline_ = AiPreprocessPipe{};
    pipe_initialized_ = false;
    output_meta_logged_ = false;
    output_score_logged_ = false;
    input_debug_logged_ = false;
    model_alloc_flag_ = SSNE_STATIC_ALLOC;
    model_input_dtype_ = -1;
    model_is_uint8_ = 0;
    for (int i = 0; i < FACE_OUTPUT_COUNT; ++i) {
        chw_output_buffers_[i].clear();
    }
    model_id_ = 0;
    model_path_.clear();
    initialized_ = false;
}

bool StrangerFaceAdapterStub::IsInitialized() const {
    return initialized_;
}

int StrangerFaceAdapterStub::infer_yuv_roi(ssne_tensor_t* input,
                                           int left,
                                           int top,
                                           int size,
                                           FaceRawTensor outputs[FACE_OUTPUT_COUNT]) {
    if (!initialized_ || input == nullptr || outputs == nullptr || size <= 0) {
        return -1;
    }
    if (!input_created_ || !IsValidTensor(input_tensor_)) {
        return -2;
    }

    FaceRoi roi;
    roi.left = left;
    roi.top = top;
    roi.size = size;
    ssne_tensor_t roi_tensor = {};
    if (!CropYuv422SquareTensor(*input, roi, &roi_tensor)) {
        return -3;
    }

    const int preprocess_ret = RunAiPreprocessPipe(pipe_offline_, roi_tensor, input_tensor_);
    release_tensor(roi_tensor);
    if (preprocess_ret != 0) {
        LOG_ERROR("stranger target preprocess failed ret=%d\n", preprocess_ret);
        return -4;
    }
    if (!input_debug_logged_) {
        LogInputTensorPreview("stranger target input(after preprocess)", input_tensor_);
        input_debug_logged_ = true;
    }
    const int infer_ret = ssne_inference(model_id_, 1, &input_tensor_);
    if (infer_ret != 0) {
        LOG_ERROR("stranger target inference failed ret=%d\n", infer_ret);
        return -5;
    }

    const int getout_ret = ssne_getoutput(model_id_, FACE_OUTPUT_COUNT, output_tensors_);
    if (getout_ret != 0) {
        LOG_ERROR("stranger target getoutput failed ret=%d\n", getout_ret);
        return -6;
    }

    if (!output_meta_logged_) {
        LOG_INFO("stranger target output layout decode: raw=NHWC, repack=CHW, swap_cls_obj=%d, decode_levels=16,32, obj=hinge_prob\n",
                 STRANGER_SWAP_CLS_OBJ_OUTPUTS);
    }
    for (int i = 0; i < FACE_OUTPUT_COUNT; ++i) {
        output_ready_[i] = true;
        if (!output_meta_logged_) {
            LogTensorInfo("stranger target output", i, output_tensors_[i]);
        }
        if (get_data_type(output_tensors_[i]) != SSNE_FLOAT32) {
            LOG_ERROR("stranger target output[%d] dtype=%s is not float32\n",
                      i,
                      TensorTypeName(get_data_type(output_tensors_[i])));
            return -7;
        }

        RepackOutputToChw(output_tensors_[i], &chw_output_buffers_[i]);
        if (chw_output_buffers_[i].empty()) {
            LOG_ERROR("stranger target output[%d] repack failed\n", i);
            return -8;
        }

        outputs[i].data = chw_output_buffers_[i].data();
        outputs[i].type = FaceTensorType::kFloat32;
        outputs[i].scale = 1.0f;
        outputs[i].channels = GetTensorChannels(output_tensors_[i]);
        outputs[i].height = static_cast<int>(get_height(output_tensors_[i]));
        outputs[i].width = static_cast<int>(get_width(output_tensors_[i]));
        if (outputs[i].channels <= 0) {
            LOG_ERROR("stranger target output[%d] invalid channels=%d\n", i, outputs[i].channels);
            return -9;
        }
    }
    output_meta_logged_ = true;
    if (!output_score_logged_) {
        OutputBinding binding;
        LogOneChannelOutputCandidates(outputs, binding);
        output_score_logged_ = true;
    }
    if (input_debug_logged_) {
        for (int level = 0; level < 3; ++level) {
            LogScoreHeadRange(outputs[level], outputs[6 + level], level);
        }
        input_debug_logged_ = false;
    }
    return 0;
}

int StrangerFaceAdapterStub::infer_rgb224(
    const uint8_t rgb224[FACE_NET_W * FACE_NET_H * 3],
    FaceRawTensor outputs[FACE_OUTPUT_COUNT]) {
    (void)rgb224;
    for (int i = 0; i < FACE_OUTPUT_COUNT; ++i) {
        outputs[i] = FaceRawTensor{};
    }
    return -1;
}
