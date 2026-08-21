#include "../include/companion_mode.hpp"
#include "../include/log.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#ifndef GESTURE_INPUT_FORMAT
#define GESTURE_INPUT_FORMAT SSNE_RGB
#endif

#ifndef GESTURE_INPUT_FORMAT_NAME
#define GESTURE_INPUT_FORMAT_NAME "RGB"
#endif

namespace {

struct CropRoi {
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;

    int Width() const { return right - left; }
    int Height() const { return bottom - top; }
    bool IsValid() const { return Width() > 0 && Height() > 0; }
};

struct TensorDebugSummary {
    bool valid = false;
    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t format = SSNE_RGB;
    uint8_t dtype = SSNE_UINT8;
    size_t channels = 3;
    std::array<double, 3> sum = {0.0, 0.0, 0.0};
    std::array<float, 3> minv = {std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max()};
    std::array<float, 3> maxv = {std::numeric_limits<float>::lowest(),
                                 std::numeric_limits<float>::lowest(),
                                 std::numeric_limits<float>::lowest()};
    std::array<std::array<float, 3>, 5> samples = {};
};

const char* SsneDataTypeName(int dtype);
const char* SsneFormatName(uint8_t format);

bool IsValidTensor(ssne_tensor_t tensor) {
    return get_data(tensor) != nullptr &&
           get_width(tensor) > 0 &&
           get_height(tensor) > 0 &&
           get_mem_size(tensor) > 0;
}

bool GestureTensorDebugEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("GESTURE_DEBUG_TENSOR");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool IsFloatLikeDType(uint8_t dtype) {
    return dtype == SSNE_FLOAT32 || dtype == SSNE_INT8 || dtype == SSNE_UINT8;
}

bool ReadTensorPixelAsRgb(ssne_tensor_t tensor, uint32_t x, uint32_t y, std::array<float, 3>* rgb) {
    if (rgb == nullptr || !IsValidTensor(tensor)) {
        return false;
    }

    const uint32_t width = get_width(tensor);
    const uint32_t height = get_height(tensor);
    if (x >= width || y >= height) {
        return false;
    }

    const uint8_t format = get_data_format(tensor);
    const uint8_t dtype = get_data_type(tensor);
    const size_t pixel_index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
    const uint8_t* base = static_cast<const uint8_t*>(get_data(tensor));
    if (base == nullptr) {
        return false;
    }

    if (dtype == SSNE_FLOAT32) {
        const float* ptr = reinterpret_cast<const float*>(base);
        const size_t offset = pixel_index * 3U;
        if (get_total_size(tensor) < offset + 3U) {
            return false;
        }
        std::array<float, 3> values = {
            ptr[offset + 0U],
            ptr[offset + 1U],
            ptr[offset + 2U],
        };
        if (format == SSNE_BGR) {
            std::swap(values[0], values[2]);
        }
        *rgb = values;
        return true;
    }

    if (dtype == SSNE_INT8) {
        const int8_t* ptr = reinterpret_cast<const int8_t*>(base);
        const size_t offset = pixel_index * 3U;
        if (get_total_size(tensor) < offset + 3U) {
            return false;
        }
        std::array<float, 3> values = {
            static_cast<float>(static_cast<int>(ptr[offset + 0U]) + 128),
            static_cast<float>(static_cast<int>(ptr[offset + 1U]) + 128),
            static_cast<float>(static_cast<int>(ptr[offset + 2U]) + 128),
        };
        if (format == SSNE_BGR) {
            std::swap(values[0], values[2]);
        }
        *rgb = values;
        return true;
    }

    if (dtype == SSNE_UINT8) {
        const uint8_t* ptr = base;
        const size_t offset = pixel_index * 3U;
        if (get_total_size(tensor) < offset + 3U) {
            return false;
        }
        std::array<float, 3> values = {
            static_cast<float>(ptr[offset + 0U]),
            static_cast<float>(ptr[offset + 1U]),
            static_cast<float>(ptr[offset + 2U]),
        };
        if (format == SSNE_BGR) {
            std::swap(values[0], values[2]);
        }
        *rgb = values;
        return true;
    }

    return false;
}

TensorDebugSummary SummarizeTensorPixels(ssne_tensor_t tensor) {
    TensorDebugSummary summary;
    if (!IsValidTensor(tensor)) {
        return summary;
    }

    summary.width = get_width(tensor);
    summary.height = get_height(tensor);
    summary.format = get_data_format(tensor);
    summary.dtype = get_data_type(tensor);
    summary.valid = summary.width > 0 && summary.height > 0 && IsFloatLikeDType(summary.dtype);
    if (!summary.valid) {
        return summary;
    }

    const std::array<std::pair<uint32_t, uint32_t>, 5> sample_points = {
        std::make_pair(0U, 0U),
        std::make_pair(summary.width > 0 ? summary.width - 1U : 0U, 0U),
        std::make_pair(0U, summary.height > 0 ? summary.height - 1U : 0U),
        std::make_pair(summary.width > 0 ? summary.width - 1U : 0U,
                       summary.height > 0 ? summary.height - 1U : 0U),
        std::make_pair(summary.width / 2U, summary.height / 2U),
    };

    for (uint32_t y = 0; y < summary.height; ++y) {
        for (uint32_t x = 0; x < summary.width; ++x) {
            std::array<float, 3> rgb = {0.0f, 0.0f, 0.0f};
            if (!ReadTensorPixelAsRgb(tensor, x, y, &rgb)) {
                summary.valid = false;
                return summary;
            }
            for (int c = 0; c < 3; ++c) {
                summary.sum[static_cast<size_t>(c)] += static_cast<double>(rgb[static_cast<size_t>(c)]);
                summary.minv[static_cast<size_t>(c)] =
                    std::min(summary.minv[static_cast<size_t>(c)], rgb[static_cast<size_t>(c)]);
                summary.maxv[static_cast<size_t>(c)] =
                    std::max(summary.maxv[static_cast<size_t>(c)], rgb[static_cast<size_t>(c)]);
            }
        }
    }

    for (size_t i = 0; i < sample_points.size(); ++i) {
        const auto& pt = sample_points[i];
        ReadTensorPixelAsRgb(tensor, pt.first, pt.second, &summary.samples[i]);
    }

    summary.channels = 3;
    return summary;
}

void LogTensorPixelSummary(const char* prefix, const TensorDebugSummary& summary) {
    if (!summary.valid) {
        LOG_WARN("%s tensor pixel summary unavailable\n", prefix != nullptr ? prefix : "tensor");
        return;
    }

    const double pixel_count = static_cast<double>(summary.width) * static_cast<double>(summary.height);
    LOG_INFO("%s pixel summary: w=%u h=%u fmt=%s dtype=%s mean=[%.3f %.3f %.3f] min=[%.3f %.3f %.3f] max=[%.3f %.3f %.3f] tl=[%.1f %.1f %.1f] tr=[%.1f %.1f %.1f] bl=[%.1f %.1f %.1f] br=[%.1f %.1f %.1f] center=[%.1f %.1f %.1f]\n",
             prefix != nullptr ? prefix : "tensor",
             summary.width,
             summary.height,
             SsneFormatName(summary.format),
             SsneDataTypeName(summary.dtype),
             summary.sum[0] / pixel_count,
             summary.sum[1] / pixel_count,
             summary.sum[2] / pixel_count,
             summary.minv[0],
             summary.minv[1],
             summary.minv[2],
             summary.maxv[0],
             summary.maxv[1],
             summary.maxv[2],
             summary.samples[0][0], summary.samples[0][1], summary.samples[0][2],
             summary.samples[1][0], summary.samples[1][1], summary.samples[1][2],
             summary.samples[2][0], summary.samples[2][1], summary.samples[2][2],
             summary.samples[3][0], summary.samples[3][1], summary.samples[3][2],
             summary.samples[4][0], summary.samples[4][1], summary.samples[4][2]);
}

const char* SsneDataTypeName(int dtype) {
    switch (dtype) {
        case SSNE_FLOAT32:
            return "FLOAT32";
        case SSNE_INT8:
            return "INT8";
        case SSNE_UINT8:
            return "UINT8";
        default:
            return "UNKNOWN";
    }
}

const char* SsneFormatName(uint8_t format) {
    switch (format) {
        case SSNE_RGB:
            return "RGB";
        case SSNE_BGR:
            return "BGR";
        case SSNE_YUV422_16:
            return "YUV422_16";
        default:
            return "UNKNOWN";
    }
}

void LogTensorSummary(const char* prefix, ssne_tensor_t tensor) {
    if (prefix == nullptr) {
        prefix = "tensor";
    }

    if (get_data(tensor) == nullptr) {
        LOG_WARN("%s tensor is null\n", prefix);
        return;
    }

    LOG_INFO("%s tensor: w=%u h=%u total=%u mem=%zu dtype=%s(%d) format=%s(%u) data=%p\n",
             prefix,
             get_width(tensor),
             get_height(tensor),
             get_total_size(tensor),
             get_mem_size(tensor),
             SsneDataTypeName(get_data_type(tensor)),
             static_cast<int>(get_data_type(tensor)),
             SsneFormatName(get_data_format(tensor)),
             static_cast<unsigned int>(get_data_format(tensor)),
             get_data(tensor));
}

uint32_t TensorFnv1a32(ssne_tensor_t tensor) {
    const uint8_t* data = static_cast<const uint8_t*>(get_data(tensor));
    const size_t size = get_mem_size(tensor);
    if (data == nullptr || size == 0U) {
        return 0U;
    }

    uint32_t hash = 2166136261U;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint32_t>(data[i]);
        hash *= 16777619U;
    }
    return hash;
}

void LogTensorFingerprint(const char* prefix, ssne_tensor_t tensor) {
    const uint8_t* data = static_cast<const uint8_t*>(get_data(tensor));
    const size_t size = get_mem_size(tensor);
    if (data == nullptr || size == 0U) {
        LOG_WARN("%s fingerprint unavailable\n", prefix != nullptr ? prefix : "tensor");
        return;
    }

    const uint8_t b0 = size > 0U ? data[0] : 0U;
    const uint8_t b1 = size > 1U ? data[1] : 0U;
    const uint8_t b2 = size > 2U ? data[2] : 0U;
    const uint8_t b3 = size > 3U ? data[3] : 0U;
    const uint8_t b4 = size > 4U ? data[4] : 0U;
    const uint8_t b5 = size > 5U ? data[5] : 0U;
    const uint8_t b6 = size > 6U ? data[6] : 0U;
    const uint8_t b7 = size > 7U ? data[7] : 0U;
    LOG_INFO("%s fingerprint: fnv32=0x%08x mem=%zu first8=[%u %u %u %u %u %u %u %u]\n",
             prefix != nullptr ? prefix : "tensor",
             TensorFnv1a32(tensor),
             size,
             static_cast<unsigned int>(b0),
             static_cast<unsigned int>(b1),
             static_cast<unsigned int>(b2),
             static_cast<unsigned int>(b3),
             static_cast<unsigned int>(b4),
             static_cast<unsigned int>(b5),
             static_cast<unsigned int>(b6),
             static_cast<unsigned int>(b7));
}

void LogGestureOutputPreview(ssne_tensor_t tensor) {
    if (!IsValidTensor(tensor) || get_total_size(tensor) != 5U) {
        LOG_WARN("gesture output preview unavailable: invalid tensor or total=%u\n",
                 get_total_size(tensor));
        return;
    }

    const uint8_t dtype = get_data_type(tensor);
    const void* data = get_data(tensor);
    const uint32_t total = get_total_size(tensor);
    if (data == nullptr) {
        LOG_WARN("gesture output preview unavailable: null data\n");
        return;
    }

    if (dtype == SSNE_FLOAT32) {
        const float* ptr = reinterpret_cast<const float*>(data);
        LOG_INFO("gesture output preview: decode=float32 total=%u order=[down,left,right,up,none] logits5=[%.6f %.6f %.6f %.6f %.6f]\n",
                 total,
                 ptr[0],
                 ptr[1],
                 ptr[2],
                 ptr[3],
                 ptr[4]);
        return;
    }

    if (dtype == SSNE_INT8) {
        const int8_t* ptr = reinterpret_cast<const int8_t*>(data);
        LOG_INFO("gesture output preview: decode=int8_raw total=%u order=[down,left,right,up,none] logits5=[%d %d %d %d %d] note=check SDK quant scale if this is not float32\n",
                 total,
                 static_cast<int>(ptr[0]),
                 static_cast<int>(ptr[1]),
                 static_cast<int>(ptr[2]),
                 static_cast<int>(ptr[3]),
                 static_cast<int>(ptr[4]));
        return;
    }

    if (dtype == SSNE_UINT8) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);
        LOG_INFO("gesture output preview: decode=uint8_raw total=%u order=[down,left,right,up,none] logits5=[%u %u %u %u %u] note=check SDK quant scale/zero point if this is not float32\n",
                 total,
                 static_cast<unsigned int>(ptr[0]),
                 static_cast<unsigned int>(ptr[1]),
                 static_cast<unsigned int>(ptr[2]),
                 static_cast<unsigned int>(ptr[3]),
                 static_cast<unsigned int>(ptr[4]));
        return;
    }

    LOG_WARN("gesture output preview unsupported dtype=%s(%d)\n",
             SsneDataTypeName(dtype),
             static_cast<int>(dtype));
}

void ReleaseOutputTensors(ssne_tensor_t* outputs, int count) {
    if (outputs == nullptr) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        release_tensor(outputs[i]);
        outputs[i] = ssne_tensor_t{};
    }
}

CropRoi BuildGestureRoi(const std::array<float, 4>* focus_box,
                        const std::array<int, 2>& img_shape) {
    CropRoi roi;
    const int img_w = img_shape[0];
    const int img_h = img_shape[1];

    if (focus_box == nullptr) {
        const int side = std::min(img_w, img_h);
        roi.left = std::max(0, (img_w - side) / 2);
        roi.top = std::max(0, (img_h - side) / 2);
        roi.right = std::min(img_w, roi.left + side);
        roi.bottom = std::min(img_h, roi.top + side);
        return roi;
    }

    const float x1 = std::max(0.0f, std::min((*focus_box)[0], static_cast<float>(img_w)));
    const float y1 = std::max(0.0f, std::min((*focus_box)[1], static_cast<float>(img_h)));
    const float x2 = std::max(0.0f, std::min((*focus_box)[2], static_cast<float>(img_w)));
    const float y2 = std::max(0.0f, std::min((*focus_box)[3], static_cast<float>(img_h)));

    float left = x1;
    float top = y1;
    float right = x2;
    float bottom = y2;

    if (left < 0.0f) {
        right -= left;
        left = 0.0f;
    }
    if (top < 0.0f) {
        bottom -= top;
        top = 0.0f;
    }
    if (right > static_cast<float>(img_w)) {
        const float overflow = right - static_cast<float>(img_w);
        left = std::max(0.0f, left - overflow);
        right = static_cast<float>(img_w);
    }
    if (bottom > static_cast<float>(img_h)) {
        const float overflow = bottom - static_cast<float>(img_h);
        top = std::max(0.0f, top - overflow);
        bottom = static_cast<float>(img_h);
    }

    roi.left = std::max(0, static_cast<int>(std::floor(left)));
    roi.top = std::max(0, static_cast<int>(std::floor(top)));
    roi.right = std::min(img_w, static_cast<int>(std::ceil(right)));
    roi.bottom = std::min(img_h, static_cast<int>(std::ceil(bottom)));

    roi.left &= ~1;
    roi.right &= ~1;
    roi.top = std::max(0, roi.top);
    roi.bottom = std::min(img_h, roi.bottom);
    if (roi.right <= roi.left) {
        roi.right = std::min(img_w, roi.left + 8);
    }

    int roi_width = roi.Width();
    if (roi_width >= 8) {
        const int aligned_width = roi_width & ~7;
        if (aligned_width > 0 && aligned_width != roi_width) {
            const int trim = roi_width - aligned_width;
            roi.left += trim / 2;
            roi.left &= ~1;
            roi.right = roi.left + aligned_width;
            if (roi.right > img_w) {
                roi.right = img_w & ~1;
                roi.left = std::max(0, roi.right - aligned_width);
                roi.left &= ~1;
                roi.right = roi.left + aligned_width;
            }
        }
    }
    return roi;
}

bool CropYuv422Tensor(ssne_tensor_t input, const CropRoi& roi, ssne_tensor_t* cropped) {
    if (cropped == nullptr || !roi.IsValid()) {
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
        roi.right > static_cast<int>(src_w) ||
        roi.bottom > static_cast<int>(src_h)) {
        return false;
    }

    ssne_tensor_t roi_tensor = create_tensor(static_cast<uint32_t>(roi.Width()),
                                             static_cast<uint32_t>(roi.Height()),
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
    const size_t dst_stride = get_mem_size(roi_tensor) / static_cast<size_t>(roi.Height());
    const size_t row_bytes = static_cast<size_t>(roi.Width()) * 2U;

    for (int y = 0; y < roi.Height(); ++y) {
        const uint8_t* src_row =
            src + static_cast<size_t>(roi.top + y) * src_stride + static_cast<size_t>(roi.left) * 2U;
        uint8_t* dst_row = dst + static_cast<size_t>(y) * dst_stride;
        std::memcpy(dst_row, src_row, row_bytes);
    }

    *cropped = roi_tensor;
    return true;
}

bool CopyOutputToFloatArray(ssne_tensor_t tensor, std::array<float, 5>* values) {
    if (values == nullptr || !IsValidTensor(tensor)) {
        return false;
    }

    if (get_total_size(tensor) != 5U) {
        return false;
    }

    const uint8_t dtype = get_data_type(tensor);
    if (dtype == SSNE_FLOAT32) {
        const float* ptr = reinterpret_cast<const float*>(get_data(tensor));
        if (ptr == nullptr) {
            return false;
        }
        for (int i = 0; i < 5; ++i) {
            (*values)[static_cast<size_t>(i)] = ptr[i];
        }
        return true;
    }

    if (dtype == SSNE_INT8) {
        const int8_t* ptr = reinterpret_cast<const int8_t*>(get_data(tensor));
        if (ptr == nullptr) {
            return false;
        }
        for (int i = 0; i < 5; ++i) {
            (*values)[static_cast<size_t>(i)] = static_cast<float>(ptr[i]);
        }
        return true;
    }

    if (dtype == SSNE_UINT8) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(get_data(tensor));
        if (ptr == nullptr) {
            return false;
        }
        for (int i = 0; i < 5; ++i) {
            (*values)[static_cast<size_t>(i)] = static_cast<float>(ptr[i]);
        }
        return true;
    }

    return false;
}

bool DumpInputTensorToPpm(ssne_tensor_t tensor, const char* path) {
    if (path == nullptr || !IsValidTensor(tensor)) {
        return false;
    }
    const uint32_t width = get_width(tensor);
    const uint32_t height = get_height(tensor);
    const uint8_t format = get_data_format(tensor);
    const uint8_t dtype = get_data_type(tensor);
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t expected_elements = pixel_count * 3U;
    const size_t expected_bytes =
        (dtype == SSNE_FLOAT32) ? expected_elements * sizeof(float) : expected_elements;
    if ((format != SSNE_RGB && format != SSNE_BGR) ||
        get_data(tensor) == nullptr ||
        get_mem_size(tensor) < expected_bytes) {
        return false;
    }

    FILE* fp = std::fopen(path, "wb");
    if (fp == nullptr) {
        return false;
    }

    std::fprintf(fp, "P6\n%u %u\n255\n", width, height);
    std::vector<uint8_t> rgb8(expected_elements, 0U);

    float min_value = std::numeric_limits<float>::max();
    float max_value = std::numeric_limits<float>::lowest();
    if (dtype == SSNE_FLOAT32) {
        const float* src = reinterpret_cast<const float*>(get_data(tensor));
        for (size_t i = 0; i < expected_elements; ++i) {
            min_value = std::min(min_value, src[i]);
            max_value = std::max(max_value, src[i]);
        }
        const float scale = (max_value > min_value) ? (255.0f / (max_value - min_value)) : 0.0f;
        for (size_t i = 0; i < expected_elements; ++i) {
            const float normalized = (scale > 0.0f) ? ((src[i] - min_value) * scale) : 128.0f;
            const float clamped = std::max(0.0f, std::min(normalized, 255.0f));
            rgb8[i] = static_cast<uint8_t>(std::lround(clamped));
        }
    } else if (dtype == SSNE_INT8) {
        const int8_t* src = reinterpret_cast<const int8_t*>(get_data(tensor));
        for (size_t i = 0; i < expected_elements; ++i) {
            const int value = static_cast<int>(src[i]) + 128;
            rgb8[i] = static_cast<uint8_t>(std::max(0, std::min(value, 255)));
        }
    } else {
        const uint8_t* src = static_cast<const uint8_t*>(get_data(tensor));
        std::memcpy(rgb8.data(), src, expected_elements);
    }

    if (format == SSNE_BGR) {
        for (size_t i = 0; i < expected_elements; i += 3U) {
            std::swap(rgb8[i], rgb8[i + 2U]);
        }
    }
    std::fwrite(rgb8.data(), 1, rgb8.size(), fp);

    std::fclose(fp);
    return true;
}

float Sigmoid(float value) {
    if (value >= 0.0f) {
        const float exp_neg = std::exp(-value);
        return 1.0f / (1.0f + exp_neg);
    }
    const float exp_pos = std::exp(value);
    return exp_pos / (1.0f + exp_pos);
}

GestureCommand GestureClassIndexToCommand(int class_index) {
    // Model output order is [down, left, right, up, none].
    switch (class_index) {
        case 0:
            return GestureCommand::TD;
        case 1:
            return GestureCommand::TL;
        case 2:
            return GestureCommand::TR;
        case 3:
            return GestureCommand::TU;
        case 4:
            return GestureCommand::NONE;
        default:
            return GestureCommand::NONE;
    }
}

}  // namespace

const char* GestureCommandName(GestureCommand command) {
    switch (command) {
        case GestureCommand::TU:
            return "TU";
        case GestureCommand::TD:
            return "TD";
        case GestureCommand::TL:
            return "TL";
        case GestureCommand::TR:
            return "TR";
        default:
            return "NONE";
    }
}

const char* SnakeDirectionName(SnakeDirection direction) {
    switch (direction) {
        case SnakeDirection::UP:
            return "UP";
        case SnakeDirection::DOWN:
            return "DOWN";
        case SnakeDirection::LEFT:
            return "LEFT";
        case SnakeDirection::RIGHT:
            return "RIGHT";
        default:
            return "UNKNOWN";
    }
}

SnakeDirection GestureToSnakeDirection(GestureCommand command, SnakeDirection fallback) {
    switch (command) {
        case GestureCommand::TU:
            return SnakeDirection::UP;
        case GestureCommand::TD:
            return SnakeDirection::DOWN;
        case GestureCommand::TL:
            return SnakeDirection::LEFT;
        case GestureCommand::TR:
            return SnakeDirection::RIGHT;
        default:
            return fallback;
    }
}

GestureCommand SnakeDirectionToGesture(SnakeDirection direction) {
    switch (direction) {
        case SnakeDirection::UP:
            return GestureCommand::TU;
        case SnakeDirection::DOWN:
            return GestureCommand::TD;
        case SnakeDirection::LEFT:
            return GestureCommand::TL;
        case SnakeDirection::RIGHT:
            return GestureCommand::TR;
        default:
            return GestureCommand::NONE;
    }
}

void GestureClassifier::Initialize(std::string& model_path,
                                   std::array<int, 2>* in_img_shape,
                                   std::array<int, 2>* in_det_shape,
                                   bool use_normalize,
                                   uint8_t in_input_format) {
    initialized = false;
    img_shape = *in_img_shape;
    det_shape = *in_det_shape;
    normalize_enabled = use_normalize;
    input_format = in_input_format;
    w_scale = static_cast<float>(img_shape[0]) / static_cast<float>(det_shape[0]);
    h_scale = static_cast<float>(img_shape[1]) / static_cast<float>(det_shape[1]);

    pipe_offline = GetAIPreprocessPipe();

    char* model_path_char = const_cast<char*>(model_path.c_str());
    model_id = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);
    LOG_INFO("gesture model load: path=%s model_id=%u\n",
             model_path.c_str(),
             static_cast<unsigned int>(model_id));
    const int move_to_sram_ret = ssne_movemodeltosram(model_id);
    if (move_to_sram_ret == 0) {
        LOG_INFO("gesture model moved to SRAM for faster access\n");
    } else {
        LOG_WARN("gesture ssne_movemodeltosram ret=%d, continue with default placement\n",
                 move_to_sram_ret);
    }
    if (normalize_enabled) {
        SetNormalize(pipe_offline, model_id);
    } else {
        LOG_WARN("gesture SetNormalize disabled: use raw SDK resize/color conversion path for mobilenet preprocess check\n");
    }

    const uint32_t det_width = static_cast<uint32_t>(det_shape[0]);
    const uint32_t det_height = static_cast<uint32_t>(det_shape[1]);
    inputs[0] = create_tensor(det_width, det_height, input_format, SSNE_BUF_AI);
    if (!IsValidTensor(inputs[0])) {
        LOG_ERROR("gesture input tensor allocation failed for [%u x %u], mem=%zu\n",
                  det_width, det_height, get_mem_size(inputs[0]));
        ReleaseAIPreprocessPipe(pipe_offline);
        pipe_offline = AiPreprocessPipe{};
        model_id = 0;
        return;
    }

    int dtype = -1;
    ssne_get_model_input_dtype(model_id, &dtype);
    set_data_type(inputs[0], dtype);

    LOG_INFO("gesture initialized\n");
    LOG_INFO("gesture expected input nchw=[1,3,%u,%u], tensor=[w=%u,h=%u]\n",
             det_height,
             det_width,
             det_width,
             det_height);
    LOG_INFO("model=%s crop=[%d,%d] det=[%d,%d] input=[%u,%u] format=%s normalize=%s input_dtype=%s(%d)\n",
             model_path.c_str(),
             img_shape[0],
             img_shape[1],
             det_shape[0],
             det_shape[1],
             det_width,
             det_height,
             SsneFormatName(input_format),
             normalize_enabled ? "on" : "off",
             SsneDataTypeName(dtype),
             dtype);
    LogTensorSummary("gesture input", inputs[0]);
    initialized = true;
}

void GestureClassifier::SetFocusBox(const std::array<float, 4>* in_focus_box) {
    if (in_focus_box == nullptr) {
        focus_valid = false;
        focus_box = {0.0f, 0.0f, 0.0f, 0.0f};
        return;
    }

    focus_box = *in_focus_box;
    focus_box[0] = std::max(0.0f, std::min(focus_box[0], static_cast<float>(img_shape[0])));
    focus_box[1] = std::max(0.0f, std::min(focus_box[1], static_cast<float>(img_shape[1])));
    focus_box[2] = std::max(0.0f, std::min(focus_box[2], static_cast<float>(img_shape[0])));
    focus_box[3] = std::max(0.0f, std::min(focus_box[3], static_cast<float>(img_shape[1])));
    focus_valid = focus_box[2] > focus_box[0] && focus_box[3] > focus_box[1];
}

void GestureClassifier::Predict(ssne_tensor_t* img, GestureResult* result, float conf_threshold) {
    (void)conf_threshold;
    if (result == nullptr) {
        LOG_ERROR("gesture predict got null result pointer\n");
        return;
    }

    *result = GestureResult{};
    if (img == nullptr || !IsValidTensor(*img)) {
        LOG_ERROR("gesture predict got invalid input tensor\n");
        return;
    }
    if (!IsValidTensor(inputs[0])) {
        LOG_ERROR("gesture input tensor is invalid, skip this frame\n");
        return;
    }

    static bool logged_runtime_input = false;
    if (!logged_runtime_input) {
        LogTensorSummary("gesture source", *img);
        LogTensorSummary("gesture model input(before preprocess)", inputs[0]);
        logged_runtime_input = true;
    }

    static bool logged_roi_info = false;
    if (!logged_roi_info) {
        LOG_INFO("gesture roi: manual_crop=1 official_preprocess=1 focus_valid=%d configured_img=[%d,%d] runtime_img=[%u,%u] box_crop=[%.1f,%.1f,%.1f,%.1f] note=manual ROI crop then RunAiPreprocessPipe resize to model input\n",
                 focus_valid ? 1 : 0,
                 img_shape[0],
                 img_shape[1],
                 get_width(*img),
                 get_height(*img),
                 focus_box[0],
                 focus_box[1],
                 focus_box[2],
                 focus_box[3]);
        logged_roi_info = true;
    }

    // Crop the selected source ROI first. The official preprocessing pipe then
    // resizes that ROI to the model's required 256x256 RGB tensor.
    const CropRoi roi = BuildGestureRoi(focus_valid ? &focus_box : nullptr, img_shape);
    ssne_tensor_t preprocess_source = *img;
    ssne_tensor_t cropped_source = ssne_tensor_t{};
    bool owns_cropped_source = false;
    if (roi.IsValid() &&
        (roi.left != 0 || roi.top != 0 ||
         roi.right != static_cast<int>(get_width(*img)) ||
         roi.bottom != static_cast<int>(get_height(*img)))) {
        if (!CropYuv422Tensor(*img, roi, &cropped_source)) {
            LOG_ERROR("gesture ROI crop failed: roi=[l=%d t=%d r=%d b=%d] source=[%u,%u] format=%s\n",
                      roi.left, roi.top, roi.right, roi.bottom,
                      get_width(*img), get_height(*img),
                      SsneFormatName(get_data_format(*img)));
            return;
        }
        preprocess_source = cropped_source;
        owns_cropped_source = true;
    }

    static bool logged_preprocess_roi = false;
    static CropRoi last_preprocess_roi;
    const bool roi_changed =
        !logged_preprocess_roi ||
        last_preprocess_roi.left != roi.left ||
        last_preprocess_roi.top != roi.top ||
        last_preprocess_roi.right != roi.right ||
        last_preprocess_roi.bottom != roi.bottom;
    if (roi_changed) {
        LOG_INFO("gesture preprocess ROI: focus_valid=%d roi=[l=%d t=%d r=%d b=%d w=%d h=%d] source=[%u,%u]\n",
                 focus_valid ? 1 : 0,
                 roi.left, roi.top, roi.right, roi.bottom,
                 roi.Width(), roi.Height(),
                 get_width(preprocess_source), get_height(preprocess_source));
        last_preprocess_roi = roi;
        logged_preprocess_roi = true;
    }

    const int preprocess_ret = RunAiPreprocessPipe(pipe_offline, preprocess_source, inputs[0]);
    if (owns_cropped_source) {
        release_tensor(cropped_source);
    }
    static bool logged_preprocess_output = false;
    if (!logged_preprocess_output) {
        LOG_INFO("gesture preprocess ret=%d\n", preprocess_ret);
        LogTensorSummary("gesture model input(after preprocess)", inputs[0]);
        logged_preprocess_output = true;
    }
    if (preprocess_ret != 0) {
        LOG_ERROR("gesture preprocess failed, ret=%d\n", preprocess_ret);
        return;
    }

    int dtype = -1;
    ssne_get_model_input_dtype(model_id, &dtype);
    set_data_type(inputs[0], dtype);

    static int preprocessed_frame_count = 0;
    static int dumped_input_frames = 0;
    const bool dump_numbered_snapshot = dumped_input_frames < 5;
    const bool refresh_latest_snapshot = (preprocessed_frame_count % 45) == 0;
    if (dump_numbered_snapshot || refresh_latest_snapshot) {
        LogTensorFingerprint("gesture model input(after preprocess)", inputs[0]);
        if (GestureTensorDebugEnabled()) {
            LogTensorPixelSummary("gesture model input(after preprocess)", SummarizeTensorPixels(inputs[0]));
        }
        char dump_path[64] = {};
        std::snprintf(dump_path,
                      sizeof(dump_path),
                      "/tmp/gesture_input_%03d.ppm",
                      dumped_input_frames);
        const bool latest_ok = DumpInputTensorToPpm(inputs[0], "/tmp/gesture_input_latest.ppm");
        const bool numbered_ok =
            dump_numbered_snapshot ? DumpInputTensorToPpm(inputs[0], dump_path) : true;
        if (latest_ok && numbered_ok) {
            LOG_INFO("gesture preprocessed input dumped: %s and /tmp/gesture_input_latest.ppm\n",
                     dump_numbered_snapshot ? dump_path : "periodic");
        } else {
            LOG_WARN("gesture preprocessed input dump failed: %s\n", dump_path);
        }
        if (dump_numbered_snapshot) {
            ++dumped_input_frames;
        }
    }
    ++preprocessed_frame_count;

    int ret = ssne_inference(model_id, 1, inputs);
    if (ret != 0) {
        LOG_ERROR("gesture inference context: model_id=%u input_dtype=%s(%d) input_format=%s(%u) input_mem=%zu input_total=%u\n",
                  static_cast<unsigned int>(model_id),
                  SsneDataTypeName(get_data_type(inputs[0])),
                  static_cast<int>(get_data_type(inputs[0])),
                  SsneFormatName(get_data_format(inputs[0])),
                  static_cast<unsigned int>(get_data_format(inputs[0])),
                  get_mem_size(inputs[0]),
                  get_total_size(inputs[0]));
        LOG_ERROR("gesture inference failed, ret=%d\n", ret);
        return;
    }

    ret = ssne_getoutput(model_id, 1, outputs);
    if (ret != 0) {
        LOG_ERROR("gesture getoutput failed, ret=%d output_count=1\n", ret);
        return;
    }

    static bool logged_output_tensor = false;
    if (!logged_output_tensor) {
        LogTensorSummary("gesture output[0]", outputs[0]);
        LogGestureOutputPreview(outputs[0]);
        logged_output_tensor = true;
    }

    std::array<float, 5> logits = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (!CopyOutputToFloatArray(outputs[0], &logits)) {
        LOG_ERROR("gesture output decode failed: expected exactly 5 values [down,left,right,up,none], actual_total=%u\n",
                  get_total_size(outputs[0]));
        return;
    }

    std::array<float, 5> scores = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 5; ++i) {
        scores[static_cast<size_t>(i)] =
            Sigmoid(logits[static_cast<size_t>(i)]);
    }

    int best_index = 0;
    for (int i = 1; i < 5; ++i) {
        if (scores[static_cast<size_t>(i)] > scores[static_cast<size_t>(best_index)]) {
            best_index = i;
        }
    }

    result->logits = logits;
    result->probabilities = scores;
    result->confidence = scores[static_cast<size_t>(best_index)];
    if (best_index == 4 || result->confidence < conf_threshold) {
        result->command = GestureCommand::NONE;
        result->valid = false;
        return;
    }

    result->valid = true;
    result->command = GestureClassIndexToCommand(best_index);
}

void GestureClassifier::Release() {
    initialized = false;
    release_tensor(inputs[0]);
    inputs[0] = ssne_tensor_t{};
    ReleaseOutputTensors(outputs, 1);
    ReleaseAIPreprocessPipe(pipe_offline);
}

GestureTemporalFilter::GestureTemporalFilter(int history_size_in,
                                             int required_hits_in,
                                             int emit_cooldown_frames_in)
    : history_size(std::max(1, history_size_in)),
      required_hits(std::max(1, required_hits_in)),
      emit_cooldown_frames(std::max(0, emit_cooldown_frames_in)) {}

GestureCommand GestureTemporalFilter::Push(const GestureResult& result) {
    if (cooldown_left > 0) {
        --cooldown_left;
    }

    const GestureCommand current = result.valid ? result.command : GestureCommand::NONE;
    history.push_back(current);
    while (static_cast<int>(history.size()) > history_size) {
        history.pop_front();
    }

    if (current == GestureCommand::NONE) {
        return GestureCommand::NONE;
    }

    int hits = 0;
    for (std::deque<GestureCommand>::const_reverse_iterator it = history.rbegin();
         it != history.rend(); ++it) {
        if (*it == current) {
            ++hits;
        }
    }

    if (hits < required_hits) {
        return GestureCommand::NONE;
    }
    if (cooldown_left > 0) {
        return GestureCommand::NONE;
    }

    last_emitted = current;
    cooldown_left = emit_cooldown_frames;
    return current;
}

void GestureTemporalFilter::Reset() {
    history.clear();
    cooldown_left = 0;
    last_emitted = GestureCommand::NONE;
}

SnakeGame::SnakeGame() : rng(static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count())) {}

void SnakeGame::Initialize(int cols_in, int rows_in, uint32_t seed) {
    cols = std::max(6, cols_in);
    rows = std::max(6, rows_in);
    if (seed != 0U) {
        rng.seed(seed);
    }
    initialized = true;
    Reset();
}

void SnakeGame::Reset() {
    if (!initialized) {
        return;
    }

    body.clear();
    const int center_x = cols / 2;
    const int center_y = rows / 2;
    body.push_back(SnakeCell{center_x, center_y});
    body.push_back(SnakeCell{center_x - 1, center_y});
    body.push_back(SnakeCell{center_x - 2, center_y});
    direction = SnakeDirection::RIGHT;
    pending_direction = SnakeDirection::RIGHT;
    paused = false;
    game_over = false;
    score = 0;
    grow_pending = 0;
    SpawnFood();
}

void SnakeGame::SetPaused(bool paused_in) {
    paused = paused_in;
}

int SnakeGame::TickIntervalMs() const {
    const int start_ms = 800;
    const int min_ms = 450;
    const int step_ms = 15;
    const int speedup = std::min(score, 12) * step_ms;
    return std::max(min_ms, start_ms - speedup);
}

bool SnakeGame::IsOpposite(SnakeDirection a, SnakeDirection b) const {
    return (a == SnakeDirection::UP && b == SnakeDirection::DOWN) ||
           (a == SnakeDirection::DOWN && b == SnakeDirection::UP) ||
           (a == SnakeDirection::LEFT && b == SnakeDirection::RIGHT) ||
           (a == SnakeDirection::RIGHT && b == SnakeDirection::LEFT);
}

void SnakeGame::SetDirection(SnakeDirection next_direction) {
    if (!initialized || game_over || body.empty()) {
        return;
    }
    if (IsOpposite(direction, next_direction)) {
        // Reject a 180-degree turn. Reversing the deque changes the head/tail
        // semantics and can make collision detection inconsistent.
        return;
    }
    pending_direction = next_direction;
}

bool SnakeGame::IsOccupied(int x, int y) const {
    for (size_t i = 0; i < body.size(); ++i) {
        if (body[i].x == x && body[i].y == y) {
            return true;
        }
    }
    return false;
}

void SnakeGame::SpawnFood() {
    has_food = false;
    if (static_cast<int>(body.size()) >= cols * rows) {
        return;
    }

    std::vector<SnakeCell> free_cells;
    free_cells.reserve(static_cast<size_t>(cols * rows) - body.size());
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (!IsOccupied(x, y)) {
                free_cells.push_back(SnakeCell{x, y});
            }
        }
    }

    if (free_cells.empty()) {
        return;
    }

    std::uniform_int_distribution<int> dist(0, static_cast<int>(free_cells.size()) - 1);
    food = free_cells[static_cast<size_t>(dist(rng))];
    has_food = true;
}

bool SnakeGame::Tick() {
    if (!initialized || paused || game_over || body.empty()) {
        return false;
    }

    if (!IsOpposite(direction, pending_direction)) {
        direction = pending_direction;
    }

    SnakeCell next = body.front();
    switch (direction) {
        case SnakeDirection::UP:
            --next.y;
            break;
        case SnakeDirection::DOWN:
            ++next.y;
            break;
        case SnakeDirection::LEFT:
            --next.x;
            break;
        case SnakeDirection::RIGHT:
            ++next.x;
            break;
    }

    if (next.x < 0 || next.x >= cols || next.y < 0 || next.y >= rows) {
        game_over = true;
        return false;
    }

    const bool will_grow = has_food && next.x == food.x && next.y == food.y;
    const size_t self_check_count = will_grow ? body.size() : body.size() - 1;
    for (size_t i = 0; i < self_check_count; ++i) {
        if (body[i].x == next.x && body[i].y == next.y) {
            game_over = true;
            return false;
        }
    }

    body.push_front(next);
    if (will_grow) {
        ++score;
        best_score = std::max(best_score, score);
        grow_pending += 2;
        SpawnFood();
    }

    if (grow_pending > 0) {
        --grow_pending;
    } else {
        body.pop_back();
    }

    return true;
}

SnakeRenderData SnakeGame::BuildRenderData() const {
    SnakeRenderData data;
    data.board_cols = cols;
    data.board_rows = rows;
    data.score = score;
    data.best_score = best_score;
    data.paused = paused;
    data.game_over = game_over;
    data.has_food = has_food;
    data.food = food;
    data.snake.assign(body.begin(), body.end());
    return data;
}
