#include "../include/image_enhance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace image_enhance {
namespace {

constexpr size_t kHistBins = 256;
constexpr uint8_t kDarkThreshold = 50;
constexpr uint8_t kBrightThreshold = 225;
constexpr uint8_t kLowLightFocusUsmThreshold = 52;
constexpr uint8_t kLowLightBackgroundUsmThreshold = 96;
constexpr float kToeLinearEnd = 0.08f;
constexpr size_t kLowLightPresetCount = 6;

struct LowLightPreset {
    float mean_luma_upper;
    float gamma;
    float shadow_boost;
    float toe_lift;
    float mid_boost;

    LowLightPreset()
        : mean_luma_upper(255.0f),
          gamma(1.0f),
          shadow_boost(0.0f),
          toe_lift(0.0f),
          mid_boost(0.0f) {}

    LowLightPreset(float mean_luma_upper_in,
                   float gamma_in,
                   float shadow_boost_in,
                   float toe_lift_in,
                   float mid_boost_in)
        : mean_luma_upper(mean_luma_upper_in),
          gamma(gamma_in),
          shadow_boost(shadow_boost_in),
          toe_lift(toe_lift_in),
          mid_boost(mid_boost_in) {}
};

const std::array<LowLightPreset, kLowLightPresetCount> kLowLightPresets = {{
    // mean 越低 gamma 越激进，同时用 toe_lift 保护极暗部，避免纯幂函数把底噪无限放大。
    {22.0f, 0.28f, 0.32f, 0.10f, 0.12f},
    {30.0f, 0.34f, 0.28f, 0.08f, 0.10f},
    {40.0f, 0.40f, 0.24f, 0.06f, 0.09f},
    {50.0f, 0.48f, 0.20f, 0.05f, 0.07f},
    {62.0f, 0.56f, 0.16f, 0.04f, 0.05f},
    {255.0f, 0.64f, 0.12f, 0.03f, 0.03f},
}};

inline float ClampFloat(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

inline uint8_t ToByte(float value) {
    return static_cast<uint8_t>(std::round(ClampFloat(value, 0.0f, 255.0f)));
}

inline float NormalizeLuma(int value, int black, int white) {
    if (white <= black) {
        return static_cast<float>(value) / 255.0f;
    }
    return ClampFloat(static_cast<float>(value - black) / static_cast<float>(white - black), 0.0f, 1.0f);
}

const LowLightPreset& SelectLowLightPreset(const SceneStats& stats) {
    size_t preset_index = 0;
    while (preset_index + 1 < kLowLightPresets.size() &&
           stats.mean_luma > kLowLightPresets[preset_index].mean_luma_upper) {
        ++preset_index;
    }

    if ((stats.dark_ratio > 0.82f || stats.p95 < 80) && preset_index > 0) {
        --preset_index;
    } else if ((stats.dark_ratio > 0.70f || stats.p95 < 105) && preset_index > 1) {
        --preset_index;
    }

    return kLowLightPresets[preset_index];
}

const std::array<uint8_t, 256>& GetLowLightGammaLut(float gamma) {
    // gamma 只在建表时使用 pow，逐像素增强阶段只查 LUT，保证低算力平台可承受。
    struct CachedGammaLut {
        float gamma = -1.0f;
        std::array<uint8_t, 256> lut = {};
    };

    static std::array<CachedGammaLut, kLowLightPresetCount> cache = {};

    for (auto& entry : cache) {
        if (std::fabs(entry.gamma - gamma) < 1e-6f) {
            return entry.lut;
        }
    }

    for (auto& entry : cache) {
        if (entry.gamma < 0.0f) {
            entry.gamma = gamma;
            for (int i = 0; i < 256; ++i) {
                const float source = static_cast<float>(i) / 255.0f;
                entry.lut[static_cast<size_t>(i)] = ToByte(std::pow(source, gamma) * 255.0f);
            }
            return entry.lut;
        }
    }

    cache[0].gamma = gamma;
    for (int i = 0; i < 256; ++i) {
        const float source = static_cast<float>(i) / 255.0f;
        cache[0].lut[static_cast<size_t>(i)] = ToByte(std::pow(source, gamma) * 255.0f);
    }
    return cache[0].lut;
}

inline uint8_t ApplyPiecewiseToe(const std::array<uint8_t, 256>& gamma_lut,
                                 float normalized_value) {
    const float clamped = ClampFloat(normalized_value, 0.0f, 1.0f);
    if (clamped <= 0.0f) {
        return 0;
    }

    if (clamped < kToeLinearEnd) {
        const int toe_end_index = static_cast<int>(kToeLinearEnd * 255.0f + 0.5f);
        const float toe_end_value =
            static_cast<float>(gamma_lut[static_cast<size_t>(toe_end_index)]) / 255.0f;
        const float linear_scale = toe_end_value / kToeLinearEnd;
        return ToByte(clamped * linear_scale * 255.0f);
    }

    const int lut_index = static_cast<int>(clamped * 255.0f + 0.5f);
    return gamma_lut[static_cast<size_t>(std::min(255, std::max(0, lut_index)))];
}

uint8_t PercentileFromHist(const std::array<uint32_t, kHistBins>& hist,
                           uint64_t total,
                           float percentile) {
    if (total == 0) {
        return 0;
    }

    const uint64_t target = static_cast<uint64_t>(std::round(percentile * static_cast<float>(total - 1)));
    uint64_t accum = 0;
    for (size_t i = 0; i < hist.size(); ++i) {
        accum += hist[i];
        if (accum > target) {
            return static_cast<uint8_t>(i);
        }
    }
    return 255;
}

struct RoiBounds {
    uint32_t left = 0;
    uint32_t top = 0;
    uint32_t right = 0;
    uint32_t bottom = 0;
    bool valid = false;
};

inline bool IsInsideRoi(uint32_t x, uint32_t y, const RoiBounds& roi) {
    return roi.valid &&
           x >= roi.left && x < roi.right &&
           y >= roi.top && y < roi.bottom;
}

RoiBounds ResolveFocusRoi(uint32_t width,
                          uint32_t height,
                          const std::array<float, 4>* focus_roi) {
    RoiBounds roi;
    if (focus_roi == nullptr) {
        return roi;
    }

    const float x1 = std::max(0.0f, std::min((*focus_roi)[0], static_cast<float>(width)));
    const float y1 = std::max(0.0f, std::min((*focus_roi)[1], static_cast<float>(height)));
    const float x2 = std::max(0.0f, std::min((*focus_roi)[2], static_cast<float>(width)));
    const float y2 = std::max(0.0f, std::min((*focus_roi)[3], static_cast<float>(height)));
    if (x2 - x1 < 4.0f || y2 - y1 < 4.0f) {
        return roi;
    }

    roi.left = static_cast<uint32_t>(std::floor(x1));
    roi.top = static_cast<uint32_t>(std::floor(y1));
    roi.right = std::min(static_cast<uint32_t>(std::ceil(x2)), width);
    roi.bottom = std::min(static_cast<uint32_t>(std::ceil(y2)), height);
    roi.valid = roi.right > roi.left && roi.bottom > roi.top;
    return roi;
}

float ComputeLowLightUsmStrength(const SceneStats& stats) {
    float strength = 0.50f;
    strength += ClampFloat((42.0f - stats.mean_luma) / 30.0f, 0.0f, 1.0f) * 0.18f;
    strength += ClampFloat((0.72f - stats.dark_ratio) / 0.42f, 0.0f, 1.0f) * 0.08f;
    if (stats.used_focus_roi) {
        strength += 0.10f;
    }
    return ClampFloat(strength, 0.45f, 0.82f);
}

bool IsExtremeLowLight(const SceneStats& stats) {
    return stats.mean_luma < 35.0f ||
           stats.dark_ratio > 0.72f ||
           stats.p95 < 90;
}

}  // namespace

const char* AdaptiveImageEnhancer::SceneTypeName(SceneType scene) {
    switch (scene) {
        case SceneType::kNormal:
            return "normal";
        case SceneType::kBacklight:
            return "backlight";
        case SceneType::kLowLight:
            return "lowlight";
        case SceneType::kOverexposed:
            return "overexposed";
        default:
            return "unknown";
    }
}

bool AdaptiveImageEnhancer::PrepareForInference(ssne_tensor_t input,
                                                ssne_tensor_t* prepared,
                                                SceneStats* stats) {
    return PrepareForInference(input, prepared, stats, nullptr);
}

bool AdaptiveImageEnhancer::PrepareForInference(ssne_tensor_t input,
                                                ssne_tensor_t* prepared,
                                                SceneStats* stats,
                                                const std::array<float, 4>* focus_roi) {
    if (prepared == nullptr) {
        return false;
    }

    *prepared = input;
    if (!IsSupported(input)) {
        if (stats != nullptr) {
            *stats = SceneStats();
        }
        return false;
    }

    // 先分析当前帧属于哪种亮度场景，再决定是否增强，避免正常光照也付出增强成本。
    const SceneStats scene_stats = Analyze(input, focus_roi);
    if (stats != nullptr) {
        *stats = scene_stats;
    }
    bool should_enhance = false;
    switch (scene_stats.scene) {
        case SceneType::kLowLight:
            should_enhance =
                scene_stats.mean_luma < 55.0f ||
                scene_stats.dark_ratio > 0.48f ||
                scene_stats.p95 < 115;
            break;
        case SceneType::kBacklight:
        case SceneType::kOverexposed:
            should_enhance = true;
            break;
        case SceneType::kNormal:
        default:
            should_enhance = false;
            break;
    }

    if (!should_enhance) {
        return false;
    }

    // 在副本上增强，避免改写在线输入帧
    ssne_tensor_t working = copy_tensor(input);
    if (get_data(working) == nullptr ||
        get_width(working) == 0 ||
        get_height(working) == 0 ||
        get_mem_size(working) == 0 ||
        get_data(working) == get_data(input)) {
        return false;
    }

    uint8_t lut[256] = {};
    BuildLut(scene_stats, lut);
    if (scene_stats.scene == SceneType::kLowLight) {
        // 当前项目依赖 ISP 去噪，因此低照增强只做 LUT 拉伸和轻 USM，不再做软件时域降噪。
        const bool extreme_low_light = IsExtremeLowLight(scene_stats);
        ApplyLut(working, scene_stats.y_parity, lut);
        if (!extreme_low_light) {
            ApplyFastUSM(working,
                         scene_stats.y_parity,
                         ComputeLowLightUsmStrength(scene_stats),
                         focus_roi,
                         scene_stats.used_focus_roi ? 0.35f : 0.75f,
                         scene_stats.used_focus_roi ? kLowLightFocusUsmThreshold : 64,
                         kLowLightBackgroundUsmThreshold);
        }
    } else if (scene_stats.scene == SceneType::kBacklight) {
        ApplyLut(working, scene_stats.y_parity, lut);
        ApplyFastUSM(working, scene_stats.y_parity, 0.45f);
    } else if (scene_stats.scene == SceneType::kOverexposed) {
        ApplyLut(working, scene_stats.y_parity, lut);
        ApplyFastUSM(working, scene_stats.y_parity, 0.35f);
    } else {
        ApplyLut(working, scene_stats.y_parity, lut);
    }

    *prepared = working;
    return true;
}

bool AdaptiveImageEnhancer::IsSupported(ssne_tensor_t input) {
    if (get_data(input) == nullptr) {
        return false;
    }
    if (get_data_format(input) != SSNE_YUV422_16) {
        return false;
    }
    if (get_width(input) == 0 || get_height(input) == 0) {
        return false;
    }
    return get_mem_size(input) >= static_cast<size_t>(get_width(input)) *
                                      static_cast<size_t>(get_height(input)) * 2U;
}

SceneStats AdaptiveImageEnhancer::Analyze(ssne_tensor_t input) {
    return Analyze(input, nullptr);
}

SceneStats AdaptiveImageEnhancer::Analyze(ssne_tensor_t input,
                                          const std::array<float, 4>* focus_roi) {
    SceneStats stats;

    const uint32_t width = get_width(input);
    const uint32_t height = get_height(input);
    const size_t mem_size = get_mem_size(input);
    if (width == 0 || height == 0 || mem_size == 0) {
        return stats;
    }

    const size_t row_stride = mem_size / static_cast<size_t>(height);
    if (row_stride < static_cast<size_t>(width) * 2U) {
        return stats;
    }

    const uint8_t* data = static_cast<const uint8_t*>(get_data(input));
    if (data == nullptr) {
        return stats;
    }

    stats.y_parity = DetectYParity(data, width, height, row_stride);

    std::array<uint32_t, kHistBins> hist = {};
    uint64_t luma_sum = 0;
    uint64_t dark_count = 0;
    uint64_t bright_count = 0;
    uint64_t center_sum = 0;
    uint64_t edge_sum = 0;
    uint64_t center_count = 0;
    uint64_t edge_count = 0;

    // 有人体 ROI 时把该区域当成“中心”，否则用画面中心区域估计背光/暗光。
    RoiBounds roi = ResolveFocusRoi(width, height, focus_roi);
    stats.used_focus_roi = roi.valid;

    const uint32_t center_left = roi.valid ? roi.left : width / 4;
    const uint32_t center_right = roi.valid ? roi.right : (width - width / 4);
    const uint32_t center_top = roi.valid ? roi.top : height / 4;
    const uint32_t center_bottom = roi.valid ? roi.bottom : (height - height / 4);

    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* row = data + static_cast<size_t>(y) * row_stride + static_cast<size_t>(stats.y_parity);
        for (uint32_t x = 0; x < width; ++x) {
            const uint8_t luma = row[static_cast<size_t>(x) * 2U];
            hist[luma] += 1U;
            luma_sum += luma;
            dark_count += (luma <= kDarkThreshold) ? 1U : 0U;
            bright_count += (luma >= kBrightThreshold) ? 1U : 0U;

            const bool in_center = x >= center_left && x < center_right &&
                                   y >= center_top && y < center_bottom;
            if (in_center) {
                center_sum += luma;
                ++center_count;
            } else {
                edge_sum += luma;
                ++edge_count;
            }
        }
    }

    const uint64_t total_pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    if (total_pixels == 0) {
        return stats;
    }

    stats.mean_luma = static_cast<float>(luma_sum) / static_cast<float>(total_pixels);
    stats.center_luma = center_count > 0
        ? static_cast<float>(center_sum) / static_cast<float>(center_count)
        : stats.mean_luma;
    stats.edge_luma = edge_count > 0
        ? static_cast<float>(edge_sum) / static_cast<float>(edge_count)
        : stats.mean_luma;
    stats.dark_ratio = static_cast<float>(dark_count) / static_cast<float>(total_pixels);
    stats.bright_ratio = static_cast<float>(bright_count) / static_cast<float>(total_pixels);
    stats.p05 = PercentileFromHist(hist, total_pixels, 0.05f);
    stats.p10 = PercentileFromHist(hist, total_pixels, 0.10f);
    stats.p90 = PercentileFromHist(hist, total_pixels, 0.90f);
    stats.p95 = PercentileFromHist(hist, total_pixels, 0.95f);

    const float center_gap = stats.edge_luma - stats.center_luma;
    const bool is_backlight =
        center_gap > 26.0f &&
        stats.edge_luma > 125.0f &&
        stats.center_luma < stats.edge_luma * 0.85f &&
        stats.p95 > 185;
    const bool is_overexposed =
        stats.bright_ratio > 0.12f ||
        stats.p95 >= 248 ||
        (stats.mean_luma > 165.0f && stats.bright_ratio > 0.05f);
    const bool is_low_light =
        stats.mean_luma < 78.0f ||
        stats.dark_ratio > 0.35f ||
        stats.p95 < 150;

    if (is_backlight) {
        stats.scene = SceneType::kBacklight;
    } else if (is_overexposed) {
        stats.scene = SceneType::kOverexposed;
    } else if (is_low_light) {
        stats.scene = SceneType::kLowLight;
    } else {
        stats.scene = SceneType::kNormal;
    }

    return stats;
}

int AdaptiveImageEnhancer::DetectYParity(const uint8_t* data,
                                         uint32_t width,
                                         uint32_t height,
                                         size_t row_stride) {
    double score[2] = {0.0, 0.0};
    const uint32_t row_sample_step = height > 720 ? 2U : 1U;
    const uint32_t col_sample_step = width > 720 ? 2U : 1U;

    for (int parity = 0; parity < 2; ++parity) {
        uint64_t count = 0;
        uint64_t diff_count = 0;
        double sum = 0.0;
        double sum_sq = 0.0;
        double diff_sum = 0.0;

        for (uint32_t y = 0; y < height; y += row_sample_step) {
            const uint8_t* row = data + static_cast<size_t>(y) * row_stride + static_cast<size_t>(parity);
            for (uint32_t x = 0; x < width; x += col_sample_step) {
                const uint8_t value = row[static_cast<size_t>(x) * 2U];
                sum += value;
                sum_sq += static_cast<double>(value) * static_cast<double>(value);
                ++count;

                if (x + 2U * col_sample_step < width) {
                    const uint8_t next = row[static_cast<size_t>(x + 2U * col_sample_step) * 2U];
                    diff_sum += std::abs(static_cast<int>(value) - static_cast<int>(next));
                    ++diff_count;
                }
            }
        }

        if (count == 0) {
            continue;
        }

        const double mean = sum / static_cast<double>(count);
        const double variance = std::max(0.0, sum_sq / static_cast<double>(count) - mean * mean);
        const double stddev = std::sqrt(variance);
        const double avg_diff = diff_count > 0 ? diff_sum / static_cast<double>(diff_count) : 0.0;
        score[parity] = stddev + avg_diff * 0.35 + std::abs(mean - 128.0) * 0.05;
    }

    return (score[1] > score[0]) ? 1 : 0;
}

void AdaptiveImageEnhancer::BuildLut(const SceneStats& stats, uint8_t lut[256]) {
    switch (stats.scene) {
        case SceneType::kBacklight:
            BuildBacklightLut(stats, lut);
            break;
        case SceneType::kLowLight:
            BuildLowLightLut(stats, lut);
            break;
        case SceneType::kOverexposed:
            BuildOverexposedLut(stats, lut);
            break;
        case SceneType::kNormal:
        default:
            for (int i = 0; i < 256; ++i) {
                lut[i] = static_cast<uint8_t>(i);
            }
            break;
    }
}

void AdaptiveImageEnhancer::ApplyLut(ssne_tensor_t tensor, int y_parity, const uint8_t lut[256]) {
    const uint32_t width = get_width(tensor);
    const uint32_t height = get_height(tensor);
    const size_t mem_size = get_mem_size(tensor);
    if (width == 0 || height == 0 || mem_size == 0) {
        return;
    }

    const size_t row_stride = mem_size / static_cast<size_t>(height);
    uint8_t* data = static_cast<uint8_t*>(get_data(tensor));
    if (data == nullptr || row_stride < static_cast<size_t>(width) * 2U) {
        return;
    }

    for (uint32_t y = 0; y < height; ++y) {
        uint8_t* row = data + static_cast<size_t>(y) * row_stride + static_cast<size_t>(y_parity);
        for (uint32_t x = 0; x < width; ++x) {
            uint8_t& luma = row[static_cast<size_t>(x) * 2U];
            luma = lut[luma];
        }
    }
}

void AdaptiveImageEnhancer::ApplyFastUSM(ssne_tensor_t tensor,
                                         int y_parity,
                                         float strength,
                                         const std::array<float, 4>* focus_roi,
                                         float background_scale,
                                         uint8_t focus_luma_threshold,
                                         uint8_t background_luma_threshold) {
    const uint32_t width = get_width(tensor);
    const uint32_t height = get_height(tensor);
    const size_t mem_size = get_mem_size(tensor);
    if (width < 3 || height < 3 || mem_size == 0) {
        return;
    }

    const size_t row_stride = mem_size / static_cast<size_t>(height);
    uint8_t* data = static_cast<uint8_t*>(get_data(tensor));
    if (data == nullptr || row_stride < static_cast<size_t>(width) * 2U) {
        return;
    }

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    // ROI 内保留较强锐化，背景区域降强度，减少暗部噪声被 YOLO 误识别成目标。
    const RoiBounds roi = ResolveFocusRoi(width, height, focus_roi);
    std::vector<uint8_t> src_y(pixel_count, 0);

    size_t idx = 0;
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* row = data + static_cast<size_t>(y) * row_stride + static_cast<size_t>(y_parity);
        for (uint32_t x = 0; x < width; ++x) {
            src_y[idx++] = row[static_cast<size_t>(x) * 2U];
        }
    }

    for (uint32_t y = 1; y + 1 < height; ++y) {
        uint8_t* row = data + static_cast<size_t>(y) * row_stride + static_cast<size_t>(y_parity);
        for (uint32_t x = 1; x + 1 < width; ++x) {
            const size_t center_idx = static_cast<size_t>(y) * width + static_cast<size_t>(x);
            const uint8_t original_y = src_y[center_idx];
            const bool in_focus = IsInsideRoi(x, y, roi);
            const float local_strength = strength * (in_focus ? 1.0f : background_scale);
            const uint8_t local_threshold = in_focus ? focus_luma_threshold : background_luma_threshold;
            if (local_strength <= 0.01f || original_y <= local_threshold) {
                continue;
            }

            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const size_t sample_idx =
                        static_cast<size_t>(static_cast<int>(y) + ky) * width +
                        static_cast<size_t>(static_cast<int>(x) + kx);
                    sum += static_cast<int>(src_y[sample_idx]);
                }
            }

            const float blur_y = static_cast<float>(sum) / 9.0f;
            const float detail = static_cast<float>(original_y) - blur_y;
            const float output_y = static_cast<float>(original_y) + detail * local_strength;
            row[static_cast<size_t>(x) * 2U] = ToByte(output_y);
        }
    }
}

void AdaptiveImageEnhancer::BuildLowLightLut(const SceneStats& stats, uint8_t lut[256]) {
    const LowLightPreset& preset = SelectLowLightPreset(stats);
    const bool extreme_low_light = IsExtremeLowLight(stats);
    const float gamma = extreme_low_light ? 0.24f : preset.gamma;
    const std::array<uint8_t, 256>& gamma_lut = GetLowLightGammaLut(gamma);
    // 黑白点来自分位数而不是固定阈值，尽量适应窗帘、背光和局部 ROI 的亮度差异。
    const int black = extreme_low_light
        ? std::max(0, static_cast<int>(stats.p05) - 2)
        : std::max(0, static_cast<int>(stats.p05) - 4);
    const int white = extreme_low_light
        ? std::min(128, std::max(black + 24, static_cast<int>(stats.p95) + 10))
        : std::min(255, std::max(128, static_cast<int>(stats.p95) + 10));
    const float roi_protect = stats.used_focus_roi ? 0.85f : 1.0f;
    const float gain = extreme_low_light
        ? (stats.used_focus_roi ? 1.26f : 1.34f)
        : (stats.used_focus_roi ? 1.03f : 1.06f);

    for (int i = 0; i < 256; ++i) {
        const float source = NormalizeLuma(i, black, white);
        float mapped = static_cast<float>(ApplyPiecewiseToe(gamma_lut, source)) / 255.0f;
        const float shadow_weight = std::max(0.0f, 0.70f - source) / 0.70f;
        const float shadow_boost = extreme_low_light
            ? std::max(0.42f, preset.shadow_boost + 0.14f)
            : preset.shadow_boost;
        mapped += shadow_boost * roi_protect * shadow_weight * shadow_weight;
        if (source < 0.42f) {
            const float mid_boost = extreme_low_light
                ? std::max(0.18f, preset.mid_boost + 0.08f)
                : preset.mid_boost;
            mapped += mid_boost * roi_protect * (0.42f - source) / 0.42f;
        }
        if (extreme_low_light) {
            mapped += 0.18f * shadow_weight;
            if (source < 0.28f) {
                mapped += 0.12f * (0.28f - source) / 0.28f;
            }
        }

        const float toe_lift = (extreme_low_light ? std::max(0.16f, preset.toe_lift + 0.08f) : preset.toe_lift) *
                               roi_protect;
        mapped = mapped * (1.0f - toe_lift) + toe_lift;
        if (!extreme_low_light && source < kToeLinearEnd) {
            const float toe_mix = source / kToeLinearEnd;
            mapped = mapped * toe_mix + source * (1.0f - toe_mix);
        }
        mapped = std::min(1.0f, mapped * gain + (extreme_low_light ? 0.04f : 0.01f));
        lut[i] = ToByte(mapped * 255.0f);
    }
}

void AdaptiveImageEnhancer::BuildBacklightLut(const SceneStats& stats, uint8_t lut[256]) {
    // 背光场景: 优先提阴影，再压高光
    const int black = std::max(0, static_cast<int>(stats.p05) - 8);
    const float gap_strength = ClampFloat((stats.edge_luma - stats.center_luma) / 96.0f, 0.0f, 1.0f);
    const float gamma = 0.78f - 0.10f * gap_strength;
    const float knee = 0.74f - 0.04f * gap_strength;
    const float high_scale = 0.58f - 0.12f * gap_strength;

    for (int i = 0; i < 256; ++i) {
        const float source = NormalizeLuma(i, black, 255);
        float mapped = std::pow(source, gamma);
        if (mapped > knee) {
            mapped = knee + (mapped - knee) * high_scale;
        }

        const float shadow_weight = std::max(0.0f, 0.55f - source) / 0.55f;
        mapped = std::min(1.0f, mapped + 0.12f * gap_strength * shadow_weight * shadow_weight);
        lut[i] = ToByte(mapped * 255.0f);
    }
}

void AdaptiveImageEnhancer::BuildOverexposedLut(const SceneStats& stats, uint8_t lut[256]) {
    const float strength = ClampFloat((stats.bright_ratio - 0.08f) / 0.20f, 0.0f, 1.0f);
    const float gamma = 1.18f + 0.32f * strength;
    const float knee = 0.70f - 0.05f * strength;
    const float high_scale = 0.46f - 0.10f * strength;

    for (int i = 0; i < 256; ++i) {
        const float source = static_cast<float>(i) / 255.0f;
        float mapped = source;
        if (source >= 0.35f) {
            mapped = std::pow(source, gamma);
            mapped = mapped * 0.65f + source * 0.35f;
        } else {
            mapped = source * (0.98f - 0.03f * strength);
        }

        if (mapped > knee) {
            mapped = knee + (mapped - knee) * high_scale;
        }
        lut[i] = ToByte(mapped * 255.0f);
    }
}

}  // namespace image_enhance



