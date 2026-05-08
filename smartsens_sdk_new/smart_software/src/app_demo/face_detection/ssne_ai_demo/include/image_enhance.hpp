#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "smartsoc/ssne_api.h"

namespace image_enhance {

enum class SceneType {
    kNormal = 0,
    kBacklight,
    kLowLight,
    kOverexposed,
};

struct SceneStats {
    SceneType scene = SceneType::kNormal;
    float mean_luma = 0.0f;
    float center_luma = 0.0f;
    float edge_luma = 0.0f;
    float dark_ratio = 0.0f;
    float bright_ratio = 0.0f;
    uint8_t p05 = 0;
    uint8_t p10 = 0;
    uint8_t p90 = 0;
    uint8_t p95 = 0;
    int y_parity = 0;
    bool used_focus_roi = false;
};

class AdaptiveImageEnhancer {
  public:
    bool PrepareForInference(ssne_tensor_t input,
                             ssne_tensor_t* prepared,
                             SceneStats* stats = nullptr);
    bool PrepareForInference(ssne_tensor_t input,
                             ssne_tensor_t* prepared,
                             SceneStats* stats,
                             const std::array<float, 4>* focus_roi);

    static const char* SceneTypeName(SceneType scene);

  private:
    bool IsSupported(ssne_tensor_t input);
    SceneStats Analyze(ssne_tensor_t input);
    SceneStats Analyze(ssne_tensor_t input, const std::array<float, 4>* focus_roi);
    int DetectYParity(const uint8_t* data, uint32_t width, uint32_t height, size_t row_stride);

    void BuildLut(const SceneStats& stats, uint8_t lut[256]);
    void ApplyLut(ssne_tensor_t tensor, int y_parity, const uint8_t lut[256]);
    void ApplyFastUSM(ssne_tensor_t tensor,
                      int y_parity,
                      float strength,
                      const std::array<float, 4>* focus_roi = nullptr,
                      float background_scale = 1.0f,
                      uint8_t focus_luma_threshold = 100,
                      uint8_t background_luma_threshold = 100);

    void BuildLowLightLut(const SceneStats& stats, uint8_t lut[256]);
    void BuildBacklightLut(const SceneStats& stats, uint8_t lut[256]);
    void BuildOverexposedLut(const SceneStats& stats, uint8_t lut[256]);
};

}  // namespace image_enhance
