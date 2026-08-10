#pragma once

#include <cstdint>

#include "face_recognition_business.h"
#include "smartsoc/ssne_api.h"

namespace a1face {

struct RecognizerConfig {
  int intermediate_roi_size = 160;
  float roi_expand_scale = 1.45f;
  float int8_output_scale = 0.03260263800621033f;
  QualityConfig quality{};
  VoteConfig vote{};
};

struct RecognitionResult {
  int status = 0;
  float similarity = -1.0f;
  bool valid_face = false;
  bool instant_owner = false;
  bool confirmed_owner = false;
  int pass_count = 0;
  int sample_count = 0;
};

class A1FaceRecognizer {
 public:
  A1FaceRecognizer();
  ~A1FaceRecognizer();

  int Initialize(const char* model_path, const char* owner_template_path,
                 RecognizerConfig config = {});
  void Release();
  bool IsInitialized() const { return initialized_; }
  void ResetVoting();

  int RecognizeYuv422(ssne_tensor_t frame_yuv422, const Box& face,
                      const Point landmarks[5], RecognitionResult* result);

  int RecognizeAlignedRgb112(
      const uint8_t rgb112[kFaceInputWidth * kFaceInputHeight * 3],
      RecognitionResult* result);

 private:
  int InferEmbedding(const uint8_t rgb112[kFaceInputWidth * kFaceInputHeight * 3],
                     float embedding[kEmbeddingDimension]);
  int FillModelInput(const uint8_t* rgb112);
  int CropYuv422Square(ssne_tensor_t source, int left, int top, int side,
                       ssne_tensor_t* destination);
  bool BuildSquareRoi(const Box& face, const Point landmarks[5], int image_width,
                      int image_height, int* left, int* top, int* side) const;
  void InvalidateResult(int status, RecognitionResult* result);

  RecognizerConfig config_{};
  VoteFilter vote_{};
  float owner_template_[kEmbeddingDimension]{};
  uint16_t model_id_ = 0;
  uint8_t model_alloc_flag_ = SSNE_STATIC_ALLOC;
  int model_input_dtype_ = -1;

  AiPreprocessPipe roi_pipe_{};
  bool roi_pipe_created_ = false;
  ssne_tensor_t roi_rgb_tensor_{};
  bool roi_rgb_created_ = false;
  ssne_tensor_t model_input_tensor_{};
  bool model_input_created_ = false;
  ssne_tensor_t output_tensor_{};
  bool output_ready_ = false;

  uint8_t aligned_rgb_[kFaceInputWidth * kFaceInputHeight * 3]{};
  bool initialized_ = false;
};

}  // namespace a1face
