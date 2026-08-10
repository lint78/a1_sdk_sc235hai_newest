#pragma once

#include <cstddef>
#include <cstdint>

namespace a1face {

constexpr int kFaceInputWidth = 112;
constexpr int kFaceInputHeight = 112;
constexpr int kFaceInputChannels = 3;
constexpr int kEmbeddingDimension = 512;
constexpr int kMaxVoteWindow = 16;

struct Point {
  Point() : x(0.0f), y(0.0f) {}
  Point(float x_value, float y_value) : x(x_value), y(y_value) {}
  float x;
  float y;
};

struct Box {
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float detection_score = 0.0f;
};

struct QualityConfig {
  float minimum_detection_score = 0.60f;
  float minimum_face_side_pixels = 64.0f;
  float minimum_eye_distance_pixels = 16.0f;
  float maximum_absolute_roll_degrees = 35.0f;
};

struct VoteConfig {
  float similarity_threshold = 0.26f;
  int window_size = 5;
  int required_passes = 3;
};

struct VoteResult {
  float score = 0.0f;
  bool instant_owner = false;
  bool confirmed_owner = false;
  int pass_count = 0;
  int sample_count = 0;
};

bool ValidateFace(const Box& box,
                  const Point landmarks[5],
                  int image_width,
                  int image_height,
                  const QualityConfig& config);

bool AlignFaceRgb112(const uint8_t* source_rgb,
                     int source_width,
                     int source_height,
                     int source_stride_bytes,
                     const Point landmarks[5],
                     uint8_t output_rgb[kFaceInputWidth * kFaceInputHeight *
                                        kFaceInputChannels]);

bool L2Normalize(float* values, int count);
float DotProduct(const float* lhs, const float* rhs, int count);
int LoadOwnerTemplate(const char* path, float output[kEmbeddingDimension]);

class VoteFilter {
 public:
  explicit VoteFilter(VoteConfig config = {});

  void Configure(VoteConfig config);
  void Reset();
  VoteResult Update(float score, bool valid_sample);

 private:
  VoteConfig config_{};
  bool passes_[kMaxVoteWindow]{};
  int next_index_ = 0;
  int sample_count_ = 0;
};

}  // namespace a1face
