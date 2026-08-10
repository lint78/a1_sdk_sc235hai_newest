#include "face_recognition_business.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace a1face {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDestinationLandmarks[5][2] = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f},
};

float Square(float value) { return value * value; }

float Distance(const Point& lhs, const Point& rhs) {
  return std::sqrt(Square(lhs.x - rhs.x) + Square(lhs.y - rhs.y));
}

void CanonicalizeLandmarkOrder(const Point input[5], Point output[5]) {
  std::memcpy(output, input, sizeof(Point) * 5U);
  if (output[0].x > output[1].x) std::swap(output[0], output[1]);
  if (output[3].x > output[4].x) std::swap(output[3], output[4]);
}

bool SolveLinear4x4(float matrix[4][4], float rhs[4], float solution[4]) {
  float augmented[4][5] = {};
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      augmented[row][col] = matrix[row][col];
    }
    augmented[row][4] = rhs[row];
  }

  for (int col = 0; col < 4; ++col) {
    int pivot = col;
    for (int row = col + 1; row < 4; ++row) {
      if (std::fabs(augmented[row][col]) > std::fabs(augmented[pivot][col])) {
        pivot = row;
      }
    }
    if (std::fabs(augmented[pivot][col]) < 1e-8f) {
      return false;
    }
    if (pivot != col) {
      for (int item = col; item < 5; ++item) {
        std::swap(augmented[col][item], augmented[pivot][item]);
      }
    }
    const float divisor = augmented[col][col];
    for (int item = col; item < 5; ++item) {
      augmented[col][item] /= divisor;
    }
    for (int row = 0; row < 4; ++row) {
      if (row == col) {
        continue;
      }
      const float factor = augmented[row][col];
      for (int item = col; item < 5; ++item) {
        augmented[row][item] -= factor * augmented[col][item];
      }
    }
  }

  for (int row = 0; row < 4; ++row) {
    solution[row] = augmented[row][4];
  }
  return true;
}

bool EstimateSimilarity(const Point source[5], float* a, float* b, float* tx,
                        float* ty) {
  float normal[4][4] = {};
  float rhs[4] = {};

  for (int i = 0; i < 5; ++i) {
    const float x = source[i].x;
    const float y = source[i].y;
    const float u = kDestinationLandmarks[i][0];
    const float v = kDestinationLandmarks[i][1];
    const float rows[2][4] = {
        {x, -y, 1.0f, 0.0f},
        {y, x, 0.0f, 1.0f},
    };
    const float targets[2] = {u, v};
    for (int r = 0; r < 2; ++r) {
      for (int col = 0; col < 4; ++col) {
        rhs[col] += rows[r][col] * targets[r];
        for (int other = 0; other < 4; ++other) {
          normal[col][other] += rows[r][col] * rows[r][other];
        }
      }
    }
  }

  float solution[4] = {};
  if (!SolveLinear4x4(normal, rhs, solution)) {
    return false;
  }
  *a = solution[0];
  *b = solution[1];
  *tx = solution[2];
  *ty = solution[3];
  return (*a * *a + *b * *b) > 1e-8f;
}

uint8_t BilinearSample(const uint8_t* source, int width, int height, int stride,
                       float x, float y, int channel) {
  if (x < 0.0f || y < 0.0f || x > static_cast<float>(width - 1) ||
      y > static_cast<float>(height - 1)) {
    return 0;
  }
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, width - 1);
  const int y1 = std::min(y0 + 1, height - 1);
  const float dx = x - static_cast<float>(x0);
  const float dy = y - static_cast<float>(y0);
  const float p00 = source[y0 * stride + x0 * 3 + channel];
  const float p01 = source[y0 * stride + x1 * 3 + channel];
  const float p10 = source[y1 * stride + x0 * 3 + channel];
  const float p11 = source[y1 * stride + x1 * 3 + channel];
  const float top = p00 + (p01 - p00) * dx;
  const float bottom = p10 + (p11 - p10) * dx;
  const float value = top + (bottom - top) * dy;
  return static_cast<uint8_t>(
      std::max(0.0f, std::min(255.0f, std::round(value))));
}

VoteConfig SanitizeVoteConfig(VoteConfig config) {
  config.window_size = std::max(1, std::min(config.window_size, kMaxVoteWindow));
  config.required_passes =
      std::max(1, std::min(config.required_passes, config.window_size));
  return config;
}

}  // namespace

bool ValidateFace(const Box& box, const Point landmarks[5], int image_width,
                  int image_height, const QualityConfig& config) {
  if (!landmarks || image_width <= 0 || image_height <= 0) {
    return false;
  }
  if (box.detection_score < config.minimum_detection_score ||
      std::min(box.width, box.height) < config.minimum_face_side_pixels) {
    return false;
  }

  Point ordered[5];
  CanonicalizeLandmarkOrder(landmarks, ordered);
  for (int i = 0; i < 5; ++i) {
    if (!std::isfinite(ordered[i].x) || !std::isfinite(ordered[i].y) ||
        ordered[i].x < 0.0f || ordered[i].x >= static_cast<float>(image_width) ||
        ordered[i].y < 0.0f ||
        ordered[i].y >= static_cast<float>(image_height)) {
      return false;
    }
  }

  const float eye_distance = Distance(ordered[0], ordered[1]);
  if (eye_distance < config.minimum_eye_distance_pixels) {
    return false;
  }
  const float roll =
      std::atan2(ordered[1].y - ordered[0].y, ordered[1].x - ordered[0].x) *
      180.0f / kPi;
  return std::fabs(roll) <= config.maximum_absolute_roll_degrees;
}

bool AlignFaceRgb112(const uint8_t* source_rgb, int source_width,
                     int source_height, int source_stride_bytes,
                     const Point landmarks[5],
                     uint8_t output_rgb[kFaceInputWidth * kFaceInputHeight *
                                        kFaceInputChannels]) {
  if (!source_rgb || !landmarks || !output_rgb || source_width <= 1 ||
      source_height <= 1 || source_stride_bytes < source_width * 3) {
    return false;
  }

  Point ordered[5];
  CanonicalizeLandmarkOrder(landmarks, ordered);

  float a = 0.0f;
  float b = 0.0f;
  float tx = 0.0f;
  float ty = 0.0f;
  if (!EstimateSimilarity(ordered, &a, &b, &tx, &ty)) {
    return false;
  }
  const float denominator = a * a + b * b;

  for (int y = 0; y < kFaceInputHeight; ++y) {
    for (int x = 0; x < kFaceInputWidth; ++x) {
      const float du = static_cast<float>(x) - tx;
      const float dv = static_cast<float>(y) - ty;
      const float source_x = (a * du + b * dv) / denominator;
      const float source_y = (-b * du + a * dv) / denominator;
      uint8_t* destination = output_rgb + (y * kFaceInputWidth + x) * 3;
      for (int channel = 0; channel < 3; ++channel) {
        destination[channel] =
            BilinearSample(source_rgb, source_width, source_height,
                           source_stride_bytes, source_x, source_y, channel);
      }
    }
  }
  return true;
}

bool L2Normalize(float* values, int count) {
  if (!values || count <= 0) {
    return false;
  }
  double squared_norm = 0.0;
  for (int i = 0; i < count; ++i) {
    squared_norm += static_cast<double>(values[i]) * values[i];
  }
  if (!std::isfinite(squared_norm) || squared_norm <= 1e-20) {
    return false;
  }
  const float inverse_norm = 1.0f / static_cast<float>(std::sqrt(squared_norm));
  for (int i = 0; i < count; ++i) {
    values[i] *= inverse_norm;
  }
  return true;
}

float DotProduct(const float* lhs, const float* rhs, int count) {
  if (!lhs || !rhs || count <= 0) {
    return -1.0f;
  }
  double result = 0.0;
  for (int i = 0; i < count; ++i) {
    result += static_cast<double>(lhs[i]) * rhs[i];
  }
  return static_cast<float>(result);
}

int LoadOwnerTemplate(const char* path, float output[kEmbeddingDimension]) {
  if (!path || !output) {
    return -1;
  }
  FILE* file = std::fopen(path, "rb");
  if (!file) {
    return -2;
  }
  const size_t count = std::fread(output, sizeof(float), kEmbeddingDimension, file);
  const int extra = std::fgetc(file);
  std::fclose(file);
  if (count != kEmbeddingDimension || extra != EOF) {
    return -3;
  }
  return L2Normalize(output, kEmbeddingDimension) ? 0 : -4;
}

VoteFilter::VoteFilter(VoteConfig config) { Configure(config); }

void VoteFilter::Configure(VoteConfig config) {
  config_ = SanitizeVoteConfig(config);
  Reset();
}

void VoteFilter::Reset() {
  std::memset(passes_, 0, sizeof(passes_));
  next_index_ = 0;
  sample_count_ = 0;
}

VoteResult VoteFilter::Update(float score, bool valid_sample) {
  VoteResult result;
  result.score = score;
  if (!valid_sample || !std::isfinite(score)) {
    Reset();
    return result;
  }

  const bool pass = score >= config_.similarity_threshold;
  passes_[next_index_] = pass;
  next_index_ = (next_index_ + 1) % config_.window_size;
  sample_count_ = std::min(sample_count_ + 1, config_.window_size);

  int pass_count = 0;
  for (int i = 0; i < sample_count_; ++i) {
    pass_count += passes_[i] ? 1 : 0;
  }
  result.instant_owner = pass;
  result.pass_count = pass_count;
  result.sample_count = sample_count_;
  result.confirmed_owner = sample_count_ == config_.window_size &&
                           pass_count >= config_.required_passes;
  return result;
}

}  // namespace a1face
