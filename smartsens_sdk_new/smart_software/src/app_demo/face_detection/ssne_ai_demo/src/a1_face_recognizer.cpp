#include "a1_face_recognizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

namespace a1face {
namespace {

constexpr int kRgbBytes = kFaceInputWidth * kFaceInputHeight * 3;

bool IsValidTensor(ssne_tensor_t tensor) {
  return get_data(tensor) != nullptr && get_width(tensor) > 0 &&
         get_height(tensor) > 0 && get_mem_size(tensor) > 0;
}

float Clamp(float value, float low, float high) {
  return std::max(low, std::min(value, high));
}

}  // namespace

A1FaceRecognizer::A1FaceRecognizer() : vote_(config_.vote) {}

A1FaceRecognizer::~A1FaceRecognizer() { Release(); }

int A1FaceRecognizer::Initialize(const char* model_path,
                                 const char* owner_template_path,
                                 RecognizerConfig config) {
  Release();
  if (!model_path || !owner_template_path ||
      config.intermediate_roi_size < kFaceInputWidth ||
      config.roi_expand_scale < 1.0f) {
    return -1;
  }
  config_ = config;
  vote_.Configure(config_.vote);

  const int template_status =
      LoadOwnerTemplate(owner_template_path, owner_template_);
  if (template_status != 0) {
    std::fprintf(stderr,
                 "[face-id] owner template load failed: %d path=%s\n",
                 template_status, owner_template_path);
    Release();
    return -2;
  }

  struct stat model_stat = {};
  if (stat(model_path, &model_stat) != 0) {
    std::fprintf(stderr, "[face-id] model file not found: %s\n", model_path);
    Release();
    return -3;
  }

  char* mutable_path = const_cast<char*>(model_path);
  model_alloc_flag_ = SSNE_STATIC_ALLOC;
  model_id_ = ssne_loadmodel(mutable_path, model_alloc_flag_);
  if (model_id_ == 0) {
    model_alloc_flag_ = SSNE_DYNAMIC_ALLOC;
    model_id_ = ssne_loadmodel(mutable_path, model_alloc_flag_);
  }
  if (model_id_ == 0) {
    std::fprintf(stderr, "[face-id] model load failed: %s\n", model_path);
    Release();
    return -4;
  }

  const int input_count = ssne_get_model_input_num(model_id_);
  if (input_count != 1 ||
      ssne_get_model_input_dtype(model_id_, &model_input_dtype_) != 0) {
    std::fprintf(stderr,
                 "[face-id] unexpected model input metadata: count=%d dtype=%d\n",
                 input_count, model_input_dtype_);
    Release();
    return -5;
  }

  roi_pipe_ = GetAIPreprocessPipe();
  roi_pipe_created_ = roi_pipe_ != nullptr;
  if (!roi_pipe_created_) {
    Release();
    return -6;
  }

  roi_rgb_tensor_ = create_tensor(
      static_cast<uint32_t>(config_.intermediate_roi_size),
      static_cast<uint32_t>(config_.intermediate_roi_size), SSNE_RGB,
      SSNE_BUF_AI);
  roi_rgb_created_ = IsValidTensor(roi_rgb_tensor_);
  if (!roi_rgb_created_) {
    Release();
    return -7;
  }
  set_data_type(roi_rgb_tensor_, SSNE_UINT8);

  model_input_tensor_ =
      create_tensor(kFaceInputWidth, kFaceInputHeight, SSNE_RGB, SSNE_BUF_AI);
  model_input_created_ = IsValidTensor(model_input_tensor_);
  if (!model_input_created_) {
    Release();
    return -8;
  }
  if (set_data_type(model_input_tensor_, static_cast<uint8_t>(model_input_dtype_)) != 0) {
    Release();
    return -9;
  }

  std::printf(
      "[face-id] initialized model=%s bytes=%lld input_dtype=%d input=RGB112 output=512 "
      "threshold=%.3f vote=%d/%d\n",
      model_path, static_cast<long long>(model_stat.st_size),
      model_input_dtype_, config_.vote.similarity_threshold,
      config_.vote.required_passes, config_.vote.window_size);
  initialized_ = true;
  return 0;
}

void A1FaceRecognizer::Release() {
  initialized_ = false;
  vote_.Reset();
  if (output_ready_) {
    release_tensor(output_tensor_);
  }
  output_tensor_ = ssne_tensor_t{};
  output_ready_ = false;
  if (model_input_created_) {
    release_tensor(model_input_tensor_);
  }
  model_input_tensor_ = ssne_tensor_t{};
  model_input_created_ = false;
  if (roi_rgb_created_) {
    release_tensor(roi_rgb_tensor_);
  }
  roi_rgb_tensor_ = ssne_tensor_t{};
  roi_rgb_created_ = false;
  if (roi_pipe_created_) {
    ReleaseAIPreprocessPipe(roi_pipe_);
  }
  roi_pipe_ = AiPreprocessPipe{};
  roi_pipe_created_ = false;
  model_id_ = 0;
  model_input_dtype_ = -1;
  std::memset(owner_template_, 0, sizeof(owner_template_));
  std::memset(aligned_rgb_, 0, sizeof(aligned_rgb_));
}

void A1FaceRecognizer::ResetVoting() { vote_.Reset(); }

void A1FaceRecognizer::InvalidateResult(int status, RecognitionResult* result) {
  vote_.Update(0.0f, false);
  if (result) {
    *result = RecognitionResult{};
    result->status = status;
  }
}

bool A1FaceRecognizer::BuildSquareRoi(const Box& face, const Point landmarks[5],
                                      int image_width, int image_height, int* left,
                                      int* top, int* side) const {
  if (!landmarks || !left || !top || !side || image_width < 2 ||
      image_height < 2) {
    return false;
  }
  float minimum_x = face.x;
  float minimum_y = face.y;
  float maximum_x = face.x + face.width;
  float maximum_y = face.y + face.height;
  for (int i = 0; i < 5; ++i) {
    minimum_x = std::min(minimum_x, landmarks[i].x);
    minimum_y = std::min(minimum_y, landmarks[i].y);
    maximum_x = std::max(maximum_x, landmarks[i].x);
    maximum_y = std::max(maximum_y, landmarks[i].y);
  }

  const float center_x = (minimum_x + maximum_x) * 0.5f;
  const float center_y = (minimum_y + maximum_y) * 0.5f;
  int requested_side = static_cast<int>(std::ceil(
      std::max(maximum_x - minimum_x, maximum_y - minimum_y) *
      config_.roi_expand_scale));
  requested_side = std::max(requested_side, 2);
  requested_side =
      std::min(requested_side, std::min(image_width, image_height));
  requested_side &= ~1;
  if (requested_side < 2) {
    return false;
  }

  int requested_left =
      static_cast<int>(std::lround(center_x - requested_side * 0.5f));
  int requested_top =
      static_cast<int>(std::lround(center_y - requested_side * 0.5f));
  requested_left =
      std::max(0, std::min(requested_left, image_width - requested_side));
  requested_top =
      std::max(0, std::min(requested_top, image_height - requested_side));
  requested_left &= ~1;
  if (requested_left + requested_side > image_width) {
    requested_left = (image_width - requested_side) & ~1;
  }
  requested_left = std::max(0, requested_left);

  *left = requested_left;
  *top = requested_top;
  *side = requested_side;
  return true;
}

int A1FaceRecognizer::CropYuv422Square(ssne_tensor_t source, int left, int top,
                                       int side, ssne_tensor_t* destination) {
  if (!destination || !IsValidTensor(source) ||
      get_data_format(source) != SSNE_YUV422_16 || left < 0 || top < 0 ||
      side <= 0 || (left & 1) || (side & 1) ||
      left + side > static_cast<int>(get_width(source)) ||
      top + side > static_cast<int>(get_height(source))) {
    return -1;
  }

  ssne_tensor_t crop = create_tensor(static_cast<uint32_t>(side),
                                     static_cast<uint32_t>(side), SSNE_YUV422_16,
                                     SSNE_BUF_AI);
  if (!IsValidTensor(crop)) {
    return -2;
  }
  const uint8_t* source_data = static_cast<const uint8_t*>(get_data(source));
  uint8_t* crop_data = static_cast<uint8_t*>(get_data(crop));
  const size_t source_stride = get_mem_size(source) / get_height(source);
  const size_t crop_stride = get_mem_size(crop) / get_height(crop);
  const size_t row_bytes = static_cast<size_t>(side) * 2U;
  if (source_stride < static_cast<size_t>(get_width(source)) * 2U ||
      crop_stride < row_bytes) {
    release_tensor(crop);
    return -3;
  }

  for (int y = 0; y < side; ++y) {
    std::memcpy(crop_data + static_cast<size_t>(y) * crop_stride,
                source_data + static_cast<size_t>(top + y) * source_stride +
                    static_cast<size_t>(left) * 2U,
                row_bytes);
  }
  *destination = crop;
  return 0;
}

int A1FaceRecognizer::FillModelInput(const uint8_t* rgb112) {
  if (!rgb112 || !model_input_created_ || !IsValidTensor(model_input_tensor_)) {
    return -1;
  }
  void* data = get_data(model_input_tensor_);
  const size_t memory_size = get_mem_size(model_input_tensor_);

  if (model_input_dtype_ == SSNE_INT8) {
    if (memory_size < static_cast<size_t>(kRgbBytes)) {
      return -2;
    }
    int8_t* output = static_cast<int8_t*>(data);
    for (int i = 0; i < kRgbBytes; ++i) {
      output[i] = static_cast<int8_t>(static_cast<int>(rgb112[i]) - 128);
    }
    return 0;
  }
  if (model_input_dtype_ == SSNE_FLOAT32) {
    if (memory_size < static_cast<size_t>(kRgbBytes) * sizeof(float)) {
      return -3;
    }
    float* output = static_cast<float*>(data);
    for (int i = 0; i < kRgbBytes; ++i) {
      output[i] = (static_cast<float>(rgb112[i]) - 127.5f) / 127.5f;
    }
    return 0;
  }
  if (model_input_dtype_ == SSNE_UINT8) {
    if (memory_size < static_cast<size_t>(kRgbBytes)) {
      return -4;
    }
    std::memcpy(data, rgb112, kRgbBytes);
    return 0;
  }
  return -5;
}

int A1FaceRecognizer::InferEmbedding(const uint8_t* rgb112, float* embedding) {
  if (!initialized_ || !embedding) {
    return -1;
  }
  const int input_status = FillModelInput(rgb112);
  if (input_status != 0) {
    return -10 + input_status;
  }
  const int inference_status = ssne_inference(model_id_, 1, &model_input_tensor_);
  if (inference_status != 0) {
    return -20;
  }
  if (output_ready_) {
    release_tensor(output_tensor_);
    output_tensor_ = ssne_tensor_t{};
    output_ready_ = false;
  }
  const int output_status = ssne_getoutput(model_id_, 1, &output_tensor_);
  if (output_status != 0) {
    return -21;
  }
  output_ready_ = true;

  if (!IsValidTensor(output_tensor_) ||
      get_total_size(output_tensor_) != kEmbeddingDimension) {
    return -22;
  }
  const uint8_t output_dtype = get_data_type(output_tensor_);
  const void* source = get_data(output_tensor_);
  if (output_dtype == SSNE_FLOAT32) {
    std::memcpy(embedding, source, sizeof(float) * kEmbeddingDimension);
  } else if (output_dtype == SSNE_INT8) {
    const int8_t* quantized = static_cast<const int8_t*>(source);
    for (int i = 0; i < kEmbeddingDimension; ++i) {
      embedding[i] =
          static_cast<float>(quantized[i]) * config_.int8_output_scale;
    }
  } else {
    return -23;
  }
  return L2Normalize(embedding, kEmbeddingDimension) ? 0 : -24;
}

int A1FaceRecognizer::RecognizeAlignedRgb112(
    const uint8_t* rgb112, RecognitionResult* result) {
  if (!result) {
    return -1;
  }
  *result = RecognitionResult{};
  if (!initialized_ || !rgb112) {
    InvalidateResult(-2, result);
    return -2;
  }

  float embedding[kEmbeddingDimension] = {};
  const int status = InferEmbedding(rgb112, embedding);
  if (status != 0) {
    InvalidateResult(status, result);
    return status;
  }

  const float score =
      Clamp(DotProduct(embedding, owner_template_, kEmbeddingDimension), -1.0f,
            1.0f);
  const VoteResult vote_result = vote_.Update(score, true);
  result->status = 0;
  result->similarity = score;
  result->valid_face = true;
  result->instant_owner = vote_result.instant_owner;
  result->confirmed_owner = vote_result.confirmed_owner;
  result->pass_count = vote_result.pass_count;
  result->sample_count = vote_result.sample_count;
  return 0;
}

int A1FaceRecognizer::RecognizeYuv422(ssne_tensor_t frame, const Box& face,
                                      const Point landmarks[5],
                                      RecognitionResult* result) {
  if (!result) {
    return -1;
  }
  *result = RecognitionResult{};
  if (!initialized_ || !IsValidTensor(frame) || !landmarks) {
    InvalidateResult(-2, result);
    return -2;
  }

  const int frame_width = static_cast<int>(get_width(frame));
  const int frame_height = static_cast<int>(get_height(frame));
  if (!ValidateFace(face, landmarks, frame_width, frame_height,
                    config_.quality)) {
    InvalidateResult(-3, result);
    return -3;
  }

  int left = 0;
  int top = 0;
  int side = 0;
  if (!BuildSquareRoi(face, landmarks, frame_width, frame_height, &left, &top,
                      &side)) {
    InvalidateResult(-4, result);
    return -4;
  }

  ssne_tensor_t yuv_roi = {};
  const int crop_status = CropYuv422Square(frame, left, top, side, &yuv_roi);
  if (crop_status != 0) {
    InvalidateResult(-10 + crop_status, result);
    return -10 + crop_status;
  }

  const int preprocess_status =
      RunAiPreprocessPipe(roi_pipe_, yuv_roi, roi_rgb_tensor_);
  release_tensor(yuv_roi);
  if (preprocess_status != 0) {
    InvalidateResult(-20, result);
    return -20;
  }

  const float roi_scale =
      static_cast<float>(config_.intermediate_roi_size) /
      static_cast<float>(side);
  Point roi_landmarks[5] = {};
  for (int i = 0; i < 5; ++i) {
    roi_landmarks[i].x = (landmarks[i].x - static_cast<float>(left)) * roi_scale;
    roi_landmarks[i].y = (landmarks[i].y - static_cast<float>(top)) * roi_scale;
  }
  const int roi_stride =
      static_cast<int>(get_mem_size(roi_rgb_tensor_) / get_height(roi_rgb_tensor_));
  if (!AlignFaceRgb112(static_cast<const uint8_t*>(get_data(roi_rgb_tensor_)),
                       config_.intermediate_roi_size,
                       config_.intermediate_roi_size, roi_stride,
                       roi_landmarks, aligned_rgb_)) {
    InvalidateResult(-21, result);
    return -21;
  }

  return RecognizeAlignedRgb112(aligned_rgb_, result);
}

}  // namespace a1face
