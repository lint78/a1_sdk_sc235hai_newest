#include "stranger_mode.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "log.hpp"
#include "model_quant_config.h"

namespace {

constexpr const char* kDefaultGenericFaceModelPath =
    "/app_demo/app_assets/models/yunet_160x120.m1model";
constexpr float kVoteSimilarityThreshold = 0.26f;
constexpr int kVoteWindowSize = 5;
constexpr int kVoteRequiredPasses = 3;
constexpr float kMinimumDetectionScore = 0.60f;
constexpr float kMinimumFaceSidePixels = 64.0f;
constexpr float kMinimumEyeDistancePixels = 16.0f;
constexpr float kMaximumAbsoluteRollDegrees = 35.0f;

bool IsValidTensor(ssne_tensor_t tensor) {
  return get_data(tensor) != nullptr && get_width(tensor) > 0 &&
         get_height(tensor) > 0 && get_mem_size(tensor) > 0;
}

float ClampFloat(float value, float low, float high) {
  return std::max(low, std::min(value, high));
}

float FaceArea(const OfficialYuNet160Detection& det) {
  return std::max(0.0f, det.box[2] - det.box[0]) *
         std::max(0.0f, det.box[3] - det.box[1]);
}

const char* RecognizeStatusText(int status) {
  switch (status) {
    case 0:
      return "recognition_ok";
    case -2:
      return "recognizer_input_invalid";
    case -3:
      return "face_quality_gate_rejected";
    case -4:
      return "roi_build_failed";
    case -11:
    case -12:
    case -13:
      return "yuv_roi_crop_failed";
    case -20:
      return "roi_preprocess_failed";
    case -21:
      return "embedding_inference_failed";
    case -22:
      return "embedding_output_invalid";
    case -23:
      return "embedding_output_dtype_unsupported";
    case -24:
      return "embedding_l2_invalid";
    default:
      return "recognition_failed";
  }
}

}  // namespace

void StrangerModeRunner::Initialize(const std::string& model_path) {
  Release();
  model_path_ = model_path;
  generic_face_model_path_ = kDefaultGenericFaceModelPath;
  owner_template_path_ =
      std::string("/app_demo/app_assets/models/") +
      kStrangerFaceOwnerTemplateName;

  a1face::RecognizerConfig config;
  config.vote.similarity_threshold = kVoteSimilarityThreshold;
  config.vote.window_size = kVoteWindowSize;
  config.vote.required_passes = kVoteRequiredPasses;
  config.quality.minimum_detection_score = kMinimumDetectionScore;
  config.quality.minimum_face_side_pixels = kMinimumFaceSidePixels;
  config.quality.minimum_eye_distance_pixels = kMinimumEyeDistancePixels;
  config.quality.maximum_absolute_roll_degrees = kMaximumAbsoluteRollDegrees;

  LOG_INFO("stranger mode setup begin detector=%s recognizer=%s owner=%s\n",
           generic_face_model_path_.c_str(), model_path_.c_str(),
           owner_template_path_.c_str());
  const int init_ret =
      recognizer_.Initialize(model_path_.c_str(), owner_template_path_.c_str(),
                             config);
  if (init_ret != 0) {
    last_error_ = "face recognizer init failed";
    LOG_ERROR("stranger mode recognizer init failed ret=%d model=%s owner=%s\n",
              init_ret, model_path_.c_str(), owner_template_path_.c_str());
  } else {
    LOG_INFO("stranger mode recognizer ready model=%s owner=%s\n",
             model_path_.c_str(), owner_template_path_.c_str());
  }

  initialized_ = true;
}

void StrangerModeRunner::Release() {
  recognizer_.Release();
  generic_face_detector_.Release();
  initialized_ = false;
  model_path_.clear();
  owner_template_path_.clear();
  generic_face_model_path_.clear();
  last_error_.clear();
  generic_face_ready_ = false;
  last_similarity_ = 0.0f;
  last_recognize_status_ = 0;
  last_instant_owner_ = false;
  last_confirmed_owner_ = false;
  last_vote_pass_count_ = 0;
  last_vote_sample_count_ = 0;
  last_primary_face_index_ = -1;
}

bool StrangerModeRunner::IsInitialized() const { return initialized_; }

const std::string& StrangerModeRunner::ModelPath() const { return model_path_; }

const std::string& StrangerModeRunner::LastError() const { return last_error_; }

float StrangerModeRunner::LastSimilarity() const { return last_similarity_; }

int StrangerModeRunner::LastRecognizeStatus() const { return last_recognize_status_; }

bool StrangerModeRunner::LastInstantOwner() const { return last_instant_owner_; }

bool StrangerModeRunner::LastConfirmedOwner() const { return last_confirmed_owner_; }

int StrangerModeRunner::LastVotePassCount() const { return last_vote_pass_count_; }

int StrangerModeRunner::LastVoteSampleCount() const { return last_vote_sample_count_; }

int StrangerModeRunner::ProcessFrame(ssne_tensor_t* img, int crop_offset_x,
                                     int img_width, int img_height,
                                     uint16_t frame_id, FaceResult* result) {
  if (result != nullptr) {
    *result = FaceResult{};
    result->frame_id = frame_id;
  }
  last_similarity_ = 0.0f;
  last_recognize_status_ = 0;
  last_instant_owner_ = false;
  last_confirmed_owner_ = false;
  last_vote_pass_count_ = 0;
  last_vote_sample_count_ = 0;
  last_primary_face_index_ = -1;

  if (!initialized_) {
    last_error_ = "stranger mode runtime not initialized";
    return -1;
  }
  if (img == nullptr || !IsValidTensor(*img)) {
    last_error_ = "stranger mode got invalid image tensor";
    recognizer_.ResetVoting();
    return -2;
  }
  if (result == nullptr) {
    last_error_ = "stranger mode got null result";
    recognizer_.ResetVoting();
    return -3;
  }
  if (!recognizer_.IsInitialized()) {
    last_error_ = "face recognizer is not initialized";
    recognizer_.ResetVoting();
    return -4;
  }

  if (!generic_face_ready_) {
    const std::array<int, 2> crop_shape = {1080, 1080};
    const std::array<int, 2> generic_det_shape = {160, 120};
    LOG_INFO("stranger mode lazy init generic detector model=%s\n",
             generic_face_model_path_.c_str());
    generic_face_detector_.Initialize(generic_face_model_path_, crop_shape,
                                      generic_det_shape);
    generic_face_ready_ = generic_face_detector_.IsInitialized();
    if (!generic_face_ready_) {
      last_error_ = "generic yunet face detector init failed";
      recognizer_.ResetVoting();
      return -5;
    }
    LOG_INFO("stranger mode generic detector ready\n");
  }

  std::vector<OfficialYuNet160Detection> faces;
  if (!generic_face_detector_.Predict(img, &faces, &last_error_)) {
    recognizer_.ResetVoting();
    return -6;
  }

  const int final_count =
      std::min(static_cast<int>(faces.size()), FACE_MAX_DETECTIONS);
  result->count = final_count;
  for (int i = 0; i < final_count; ++i) {
    const OfficialYuNet160Detection& face = faces[static_cast<size_t>(i)];
    FaceDetection det;
    det.x = ClampFloat(face.box[0] + static_cast<float>(crop_offset_x), 0.0f,
                       static_cast<float>(img_width));
    det.y = ClampFloat(face.box[1], 0.0f, static_cast<float>(img_height));
    const float x2 = ClampFloat(face.box[2] + static_cast<float>(crop_offset_x),
                                0.0f, static_cast<float>(img_width));
    const float y2 = ClampFloat(face.box[3], 0.0f,
                                static_cast<float>(img_height));
    det.w = std::max(0.0f, x2 - det.x);
    det.h = std::max(0.0f, y2 - det.y);
    det.score = face.score;
    det.identity = FaceIdentity::kUnknown;
    for (int p = 0; p < 5; ++p) {
      det.landmarks[p].x = ClampFloat(
          face.landmarks[static_cast<size_t>(p) * 2U] +
              static_cast<float>(crop_offset_x),
          0.0f, static_cast<float>(img_width));
      det.landmarks[p].y = ClampFloat(
          face.landmarks[static_cast<size_t>(p) * 2U + 1U], 0.0f,
          static_cast<float>(img_height));
    }
    result->faces[i] = det;
  }

  if (final_count <= 0) {
    recognizer_.ResetVoting();
    last_error_ = "generic face detector active; no face detected";
    return 1;
  }

  int best_index = 0;
  float best_area = FaceArea(faces[0]);
  for (int i = 1; i < final_count; ++i) {
    const float area = FaceArea(faces[static_cast<size_t>(i)]);
    if (area > best_area) {
      best_area = area;
      best_index = i;
    }
  }
  last_primary_face_index_ = best_index;

  const OfficialYuNet160Detection& primary = faces[static_cast<size_t>(best_index)];
  a1face::Box face_box;
  face_box.x = primary.box[0];
  face_box.y = primary.box[1];
  face_box.width = std::max(0.0f, primary.box[2] - primary.box[0]);
  face_box.height = std::max(0.0f, primary.box[3] - primary.box[1]);
  face_box.detection_score = primary.score;

  a1face::Point landmarks[5];
  for (int p = 0; p < 5; ++p) {
    landmarks[p].x = primary.landmarks[static_cast<size_t>(p) * 2U];
    landmarks[p].y = primary.landmarks[static_cast<size_t>(p) * 2U + 1U];
  }

  a1face::RecognitionResult recognition;
  const int recognize_ret =
      recognizer_.RecognizeYuv422(*img, face_box, landmarks, &recognition);
  last_recognize_status_ = recognize_ret;
  last_similarity_ = recognition.similarity;
  last_instant_owner_ = recognition.instant_owner;
  last_confirmed_owner_ = recognition.confirmed_owner;
  last_vote_pass_count_ = recognition.pass_count;
  last_vote_sample_count_ = recognition.sample_count;

  FaceDetection& primary_output = result->faces[best_index];
  if (recognize_ret != 0) {
    primary_output.identity = FaceIdentity::kUnknown;
    primary_output.score = primary.score;
    last_error_ = std::string("generic face detector active; ") +
                  RecognizeStatusText(recognize_ret);
    LOG_WARN(
        "stranger mode recognize failed ret=%d score=%.3f face=[x=%.1f y=%.1f "
        "w=%.1f h=%.1f]\n",
        recognize_ret, primary.score, face_box.x, face_box.y, face_box.width,
        face_box.height);
    return 1;
  }

  primary_output.score =
      std::max(primary.score, std::max(0.0f, recognition.similarity));
  if (recognition.confirmed_owner) {
    primary_output.identity = FaceIdentity::kKnown;
    last_error_ =
        "generic face detector active; owner confirmed by embedding vote";
  } else if (recognition.instant_owner) {
    primary_output.identity = FaceIdentity::kUnknown;
    last_error_ =
        "generic face detector active; owner score passed but vote is warming up";
  } else {
    primary_output.identity = FaceIdentity::kStranger;
    last_error_ =
        "generic face detector active; embedding did not match owner template";
  }

  LOG_INFO(
      "stranger face[%d] det=%.3f sim=%.3f instant=%d confirmed=%d vote=%d/%d "
      "status=%s\n",
      best_index, primary.score, recognition.similarity,
      recognition.instant_owner ? 1 : 0,
      recognition.confirmed_owner ? 1 : 0, recognition.pass_count,
      recognition.sample_count, last_error_.c_str());
  return 1;
}
