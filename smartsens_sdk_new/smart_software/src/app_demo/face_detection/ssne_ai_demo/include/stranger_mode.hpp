#pragma once

#include <string>
#include "a1_face_recognizer.h"
#include "face_business.h"
#include "official_yunet160.hpp"
#include "smartsoc/ssne_api.h"

class StrangerModeRunner {
 public:
  StrangerModeRunner() = default;

  void Initialize(const std::string& model_path);
  void Release();
  bool IsInitialized() const;
  const std::string& ModelPath() const;
  const std::string& LastError() const;
  float LastSimilarity() const;
  int LastRecognizeStatus() const;
  bool LastInstantOwner() const;
  bool LastConfirmedOwner() const;
  int LastVotePassCount() const;
  int LastVoteSampleCount() const;
  int ProcessFrame(ssne_tensor_t* img,
                   int crop_offset_x,
                   int img_width,
                   int img_height,
                   uint16_t frame_id,
                   FaceResult* result);

 private:
  bool initialized_ = false;
  std::string model_path_;
  std::string owner_template_path_;
  std::string generic_face_model_path_;
  std::string last_error_;
  bool generic_face_ready_ = false;
  float last_similarity_ = 0.0f;
  int last_recognize_status_ = 0;
  bool last_instant_owner_ = false;
  bool last_confirmed_owner_ = false;
  int last_vote_pass_count_ = 0;
  int last_vote_sample_count_ = 0;
  int last_primary_face_index_ = -1;
  a1face::A1FaceRecognizer recognizer_;
  OfficialYuNet160FaceDetector generic_face_detector_;
};
