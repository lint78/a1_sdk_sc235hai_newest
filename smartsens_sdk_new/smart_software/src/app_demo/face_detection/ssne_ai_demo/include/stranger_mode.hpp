#pragma once

#include <array>
#include <string>
#include <vector>
#include "a1_face_adapter.hpp"
#include "official_yunet160.hpp"
#include "smartsoc/ssne_api.h"

class StrangerFaceAdapterStub : public A1FaceAdapter {
 public:
  void Initialize(const std::string& model_path);
  void Release();
  bool IsInitialized() const;
  int infer_yuv_roi(ssne_tensor_t* input,
                    int left,
                    int top,
                    int size,
                    FaceRawTensor outputs[FACE_OUTPUT_COUNT]);
  int infer_rgb224(const uint8_t rgb224[FACE_NET_W * FACE_NET_H * 3],
                   FaceRawTensor outputs[FACE_OUTPUT_COUNT]) override;

 private:
  std::string model_path_;
  bool initialized_ = false;
  uint16_t model_id_ = 0;
  uint8_t model_alloc_flag_ = SSNE_STATIC_ALLOC;
  ssne_tensor_t input_tensor_ = {};
  ssne_tensor_t output_tensors_[FACE_OUTPUT_COUNT] = {};
  AiPreprocessPipe pipe_offline_ = AiPreprocessPipe{};
  bool pipe_initialized_ = false;
  bool input_created_ = false;
  bool output_ready_[FACE_OUTPUT_COUNT] = {};
  bool output_meta_logged_ = false;
  bool output_score_logged_ = false;
  bool input_debug_logged_ = false;
  int model_input_dtype_ = -1;
  int model_is_uint8_ = 0;
  std::array<std::vector<float>, FACE_OUTPUT_COUNT> chw_output_buffers_;
};

class StrangerModeRunner {
 public:
  StrangerModeRunner() = default;

  void Initialize(const std::string& model_path);
  void Release();
  bool IsInitialized() const;
  const std::string& ModelPath() const;
  const std::string& LastError() const;
  int ProcessFrame(ssne_tensor_t* img,
                   int crop_offset_x,
                   int img_width,
                   int img_height,
                   uint16_t frame_id,
                   FaceResult* result);

 private:
  StrangerFaceAdapterStub adapter_;
  bool initialized_ = false;
  std::string model_path_;
  std::string generic_face_model_path_;
  std::string last_error_;
  bool warned_process_placeholder_ = false;
  bool generic_face_ready_ = false;
  uint16_t last_second_stage_frame_id_ = 0;
  bool has_last_second_stage_frame_ = false;
  bool last_second_stage_ok_ = false;
  FaceResult last_second_stage_result_ = {};
  float last_second_stage_cls_best_ = 0.0f;
  float last_second_stage_obj_best_ = 0.0f;
  float last_second_stage_fused_best_ = 0.0f;
  OfficialYuNet160FaceDetector generic_face_detector_;
};
