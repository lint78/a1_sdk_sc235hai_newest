#pragma once

#include "face_business.h"

class A1FaceAdapter {
 public:
  virtual ~A1FaceAdapter() = default;
  virtual int infer_rgb224(const uint8_t rgb224[FACE_NET_W * FACE_NET_H * 3],
                           FaceRawTensor outputs[FACE_OUTPUT_COUNT]) = 0;
};

class FacePipeline {
 public:
  explicit FacePipeline(A1FaceAdapter& adapter, FaceConfig cfg = {})
      : adapter_(adapter), cfg_(cfg) {}

  int process_rgb(const uint8_t* rgb,
                  int width,
                  int height,
                  int stride,
                  uint16_t frame_id,
                  FaceResult* result);

 private:
  A1FaceAdapter& adapter_;
  FaceConfig cfg_;
  uint8_t input_[FACE_NET_W * FACE_NET_H * 3] = {};
};
