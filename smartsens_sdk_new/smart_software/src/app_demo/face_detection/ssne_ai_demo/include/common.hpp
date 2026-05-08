/*
 * @Filename: common.hpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 */
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include "smartsoc/ssne_api.h"

struct PoseKeyPoint {
  float x = 0.0f;
  float y = 0.0f;
  float conf = 0.0f;
};

struct PoseDetection {
  std::array<float, 4> box = {0.0f, 0.0f, 0.0f, 0.0f};
  float score = 0.0f;
  int class_id = 0;
  std::array<PoseKeyPoint, 17> keypoints;
};

struct ObjectDetection {
  std::array<float, 4> box = {0.0f, 0.0f, 0.0f, 0.0f};
  float score = 0.0f;
  int class_id = -1;
};

struct FaceDetectionResult {
  std::vector<PoseDetection> detections;
  std::vector<std::array<float, 4>> boxes;
  std::vector<std::array<PoseKeyPoint, 17>> keypoints;
  std::vector<float> scores;
  std::vector<int> class_ids;

  FaceDetectionResult() = default;
  FaceDetectionResult(const FaceDetectionResult& res);

  void Clear();
  void Free();
  void Reserve(int size);
  void Resize(int size);
};

struct ObjectDetectionResult {
  std::vector<ObjectDetection> detections;
  std::vector<std::array<float, 4>> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  ObjectDetectionResult() = default;
  ObjectDetectionResult(const ObjectDetectionResult& res);

  void Clear();
  void Free();
  void Reserve(int size);
  void Resize(int size);
};

class IMAGEPROCESSOR {
 public:
  void Initialize(std::array<int, 2>* in_img_shape);
  void GetImage(ssne_tensor_t* img_sensor);
  void Release();

  std::array<int, 2> img_shape;

 private:
  uint8_t format_online = SSNE_YUV422_16;
};

class YUNET {
 public:
  std::string ModelName() const { return "yolov8_pose"; }

  void Predict(ssne_tensor_t* img_in, FaceDetectionResult* result, float conf_threshold = 0.6f);
  void Initialize(std::string& model_path, std::array<int, 2>* in_img_shape,
                  std::array<int, 2>* in_det_shape, bool in_use_kps,
                  int in_box_len);
  void SetEnhanceFocusBox(const std::array<float, 4>* focus_box);

  float nms_threshold = 0.45f;
  int keep_top_k = 100;
  int top_k = 300;

  std::array<int, 2> img_shape = {0, 0};
  std::array<int, 2> det_shape = {0, 0};
  int box_len = 0;
  float w_scale = 1.0f;
  float h_scale = 1.0f;
  bool use_kps = true;

  std::vector<std::vector<int>> min_sizes;
  std::vector<int> steps;
  std::vector<float> variance;

  void Release();
  void saveImageBin(const void* data, int w, int h, const char* filename);
  void saveFloatBin(const float* data, int length, const char* filename);

 private:
  uint16_t model_id = 0;
  ssne_tensor_t inputs[1] = {};
  ssne_tensor_t outputs[9] = {};
  AiPreprocessPipe pipe_offline = GetAIPreprocessPipe();
  bool enhance_focus_valid = false;
  std::array<float, 4> enhance_focus_box = {0.0f, 0.0f, 0.0f, 0.0f};
};

class YOLOV8NANO {
 public:
  std::string ModelName() const { return "yolov8nano"; }

  void Predict(ssne_tensor_t* img_in, ObjectDetectionResult* result,
               float conf_threshold = 0.25f,
               float person_conf_threshold = -1.0f);
  void Initialize(std::string& model_path, std::array<int, 2>* in_img_shape,
                  std::array<int, 2>* in_det_shape, int in_box_len,
                  int in_num_classes = 7);

  float nms_threshold = 0.45f;
  int keep_top_k = 100;
  int top_k = 300;

  std::array<int, 2> img_shape = {0, 0};
  std::array<int, 2> det_shape = {0, 0};
  int box_len = 0;
  int num_classes = 7;
  float w_scale = 1.0f;
  float h_scale = 1.0f;

  void Release();

 private:
  uint16_t model_id = 0;
  ssne_tensor_t inputs[1] = {};
  ssne_tensor_t outputs[6] = {};
  AiPreprocessPipe pipe_offline = GetAIPreprocessPipe();
};
