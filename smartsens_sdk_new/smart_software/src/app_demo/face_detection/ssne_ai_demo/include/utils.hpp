/*
 * @Filename: utils.hpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 */
#pragma once

#include "common.hpp"
#include "companion_mode.hpp"
#include "face_business.h"
#include "osd-device.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#ifdef SSNE_AI_DEMO_HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace utils {
void Merge(FaceDetectionResult* result, size_t low, size_t mid, size_t high);
void MergeSort(FaceDetectionResult* result, size_t low, size_t high);
void SortDetectionResult(FaceDetectionResult* result);
void NMS(FaceDetectionResult* result, float iou_threshold, int top_k);
}  // namespace utils

class VISUALIZER {
public:
    static const int DETECTION_LAYER_ID = 0;

    void Initialize(std::array<int, 2>& in_img_shape,
                    const std::string& bitmap_lut_path = "");
    void Release();

    void Draw();
    void Draw(const std::vector<std::array<float, 4>>& boxes);
    void Draw(const std::vector<ObjectDetection>& detections);
    void Draw(const FaceResult& face_result);
    void Draw(const std::vector<PoseDetection>& detections,
              float kpt_conf_threshold = 0.5f);
    void Draw(const std::vector<ObjectDetection>& detections,
              const std::vector<PoseDetection>& poses,
              float kpt_conf_threshold = 0.5f);

#ifdef SSNE_AI_DEMO_HAS_OPENCV
    void DrawPose(cv::Mat& image,
                  const std::vector<PoseDetection>& detections,
                  float kpt_conf_threshold = 0.5f);
#endif

    void DrawFixedSquare(int x_min,
                         int y_min,
                         int x_max,
                         int y_max,
                         int layer_id = 1);
    void DrawBitmap(const std::string& bitmap_path,
                    const std::string& lut_path = "",
                    int pos_x = 0,
                    int pos_y = 0,
                    int layer_id = 2,
                    bool flush = true);
    void DrawSnakeGame(const SnakeRenderData& game);
    void ClearLayer(int layer_id);

private:
    sst::device::osd::OsdDevice osd_device;
    int m_width = 0;
    int m_height = 0;
    std::string m_bitmap_lut_path_full;
};
