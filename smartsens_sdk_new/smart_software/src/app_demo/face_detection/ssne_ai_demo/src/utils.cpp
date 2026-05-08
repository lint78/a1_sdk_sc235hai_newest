/*
 * @Filename: utils.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 */
#include "../include/utils.hpp"
#include "../include/log.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdio>

namespace utils {

/**
 * @brief 归并排序的合并操作
 * @param result 人脸/目标检测结果结构体指针
 * @param low 合并区间的起始索引
 * @param mid 合并区间的中间索引
 * @param high 合并区间的结束索引
 * @description 将两个已排序的子数组合并成一个有序数组，按照分数从高到低排序
 */
void Merge(FaceDetectionResult* result, size_t low, size_t mid, size_t high) {
  std::vector<PoseDetection>& detections = result->detections;
  std::vector<std::array<float, 4>>& boxes = result->boxes;
  std::vector<float>& scores = result->scores;
  std::vector<int>& class_ids = result->class_ids;
  std::vector<std::array<PoseKeyPoint, 17>>& keypoints = result->keypoints;

  std::vector<PoseDetection> temp_detections(detections);
  std::vector<std::array<float, 4>> temp_boxes(boxes);
  std::vector<float> temp_scores(scores);
  std::vector<int> temp_class_ids(class_ids);
  std::vector<std::array<PoseKeyPoint, 17>> temp_keypoints(keypoints);

  size_t i = low;
  size_t j = mid + 1;
  size_t k = i;

  for (; i <= mid && j <= high; k++) {
    if (temp_scores[i] >= temp_scores[j]) {
      detections[k] = temp_detections[i];
      scores[k] = temp_scores[i];
      boxes[k] = temp_boxes[i];
      class_ids[k] = temp_class_ids[i];
      keypoints[k] = temp_keypoints[i];
      i++;
    } else {
      detections[k] = temp_detections[j];
      scores[k] = temp_scores[j];
      boxes[k] = temp_boxes[j];
      class_ids[k] = temp_class_ids[j];
      keypoints[k] = temp_keypoints[j];
      j++;
    }
  }

  while (i <= mid) {
    detections[k] = temp_detections[i];
    scores[k] = temp_scores[i];
    boxes[k] = temp_boxes[i];
    class_ids[k] = temp_class_ids[i];
    keypoints[k] = temp_keypoints[i];
    k++;
    i++;
  }

  while (j <= high) {
    detections[k] = temp_detections[j];
    scores[k] = temp_scores[j];
    boxes[k] = temp_boxes[j];
    class_ids[k] = temp_class_ids[j];
    keypoints[k] = temp_keypoints[j];
    k++;
    j++;
  }
}

/**
 * @brief 归并排序递归函数
 * @param result 检测结果结构体指针
 * @param low 排序区间的起始索引
 * @param high 排序区间的结束索引
 */
void MergeSort(FaceDetectionResult* result, size_t low, size_t high) {
  if (low < high) {
    size_t mid = (high - low) / 2 + low;
    MergeSort(result, low, mid);
    MergeSort(result, mid + 1, high);
    Merge(result, low, mid, high);
  }
}

/**
 * @brief 对检测结果进行排序
 * @param result 检测结果结构体指针
 * @description 按照检测分数从高到低对检测结果进行排序
 */
void SortDetectionResult(FaceDetectionResult* result) {
  size_t low = 0;
  size_t high = result->scores.size();
  if (high == 0) {
    return;
  }
  high = high - 1;
  MergeSort(result, low, high);
}

/**
 * @brief 非极大值抑制（NMS）算法
 * @param result 检测结果结构体指针
 * @param iou_threshold IoU阈值
 * @param top_k 保留前k个检测结果
 *
 * @note
 * 这版保留了 class_ids 的同步处理。
 * 这个函数即使暂时不用，但是保留可编译状态。
 */
void NMS(FaceDetectionResult* result, float iou_threshold, int top_k) {
  SortDetectionResult(result);

  int res_count = static_cast<int>(result->boxes.size());
  result->Resize(std::min(res_count, top_k));

  std::vector<float> area_of_boxes(result->boxes.size());
  std::vector<int> suppressed(result->boxes.size(), 0);

  for (size_t i = 0; i < result->boxes.size(); ++i) {
    area_of_boxes[i] = (result->boxes[i][2] - result->boxes[i][0] + 1) *
                       (result->boxes[i][3] - result->boxes[i][1] + 1);
  }

  for (size_t i = 0; i < result->boxes.size(); ++i) {
    if (suppressed[i] == 1) {
      continue;
    }
    for (size_t j = i + 1; j < result->boxes.size(); ++j) {
      if (suppressed[j] == 1) {
        continue;
      }

      // 同类 NMS：只有类别相同才比较抑制
      if (result->class_ids[i] != result->class_ids[j]) {
        continue;
      }

      float xmin = std::max(result->boxes[i][0], result->boxes[j][0]);
      float ymin = std::max(result->boxes[i][1], result->boxes[j][1]);
      float xmax = std::min(result->boxes[i][2], result->boxes[j][2]);
      float ymax = std::min(result->boxes[i][3], result->boxes[j][3]);
      float overlap_w = std::max(0.0f, xmax - xmin + 1);
      float overlap_h = std::max(0.0f, ymax - ymin + 1);
      float overlap_area = overlap_w * overlap_h;

      float overlap_ratio =
          overlap_area / (area_of_boxes[i] + area_of_boxes[j] - overlap_area);

      if (overlap_ratio > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }

  FaceDetectionResult backup(*result);

  result->Clear();
  result->Reserve(suppressed.size());

  for (size_t i = 0; i < suppressed.size(); ++i) {
    if (suppressed[i] == 1) {
      continue;
    }

    result->detections.emplace_back(backup.detections[i]);
    result->boxes.emplace_back(backup.boxes[i]);
    result->scores.push_back(backup.scores[i]);
    result->class_ids.push_back(backup.class_ids[i]);
    result->keypoints.emplace_back(backup.keypoints[i]);
  }
}

}  // namespace utils


/**
 * @brief 释放 FaceDetectionResult 的内存
 * @description 使用 swap 技巧释放 vector 占用的内存
 */
void FaceDetectionResult::Free() {
  std::vector<PoseDetection>().swap(detections);
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<int>().swap(class_ids);
  std::vector<std::array<PoseKeyPoint, 17>>().swap(keypoints);
}

/**
 * @brief 清空 FaceDetectionResult 的内容
 * @description 清空所有检测框、分数、类别和关键点，但保留内存分配
 */
void FaceDetectionResult::Clear() {
  detections.clear();
  boxes.clear();
  scores.clear();
  class_ids.clear();
  keypoints.clear();
}

/**
 * @brief 预分配内存空间
 * @param size 要保留的元素数量
 */
void FaceDetectionResult::Reserve(int size) {
  detections.reserve(size);
  boxes.reserve(size);
  scores.reserve(size);
  class_ids.reserve(size);
  keypoints.reserve(size);
}

/**
 * @brief 调整 FaceDetectionResult 的大小
 * @param size 新的元素数量
 */
void FaceDetectionResult::Resize(int size) {
  detections.resize(size);
  boxes.resize(size);
  scores.resize(size);
  class_ids.resize(size);
  keypoints.resize(size);
}

/**
 * @brief FaceDetectionResult 的拷贝构造函数
 * @param res 要拷贝的 FaceDetectionResult 对象
 */
FaceDetectionResult::FaceDetectionResult(const FaceDetectionResult& res) {
  detections.assign(res.detections.begin(), res.detections.end());
  boxes.assign(res.boxes.begin(), res.boxes.end());
  keypoints.assign(res.keypoints.begin(), res.keypoints.end());
  scores.assign(res.scores.begin(), res.scores.end());
  class_ids.assign(res.class_ids.begin(), res.class_ids.end());
}

void ObjectDetectionResult::Free() {
  std::vector<ObjectDetection>().swap(detections);
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<int>().swap(class_ids);
}

void ObjectDetectionResult::Clear() {
  detections.clear();
  boxes.clear();
  scores.clear();
  class_ids.clear();
}

void ObjectDetectionResult::Reserve(int size) {
  detections.reserve(size);
  boxes.reserve(size);
  scores.reserve(size);
  class_ids.reserve(size);
}

void ObjectDetectionResult::Resize(int size) {
  detections.resize(size);
  boxes.resize(size);
  scores.resize(size);
  class_ids.resize(size);
}

ObjectDetectionResult::ObjectDetectionResult(const ObjectDetectionResult& res) {
  detections.assign(res.detections.begin(), res.detections.end());
  boxes.assign(res.boxes.begin(), res.boxes.end());
  scores.assign(res.scores.begin(), res.scores.end());
  class_ids.assign(res.class_ids.begin(), res.class_ids.end());
}

namespace {

constexpr int kBoxColor = 1;
constexpr int kLeftColor = 2;
constexpr int kRightColor = 3;
constexpr int kCenterColor = 4;
constexpr int kPointColor = 5;
constexpr int kWarnColor = kLeftColor;
constexpr int kAlertColor = kWarnColor;
constexpr float kSkeletonLineThickness = 2.0f;
constexpr int kClassCat = 0;
constexpr int kClassDog = 1;
constexpr int kClassSnake = 2;
constexpr int kClassMouse = 3;
constexpr int kClassFire = 5;
constexpr float kGreenScoreThreshold = 0.50f;

bool IsLeftPointIndex(int idx) {
    switch (idx) {
        case 1: case 3: case 5: case 7: case 9: case 11: case 13: case 15:
            return true;
        default:
            return false;
    }
}

bool IsRightPointIndex(int idx) {
    switch (idx) {
        case 2: case 4: case 6: case 8: case 10: case 12: case 14: case 16:
            return true;
        default:
            return false;
    }
}

int GetSkeletonColorIndex(int a, int b) {
    if (IsLeftPointIndex(a) && IsLeftPointIndex(b)) {
        return kLeftColor;
    }
    if (IsRightPointIndex(a) && IsRightPointIndex(b)) {
        return kRightColor;
    }
    return kCenterColor;
}

bool IsAlertClass(int class_id) {
    // 只有蛇、鼠和火焰属于业务告警；猫狗只画普通检测框，不显示左上角叹号。
    return class_id == kClassMouse ||
           class_id == kClassSnake ||
           class_id == kClassFire;
}

int GetDetectionColorIndex(const ObjectDetection& det) {
    // 告警类固定红框；普通目标按置信度区分绿框和黄框。
    if (IsAlertClass(det.class_id)) {
        return kAlertColor;
    }
    return det.score >= kGreenScoreThreshold ? kBoxColor : kWarnColor;
}

fdevice::COVER_ATTR_S MakeSolidRectCover(float x1, float y1, float x2, float y2, int color) {
    fdevice::COVER_ATTR_S cover = {};

    const int left = static_cast<int>(std::floor(std::min(x1, x2)));
    const int right = static_cast<int>(std::ceil(std::max(x1, x2)));
    const int top = static_cast<int>(std::floor(std::min(y1, y2)));
    const int bottom = static_cast<int>(std::ceil(std::max(y1, y2)));

    cover.colorIdx = color;
    cover.eSolid = fdevice::TYPE_SOLID;
    cover.alpha = fdevice::TYPE_ALPHA100;

    cover.vertex_out.points[0] = {left, top};
    cover.vertex_out.points[1] = {left, bottom};
    cover.vertex_out.points[2] = {right, bottom};
    cover.vertex_out.points[3] = {right, top};
    cover.vertex_in = cover.vertex_out;
    return cover;
}

fdevice::COVER_ATTR_S MakeHollowBoxCover(const std::array<float, 4>& box, int border, int color) {
    fdevice::COVER_ATTR_S cover = {};

    const int x1 = static_cast<int>(box[0]);
    const int y1 = static_cast<int>(box[1]);
    const int x2 = static_cast<int>(box[2]);
    const int y2 = static_cast<int>(box[3]);

    cover.colorIdx = color;
    cover.eSolid = fdevice::TYPE_HOLLOW;
    cover.alpha = fdevice::TYPE_ALPHA75;

    cover.vertex_in.points[0] = {x1 + border, y1 + border};
    cover.vertex_in.points[1] = {x1 + border, y2 - border};
    cover.vertex_in.points[2] = {x2 - border, y2 - border};
    cover.vertex_in.points[3] = {x2 - border, y1 + border};

    cover.vertex_out.points[0] = {x1 - border, y1 - border};
    cover.vertex_out.points[1] = {x1 - border, y2 + border};
    cover.vertex_out.points[2] = {x2 + border, y2 + border};
    cover.vertex_out.points[3] = {x2 + border, y1 - border};
    return cover;
}

fdevice::COVER_ATTR_S MakeLineCover(float x1, float y1, float x2, float y2, float thickness, int color);

fdevice::COVER_ATTR_S MakeLineCover(float x1, float y1, float x2, float y2, float thickness, int color) {
    fdevice::COVER_ATTR_S cover = {};

    const float dx = x2 - x1;
    const float dy = y2 - y1;
    const float len = std::sqrt(dx * dx + dy * dy);
    const float half = thickness * 0.5f;

    float nx = 0.0f;
    float ny = 0.0f;
    if (len > 1e-6f) {
        nx = -dy / len * half;
        ny = dx / len * half;
    }

    cover.colorIdx = color;
    cover.eSolid = fdevice::TYPE_SOLID;
    cover.alpha = fdevice::TYPE_ALPHA100;

    cover.vertex_out.points[0] = {static_cast<int>(std::round(x1 + nx)), static_cast<int>(std::round(y1 + ny))};
    cover.vertex_out.points[1] = {static_cast<int>(std::round(x1 - nx)), static_cast<int>(std::round(y1 - ny))};
    cover.vertex_out.points[2] = {static_cast<int>(std::round(x2 - nx)), static_cast<int>(std::round(y2 - ny))};
    cover.vertex_out.points[3] = {static_cast<int>(std::round(x2 + nx)), static_cast<int>(std::round(y2 + ny))};
    cover.vertex_in = cover.vertex_out;
    return cover;
}

void AppendWarningIconCovers(const std::array<float, 4>& box,
                             std::vector<fdevice::COVER_ATTR_S>* covers) {
    if (covers == nullptr) {
        return;
    }

    // 用简单 cover 拼出“!”图标，避免额外加载位图资源占用 OSD 图层和内存。
    const float left = std::max(0.0f, box[0] - 34.0f);
    const float top = std::max(0.0f, box[1] - 34.0f);
    const float size = 28.0f;
    const float cx = left + size * 0.5f;

    covers->emplace_back(MakeSolidRectCover(left, top, left + size, top + size, kAlertColor));
    covers->emplace_back(MakeSolidRectCover(cx - 3.0f, top + 5.0f, cx + 3.0f, top + 18.0f, kPointColor));
    covers->emplace_back(MakeSolidRectCover(cx - 3.0f, top + 21.0f, cx + 3.0f, top + 26.0f, kPointColor));
}

}  // namespace


/**
 * @brief OSD 可视化器初始化函数
 * @param in_img_shape 图像尺寸 [宽度, 高度]
 */
void VISUALIZER::Initialize(std::array<int, 2>& in_img_shape, const std::string& bitmap_lut_path) {
    if (bitmap_lut_path.empty()) {
        osd_device.Initialize(in_img_shape[0], in_img_shape[1], nullptr);
    } else {
        osd_device.Initialize(in_img_shape[0], in_img_shape[1], bitmap_lut_path.c_str());
    }
}


/**
 * @brief 绘制测试矩形框（用于测试 OSD 功能）
 */
void VISUALIZER::Draw() {
    std::vector<sst::device::osd::OsdQuadRangle> quad_rangle_vec;

    sst::device::osd::OsdQuadRangle q;

    q.color = 0;
    q.box = {100, 100, 200, 200};
    q.border = 3;
    q.alpha = fdevice::TYPE_ALPHA75;
    q.type = fdevice::TYPE_HOLLOW;
    quad_rangle_vec.emplace_back(q);

    osd_device.Draw(quad_rangle_vec);
}

/**
 * @brief 根据检测框绘制 OSD 矩形
 * @param boxes 检测框向量，每个元素为[xmin, ymin, xmax, ymax]
 */
void VISUALIZER::Draw(const std::vector<std::array<float, 4>>& boxes) {
    std::vector<sst::device::osd::OsdQuadRangle> quad_rangle_vec;

    for (size_t i = 0; i < boxes.size(); i++) {
        sst::device::osd::OsdQuadRangle q;

        int xmin = static_cast<int>(boxes[i][0]);
        int ymin = static_cast<int>(boxes[i][1]);
        int xmax = static_cast<int>(boxes[i][2]);
        int ymax = static_cast<int>(boxes[i][3]);

        q.box = {
            static_cast<float>(xmin),
            static_cast<float>(ymin),
            static_cast<float>(xmax),
            static_cast<float>(ymax)
        };
        q.color = 1;
        q.border = 3;
        q.alpha = fdevice::TYPE_ALPHA75;
        q.type = fdevice::TYPE_HOLLOW;

        quad_rangle_vec.emplace_back(q);
    }

    osd_device.Draw(quad_rangle_vec);
}

void VISUALIZER::Draw(const std::vector<ObjectDetection>& detections) {
    std::vector<fdevice::COVER_ATTR_S> covers;
    covers.reserve(detections.size() * 4);

    for (const auto& det : detections) {
        const int border = IsAlertClass(det.class_id) ? 5 : 3;
        covers.emplace_back(MakeHollowBoxCover(det.box, border, GetDetectionColorIndex(det)));
        if (IsAlertClass(det.class_id)) {
            AppendWarningIconCovers(det.box, &covers);
        }
    }

    osd_device.DrawCovers(covers, DETECTION_LAYER_ID);
}

void VISUALIZER::Draw(const std::vector<PoseDetection>& detections, float kpt_conf_threshold) {
    // COCO 17 点骨架连接表，画线前会按关键点置信度过滤，避免低置信节点把骨架糊成一团。
    static const std::array<std::array<int, 2>, 16> kSkeleton = {{
        {{0, 1}}, {{0, 2}}, {{1, 3}}, {{2, 4}},
        {{5, 6}}, {{5, 7}}, {{7, 9}}, {{6, 8}},
        {{8, 10}}, {{5, 11}}, {{6, 12}}, {{11, 12}},
        {{11, 13}}, {{13, 15}}, {{12, 14}}, {{14, 16}}
    }};

    std::vector<fdevice::COVER_ATTR_S> covers;
    covers.reserve(detections.size() * 40);

    for (const auto& det : detections) {
        covers.emplace_back(MakeHollowBoxCover(det.box, 3, kBoxColor));

        for (const auto& edge : kSkeleton) {
            const PoseKeyPoint& p1 = det.keypoints[edge[0]];
            const PoseKeyPoint& p2 = det.keypoints[edge[1]];
            if (p1.conf < kpt_conf_threshold || p2.conf < kpt_conf_threshold) {
                continue;
            }

            covers.emplace_back(
                MakeLineCover(p1.x, p1.y, p2.x, p2.y,
                              kSkeletonLineThickness,
                              GetSkeletonColorIndex(edge[0], edge[1])));
        }

        for (size_t i = 0; i < det.keypoints.size(); ++i) {
            const PoseKeyPoint& kp = det.keypoints[i];
            if (kp.conf < kpt_conf_threshold) {
                continue;
            }

            int point_color = kPointColor;
            if (IsLeftPointIndex(static_cast<int>(i))) {
                point_color = kLeftColor;
            } else if (IsRightPointIndex(static_cast<int>(i))) {
                point_color = kRightColor;
            }

            covers.emplace_back(MakeSolidRectCover(kp.x - 4.0f, kp.y - 4.0f,
                                                   kp.x + 4.0f, kp.y + 4.0f,
                                                   point_color));
        }
    }

    osd_device.DrawCovers(covers, DETECTION_LAYER_ID);
}

void VISUALIZER::Draw(const std::vector<ObjectDetection>& detections,
                      const std::vector<PoseDetection>& poses,
                      float kpt_conf_threshold) {
    static const std::array<std::array<int, 2>, 16> kSkeleton = {{
        {{0, 1}}, {{0, 2}}, {{1, 3}}, {{2, 4}},
        {{5, 6}}, {{5, 7}}, {{7, 9}}, {{6, 8}},
        {{8, 10}}, {{5, 11}}, {{6, 12}}, {{11, 12}},
        {{11, 13}}, {{13, 15}}, {{12, 14}}, {{14, 16}}
    }};

    std::vector<fdevice::COVER_ATTR_S> covers;
    covers.reserve(detections.size() + poses.size() * 40);

    for (const auto& det : detections) {
        const int border = IsAlertClass(det.class_id) ? 5 : 3;
        covers.emplace_back(MakeHollowBoxCover(det.box, border, GetDetectionColorIndex(det)));
        if (IsAlertClass(det.class_id)) {
            AppendWarningIconCovers(det.box, &covers);
        }
    }

    for (const auto& det : poses) {
        covers.emplace_back(MakeHollowBoxCover(det.box, 3, kBoxColor));

        for (const auto& edge : kSkeleton) {
            const PoseKeyPoint& p1 = det.keypoints[edge[0]];
            const PoseKeyPoint& p2 = det.keypoints[edge[1]];
            if (p1.conf < kpt_conf_threshold || p2.conf < kpt_conf_threshold) {
                continue;
            }

            covers.emplace_back(
                MakeLineCover(p1.x, p1.y, p2.x, p2.y,
                              kSkeletonLineThickness,
                              GetSkeletonColorIndex(edge[0], edge[1])));
        }

        for (size_t i = 0; i < det.keypoints.size(); ++i) {
            const PoseKeyPoint& kp = det.keypoints[i];
            if (kp.conf < kpt_conf_threshold) {
                continue;
            }

            int point_color = kPointColor;
            if (IsLeftPointIndex(static_cast<int>(i))) {
                point_color = kLeftColor;
            } else if (IsRightPointIndex(static_cast<int>(i))) {
                point_color = kRightColor;
            }

            covers.emplace_back(MakeSolidRectCover(kp.x - 4.0f, kp.y - 4.0f,
                                                   kp.x + 4.0f, kp.y + 4.0f,
                                                   point_color));
        }
    }

    osd_device.DrawCovers(covers, DETECTION_LAYER_ID);
}

#ifdef SSNE_AI_DEMO_HAS_OPENCV
void VISUALIZER::DrawPose(cv::Mat& image,
                          const std::vector<PoseDetection>& detections,
                          float kpt_conf_threshold) {
    static const std::array<std::array<int, 2>, 19> kSkeleton = {{
        {{15, 13}}, {{13, 11}}, {{16, 14}}, {{14, 12}}, {{11, 12}},
        {{5, 11}},  {{6, 12}},  {{5, 6}},   {{5, 7}},   {{6, 8}},
        {{7, 9}},   {{8, 10}},  {{1, 2}},   {{0, 1}},   {{0, 2}},
        {{1, 3}},   {{2, 4}},   {{3, 5}},   {{4, 6}}
    }};

    const cv::Scalar kLeftColor(255, 128, 0);
    const cv::Scalar kRightColor(0, 220, 255);
    const cv::Scalar kMidColor(0, 255, 0);

    for (size_t i = 0; i < detections.size(); ++i) {
        const PoseDetection& det = detections[i];
        const cv::Rect rect(cv::Point(static_cast<int>(det.box[0]), static_cast<int>(det.box[1])),
                            cv::Point(static_cast<int>(det.box[2]), static_cast<int>(det.box[3])));
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);

        for (size_t k = 0; k < det.keypoints.size(); ++k) {
            const PoseKeyPoint& kp = det.keypoints[k];
            if (kp.conf < kpt_conf_threshold) {
                continue;
            }

            cv::Scalar color = kMidColor;
            if (k == 5 || k == 7 || k == 9 || k == 11 || k == 13 || k == 15) {
                color = kLeftColor;
            } else if (k == 6 || k == 8 || k == 10 || k == 12 || k == 14 || k == 16) {
                color = kRightColor;
            }

            cv::circle(image,
                       cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                       3,
                       color,
                       -1);
        }

        for (size_t e = 0; e < kSkeleton.size(); ++e) {
            const int a = kSkeleton[e][0];
            const int b = kSkeleton[e][1];
            const PoseKeyPoint& p1 = det.keypoints[a];
            const PoseKeyPoint& p2 = det.keypoints[b];
            if (p1.conf < kpt_conf_threshold || p2.conf < kpt_conf_threshold) {
                continue;
            }

            cv::Scalar color = kMidColor;
            if ((a % 2) == 1 || (b % 2) == 1) {
                color = kLeftColor;
            }
            if ((a % 2) == 0 && (b % 2) == 0 && a > 4 && b > 4) {
                color = kRightColor;
            }

            cv::line(image,
                     cv::Point(static_cast<int>(p1.x), static_cast<int>(p1.y)),
                     cv::Point(static_cast<int>(p2.x), static_cast<int>(p2.y)),
                     color,
                     2,
                     cv::LINE_AA);
        }
    }
}
#endif

/**
 * @brief 释放 OSD 可视化器资源
 */
void VISUALIZER::Release() {
    osd_device.Release();
}
