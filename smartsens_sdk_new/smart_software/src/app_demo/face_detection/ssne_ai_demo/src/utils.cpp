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
#include <sys/stat.h>
#include <utility>

namespace utils {

/**
 * @brief 褰掑苟鎺掑簭鐨勫悎骞舵搷浣? * @param result 浜鸿劯/鐩爣妫€娴嬬粨鏋滅粨鏋勪綋鎸囬拡
 * @param low 鍚堝苟鍖洪棿鐨勮捣濮嬬储寮? * @param mid 鍚堝苟鍖洪棿鐨勪腑闂寸储寮? * @param high 鍚堝苟鍖洪棿鐨勭粨鏉熺储寮? * @description 灏嗕袱涓凡鎺掑簭鐨勫瓙鏁扮粍鍚堝苟鎴愪竴涓湁搴忔暟缁勶紝鎸夌収鍒嗘暟浠庨珮鍒颁綆鎺掑簭
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
 * @brief 褰掑苟鎺掑簭閫掑綊鍑芥暟
 * @param result 妫€娴嬬粨鏋滅粨鏋勪綋鎸囬拡
 * @param low 鎺掑簭鍖洪棿鐨勮捣濮嬬储寮? * @param high 鎺掑簭鍖洪棿鐨勭粨鏉熺储寮? */
void MergeSort(FaceDetectionResult* result, size_t low, size_t high) {
  if (low < high) {
    size_t mid = (high - low) / 2 + low;
    MergeSort(result, low, mid);
    MergeSort(result, mid + 1, high);
    Merge(result, low, mid, high);
  }
}

/**
 * @brief 瀵规娴嬬粨鏋滆繘琛屾帓搴? * @param result 妫€娴嬬粨鏋滅粨鏋勪綋鎸囬拡
 * @description 鎸夌収妫€娴嬪垎鏁颁粠楂樺埌浣庡妫€娴嬬粨鏋滆繘琛屾帓搴? */
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
 * @brief 闈炴瀬澶у€兼姂鍒讹紙NMS锛夌畻娉? * @param result 妫€娴嬬粨鏋滅粨鏋勪綋鎸囬拡
 * @param iou_threshold IoU闃堝€? * @param top_k 淇濈暀鍓峩涓娴嬬粨鏋? *
 * @note
 * 杩欑増淇濈暀浜?class_ids 鐨勫悓姝ュ鐞嗐€? * 杩欎釜鍑芥暟鍗充娇鏆傛椂涓嶇敤锛屼絾鏄繚鐣欏彲缂栬瘧鐘舵€併€? */
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

      // 鍚岀被 NMS锛氬彧鏈夌被鍒浉鍚屾墠姣旇緝鎶戝埗
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
 * @brief 閲婃斁 FaceDetectionResult 鐨勫唴瀛? * @description 浣跨敤 swap 鎶€宸ч噴鏀?vector 鍗犵敤鐨勫唴瀛? */
void FaceDetectionResult::Free() {
  std::vector<PoseDetection>().swap(detections);
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<int>().swap(class_ids);
  std::vector<std::array<PoseKeyPoint, 17>>().swap(keypoints);
}

/**
 * @brief 娓呯┖ FaceDetectionResult 鐨勫唴瀹? * @description 娓呯┖鎵€鏈夋娴嬫銆佸垎鏁般€佺被鍒拰鍏抽敭鐐癸紝浣嗕繚鐣欏唴瀛樺垎閰? */
void FaceDetectionResult::Clear() {
  detections.clear();
  boxes.clear();
  scores.clear();
  class_ids.clear();
  keypoints.clear();
}

/**
 * @brief 棰勫垎閰嶅唴瀛樼┖闂? * @param size 瑕佷繚鐣欑殑鍏冪礌鏁伴噺
 */
void FaceDetectionResult::Reserve(int size) {
  detections.reserve(size);
  boxes.reserve(size);
  scores.reserve(size);
  class_ids.reserve(size);
  keypoints.reserve(size);
}

/**
 * @brief 璋冩暣 FaceDetectionResult 鐨勫ぇ灏? * @param size 鏂扮殑鍏冪礌鏁伴噺
 */
void FaceDetectionResult::Resize(int size) {
  detections.resize(size);
  boxes.resize(size);
  scores.resize(size);
  class_ids.resize(size);
  keypoints.resize(size);
}

/**
 * @brief FaceDetectionResult 鐨勬嫹璐濇瀯閫犲嚱鏁? * @param res 瑕佹嫹璐濈殑 FaceDetectionResult 瀵硅薄
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
constexpr int kSnakeBodyColor = kBoxColor;
constexpr int kSnakeHeadColor = kCenterColor;
constexpr int kSnakeFoodColor = kPointColor;
constexpr int kSnakeBoardColor = kRightColor;
constexpr float kSkeletonLineThickness = 2.0f;
constexpr int kSnakeBitmapLayerId = 2;
constexpr int kSnakeGraphicLayerId = 1;
constexpr int kSnakeSpriteCanvasSize = 48;
constexpr int kDesignWidth = 1920;
constexpr int kDesignHeight = 1080;
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
    return class_id == kClassMouse ||
           class_id == kClassSnake ||
           class_id == kClassFire;
}

int GetFaceIdentityColorIndex(FaceIdentity identity) {
    switch (identity) {
        case FaceIdentity::kKnown:
            return kBoxColor;
        case FaceIdentity::kStranger:
            return kAlertColor;
        case FaceIdentity::kUnknown:
        default:
            return kRightColor;
    }
}

int GetDetectionColorIndex(const ObjectDetection& det) {
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

bool ClampBoxToCanvas(const std::array<float, 4>& box,
                      int canvas_width,
                      int canvas_height,
                      std::array<int, 4>* clamped_box) {
    if (clamped_box == nullptr || canvas_width <= 0 || canvas_height <= 0) {
        return false;
    }

    const int max_x = std::max(0, canvas_width - 1);
    const int max_y = std::max(0, canvas_height - 1);
    const int x1 = std::max(0, std::min(max_x, static_cast<int>(std::floor(std::min(box[0], box[2])))));
    const int y1 = std::max(0, std::min(max_y, static_cast<int>(std::floor(std::min(box[1], box[3])))));
    const int x2 = std::max(0, std::min(max_x, static_cast<int>(std::ceil(std::max(box[0], box[2])))));
    const int y2 = std::max(0, std::min(max_y, static_cast<int>(std::ceil(std::max(box[1], box[3])))));

    if (x2 <= x1 || y2 <= y1) {
        return false;
    }

    *clamped_box = {x1, y1, x2, y2};
    return true;
}

bool MakeHollowBoxCover(const std::array<float, 4>& box,
                        int border,
                        int color,
                        int canvas_width,
                        int canvas_height,
                        fdevice::COVER_ATTR_S* cover) {
    if (cover == nullptr) {
        return false;
    }

    std::array<int, 4> clamped_box = {0, 0, 0, 0};
    if (!ClampBoxToCanvas(box, canvas_width, canvas_height, &clamped_box)) {
        return false;
    }

    const int x1 = clamped_box[0];
    const int y1 = clamped_box[1];
    const int x2 = clamped_box[2];
    const int y2 = clamped_box[3];
    const int box_w = x2 - x1;
    const int box_h = y2 - y1;
    const int safe_border = std::max(1, std::min(border, std::max(1, std::min(box_w, box_h) / 2 - 1)));

    if (box_w <= safe_border * 2 || box_h <= safe_border * 2) {
        return false;
    }

    *cover = {};
    cover->colorIdx = color;
    cover->eSolid = fdevice::TYPE_HOLLOW;
    cover->alpha = fdevice::TYPE_ALPHA75;

    cover->vertex_in.points[0] = {x1 + safe_border, y1 + safe_border};
    cover->vertex_in.points[1] = {x1 + safe_border, y2 - safe_border};
    cover->vertex_in.points[2] = {x2 - safe_border, y2 - safe_border};
    cover->vertex_in.points[3] = {x2 - safe_border, y1 + safe_border};

    cover->vertex_out.points[0] = {x1 - safe_border, y1 - safe_border};
    cover->vertex_out.points[1] = {x1 - safe_border, y2 + safe_border};
    cover->vertex_out.points[2] = {x2 + safe_border, y2 + safe_border};
    cover->vertex_out.points[3] = {x2 + safe_border, y1 - safe_border};
    return true;
}

fdevice::COVER_ATTR_S MakeHollowBoxCover(const std::array<float, 4>& box, int border, int color) {
    fdevice::COVER_ATTR_S cover = {};
    (void)MakeHollowBoxCover(box, border, color, 1 << 14, 1 << 14, &cover);
    return cover;
}

bool MakeHollowQuadRangle(const std::array<float, 4>& box,
                          int border,
                          int color,
                          int canvas_width,
                          int canvas_height,
                          sst::device::osd::OsdQuadRangle* quad) {
    if (quad == nullptr) {
        return false;
    }

    std::array<int, 4> clamped_box = {0, 0, 0, 0};
    if (!ClampBoxToCanvas(box, canvas_width, canvas_height, &clamped_box)) {
        return false;
    }

    const int x1 = clamped_box[0];
    const int y1 = clamped_box[1];
    const int x2 = clamped_box[2];
    const int y2 = clamped_box[3];
    const int box_w = x2 - x1;
    const int box_h = y2 - y1;
    const int safe_border = std::max(1, std::min(border, std::max(1, std::min(box_w, box_h) / 2 - 1)));
    if (box_w <= safe_border * 2 || box_h <= safe_border * 2) {
        return false;
    }

    *quad = {};
    quad->box = {
        static_cast<float>(x1),
        static_cast<float>(y1),
        static_cast<float>(x2),
        static_cast<float>(y2)
    };
    quad->border = safe_border;
    quad->layer_id = VISUALIZER::DETECTION_LAYER_ID;
    quad->type = fdevice::TYPE_HOLLOW;
    quad->alpha = fdevice::TYPE_ALPHA75;
    quad->color = color;
    return true;
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

    const float left = std::max(0.0f, box[0] - 34.0f);
    const float top = std::max(0.0f, box[1] - 34.0f);
    const float size = 28.0f;
    const float cx = left + size * 0.5f;

    covers->emplace_back(MakeSolidRectCover(left, top, left + size, top + size, kAlertColor));
    covers->emplace_back(MakeSolidRectCover(cx - 3.0f, top + 5.0f, cx + 3.0f, top + 18.0f, kPointColor));
    covers->emplace_back(MakeSolidRectCover(cx - 3.0f, top + 21.0f, cx + 3.0f, top + 26.0f, kPointColor));
}

int ScaleDesignX(int width, float x) {
    return static_cast<int>(std::round(x * static_cast<float>(width) /
                                       static_cast<float>(kDesignWidth)));
}

int ScaleDesignY(int height, float y) {
    return static_cast<int>(std::round(y * static_cast<float>(height) /
                                       static_cast<float>(kDesignHeight)));
}

std::string ResolveAppAssetPath(const std::string& asset_path) {
    if (asset_path.empty()) {
        return std::string();
    }
    if (asset_path[0] == '/') {
        return asset_path;
    }
    return std::string("/app_demo/app_assets/") + asset_path;
}

bool FileExists(const std::string& path, long* size_bytes = nullptr) {
    if (path.empty()) {
        return false;
    }
    struct stat file_stat;
    if (stat(path.c_str(), &file_stat) != 0) {
        return false;
    }
    if (size_bytes != nullptr) {
        *size_bytes = static_cast<long>(file_stat.st_size);
    }
    return file_stat.st_size > 0;
}

std::string SnakeAsset(const std::string& relative_path) {
    return std::string("ui/snake/") + relative_path;
}

std::string DigitBitmapPath(int digit) {
    const int clamped_digit = std::max(0, std::min(9, digit));
    return SnakeAsset("digits/digit_" + std::to_string(clamped_digit) + ".ssbmp");
}

std::string GestureBitmapPath(GestureCommand command) {
    switch (command) {
        case GestureCommand::TU:
            return SnakeAsset("gestures/gesture_up.ssbmp");
        case GestureCommand::TD:
            return SnakeAsset("gestures/gesture_down.ssbmp");
        case GestureCommand::TL:
            return SnakeAsset("gestures/gesture_left.ssbmp");
        case GestureCommand::TR:
            return SnakeAsset("gestures/gesture_right.ssbmp");
        case GestureCommand::NONE:
        default:
            return std::string();
    }
}

int ClampSpriteX(float x, float board_left, float board_width) {
    const int min_x = static_cast<int>(std::round(board_left));
    const int max_x = static_cast<int>(std::round(board_left + board_width)) - kSnakeSpriteCanvasSize;
    return std::max(min_x, std::min(max_x, static_cast<int>(std::round(x))));
}

int ClampSpriteY(float y, float board_top, float board_height) {
    const int min_y = static_cast<int>(std::round(board_top));
    const int max_y = static_cast<int>(std::round(board_top + board_height)) - kSnakeSpriteCanvasSize;
    return std::max(min_y, std::min(max_y, static_cast<int>(std::round(y))));
}

fdevice::COVER_ATTR_S MakeSnakeCellCover(const SnakeCell& cell,
                                         float board_left,
                                         float board_top,
                                         float cell_w,
                                         float cell_h,
                                         int color) {
    const float inset = std::max(2.0f, std::min(cell_w, cell_h) * 0.12f);
    const float left = board_left + static_cast<float>(cell.x) * cell_w + inset;
    const float top = board_top + static_cast<float>(cell.y) * cell_h + inset;
    const float right = board_left + static_cast<float>(cell.x + 1) * cell_w - inset;
    const float bottom = board_top + static_cast<float>(cell.y + 1) * cell_h - inset;
    return MakeSolidRectCover(left, top, right, bottom, color);
}

std::array<bool, 7> DigitSegments(int digit) {
    switch (digit) {
        case 0: return {{true, true, true, true, true, true, false}};
        case 1: return {{false, true, true, false, false, false, false}};
        case 2: return {{true, true, false, true, true, false, true}};
        case 3: return {{true, true, true, true, false, false, true}};
        case 4: return {{false, true, true, false, false, true, true}};
        case 5: return {{true, false, true, true, false, true, true}};
        case 6: return {{true, false, true, true, true, true, true}};
        case 7: return {{true, true, true, false, false, false, false}};
        case 8: return {{true, true, true, true, true, true, true}};
        case 9: return {{true, true, true, true, false, true, true}};
        default: return {{false, false, false, false, false, false, false}};
    }
}

void AppendSevenSegmentDigit(std::vector<fdevice::COVER_ATTR_S>* covers,
                             int digit,
                             float left,
                             float top,
                             float scale,
                             int color) {
    if (covers == nullptr) {
        return;
    }

    const std::array<bool, 7> seg = DigitSegments(digit);
    const float w = 28.0f * scale;
    const float h = 50.0f * scale;
    const float t = std::max(3.0f, 5.0f * scale);
    const float mid = top + h * 0.5f;
    const float bottom = top + h;

    if (seg[0]) covers->emplace_back(MakeSolidRectCover(left + t, top, left + w - t, top + t, color));
    if (seg[1]) covers->emplace_back(MakeSolidRectCover(left + w - t, top + t, left + w, mid - t * 0.5f, color));
    if (seg[2]) covers->emplace_back(MakeSolidRectCover(left + w - t, mid + t * 0.5f, left + w, bottom - t, color));
    if (seg[3]) covers->emplace_back(MakeSolidRectCover(left + t, bottom - t, left + w - t, bottom, color));
    if (seg[4]) covers->emplace_back(MakeSolidRectCover(left, mid + t * 0.5f, left + t, bottom - t, color));
    if (seg[5]) covers->emplace_back(MakeSolidRectCover(left, top + t, left + t, mid - t * 0.5f, color));
    if (seg[6]) covers->emplace_back(MakeSolidRectCover(left + t, mid - t * 0.5f, left + w - t, mid + t * 0.5f, color));
}

void AppendGraphicNumber(std::vector<fdevice::COVER_ATTR_S>* covers,
                         int value,
                         float right_x,
                         float top_y,
                         float scale,
                         int color) {
    if (covers == nullptr) {
        return;
    }

    const std::string text = std::to_string(std::max(0, value));
    const float spacing = 34.0f * scale;
    const float start_x = right_x - static_cast<float>(text.size()) * spacing;
    for (size_t i = 0; i < text.size(); ++i) {
        AppendSevenSegmentDigit(covers,
                                text[i] - '0',
                                start_x + static_cast<float>(i) * spacing,
                                top_y,
                                scale,
                                color);
    }
}
std::string SnakeStateBitmapPath(const SnakeRenderData& game) {
    if (game.game_over) {
        return SnakeAsset("states/state_game_over.ssbmp");
    }
    if (game.paused) {
        return SnakeAsset("states/state_paused.ssbmp");
    }
    if (game.score == 0 && game.last_command == GestureCommand::NONE) {
        return SnakeAsset("states/state_ready.ssbmp");
    }
    return SnakeAsset("states/state_playing.ssbmp");
}

bool IsStraightSegment(const SnakeCell& prev, const SnakeCell& next) {
    return prev.x == next.x || prev.y == next.y;
}

std::string SnakeSegmentBitmapPath(const SnakeRenderData& game, size_t index) {
    if (index >= game.snake.size()) {
        return std::string();
    }
    if (index == 0) {
        return SnakeAsset("snake/snake_head.ssbmp");
    }
    if (index + 1 == game.snake.size()) {
        return SnakeAsset("snake/snake_tail.ssbmp");
    }
    const SnakeCell& prev = game.snake[index - 1];
    const SnakeCell& next = game.snake[index + 1];
    return IsStraightSegment(prev, next) ?
        SnakeAsset("snake/snake_body.ssbmp") :
        SnakeAsset("snake/snake_corner.ssbmp");
}

void DrawNumberBitmaps(VISUALIZER* visualizer,
                       int value,
                       int right_x,
                       int top_y,
                       int spacing) {
    if (visualizer == nullptr) {
        return;
    }

    const std::string text = std::to_string(std::max(0, value));
    const int start_x = right_x - static_cast<int>(text.size()) * spacing;
    for (size_t i = 0; i < text.size(); ++i) {
        const int digit = text[i] - '0';
        visualizer->DrawBitmap(DigitBitmapPath(digit),
                               "",
                               start_x + static_cast<int>(i) * spacing,
                               top_y,
                               kSnakeBitmapLayerId,
                               false);
    }
}

}  // namespace


/**
 * @brief OSD 鍙鍖栧櫒鍒濆鍖栧嚱鏁? * @param in_img_shape 鍥惧儚灏哄 [瀹藉害, 楂樺害]
 */
void VISUALIZER::Initialize(std::array<int, 2>& in_img_shape, const std::string& bitmap_lut_path) {
    m_width = in_img_shape[0];
    m_height = in_img_shape[1];
    if (bitmap_lut_path.empty()) {
        m_bitmap_lut_path_full.clear();
        osd_device.Initialize(in_img_shape[0], in_img_shape[1], nullptr);
    } else {
        m_bitmap_lut_path_full = ResolveAppAssetPath(bitmap_lut_path);
        long lut_size = 0;
        if (FileExists(m_bitmap_lut_path_full, &lut_size)) {
            std::cout << "[VISUALIZER] bitmap LUT resolved: "
                      << m_bitmap_lut_path_full << " size=" << lut_size << " bytes" << std::endl;
        } else {
            std::cerr << "[VISUALIZER] WARN: bitmap LUT not found after resolve: "
                      << m_bitmap_lut_path_full << std::endl;
        }
        osd_device.Initialize(in_img_shape[0], in_img_shape[1], m_bitmap_lut_path_full.c_str());
    }
}


/**
 * @brief 缁樺埗娴嬭瘯鐭╁舰妗嗭紙鐢ㄤ簬娴嬭瘯 OSD 鍔熻兘锛? */
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
 * @brief 鏍规嵁妫€娴嬫缁樺埗 OSD 鐭╁舰
 * @param boxes 妫€娴嬫鍚戦噺锛屾瘡涓厓绱犱负[xmin, ymin, xmax, ymax]
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
    std::vector<sst::device::osd::OsdQuadRangle> quads;
    std::vector<fdevice::COVER_ATTR_S> covers;
    quads.reserve(detections.size());
    covers.reserve(detections.size() * 4);

    for (const auto& det : detections) {
        const int border = IsAlertClass(det.class_id) ? 5 : 3;
        sst::device::osd::OsdQuadRangle quad;
        if (MakeHollowQuadRangle(det.box, border, GetDetectionColorIndex(det),
                                 m_width, m_height, &quad)) {
            quads.emplace_back(quad);
        }
        if (IsAlertClass(det.class_id)) {
            AppendWarningIconCovers(det.box, &covers);
        }
    }

    if (quads.empty() && covers.empty()) {
        osd_device.ClearLayer(DETECTION_LAYER_ID);
        return;
    }
    if (!quads.empty()) {
        osd_device.Draw(quads, DETECTION_LAYER_ID);
    }
    if (!covers.empty()) {
        osd_device.DrawCovers(covers, DETECTION_LAYER_ID);
    }
}

void VISUALIZER::Draw(const FaceResult& face_result) {
    std::vector<fdevice::COVER_ATTR_S> covers;
    covers.reserve(static_cast<size_t>(std::max(face_result.count, 0)) * 5U);

    for (int i = 0; i < face_result.count; ++i) {
        const FaceDetection& det = face_result.faces[i];
        if (det.identity == FaceIdentity::kUnknown) {
            continue;
        }
        const std::array<float, 4> box = {
            det.x,
            det.y,
            det.x + det.w,
            det.y + det.h
        };
        const int color = GetFaceIdentityColorIndex(det.identity);
        const int border = det.identity == FaceIdentity::kStranger ? 5 : 3;
        fdevice::COVER_ATTR_S cover = {};
        if (MakeHollowBoxCover(box, border, color, m_width, m_height, &cover)) {
            covers.emplace_back(cover);
        }

        const float badge_left = std::max(0.0f, box[0] - 26.0f);
        const float badge_top = std::max(0.0f, box[1] - 26.0f);
        covers.emplace_back(
            MakeSolidRectCover(badge_left, badge_top,
                               badge_left + 20.0f, badge_top + 20.0f, color));

        if (det.identity == FaceIdentity::kKnown) {
            covers.emplace_back(
                MakeSolidRectCover(badge_left + 4.0f, badge_top + 4.0f,
                                   badge_left + 8.0f, badge_top + 16.0f, kPointColor));
            covers.emplace_back(
                MakeSolidRectCover(badge_left + 12.0f, badge_top + 4.0f,
                                   badge_left + 16.0f, badge_top + 16.0f, kPointColor));
        } else if (det.identity == FaceIdentity::kStranger) {
            covers.emplace_back(
                MakeSolidRectCover(badge_left + 8.0f, badge_top + 4.0f,
                                   badge_left + 12.0f, badge_top + 12.0f, kPointColor));
            covers.emplace_back(
                MakeSolidRectCover(badge_left + 8.0f, badge_top + 14.0f,
                                   badge_left + 12.0f, badge_top + 16.0f, kPointColor));
        } else {
            covers.emplace_back(
                MakeSolidRectCover(badge_left + 5.0f, badge_top + 8.0f,
                                   badge_left + 15.0f, badge_top + 12.0f, kPointColor));
        }
    }

    osd_device.DrawCovers(covers, DETECTION_LAYER_ID);
}

void VISUALIZER::Draw(const std::vector<PoseDetection>& detections, float kpt_conf_threshold) {
    static const std::array<std::array<int, 2>, 16> kSkeleton = {{
        {{0, 1}}, {{0, 2}}, {{1, 3}}, {{2, 4}},
        {{5, 6}}, {{5, 7}}, {{7, 9}}, {{6, 8}},
        {{8, 10}}, {{5, 11}}, {{6, 12}}, {{11, 12}},
        {{11, 13}}, {{13, 15}}, {{12, 14}}, {{14, 16}}
    }};

    std::vector<sst::device::osd::OsdQuadRangle> quads;
    std::vector<fdevice::COVER_ATTR_S> covers;
    quads.reserve(detections.size());
    covers.reserve(detections.size() * 40);

    for (const auto& det : detections) {
        sst::device::osd::OsdQuadRangle quad;
        if (MakeHollowQuadRangle(det.box, 3, kBoxColor, m_width, m_height, &quad)) {
            quads.emplace_back(quad);
        }

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

    if (quads.empty() && covers.empty()) {
        osd_device.ClearLayer(DETECTION_LAYER_ID);
        return;
    }
    if (!quads.empty()) {
        osd_device.Draw(quads, DETECTION_LAYER_ID);
    }
    if (!covers.empty()) {
        osd_device.DrawCovers(covers, DETECTION_LAYER_ID);
    }
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

    std::vector<sst::device::osd::OsdQuadRangle> quads;
    std::vector<fdevice::COVER_ATTR_S> covers;
    quads.reserve(detections.size() + poses.size());
    covers.reserve(detections.size() + poses.size() * 40);

    for (const auto& det : detections) {
        const int border = IsAlertClass(det.class_id) ? 5 : 3;
        sst::device::osd::OsdQuadRangle quad;
        if (MakeHollowQuadRangle(det.box, border, GetDetectionColorIndex(det),
                                 m_width, m_height, &quad)) {
            quads.emplace_back(quad);
        }
        if (IsAlertClass(det.class_id)) {
            AppendWarningIconCovers(det.box, &covers);
        }
    }

    for (const auto& det : poses) {
        sst::device::osd::OsdQuadRangle quad;
        if (MakeHollowQuadRangle(det.box, 3, kBoxColor, m_width, m_height, &quad)) {
            quads.emplace_back(quad);
        }

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

    if (quads.empty() && covers.empty()) {
        osd_device.ClearLayer(DETECTION_LAYER_ID);
        return;
    }
    osd_device.ClearLayer(DETECTION_LAYER_ID);
    if (!quads.empty()) {
        osd_device.Draw(quads, DETECTION_LAYER_ID);
    }
    if (!covers.empty()) {
        osd_device.DrawCovers(covers, DETECTION_LAYER_ID);
    }
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
 * @brief 閲婃斁 OSD 鍙鍖栧櫒璧勬簮
 */
void VISUALIZER::Release() {
    osd_device.Release();
}

void VISUALIZER::ClearLayer(int layer_id) {
    osd_device.ClearLayer(layer_id);
}

void VISUALIZER::DrawBitmap(const std::string& bitmap_path,
                            const std::string& lut_path,
                            int pos_x,
                            int pos_y,
                            int layer_id,
                            bool flush) {
    const std::string bitmap_full_path = ResolveAppAssetPath(bitmap_path);
    const std::string lut_full_path =
        lut_path.empty() ? m_bitmap_lut_path_full : ResolveAppAssetPath(lut_path);

    static int s_bitmap_log_count = 0;
    if (s_bitmap_log_count < 12) {
        long bitmap_size = 0;
        const bool bitmap_ok = FileExists(bitmap_full_path, &bitmap_size);
        std::cout << "[VISUALIZER] draw bitmap[" << s_bitmap_log_count << "]: path="
                  << bitmap_full_path << " exists=" << (bitmap_ok ? 1 : 0)
                  << " size=" << bitmap_size
                  << " pos=(" << pos_x << "," << pos_y << ") layer=" << layer_id
                  << " lut=" << (lut_full_path.empty() ? "default" : lut_full_path)
                  << std::endl;
        ++s_bitmap_log_count;
    }

    osd_device.DrawTexture(bitmap_full_path.c_str(),
                           lut_full_path.empty() ? nullptr : lut_full_path.c_str(),
                           layer_id,
                           pos_x,
                           pos_y,
                           fdevice::TYPE_ALPHA100,
                           flush);
}

void VISUALIZER::DrawSnakeGame(const SnakeRenderData& game) {
    if (game.board_cols <= 0 || game.board_rows <= 0) {
        osd_device.ClearLayer(kSnakeGraphicLayerId);
        osd_device.ClearLayer(kSnakeBitmapLayerId);
        return;
    }

    osd_device.ClearLayer(kSnakeGraphicLayerId);
    osd_device.ClearLayer(kSnakeBitmapLayerId);

    const float board_left = static_cast<float>(ScaleDesignX(m_width, 1260.0f));
    const float board_top = static_cast<float>(ScaleDesignY(m_height, 250.0f));
    const float board_width = static_cast<float>(ScaleDesignX(m_width, 640.0f));
    const float board_height = static_cast<float>(ScaleDesignY(m_height, 560.0f));
    const float cell_w = board_width / static_cast<float>(game.board_cols);
    const float cell_h = board_height / static_cast<float>(game.board_rows);
    const float sprite_half = static_cast<float>(kSnakeSpriteCanvasSize) * 0.5f;

    std::vector<fdevice::COVER_ATTR_S> snake_covers;
    snake_covers.reserve(game.snake.size() + 1U);
    for (size_t i = 0; i < game.snake.size(); ++i) {
        const SnakeCell& cell = game.snake[i];
        const int color = (i == 0) ? kSnakeHeadColor : kSnakeBodyColor;
        snake_covers.emplace_back(MakeSnakeCellCover(cell,
                                                     board_left,
                                                     board_top,
                                                     cell_w,
                                                     cell_h,
                                                     color));
    }
    AppendGraphicNumber(&snake_covers,
                        game.score,
                        static_cast<float>(ScaleDesignX(m_width, 1810.0f)),
                        static_cast<float>(ScaleDesignY(m_height, 112.0f)),
                        static_cast<float>(m_width) / static_cast<float>(kDesignWidth),
                        kSnakeHeadColor);
    AppendGraphicNumber(&snake_covers,
                        game.best_score,
                        static_cast<float>(ScaleDesignX(m_width, 1810.0f)),
                        static_cast<float>(ScaleDesignY(m_height, 182.0f)),
                        static_cast<float>(m_width) / static_cast<float>(kDesignWidth),
                        kSnakeBodyColor);
    osd_device.DrawCovers(snake_covers, kSnakeGraphicLayerId);

    if (game.has_food) {
        const float center_x = board_left + (static_cast<float>(game.food.x) + 0.5f) * cell_w;
        const float center_y = board_top + (static_cast<float>(game.food.y) + 0.5f) * cell_h;
        DrawBitmap(SnakeAsset("food/food_apple.ssbmp"),
                   "",
                   ClampSpriteX(center_x - sprite_half, board_left, board_width),
                   ClampSpriteY(center_y - sprite_half, board_top, board_height),
                   kSnakeBitmapLayerId,
                   false);
    }

    const std::string gesture_bitmap = GestureBitmapPath(game.last_command);
    if (!gesture_bitmap.empty()) {
        DrawBitmap(gesture_bitmap,
                   "",
                   ScaleDesignX(m_width, 225.0f),
                   ScaleDesignY(m_height, 410.0f),
                   kSnakeBitmapLayerId,
                   false);
    }

    osd_device.FlushTextureLayer(kSnakeBitmapLayerId);
}
