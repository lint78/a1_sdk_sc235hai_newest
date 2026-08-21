/*
 * @Filename: demo_face_yunet.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-01-20
 * @Copyright (c) 2025 SmartSens
 * @Description: YOLOv8 妫€娴?+ 浣庨 pose 婕旂ず
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include "include/face_business.h"
#include "include/model_quant_config.h"
#include "include/companion_mode.hpp"
#include "include/stranger_mode.hpp"
#include "event_recorder.hpp"
#include <unistd.h>
#include "fallen_judge.hpp"
#include "include/log.hpp"
#include "include/utils.hpp"

using namespace std;

namespace {

constexpr float kDetectFallbackConfThreshold = 0.35f;
constexpr float kDetectTemporalPersonConfThreshold = 0.20f;
constexpr float kDetectPoseAssistPersonConfThreshold = 0.14f;
constexpr std::array<float, 7> kDetectClassThresholds = {
    0.45f, 0.43f, 0.44f, 0.44f, 0.26f, 0.43f, 0.14f
};
constexpr float kPoseConfThreshold = 0.25f;
constexpr float kPoseDrawKptConfThreshold = 0.40f;
constexpr float kForcePoseDrawKptConfThreshold = 0.50f;
constexpr float kForcePoseAcceptKptConfThreshold = 0.45f;
constexpr int kForcePoseMinStrongKeypoints = 6;
constexpr int kPoseInvokeInterval = 5;
constexpr int kPoseForceInterval = 15;
constexpr int kPoseAssistDetectFrames = 15;
constexpr int kPoseVisualMaxHoldFrames = 15;
constexpr float kPoseTrackScoreFloor = 0.50f;
constexpr float kGestureConfThreshold = 0.25f;
constexpr float kGestureNoneRejectThreshold = 0.50f;
constexpr int kSnakeBoardCols = 20;
constexpr int kSnakeBoardRows = 11;
constexpr int kDetectNumClasses = 7;
// Gesture ROI is defined in the 1080x1080 tensor returned by GetImage().
// The online pipeline maps it to original x by adding crop_offset_x.
constexpr std::array<float, 4> kGestureGuideBoxCenterCrop = {
    180.0f, 160.0f, 820.0f, 800.0f
};
constexpr std::array<float, 4> kSnakeBoardBoxOriginal = {
    1260.0f, 250.0f, 1900.0f, 810.0f
};
constexpr int kSnakeClassId = 2;
constexpr int kMouseClassId = 3;
constexpr int kPersonClassId = 4;
constexpr int kFireClassId = 5;
constexpr int kDarkPersonClassId = 6;
constexpr std::array<const char*, kDetectNumClasses> kDetectClassNames = {
    "cat", "dog", "snake", "mouse", "person", "fire", "person"
};

std::array<float, 4> MapBoxToOriginal(const std::array<float, 4>& box,
                                      int crop_offset_x,
                                      int img_width,
                                      int img_height) {
    std::array<float, 4> mapped = {
        box[0] + static_cast<float>(crop_offset_x),
        box[1],
        box[2] + static_cast<float>(crop_offset_x),
        box[3]
    };

    mapped[0] = std::max(0.0f, std::min(mapped[0], static_cast<float>(img_width)));
    mapped[1] = std::max(0.0f, std::min(mapped[1], static_cast<float>(img_height)));
    mapped[2] = std::max(0.0f, std::min(mapped[2], static_cast<float>(img_width)));
    mapped[3] = std::max(0.0f, std::min(mapped[3], static_cast<float>(img_height)));
    return mapped;
}

bool BoxesOverlap(const std::array<float, 4>& lhs,
                  const std::array<float, 4>& rhs) {
    return lhs[0] < rhs[2] && lhs[2] > rhs[0] &&
           lhs[1] < rhs[3] && lhs[3] > rhs[1];
}

float HorizontalGap(const std::array<float, 4>& lhs,
                    const std::array<float, 4>& rhs) {
    if (lhs[2] <= rhs[0]) {
        return rhs[0] - lhs[2];
    }
    if (rhs[2] <= lhs[0]) {
        return lhs[0] - rhs[2];
    }
    return 0.0f;
}
ObjectDetection MapDetectionToOriginal(const ObjectDetection& det,
                                       int crop_offset_x,
                                       int img_width,
                                       int img_height) {
    ObjectDetection mapped = det;
    mapped.box = MapBoxToOriginal(det.box, crop_offset_x, img_width, img_height);
    return mapped;
}

PoseDetection MapPoseToOriginal(const PoseDetection& det,
                                int crop_offset_x,
                                int img_width,
                                int img_height) {
    PoseDetection mapped = det;
    mapped.box = MapBoxToOriginal(det.box, crop_offset_x, img_width, img_height);
    for (auto& kp : mapped.keypoints) {
        kp.x = std::max(0.0f, std::min(kp.x + static_cast<float>(crop_offset_x),
                                       static_cast<float>(img_width)));
        kp.y = std::max(0.0f, std::min(kp.y, static_cast<float>(img_height)));
    }
    return mapped;
}

struct PoseRoiCandidate {
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;

    int Width() const { return right - left; }
    int Height() const { return bottom - top; }
    bool IsValid() const { return Width() > 0 && Height() > 0; }
};

bool MapTrackedBoxToCrop(const std::array<float, 4>& original_box,
                         int crop_offset_x,
                         const std::array<int, 2>& crop_shape,
                         std::array<float, 4>* crop_box) {
    if (crop_box == nullptr) {
        return false;
    }

    const float crop_left = static_cast<float>(crop_offset_x);
    const float crop_right = static_cast<float>(crop_offset_x + crop_shape[0]);
    const float x1 = std::max(original_box[0], crop_left);
    const float x2 = std::min(original_box[2], crop_right);
    const float y1 = std::max(0.0f, std::min(original_box[1], static_cast<float>(crop_shape[1])));
    const float y2 = std::max(0.0f, std::min(original_box[3], static_cast<float>(crop_shape[1])));
    if (x2 <= x1 || y2 <= y1) {
        return false;
    }

    *crop_box = {
        x1 - crop_left,
        y1,
        x2 - crop_left,
        y2
    };
    return true;
}

PoseRoiCandidate BuildPoseRequestRoi(const std::array<float, 4>& focus_box,
                                     const std::array<int, 2>& crop_shape,
                                     const std::array<int, 2>& pose_shape) {
    PoseRoiCandidate roi;
    const float box_w = std::max(1.0f, focus_box[2] - focus_box[0]);
    const float box_h = std::max(1.0f, focus_box[3] - focus_box[1]);
    const float box_cx = 0.5f * (focus_box[0] + focus_box[2]);
    const float box_cy = 0.5f * (focus_box[1] + focus_box[3]);
    const float target_aspect =
        static_cast<float>(std::max(1, pose_shape[0])) /
        static_cast<float>(std::max(1, pose_shape[1]));
    float roi_w = std::max(box_w * 1.8f, static_cast<float>(pose_shape[0]));
    float roi_h = std::max(box_h * 1.8f, static_cast<float>(pose_shape[1]));
    if (roi_w / std::max(roi_h, 1.0f) < target_aspect) {
        roi_w = roi_h * target_aspect;
    } else {
        roi_h = roi_w / target_aspect;
    }

    float left = box_cx - 0.5f * roi_w;
    float top = box_cy - 0.5f * roi_h;
    float right = box_cx + 0.5f * roi_w;
    float bottom = box_cy + 0.5f * roi_h;

    if (left < 0.0f) {
        right -= left;
        left = 0.0f;
    }
    if (top < 0.0f) {
        bottom -= top;
        top = 0.0f;
    }
    if (right > static_cast<float>(crop_shape[0])) {
        const float overflow = right - static_cast<float>(crop_shape[0]);
        left = std::max(0.0f, left - overflow);
        right = static_cast<float>(crop_shape[0]);
    }
    if (bottom > static_cast<float>(crop_shape[1])) {
        const float overflow = bottom - static_cast<float>(crop_shape[1]);
        top = std::max(0.0f, top - overflow);
        bottom = static_cast<float>(crop_shape[1]);
    }

    roi.left = std::max(0, static_cast<int>(std::floor(left)));
    roi.top = std::max(0, static_cast<int>(std::floor(top)));
    roi.right = std::min(crop_shape[0], static_cast<int>(std::ceil(right)));
    roi.bottom = std::min(crop_shape[1], static_cast<int>(std::ceil(bottom)));

    roi.left &= ~1;
    roi.right &= ~1;
    roi.right = std::max(roi.left, roi.right);

    const int width = roi.Width();
    if (width >= 8) {
        const int aligned_width = width & ~7;
        const int trim = width - aligned_width;
        roi.left += trim / 2;
        roi.left &= ~1;
        roi.right = roi.left + aligned_width;
        if (roi.right > crop_shape[0]) {
            roi.right = crop_shape[0] & ~1;
            roi.left = std::max(0, roi.right - aligned_width);
            roi.left &= ~1;
            roi.right = roi.left + aligned_width;
        }
    }

    return roi;
}

bool IsPoseRequestRoiLegal(const std::array<float, 4>& focus_box,
                           const std::array<int, 2>& crop_shape,
                           const std::array<int, 2>& pose_shape) {
    if (focus_box[2] <= focus_box[0] || focus_box[3] <= focus_box[1]) {
        return false;
    }

    const PoseRoiCandidate roi = BuildPoseRequestRoi(focus_box, crop_shape, pose_shape);
    if (!roi.IsValid()) {
        return false;
    }

    const int roi_w = roi.Width();
    if (roi_w < 8 || (roi_w & 7) != 0) {
        return false;
    }

    return true;
}

bool IsSameSnakeCells(const std::vector<SnakeCell>& a, const std::vector<SnakeCell>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].x != b[i].x || a[i].y != b[i].y) {
            return false;
        }
    }
    return true;
}

bool IsSameSnakeRenderData(const SnakeRenderData& a, const SnakeRenderData& b) {
    return a.board_cols == b.board_cols &&
           a.board_rows == b.board_rows &&
           a.score == b.score &&
           a.best_score == b.best_score &&
           a.paused == b.paused &&
           a.game_over == b.game_over &&
           a.last_command == b.last_command &&
           a.has_food == b.has_food &&
           a.food.x == b.food.x &&
           a.food.y == b.food.y &&
           IsSameSnakeCells(a.snake, b.snake);
}
const char* DetectClassName(int class_id) {
    if (class_id < 0 || class_id >= static_cast<int>(kDetectClassNames.size())) {
        return "unknown";
    }
    return kDetectClassNames[static_cast<size_t>(class_id)];
}

std::string FormatClassLabel(int class_id) {
    std::ostringstream oss;
    oss << "class=" << class_id << "(" << DetectClassName(class_id) << ")";
    return oss.str();
}

int NormalizeClassForTracking(int class_id) {
    return class_id == kDarkPersonClassId ? kPersonClassId : class_id;
}

bool IsPersonLikeDetection(const ObjectDetection& det) {
    return NormalizeClassForTracking(det.class_id) == kPersonClassId;
}

bool HasPersonLikeDetection(const std::vector<ObjectDetection>& detections) {
    return std::any_of(detections.begin(), detections.end(), IsPersonLikeDetection);
}

bool HasFireDetection(const std::vector<ObjectDetection>& detections) {
    return std::any_of(detections.begin(),
                       detections.end(),
                       [](const ObjectDetection& det) {
                           return det.class_id == kFireClassId;
                       });
}

// Use the captured Y plane as a light sanity check for fire detections. The
// detector input may be enhanced, but this check must use the original frame.
bool IsFireBrightnessPlausible(ssne_tensor_t tensor,
                               const std::array<float, 4>& original_box,
                               int crop_offset_x) {
    if (get_data(tensor) == nullptr || get_data_format(tensor) != SSNE_YUV422_16) {
        return true;
    }

    const uint32_t width = get_width(tensor);
    const uint32_t height = get_height(tensor);
    const size_t mem_size = get_mem_size(tensor);
    if (width == 0 || height == 0 || mem_size == 0) {
        return true;
    }

    const size_t row_stride = mem_size / static_cast<size_t>(height);
    if (row_stride < static_cast<size_t>(width) * 2U) {
        return true;
    }

    // Detection boxes are in the original 1920x1080 coordinates; the raw
    // sensor tensor is the centered 1080x1080 crop used by the models.
    const int left = std::max(0, static_cast<int>(std::floor(original_box[0])) - crop_offset_x);
    const int top = std::max(0, static_cast<int>(std::floor(original_box[1])));
    const int right = std::min(static_cast<int>(width),
                               static_cast<int>(std::ceil(original_box[2])) - crop_offset_x);
    const int bottom = std::min(static_cast<int>(height),
                                static_cast<int>(std::ceil(original_box[3])));
    if (right <= left || bottom <= top) {
        return true;
    }

    const uint32_t step_x = std::max<uint32_t>(1, static_cast<uint32_t>(right - left) / 32U);
    const uint32_t step_y = std::max<uint32_t>(1, static_cast<uint32_t>(bottom - top) / 24U);
    const uint8_t* data = static_cast<const uint8_t*>(get_data(tensor));
    uint32_t samples = 0;
    uint32_t bright_pixels = 0;
    uint32_t hot_pixels = 0;
    for (int y = top; y < bottom; y += static_cast<int>(step_y)) {
        const uint8_t* row = data + static_cast<size_t>(y) * row_stride;
        for (int x = left; x < right; x += static_cast<int>(step_x)) {
            // SSNE_YUV422_16 is packed YUV422: Y0 U Y1 V. Only even bytes
            // are luma; the intervening bytes are chroma and must be ignored.
            const uint8_t raw_y = row[static_cast<size_t>(x) * 2U];
            bright_pixels += raw_y >= 128U ? 1U : 0U;
            hot_pixels += raw_y >= 180U ? 1U : 0U;
            ++samples;
        }
    }
    if (samples == 0) {
        return true;
    }

    const float bright_ratio = static_cast<float>(bright_pixels) /
                               static_cast<float>(samples);
    const bool enough_bright_area = bright_ratio >= 0.02f;
    const bool has_small_bright_core = hot_pixels >= 2U && bright_ratio >= 0.005f;
    return enough_bright_area || has_small_bright_core;
}

void ApplyFireBrightnessFallback(ssne_tensor_t raw_tensor,
                                 int crop_offset_x,
                                 std::vector<ObjectDetection>* detections) {
    if (detections == nullptr) {
        return;
    }
    detections->erase(
        std::remove_if(detections->begin(), detections->end(),
                       [raw_tensor, crop_offset_x](const ObjectDetection& det) {
                           return det.class_id == kFireClassId &&
                                  !IsFireBrightnessPlausible(raw_tensor, det.box, crop_offset_x);
                       }),
        detections->end());
}

bool HasIntrusionDetection(const std::vector<ObjectDetection>& detections) {
    return std::any_of(detections.begin(),
                       detections.end(),
                       [](const ObjectDetection& det) {
                           return det.class_id == kMouseClassId ||
                                  det.class_id == kSnakeClassId;
                       });
}

bool IsStrongKeypoint(const PoseDetection& pose, int index, float threshold) {
    return index >= 0 &&
           index < static_cast<int>(pose.keypoints.size()) &&
           pose.keypoints[static_cast<size_t>(index)].conf >= threshold;
}

int CountStrongKeypoints(const PoseDetection& pose, float threshold) {
    int count = 0;
    for (const auto& kp : pose.keypoints) {
        if (kp.conf >= threshold) {
            ++count;
        }
    }
    return count;
}

bool HasHumanLikePoseSupport(const PoseDetection& pose, float threshold) {
    const bool has_shoulder =
        IsStrongKeypoint(pose, 5, threshold) || IsStrongKeypoint(pose, 6, threshold);
    const bool has_hip =
        IsStrongKeypoint(pose, 11, threshold) || IsStrongKeypoint(pose, 12, threshold);
    const bool has_leg =
        IsStrongKeypoint(pose, 13, threshold) || IsStrongKeypoint(pose, 14, threshold) ||
        IsStrongKeypoint(pose, 15, threshold) || IsStrongKeypoint(pose, 16, threshold);
    return has_shoulder && (has_hip || has_leg);
}

bool IsPoseReliableForDisplay(const PoseDetection& pose, bool strict_force_pose) {
    if (!strict_force_pose) {
        return true;
    }
    if (pose.score < kPoseConfThreshold) {
        return false;
    }
    return CountStrongKeypoints(pose, kForcePoseAcceptKptConfThreshold) >=
               kForcePoseMinStrongKeypoints &&
           HasHumanLikePoseSupport(pose, kForcePoseAcceptKptConfThreshold);
}

std::string BuildDetectionSummary(const std::vector<ObjectDetection>& detections) {
    if (detections.empty()) {
        return "none";
    }

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (i > 0) {
            oss << ";";
        }
        oss << FormatClassLabel(detections[i].class_id) << ":" << detections[i].score;
    }
    return oss.str();
}

void UpdateBestDetectionSummary(const std::vector<ObjectDetection>& detections,
                                int* best_class_id,
                                float* best_score,
                                int* det_count) {
    if (best_class_id != nullptr) {
        *best_class_id = -1;
    }
    if (best_score != nullptr) {
        *best_score = 0.0f;
    }
    if (det_count != nullptr) {
        *det_count = static_cast<int>(detections.size());
    }

    if (detections.empty()) {
        return;
    }

    int best_index = 0;
    for (size_t i = 1; i < detections.size(); ++i) {
        if (detections[i].score > detections[static_cast<size_t>(best_index)].score) {
            best_index = static_cast<int>(i);
        }
    }

    if (best_class_id != nullptr) {
        *best_class_id = detections[static_cast<size_t>(best_index)].class_id;
    }
    if (best_score != nullptr) {
        *best_score = detections[static_cast<size_t>(best_index)].score;
    }
}

void UpdateBestPoseSummary(const std::vector<PoseDetection>& poses,
                           float* best_pose_score,
                           int* pose_count) {
    if (best_pose_score != nullptr) {
        *best_pose_score = 0.0f;
    }
    if (pose_count != nullptr) {
        *pose_count = static_cast<int>(poses.size());
    }

    if (poses.empty() || best_pose_score == nullptr) {
        return;
    }

    float score = poses[0].score;
    for (size_t i = 1; i < poses.size(); ++i) {
        score = std::max(score, poses[i].score);
    }
    *best_pose_score = score;
}

enum class DemoMode {
    GUARD = 0,
    COMPANION_SNAKE = 1,
    STRANGER_FACE = 2
};

enum class GestureRoiMode {
    CENTER = 0,
    FULL = 1
};

enum class GestureMapMode {
    RAW = 0,
    LEFT_TO_UP = 1,
    RIGHT_TO_UP = 2,
    FLIP_X = 3,
    FLIP_Y = 4,
    CUSTOM = 5
};

std::atomic<int> g_custom_map_tu(static_cast<int>(GestureCommand::TU));
std::atomic<int> g_custom_map_td(static_cast<int>(GestureCommand::TD));
std::atomic<int> g_custom_map_tl(static_cast<int>(GestureCommand::TL));
std::atomic<int> g_custom_map_tr(static_cast<int>(GestureCommand::TR));

int GestureCommandIndex(GestureCommand command) {
    switch (command) {
        case GestureCommand::TU:
            return 0;
        case GestureCommand::TD:
            return 1;
        case GestureCommand::TL:
            return 2;
        case GestureCommand::TR:
            return 3;
        default:
            return -1;
    }
}

GestureCommand GestureCommandFromIndex(int index) {
    switch (index) {
        case 0:
            return GestureCommand::TU;
        case 1:
            return GestureCommand::TD;
        case 2:
            return GestureCommand::TL;
        case 3:
            return GestureCommand::TR;
        default:
            return GestureCommand::NONE;
    }
}

GestureCommand GestureCommandFromToken(std::string token) {
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (token == "tu" || token == "up") {
        return GestureCommand::TU;
    }
    if (token == "td" || token == "down") {
        return GestureCommand::TD;
    }
    if (token == "tl" || token == "left") {
        return GestureCommand::TL;
    }
    if (token == "tr" || token == "right") {
        return GestureCommand::TR;
    }
    return GestureCommand::NONE;
}

void SetCustomGestureMapping(GestureCommand source, GestureCommand target) {
    switch (source) {
        case GestureCommand::TU:
            g_custom_map_tu.store(static_cast<int>(target));
            break;
        case GestureCommand::TD:
            g_custom_map_td.store(static_cast<int>(target));
            break;
        case GestureCommand::TL:
            g_custom_map_tl.store(static_cast<int>(target));
            break;
        case GestureCommand::TR:
            g_custom_map_tr.store(static_cast<int>(target));
            break;
        default:
            break;
    }
}

const char* GestureRoiModeName(GestureRoiMode mode) {
    switch (mode) {
        case GestureRoiMode::CENTER:
            return "center";
        case GestureRoiMode::FULL:
            return "full";
        default:
            return "unknown";
    }
}

const char* GestureMapModeName(GestureMapMode mode) {
    switch (mode) {
        case GestureMapMode::RAW:
            return "raw";
        case GestureMapMode::LEFT_TO_UP:
            return "left_to_up";
        case GestureMapMode::RIGHT_TO_UP:
            return "right_to_up";
        case GestureMapMode::FLIP_X:
            return "flip_x";
        case GestureMapMode::FLIP_Y:
            return "flip_y";
        case GestureMapMode::CUSTOM:
            return "custom";
        default:
            return "raw";
    }
}

GestureCommand MapGestureCommand(GestureCommand command, GestureMapMode mode) {
    if (command == GestureCommand::NONE || mode == GestureMapMode::RAW) {
        return command;
    }

    switch (mode) {
        case GestureMapMode::CUSTOM: {
            switch (command) {
                case GestureCommand::TU:
                    return GestureCommandFromIndex(g_custom_map_tu.load());
                case GestureCommand::TD:
                    return GestureCommandFromIndex(g_custom_map_td.load());
                case GestureCommand::TL:
                    return GestureCommandFromIndex(g_custom_map_tl.load());
                case GestureCommand::TR:
                    return GestureCommandFromIndex(g_custom_map_tr.load());
                default:
                    return GestureCommand::NONE;
            }
        }
        case GestureMapMode::LEFT_TO_UP:
            switch (command) {
                case GestureCommand::TL:
                    return GestureCommand::TU;
                case GestureCommand::TU:
                    return GestureCommand::TR;
                case GestureCommand::TR:
                    return GestureCommand::TD;
                case GestureCommand::TD:
                    return GestureCommand::TL;
                default:
                    return GestureCommand::NONE;
            }
        case GestureMapMode::RIGHT_TO_UP:
            switch (command) {
                case GestureCommand::TR:
                    return GestureCommand::TU;
                case GestureCommand::TU:
                    return GestureCommand::TL;
                case GestureCommand::TL:
                    return GestureCommand::TD;
                case GestureCommand::TD:
                    return GestureCommand::TR;
                default:
                    return GestureCommand::NONE;
            }
        case GestureMapMode::FLIP_X:
            if (command == GestureCommand::TL) {
                return GestureCommand::TR;
            }
            if (command == GestureCommand::TR) {
                return GestureCommand::TL;
            }
            return command;
        case GestureMapMode::FLIP_Y:
            if (command == GestureCommand::TU) {
                return GestureCommand::TD;
            }
            if (command == GestureCommand::TD) {
                return GestureCommand::TU;
            }
            return command;
        default:
            return command;
    }
}

GestureCommand GestureDisplayCommandFromScores(const GestureResult& result,
                                               float conf_threshold) {
    // Reject only when the explicit none class is dominant enough. Otherwise
    // choose the strongest direction among down/left/right/up.
    int best_index = 0;
    for (int i = 1; i < 4; ++i) {
        if (result.probabilities[static_cast<size_t>(i)] >
            result.probabilities[static_cast<size_t>(best_index)]) {
            best_index = i;
        }
    }

    const float best_score = result.probabilities[static_cast<size_t>(best_index)];
    if (result.probabilities[4] > kGestureNoneRejectThreshold ||
        best_score < conf_threshold) {
        return GestureCommand::NONE;
    }

    // The icon should reflect the model class semantics, not the control remap.
    switch (best_index) {
        case 0:
            return GestureCommand::TD;
        case 1:
            return GestureCommand::TL;
        case 2:
            return GestureCommand::TR;
        case 3:
            return GestureCommand::TU;
        default:
            return GestureCommand::NONE;
    }
}

GestureCommand LatestValidCommand(GestureCommand stable_command,
                                  const GestureResult& mapped_result) {
    if (stable_command != GestureCommand::NONE) {
        return stable_command;
    }
    return mapped_result.valid ? mapped_result.command : GestureCommand::NONE;
}

const std::array<float, 4>* GestureFocusBoxForMode(GestureRoiMode mode) {
    switch (mode) {
        case GestureRoiMode::CENTER:
            return &kGestureGuideBoxCenterCrop;
        case GestureRoiMode::FULL:
            return nullptr;
        default:
            return &kGestureGuideBoxCenterCrop;
    }
}

std::array<float, 4> GestureGuideBoxForDisplay(GestureRoiMode mode,
                                               const std::array<int, 2>& crop_shape) {
    const std::array<float, 4>* focus_box = GestureFocusBoxForMode(mode);
    if (focus_box != nullptr) {
        return *focus_box;
    }
    return {
        0.0f,
        0.0f,
        static_cast<float>(crop_shape[0]),
        static_cast<float>(crop_shape[1])
    };
}
std::string TrimInput(const std::string& value) {
    const char* whitespace = " \t\r\n";
    const size_t begin = value.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = value.find_last_not_of(whitespace);
    return value.substr(begin, end - begin + 1);
}

int CountFacesByIdentity(const FaceResult& result, FaceIdentity identity) {
    int count = 0;
    for (int i = 0; i < result.count; ++i) {
        if (result.faces[i].identity == identity) {
            ++count;
        }
    }
    return count;
}

float BestFaceScore(const FaceResult& result) {
    float best_score = 0.0f;
    for (int i = 0; i < result.count; ++i) {
        best_score = std::max(best_score, result.faces[i].score);
    }
    return best_score;
}

float BestFaceScoreByIdentity(const FaceResult& result, FaceIdentity identity) {
    float best_score = 0.0f;
    for (int i = 0; i < result.count; ++i) {
        if (result.faces[i].identity == identity) {
            best_score = std::max(best_score, result.faces[i].score);
        }
    }
    return best_score;
}

}  // namespace

bool g_exit_flag = false;
std::mutex g_mtx;
EventRecorder g_event_recorder;
std::atomic<int> g_demo_mode(static_cast<int>(DemoMode::GUARD));
std::atomic<bool> g_snake_reset_requested(false);
std::atomic<bool> g_snake_pause_requested(false);
std::atomic<bool> g_snake_resume_requested(false);
std::atomic<int> g_forced_snake_command(static_cast<int>(GestureCommand::NONE));
std::atomic<int> g_forced_snake_hold_frames(0);
std::atomic<int> g_gesture_roi_mode(static_cast<int>(GestureRoiMode::CENTER));
std::atomic<bool> g_gesture_roi_changed(false);
// Match the gesture training input pipeline by default. The SDK reads the
// model's configured ImageNet mean/std when SetNormalize is enabled.
std::atomic<bool> g_gesture_normalize_enabled(true);
std::atomic<int> g_gesture_input_format(SSNE_RGB);
// Keep the diagnostic build in raw mode. Calibration can be enabled after
// the raw model response changes correctly with all four gestures.
std::atomic<int> g_gesture_map_mode(static_cast<int>(GestureMapMode::RAW));
std::atomic<bool> g_companion_reinit_requested(false);
// Test mode bypasses temporal filtering so model output can be compared
// directly with the direction applied to the snake.
// Production default uses temporal confirmation; enable direct mode only for diagnosis.
std::atomic<bool> g_gesture_direct_test(false);

DemoMode GetDemoMode() {
    return static_cast<DemoMode>(g_demo_mode.load());
}

void SetDemoMode(DemoMode mode) {
    g_demo_mode.store(static_cast<int>(mode));
}

GestureRoiMode GetGestureRoiMode() {
    return static_cast<GestureRoiMode>(g_gesture_roi_mode.load());
}

void SetGestureRoiMode(GestureRoiMode mode) {
    g_gesture_roi_mode.store(static_cast<int>(mode));
    g_gesture_roi_changed.store(true);
}

bool GetGestureNormalizeEnabled() {
    return g_gesture_normalize_enabled.load();
}

uint8_t GetGestureInputFormat() {
    return static_cast<uint8_t>(g_gesture_input_format.load());
}

GestureMapMode GetGestureMapMode() {
    return static_cast<GestureMapMode>(g_gesture_map_mode.load());
}

void SetGestureNormalizeEnabled(bool enabled) {
    g_gesture_normalize_enabled.store(enabled);
    g_companion_reinit_requested.store(true);
}

void SetGestureInputFormat(uint8_t format) {
    g_gesture_input_format.store(static_cast<int>(format));
    g_companion_reinit_requested.store(true);
}

void SetGestureMapMode(GestureMapMode mode) {
    g_gesture_map_mode.store(static_cast<int>(mode));
}

void PrintCompanionHelp() {
    std::cout << "mode guard|companion|stranger\n";
    std::cout << "roi center|full\n";
    std::cout << "snake reset\n";
    std::cout << "snake pause\n";
    std::cout << "snake resume\n";
    std::cout << "snake up|down|left|right\n";
    std::cout << "gesture direct on|off\n";
    std::cout << "gesture normalize on|off\n";
    std::cout << "gesture color rgb|bgr\n";
    std::cout << "gesture map raw\n";
    std::cout << "gesture map left_to_up\n";
    std::cout << "gesture map right_to_up\n";
    std::cout << "gesture map flip_x\n";
    std::cout << "gesture map flip_y\n";
    std::cout << "gesture map <up|down|left|right> <up|down|left|right>\n";
}

bool HandleDemoCommand(const std::string& line) {
    const std::string cmd = TrimInput(line);
    if (cmd.empty()) {
        return false;
    }

    if (cmd == "mode guard") {
        SetDemoMode(DemoMode::GUARD);
        std::cout << "Switched to guard mode (YOLO)." << std::endl;
        return true;
    }
    if (cmd == "mode companion") {
        SetDemoMode(DemoMode::COMPANION_SNAKE);
        g_snake_reset_requested.store(true);
        std::cout << "Switched to companion mode (gesture/snake)." << std::endl;
        return true;
    }
    if (cmd == "mode stranger") {
        SetDemoMode(DemoMode::STRANGER_FACE);
        std::cout << "Switched to stranger mode (MobileFaceNet RGB112 identity model)." << std::endl;
        return true;
    }
    if (cmd == "snake reset") {
        g_snake_reset_requested.store(true);
        std::cout << "Snake reset requested." << std::endl;
        return true;
    }
    if (cmd == "snake pause") {
        g_snake_pause_requested.store(true);
        std::cout << "Snake pause requested." << std::endl;
        return true;
    }
    if (cmd == "gesture direct on") {
        g_gesture_direct_test.store(true);
        std::cout << "Gesture direct test: on." << std::endl;
        return true;
    }
    if (cmd == "gesture direct off") {
        g_gesture_direct_test.store(false);
        std::cout << "Gesture direct test: off." << std::endl;
        return true;
    }
    if (cmd == "roi center") {
        SetGestureRoiMode(GestureRoiMode::CENTER);
        std::cout << "Gesture ROI mode: center." << std::endl;
        return true;
    }
    if (cmd == "roi full") {
        SetGestureRoiMode(GestureRoiMode::FULL);
        std::cout << "Gesture ROI mode: full." << std::endl;
        return true;
    }
    if (cmd == "gesture normalize on") {
        SetGestureNormalizeEnabled(true);
        std::cout << "Gesture normalize: on. Reinitializing companion model." << std::endl;
        return true;
    }
    if (cmd == "gesture normalize off") {
        SetGestureNormalizeEnabled(false);
        std::cout << "Gesture normalize: off. Reinitializing companion model." << std::endl;
        return true;
    }
    if (cmd == "gesture color rgb") {
        SetGestureInputFormat(SSNE_RGB);
        std::cout << "Gesture input color: RGB. Reinitializing companion model." << std::endl;
        return true;
    }
    if (cmd == "gesture color bgr") {
        SetGestureInputFormat(SSNE_BGR);
        std::cout << "Gesture input color: BGR. Reinitializing companion model." << std::endl;
        return true;
    }
    if (cmd.rfind("gesture map ", 0) == 0) {
        std::istringstream mapping_stream(cmd);
        std::string mapping_prefix;
        std::string mapping_action;
        std::string source_token;
        std::string target_token;
        mapping_stream >> mapping_prefix >> mapping_action >> source_token >> target_token;
        const GestureCommand source = GestureCommandFromToken(source_token);
        const GestureCommand target = GestureCommandFromToken(target_token);
        if (mapping_prefix == "gesture" && mapping_action == "map" &&
            source != GestureCommand::NONE && target != GestureCommand::NONE) {
            SetCustomGestureMapping(source, target);
            SetGestureMapMode(GestureMapMode::CUSTOM);
            std::cout << "Gesture custom map: "
                      << GestureCommandName(source) << " -> "
                      << GestureCommandName(target) << "." << std::endl;
            return true;
        }
    }
    if (cmd == "gesture map raw") {
        SetGestureMapMode(GestureMapMode::RAW);
        std::cout << "Gesture map mode: raw." << std::endl;
        return true;
    }
    if (cmd == "gesture map left_to_up") {
        SetGestureMapMode(GestureMapMode::LEFT_TO_UP);
        std::cout << "Gesture map mode: left_to_up." << std::endl;
        return true;
    }
    if (cmd == "gesture map right_to_up") {
        SetGestureMapMode(GestureMapMode::RIGHT_TO_UP);
        std::cout << "Gesture map mode: right_to_up." << std::endl;
        return true;
    }
    if (cmd == "gesture map flip_x") {
        SetGestureMapMode(GestureMapMode::FLIP_X);
        std::cout << "Gesture map mode: flip_x." << std::endl;
        return true;
    }
    if (cmd == "gesture map flip_y") {
        SetGestureMapMode(GestureMapMode::FLIP_Y);
        std::cout << "Gesture map mode: flip_y." << std::endl;
        return true;
    }
    if (cmd == "snake resume") {
        g_snake_resume_requested.store(true);
        std::cout << "Snake resume requested." << std::endl;
        return true;
    }
    if (cmd == "snake up") {
        g_forced_snake_command.store(static_cast<int>(GestureCommand::TU));
        g_forced_snake_hold_frames.store(60);
        std::cout << "Forced snake direction: up." << std::endl;
        return true;
    }
    if (cmd == "snake down") {
        g_forced_snake_command.store(static_cast<int>(GestureCommand::TD));
        g_forced_snake_hold_frames.store(60);
        std::cout << "Forced snake direction: down." << std::endl;
        return true;
    }
    if (cmd == "snake left") {
        g_forced_snake_command.store(static_cast<int>(GestureCommand::TL));
        g_forced_snake_hold_frames.store(60);
        std::cout << "Forced snake direction: left." << std::endl;
        return true;
    }
    if (cmd == "snake right") {
        g_forced_snake_command.store(static_cast<int>(GestureCommand::TR));
        g_forced_snake_hold_frames.store(60);
        std::cout << "Forced snake direction: right." << std::endl;
        return true;
    }
    if (cmd == "help") {
        PrintCompanionHelp();
        return true;
    }
    return false;
}

void keyboard_listener() {
    std::string input;
    std::cout << "Keyboard listener started, input 'help' for commands, 'q' to quit..." << std::endl;

    while (std::getline(std::cin, input)) {
        bool should_exit = false;
        if (HandleDemoCommand(input)) {
            continue;
        }
        if (!g_event_recorder.HandleCommand(input, &should_exit)) {
            std::cout << "Unknown command. Input 'help' for commands." << std::endl;
            continue;
        }

        if (should_exit) {
            std::lock_guard<std::mutex> lock(g_mtx);
            g_exit_flag = true;
            break;
        }
    }
}

bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}

struct MainLoopPerfStats {
    uint64_t frames = 0;
    double capture_ms = 0.0;
    double detect_ms = 0.0;
    double pose_ms = 0.0;
    double osd_ms = 0.0;
    int last_det_count = 0;
    int last_best_class = -1;
    float last_best_score = 0.0f;
    int last_pose_count = 0;
    float last_best_pose_score = 0.0f;
    std::string last_detection_summary = "none";
    std::string last_fall_state = "NORMAL";
    std::chrono::steady_clock::time_point window_begin = std::chrono::steady_clock::now();
};

struct SnakeLoopPerfStats {
    uint64_t frames = 0;
    double capture_ms = 0.0;
    double gesture_ms = 0.0;
    double game_ms = 0.0;
    double osd_ms = 0.0;
    int last_score = 0;
    int last_best_score = 0;
    int last_snake_len = 0;
    int last_head_x = -1;
    int last_head_y = -1;
    int last_food_x = -1;
    int last_food_y = -1;
    GestureCommand last_command = GestureCommand::NONE;
    GestureCommand last_raw_command = GestureCommand::NONE;
    GestureCommand last_stable_command = GestureCommand::NONE;
    GestureCommand last_held_command = GestureCommand::NONE;
    GestureCommand last_applied_command = GestureCommand::NONE;
    SnakeDirection last_direction = SnakeDirection::RIGHT;
    float last_confidence = 0.0f;
    std::array<float, 5> last_logits = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 5> last_probs = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::string last_state = "running";
    const char* last_roi_mode = "center";
    const char* last_norm_mode = "on";
    const char* last_color_mode = "RGB";
    const char* last_map_mode = "raw";
    std::chrono::steady_clock::time_point window_begin = std::chrono::steady_clock::now();
};

struct PoseRequest {
    bool pending = false;
    std::array<float, 4> focus_box = {0.0f, 0.0f, 0.0f, 0.0f};
};

void FlushMainLoopPerfIfNeeded(MainLoopPerfStats* stats) {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    const double elapsed_ms =
        static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(now - stats->window_begin).count());
    if (elapsed_ms < 1000.0 || stats->frames == 0) {
        return;
    }

    const double inv = 1.0 / static_cast<double>(stats->frames);
    const double fps = static_cast<double>(stats->frames) * 1000.0 / elapsed_ms;
    const double avg_capture_ms = stats->capture_ms * inv;
    const double avg_detect_ms = stats->detect_ms * inv;
    const double avg_pose_ms = stats->pose_ms * inv;
    const double avg_osd_ms = stats->osd_ms * inv;
    const double avg_total_ms = avg_capture_ms + avg_detect_ms + avg_pose_ms + avg_osd_ms;
    const std::string best_class_label = FormatClassLabel(stats->last_best_class);
    LOG_INFO("serial fps=%.2f total=%.2fms capture=%.2fms detect=%.2fms pose=%.2fms osd=%.2fms det_count=%d detections=%s best=%s:%.3f pose_count=%d pose_score=%.3f fall=%s\n",
             fps,
             avg_total_ms,
             avg_capture_ms,
             avg_detect_ms,
             avg_pose_ms,
             avg_osd_ms,
             stats->last_det_count,
             stats->last_detection_summary.c_str(),
             best_class_label.c_str(),
             stats->last_best_score,
             stats->last_pose_count,
             stats->last_best_pose_score,
             stats->last_fall_state.c_str());

    stats->frames = 0;
    stats->capture_ms = 0.0;
    stats->detect_ms = 0.0;
    stats->pose_ms = 0.0;
    stats->osd_ms = 0.0;
    stats->last_det_count = 0;
    stats->last_best_class = -1;
    stats->last_best_score = 0.0f;
    stats->last_pose_count = 0;
    stats->last_best_pose_score = 0.0f;
    stats->last_detection_summary = "none";
    stats->window_begin = now;
}

void FlushSnakePerfIfNeeded(SnakeLoopPerfStats* stats) {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    const double elapsed_ms =
        static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(now - stats->window_begin).count());
    if (elapsed_ms < 1000.0 || stats->frames == 0) {
        return;
    }

    const double inv = 1.0 / static_cast<double>(stats->frames);
    const double fps = static_cast<double>(stats->frames) * 1000.0 / elapsed_ms;
    LOG_INFO("serial mode=snake build=snake_gesture_test_v35 fps=%.2f capture=%.2fms gesture=%.2fms game=%.2fms osd=%.2fms score=%d best=%d len=%d head=(%d,%d) food=(%d,%d) roi=%s norm=%s color=%s score_mode=softmax_multiclass threshold=%.2f none_reject=%.2f map=%s direct=%d display=%s raw=%s stable=%s held=%s applied=%s dir=%s conf=%.3f logits=[D %.3f L %.3f R %.3f U %.3f N %.3f] scores=[D %.3f L %.3f R %.3f U %.3f N %.3f] state=%s\n",
             fps,
             stats->capture_ms * inv,
             stats->gesture_ms * inv,
             stats->game_ms * inv,
             stats->osd_ms * inv,
             stats->last_score,
             stats->last_best_score,
             stats->last_snake_len,
             stats->last_head_x,
             stats->last_head_y,
             stats->last_food_x,
             stats->last_food_y,
             stats->last_roi_mode,
              stats->last_norm_mode,
              stats->last_color_mode,
              kGestureConfThreshold,
              kGestureNoneRejectThreshold,
              stats->last_map_mode,
              g_gesture_direct_test.load() ? 1 : 0,
              GestureCommandName(stats->last_command),
             GestureCommandName(stats->last_raw_command),
             GestureCommandName(stats->last_stable_command),
             GestureCommandName(stats->last_held_command),
             GestureCommandName(stats->last_applied_command),
             SnakeDirectionName(stats->last_direction),
             stats->last_confidence,
             stats->last_logits[0],
             stats->last_logits[1],
             stats->last_logits[2],
              stats->last_logits[3],
              stats->last_logits[4],
              stats->last_probs[0],
              stats->last_probs[1],
              stats->last_probs[2],
              stats->last_probs[3],
              stats->last_probs[4],
              stats->last_state.c_str());

    stats->frames = 0;
    stats->capture_ms = 0.0;
    stats->gesture_ms = 0.0;
    stats->game_ms = 0.0;
    stats->osd_ms = 0.0;
    stats->last_score = 0;
    stats->last_best_score = 0;
    stats->last_snake_len = 0;
    stats->last_head_x = -1;
    stats->last_head_y = -1;
    stats->last_food_x = -1;
    stats->last_food_y = -1;
    stats->last_command = GestureCommand::NONE;
    stats->last_raw_command = GestureCommand::NONE;
    stats->last_stable_command = GestureCommand::NONE;
    stats->last_held_command = GestureCommand::NONE;
    stats->last_applied_command = GestureCommand::NONE;
    stats->last_direction = SnakeDirection::RIGHT;
    stats->last_confidence = 0.0f;
    stats->last_logits = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    stats->last_probs = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    stats->last_state = "running";
    stats->last_roi_mode = "center";
    stats->last_norm_mode = "on";
    stats->last_color_mode = "RGB";
    stats->last_map_mode = "raw";
    stats->window_begin = now;
}

int main() {
    using clock = std::chrono::steady_clock;

    const int img_width = 1920;
    const int img_height = 1080;
    const int crop_offset_x = 420;

    std::array<int, 2> img_shape = {img_width, img_height};
    std::array<int, 2> crop_shape = {1080, 1080};
    std::array<int, 2> detect_shape = {256, 256};
    std::array<int, 2> gesture_shape = {640, 640};
    std::array<int, 2> pose_shape = {480, 320};

    std::string detect_model_path = "/app_demo/app_assets/models/yolov8nano.m1model";
    std::string pose_model_path = "/app_demo/app_assets/models/yolov8_pose.m1model";
    std::string gesture_model_path = "/app_demo/app_assets/models/gesture_mobilenetv1.m1model";
    std::string stranger_face_model_path =
        std::string("/app_demo/app_assets/models/") + kStrangerFaceModelName;

    IMAGEPROCESSOR processor;
    YOLOV8NANO detect_detector;
    YUNET pose_detector;
    GestureClassifier gesture_classifier;
    StrangerModeRunner stranger_mode_runner;
    ObjectDetectionResult detect_result;
    FaceDetectionResult pose_result;
    VISUALIZER visualizer;
    FaceResult stranger_face_result = {};

    bool runtime_initialized = false;
    bool guard_models_initialized = false;
    bool companion_model_initialized = false;
    bool stranger_mode_initialized = false;

    auto ReleaseGuardModels = [&]() {
        if (guard_models_initialized) {
            pose_detector.Release();
            detect_detector.Release();
            guard_models_initialized = false;
        }
    };

    auto ReleaseCompanionModel = [&]() {
        if (companion_model_initialized) {
            gesture_classifier.Release();
            companion_model_initialized = false;
        }
    };

    auto ReleaseStrangerMode = [&]() {
        if (stranger_mode_initialized) {
            stranger_mode_runner.Release();
            stranger_mode_initialized = false;
        }
    };

    auto ReleaseRuntime = [&]() {
        ReleaseStrangerMode();
        ReleaseCompanionModel();
        ReleaseGuardModels();
        if (runtime_initialized) {
            processor.Release();
            visualizer.Release();
            if (ssne_release()) {
                fprintf(stderr, "SSNE release failed during mode switch!\n");
            }
            runtime_initialized = false;
        }
    };

    auto InitializeCommonRuntime = [&](const std::string& bitmap_lut_path) {
        if (runtime_initialized) {
            return;
        }
        if (ssne_initial()) {
            fprintf(stderr, "SSNE initialization failed!\n");
            return;
        }
        processor.Initialize(&img_shape);
        visualizer.Initialize(img_shape, bitmap_lut_path);
        runtime_initialized = true;
    };

    auto InitializeGuardMode = [&]() {
        if (guard_models_initialized) {
            return;
        }
        InitializeCommonRuntime("");
        if (!runtime_initialized) {
            return;
        }
        detect_detector.Initialize(detect_model_path, &crop_shape, &detect_shape, 300, kDetectNumClasses);
        detect_detector.SetClassThresholds(kDetectClassThresholds);
        detect_detector.SetTemporalPersonConfThreshold(kDetectTemporalPersonConfThreshold);
        pose_detector.Initialize(pose_model_path, &crop_shape, &pose_shape, true, 300);
        guard_models_initialized = true;
        LOG_INFO("guard mode models initialized\n");
    };

    auto InitializeCompanionMode = [&]() {
        if (companion_model_initialized) {
            return;
        }
        InitializeCommonRuntime("ui/snake/shared_colorLUT.sscl");
        if (!runtime_initialized) {
            return;
        }
        gesture_classifier.Initialize(gesture_model_path,
                                      &crop_shape,
                                      &gesture_shape,
                                      GetGestureNormalizeEnabled(),
                                      GetGestureInputFormat());
        if (!gesture_classifier.IsInitialized()) {
            LOG_ERROR("companion gesture model initialization failed; path=%s\n",
                      gesture_model_path.c_str());
            return;
        }
        companion_model_initialized = true;
        LOG_INFO("companion mode model initialized\n");
    };

    auto InitializeStrangerMode = [&]() {
        if (stranger_mode_initialized) {
            return;
        }
        InitializeCommonRuntime("");
        if (!runtime_initialized) {
            return;
        }
        stranger_mode_runner.Initialize(stranger_face_model_path);
        stranger_mode_initialized = true;
        LOG_INFO("stranger mode interface initialized, model path=%s\n",
                 stranger_face_model_path.c_str());
    };

    auto PrepareModeRuntime = [&](DemoMode mode) {
        ReleaseRuntime();
        if (mode == DemoMode::COMPANION_SNAKE) {
            InitializeCompanionMode();
        } else if (mode == DemoMode::STRANGER_FACE) {
            InitializeStrangerMode();
        } else {
            InitializeGuardMode();
        }
        if (runtime_initialized) {
            LOG_INFO("warmup sleep for 1 second after mode runtime init\n");
            sleep(1);
        }
    };

    PrepareModeRuntime(GetDemoMode());

    FallJudge judge;
    FallJudgeConfig cfg;
    cfg.image_width = img_width;
    cfg.image_height = img_height;
    cfg.person_class_id = kPersonClassId;
    cfg.min_score = 0.35f;
    cfg.horizontal_ratio = 1.2f;
    cfg.downward_motion_ratio = 0.08f;
    cfg.stable_motion_ratio = 0.02f;
    cfg.keypoint_min_conf = 0.35f;
    cfg.horizontal_spine_angle_deg = 35.0f;
    cfg.upside_down_dy_ratio = 0.02f;
    cfg.track_iou_threshold = 0.10f;
    cfg.side_spine_delta_deg = 28.0f;
    cfg.side_box_ratio_with_pose = 1.10f;
    cfg.side_box_ratio_no_pose = 1.25f;
    cfg.front_head_below_margin_ratio = 0.02f;
    cfg.front_ankle_stable_ratio = 0.04f;
    cfg.front_segment_shrink_ratio = 0.82f;
    cfg.front_segment_consistency_ratio = 0.18f;
    cfg.front_aspect_change_ratio = 0.20f;
    cfg.track_high_thresh = 0.42f;
    cfg.track_low_thresh = 0.12f;
    cfg.track_high_match_iou = 0.12f;
    cfg.track_low_match_iou = 0.03f;
    cfg.track_center_distance_ratio = 0.20f;
    cfg.track_size_ratio_min = 0.35f;
    cfg.track_size_ratio_max = 2.80f;
    cfg.primary_switch_margin = 0.30f;
    cfg.track_max_age = 12;
    cfg.track_min_hits = 2;
    cfg.primary_switch_hold_frames = 12;
    cfg.suspect_frames = 2;
    cfg.confirm_frames = 4;
    cfg.reset_frames = 6;
    judge.Initialize(cfg);

    FallState last_fall_state = FallState::NORMAL;
    uint64_t frame_index = 0;
    uint64_t next_pose_frame = 0;
    uint64_t pose_assist_detect_until = 0;
    MainLoopPerfStats perf_stats;
    SnakeLoopPerfStats snake_perf_stats;
    ssne_tensor_t img_sensor = ssne_tensor_t{};
    std::vector<PoseDetection> cached_poses_original_coord;
    std::vector<PoseDetection> last_visual_poses_original_coord;
    std::vector<ObjectDetection> detections_original_coord;
    PoseRequest pose_request;
    float last_visual_pose_kpt_threshold = kPoseDrawKptConfThreshold;
    bool last_visual_pose_valid = false;
    uint64_t last_visual_pose_frame_index = 0;
    bool person_like_visual_active = false;
    GestureTemporalFilter gesture_filter(3, 2, 10);
    SnakeGame snake_game;
    snake_game.Initialize(kSnakeBoardCols, kSnakeBoardRows);
    GestureCommand last_stable_command = GestureCommand::NONE;
    GestureCommand held_snake_command = GestureCommand::NONE;
    GestureCommand last_display_command = GestureCommand::NONE;
    auto next_snake_tick = clock::now() + std::chrono::milliseconds(snake_game.TickIntervalMs());
    DemoMode last_mode = GetDemoMode();
    SnakeRenderData last_snake_render_data;
    bool has_last_snake_render_data = false;
    bool gesture_guide_visible = false;

    std::thread listener_thread(keyboard_listener);

    while (!check_exit_flag()) {
        const DemoMode current_mode = GetDemoMode();
        if (current_mode == DemoMode::COMPANION_SNAKE &&
            g_companion_reinit_requested.exchange(false)) {
            PrepareModeRuntime(current_mode);
            gesture_filter.Reset();
            snake_game.Reset();
            last_stable_command = GestureCommand::NONE;
            held_snake_command = GestureCommand::NONE;
            last_display_command = GestureCommand::NONE;
            next_snake_tick = clock::now() + std::chrono::milliseconds(snake_game.TickIntervalMs());
            has_last_snake_render_data = false;
            gesture_guide_visible = false;
            continue;
        }

        if (current_mode != last_mode) {
            PrepareModeRuntime(current_mode);
            detect_result.Clear();
            pose_result.Clear();
            cached_poses_original_coord.clear();
            last_visual_poses_original_coord.clear();
            detections_original_coord.clear();
            pose_request.pending = false;
            last_visual_pose_valid = false;
            last_visual_pose_frame_index = 0;
            person_like_visual_active = false;
            stranger_face_result = FaceResult{};
            gesture_filter.Reset();
            last_stable_command = GestureCommand::NONE;
            held_snake_command = GestureCommand::NONE;
            last_display_command = GestureCommand::NONE;
            has_last_snake_render_data = false;
            gesture_guide_visible = false;
            if (current_mode == DemoMode::COMPANION_SNAKE) {
                snake_game.Reset();
                held_snake_command = GestureCommand::NONE;
                last_display_command = GestureCommand::NONE;
                next_snake_tick = clock::now() + std::chrono::milliseconds(snake_game.TickIntervalMs());
            } else {
                next_pose_frame = 0;
                pose_assist_detect_until = 0;
            }
            last_mode = current_mode;
            continue;
        }

        if (!runtime_initialized) {
            sleep(1);
            continue;
        }

        const auto capture_begin = clock::now();
        processor.GetImage(&img_sensor);
        const auto capture_end = clock::now();

        if (current_mode == DemoMode::COMPANION_SNAKE) {
            if (g_snake_reset_requested.exchange(false)) {
                snake_game.Reset();
                gesture_filter.Reset();
                last_stable_command = GestureCommand::NONE;
                held_snake_command = GestureCommand::NONE;
                last_display_command = GestureCommand::NONE;
                next_snake_tick = clock::now() + std::chrono::milliseconds(snake_game.TickIntervalMs());
                has_last_snake_render_data = false;
                gesture_guide_visible = false;
            }
            if (g_snake_pause_requested.exchange(false)) {
                snake_game.SetPaused(true);
            }
            if (g_snake_resume_requested.exchange(false)) {
                snake_game.SetPaused(false);
            }

            if (g_gesture_roi_changed.exchange(false)) {
                gesture_filter.Reset();
                last_stable_command = GestureCommand::NONE;
                held_snake_command = GestureCommand::NONE;
                last_display_command = GestureCommand::NONE;
                gesture_guide_visible = false;
            }
            const GestureRoiMode gesture_roi_mode = GetGestureRoiMode();
            const std::array<float, 4>* gesture_focus_box =
                GestureFocusBoxForMode(gesture_roi_mode);
            GestureResult gesture_result;
            gesture_classifier.SetFocusBox(gesture_focus_box);
            const auto gesture_begin = clock::now();
            gesture_classifier.Predict(&img_sensor, &gesture_result, kGestureConfThreshold);
            const auto gesture_end = clock::now();

            const GestureCommand raw_gesture_command = gesture_result.command;
            const GestureMapMode gesture_map_mode = GetGestureMapMode();
            const GestureCommand raw_display_command =
                GestureDisplayCommandFromScores(gesture_result, kGestureConfThreshold);
            gesture_result.command =
                MapGestureCommand(gesture_result.command, gesture_map_mode);
            gesture_result.valid = gesture_result.command != GestureCommand::NONE;
            const GestureCommand display_command =
                MapGestureCommand(raw_display_command, gesture_map_mode);
            // Display the current frame result. Do not retain a previous
            // direction when the current model result is NONE.
            last_display_command = display_command;
            const GestureCommand filtered_command = gesture_filter.Push(gesture_result);
            GestureCommand forced_command = GestureCommand::NONE;
            if (g_forced_snake_hold_frames.load() > 0) {
                forced_command =
                    static_cast<GestureCommand>(g_forced_snake_command.load());
                g_forced_snake_hold_frames.fetch_sub(1);
            } else {
                g_forced_snake_command.store(static_cast<int>(GestureCommand::NONE));
            }
            const bool direct_test = g_gesture_direct_test.load();
            const GestureCommand stable_command =
                forced_command != GestureCommand::NONE
                    ? forced_command
                    : (direct_test
                           ? (gesture_result.valid ? gesture_result.command : GestureCommand::NONE)
                           : filtered_command);
            GestureCommand applied_command = GestureCommand::NONE;
            const GestureCommand latest_valid_command =
                LatestValidCommand(stable_command, gesture_result);
            if (direct_test && stable_command == GestureCommand::NONE) {
                // In diagnostic mode do not keep an old command alive through
                // an invalid frame; this exposes the actual model response.
                held_snake_command = GestureCommand::NONE;
                last_stable_command = GestureCommand::NONE;
                last_display_command = GestureCommand::NONE;
            } else if (latest_valid_command != GestureCommand::NONE) {
                held_snake_command = latest_valid_command;
            }
            if (direct_test) {
                last_stable_command = stable_command;
            } else if (stable_command != GestureCommand::NONE) {
                last_stable_command = stable_command;
                if (snake_game.IsGameOver() && forced_command != GestureCommand::NONE) {
                    snake_game.Reset();
                    held_snake_command = stable_command;
                    next_snake_tick = clock::now() + std::chrono::milliseconds(snake_game.TickIntervalMs());
                    has_last_snake_render_data = false;
                }
            }

            const auto game_begin = clock::now();
            int tick_guard = 0;
            const auto snake_now = clock::now();
            while (snake_now >= next_snake_tick && tick_guard < 1) {
                if (held_snake_command != GestureCommand::NONE &&
                    !snake_game.IsGameOver()) {
                    snake_game.SetDirection(
                        GestureToSnakeDirection(held_snake_command, snake_game.Direction()));
                    applied_command = held_snake_command;
                }
                snake_game.Tick();
                next_snake_tick += std::chrono::milliseconds(snake_game.TickIntervalMs());
                ++tick_guard;
            }
            SnakeRenderData render_data = snake_game.BuildRenderData();
            render_data.last_command = last_display_command;
            render_data.last_command_confidence = gesture_result.confidence;
            const auto game_end = clock::now();

            const auto osd_begin = clock::now();
            if (!gesture_guide_visible) {
                std::vector<ObjectDetection> guide_boxes(2);
                const std::array<float, 4> gesture_guide_box =
                    GestureGuideBoxForDisplay(gesture_roi_mode, crop_shape);
                const std::array<float, 4> gesture_guide_box_original =
                    MapBoxToOriginal(gesture_guide_box,
                                     crop_offset_x,
                                     img_width,
                                     img_height);
                LOG_INFO("gesture guide box: mode=%s inference_crop_coord=[%.1f,%.1f,%.1f,%.1f] original_osd_coord=[%.1f,%.1f,%.1f,%.1f] snake_osd_coord=[%.1f,%.1f,%.1f,%.1f] gap_x=%.1f overlap=%s crop_offset_x=%d origin=top_left coord_space=crop_1080x1080\n",
                         GestureRoiModeName(gesture_roi_mode),
                         gesture_guide_box[0],
                         gesture_guide_box[1],
                         gesture_guide_box[2],
                         gesture_guide_box[3],
                         gesture_guide_box_original[0],
                         gesture_guide_box_original[1],
                         gesture_guide_box_original[2],
                         gesture_guide_box_original[3],
                         kSnakeBoardBoxOriginal[0],
                         kSnakeBoardBoxOriginal[1],
                         kSnakeBoardBoxOriginal[2],
                         kSnakeBoardBoxOriginal[3],
                         HorizontalGap(gesture_guide_box_original,
                                       kSnakeBoardBoxOriginal),
                         BoxesOverlap(gesture_guide_box_original,
                                      kSnakeBoardBoxOriginal) ? "yes" : "no",
                         crop_offset_x);
                guide_boxes[0].box = gesture_guide_box_original;
                guide_boxes[0].score = 1.0f;
                guide_boxes[0].class_id = kPersonClassId;
                guide_boxes[1].box = kSnakeBoardBoxOriginal;
                guide_boxes[1].score = 1.0f;
                guide_boxes[1].class_id = kPersonClassId;
                visualizer.ClearLayer(VISUALIZER::DETECTION_LAYER_ID);
                visualizer.Draw(guide_boxes);
                gesture_guide_visible = true;
            }
            if (!has_last_snake_render_data ||
                !IsSameSnakeRenderData(render_data, last_snake_render_data)) {
                visualizer.DrawSnakeGame(render_data);
                last_snake_render_data = render_data;
                has_last_snake_render_data = true;
            }
            const auto osd_end = clock::now();

            snake_perf_stats.frames += 1;
            snake_perf_stats.capture_ms +=
                static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(capture_end - capture_begin).count()) / 1000.0;
            snake_perf_stats.gesture_ms +=
                static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(gesture_end - gesture_begin).count()) / 1000.0;
            snake_perf_stats.game_ms +=
                static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(game_end - game_begin).count()) / 1000.0;
            snake_perf_stats.osd_ms +=
                static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(osd_end - osd_begin).count()) / 1000.0;
            snake_perf_stats.last_score = render_data.score;
            snake_perf_stats.last_best_score = render_data.best_score;
            snake_perf_stats.last_snake_len = static_cast<int>(render_data.snake.size());
            if (!render_data.snake.empty()) {
                snake_perf_stats.last_head_x = render_data.snake.front().x;
                snake_perf_stats.last_head_y = render_data.snake.front().y;
            } else {
                snake_perf_stats.last_head_x = -1;
                snake_perf_stats.last_head_y = -1;
            }
            snake_perf_stats.last_food_x = render_data.has_food ? render_data.food.x : -1;
            snake_perf_stats.last_food_y = render_data.has_food ? render_data.food.y : -1;
            snake_perf_stats.last_command = render_data.last_command;
            snake_perf_stats.last_raw_command = raw_gesture_command;
            snake_perf_stats.last_stable_command = last_stable_command;
            snake_perf_stats.last_held_command = held_snake_command;
            snake_perf_stats.last_applied_command = applied_command;
            snake_perf_stats.last_direction = snake_game.Direction();
            snake_perf_stats.last_confidence = render_data.last_command_confidence;
            snake_perf_stats.last_logits = gesture_result.logits;
            snake_perf_stats.last_probs = gesture_result.probabilities;
            snake_perf_stats.last_state =
                render_data.game_over ? "game_over" : (render_data.paused ? "paused" : "running");
            snake_perf_stats.last_roi_mode = GestureRoiModeName(gesture_roi_mode);
            snake_perf_stats.last_norm_mode =
                GetGestureNormalizeEnabled() ? "on" : "off";
            snake_perf_stats.last_color_mode =
                GetGestureInputFormat() == SSNE_BGR ? "BGR" : "RGB";
            snake_perf_stats.last_map_mode =
                GestureMapModeName(gesture_map_mode);
            FlushSnakePerfIfNeeded(&snake_perf_stats);

            ++frame_index;
            continue;
        }

        if (current_mode == DemoMode::STRANGER_FACE) {
            const int stranger_ret = stranger_mode_runner.ProcessFrame(
                &img_sensor,
                crop_offset_x,
                img_width,
                img_height,
                static_cast<uint16_t>(frame_index & 0xFFFFU),
                &stranger_face_result);
            if ((frame_index % 90U) == 0U) {
                const int known_count =
                    CountFacesByIdentity(stranger_face_result, FaceIdentity::kKnown);
                const int stranger_count =
                    CountFacesByIdentity(stranger_face_result, FaceIdentity::kStranger);
                const int unknown_count =
                    CountFacesByIdentity(stranger_face_result, FaceIdentity::kUnknown);
                const float best_score = BestFaceScore(stranger_face_result);
                const float known_best =
                    BestFaceScoreByIdentity(stranger_face_result, FaceIdentity::kKnown);
                const float stranger_best =
                    BestFaceScoreByIdentity(stranger_face_result, FaceIdentity::kStranger);
                const float unknown_best =
                    BestFaceScoreByIdentity(stranger_face_result, FaceIdentity::kUnknown);
                LOG_INFO("serial mode=stranger faces=%d known=%d stranger=%d unknown=%d best=%.3f known_best=%.3f stranger_best=%.3f unknown_best=%.3f sim=%.3f instant=%d confirmed=%d vote=%d/%d rec_ret=%d status=%s model=%s ret=%d\n",
                         stranger_face_result.count,
                         known_count,
                         stranger_count,
                         unknown_count,
                         best_score,
                         known_best,
                         stranger_best,
                         unknown_best,
                         stranger_mode_runner.LastSimilarity(),
                         stranger_mode_runner.LastInstantOwner() ? 1 : 0,
                         stranger_mode_runner.LastConfirmedOwner() ? 1 : 0,
                         stranger_mode_runner.LastVotePassCount(),
                         stranger_mode_runner.LastVoteSampleCount(),
                         stranger_mode_runner.LastRecognizeStatus(),
                         stranger_mode_runner.LastError().c_str(),
                         stranger_mode_runner.ModelPath().c_str(),
                         stranger_ret);
            }
            visualizer.Draw(stranger_face_result);
            ++frame_index;
            continue;
        }

        const bool force_pose_due =
            person_like_visual_active &&
            !pose_request.pending &&
            frame_index > 0 &&
            (frame_index % static_cast<uint64_t>(kPoseForceInterval)) == 0;
        const bool should_run_pose =
            (pose_request.pending &&
             (cached_poses_original_coord.empty() || frame_index >= next_pose_frame)) ||
            force_pose_due;
        const bool use_pose_focus = pose_request.pending;
        const bool strict_force_pose =
            should_run_pose && force_pose_due && !use_pose_focus;
        double detect_ms = 0.0;
        double pose_ms = 0.0;

        if (should_run_pose) {
            pose_detector.SetEnhanceFocusBox(use_pose_focus ? &pose_request.focus_box : nullptr);
            const auto pose_begin = clock::now();
            pose_detector.Predict(&img_sensor, &pose_result, kPoseConfThreshold);
            const auto pose_end = clock::now();
            pose_ms = static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(pose_end - pose_begin).count()) / 1000.0;

            cached_poses_original_coord.clear();
            cached_poses_original_coord.reserve(pose_result.detections.size());
            for (const auto& det : pose_result.detections) {
                PoseDetection mapped = MapPoseToOriginal(det, crop_offset_x, img_width, img_height);
                if (mapped.box[2] > mapped.box[0] &&
                    mapped.box[3] > mapped.box[1] &&
                    IsPoseReliableForDisplay(mapped, strict_force_pose)) {
                    cached_poses_original_coord.push_back(mapped);
                }
            }
            pose_request.pending = false;
            next_pose_frame = frame_index + static_cast<uint64_t>(kPoseInvokeInterval);
            if (!cached_poses_original_coord.empty()) {
                pose_assist_detect_until =
                    frame_index + static_cast<uint64_t>(kPoseAssistDetectFrames);
            }
            pose_detector.SetEnhanceFocusBox(nullptr);
        } else {
            const auto detect_begin = clock::now();
            const float active_person_detect_threshold =
                frame_index < pose_assist_detect_until ?
                kDetectPoseAssistPersonConfThreshold :
                -1.0f;
            detect_detector.Predict(&img_sensor, &detect_result,
                                    kDetectFallbackConfThreshold,
                                    active_person_detect_threshold);
            const auto detect_end = clock::now();
            detect_ms = static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(detect_end - detect_begin).count()) / 1000.0;

            detections_original_coord.clear();
            detections_original_coord.reserve(detect_result.detections.size());
            for (const auto& det : detect_result.detections) {
                ObjectDetection mapped = MapDetectionToOriginal(det, crop_offset_x, img_width, img_height);
                if (mapped.box[2] > mapped.box[0] && mapped.box[3] > mapped.box[1]) {
                    detections_original_coord.push_back(mapped);
                }
            }
            ApplyFireBrightnessFallback(img_sensor, crop_offset_x, &detections_original_coord);
            UpdateBestDetectionSummary(detections_original_coord,
                                       &perf_stats.last_best_class,
                                       &perf_stats.last_best_score,
                                       &perf_stats.last_det_count);
            perf_stats.last_detection_summary = BuildDetectionSummary(detections_original_coord);
            person_like_visual_active = HasPersonLikeDetection(detections_original_coord);
            if (!person_like_visual_active) {
                cached_poses_original_coord.clear();
                last_visual_poses_original_coord.clear();
                last_visual_pose_valid = false;
            }
            pose_detector.SetEnhanceFocusBox(nullptr);
        }
        UpdateBestPoseSummary(cached_poses_original_coord,
                              &perf_stats.last_best_pose_score,
                              &perf_stats.last_pose_count);

        std::vector<DetectionBox> judge_inputs;
        if (should_run_pose && !cached_poses_original_coord.empty()) {
            judge_inputs.reserve(cached_poses_original_coord.size());
            for (const auto& pose : cached_poses_original_coord) {
                DetectionBox d;
                d.box = pose.box;
                d.score = std::max(pose.score, kPoseTrackScoreFloor);
                d.class_id = kPersonClassId;
                judge_inputs.push_back(d);
            }
        } else {
            judge_inputs.reserve(detections_original_coord.size());
            for (const auto& det : detections_original_coord) {
                DetectionBox d;
                d.box = det.box;
                d.score = det.score;
                d.class_id = NormalizeClassForTracking(det.class_id);
                judge_inputs.push_back(d);
            }
        }

        const FallState fall_state = judge.Update(judge_inputs, cached_poses_original_coord, should_run_pose);
        g_event_recorder.Update(fall_state == FallState::CONFIRMED,
                                HasIntrusionDetection(detections_original_coord),
                                HasFireDetection(detections_original_coord),
                                frame_index);
        perf_stats.last_fall_state = judge.GetStateString();
        if (fall_state != last_fall_state) {
            if (fall_state == FallState::SUSPECT) {
                LOG_INFO("fall state: suspect\n");
            } else if (fall_state == FallState::CONFIRMED) {
                LOG_INFO("fall state: confirmed\n");
            } else {
                LOG_INFO("fall state: normal\n");
            }
            last_fall_state = fall_state;
        }

        if (!should_run_pose) {
            std::array<float, 4> tracked_box_original = {0.0f, 0.0f, 0.0f, 0.0f};
            std::array<float, 4> tracked_box_crop = {0.0f, 0.0f, 0.0f, 0.0f};
            const bool cooldown_legal = !pose_request.pending && frame_index >= next_pose_frame;
            const bool has_primary_track =
                judge.GetTrackedBox(tracked_box_original) &&
                MapTrackedBoxToCrop(tracked_box_original, crop_offset_x, crop_shape, &tracked_box_crop);
            const bool roi_legal =
                has_primary_track && IsPoseRequestRoiLegal(tracked_box_crop, crop_shape, pose_shape);

            if (cooldown_legal && roi_legal && person_like_visual_active) {
                pose_request.pending = true;
                pose_request.focus_box = tracked_box_crop;
            } else if (!has_primary_track || !person_like_visual_active) {
                pose_request.pending = false;
            }
        }

        if (should_run_pose && !cached_poses_original_coord.empty() && person_like_visual_active) {
            bool visual_pose_updated = false;
            PoseDetection tracked_pose;
            if (judge.GetTrackedPose(tracked_pose)) {
                last_visual_poses_original_coord.clear();
                last_visual_poses_original_coord.push_back(tracked_pose);
                visual_pose_updated = true;
            } else {
                auto best_pose = std::max_element(
                    cached_poses_original_coord.begin(),
                    cached_poses_original_coord.end(),
                    [](const PoseDetection& a, const PoseDetection& b) {
                        return a.score < b.score;
                    });
                if (best_pose != cached_poses_original_coord.end()) {
                    last_visual_poses_original_coord.clear();
                    last_visual_poses_original_coord.push_back(*best_pose);
                    visual_pose_updated = true;
                }
            }
            if (visual_pose_updated) {
                last_visual_pose_valid = true;
                last_visual_pose_frame_index = frame_index;
            }
            last_visual_pose_kpt_threshold =
                strict_force_pose ? kForcePoseDrawKptConfThreshold : kPoseDrawKptConfThreshold;
        } else if (should_run_pose) {
            last_visual_poses_original_coord.clear();
            last_visual_pose_valid = false;
        }

        if (last_visual_pose_valid &&
            frame_index - last_visual_pose_frame_index >
                static_cast<uint64_t>(kPoseVisualMaxHoldFrames)) {
            last_visual_poses_original_coord.clear();
            last_visual_pose_valid = false;
        }
        const std::vector<PoseDetection>& visual_poses = last_visual_poses_original_coord;

        const auto osd_begin = clock::now();
        const bool fall_alert = fall_state != FallState::NORMAL;
        std::vector<ObjectDetection> visual_detections = detections_original_coord;
        std::vector<std::array<float, 4>> fall_alert_boxes;
        if (fall_alert && judge.GetAlertBoxes(fall_alert_boxes)) {
            for (const auto& fall_alert_box : fall_alert_boxes) {
                ObjectDetection alert_det;
                alert_det.box = fall_alert_box;
                alert_det.score = 1.0f;
                alert_det.class_id = kFireClassId;
                visual_detections.push_back(alert_det);
            }
        }
        visualizer.Draw(visual_detections,
                        visual_poses,
                        last_visual_pose_kpt_threshold);
        const auto osd_end = clock::now();

        perf_stats.frames += 1;
        perf_stats.capture_ms +=
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(capture_end - capture_begin).count()) / 1000.0;
        perf_stats.detect_ms += detect_ms;
        perf_stats.pose_ms += pose_ms;
        perf_stats.osd_ms +=
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(osd_end - osd_begin).count()) / 1000.0;
        FlushMainLoopPerfIfNeeded(&perf_stats);

        ++frame_index;
    }

    if (listener_thread.joinable()) {
        listener_thread.join();
    }

    ReleaseRuntime();

    return 0;
}
