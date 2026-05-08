/*
 * @Filename: demo_face_yunet.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-01-20
 * @Copyright (c) 2025 SmartSens
 * @Description: YOLOv8 检测 + 低频 pose 演示
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include "event_recorder.hpp"
#include <unistd.h>
#include "fallen_judge.hpp"
#include "include/log.hpp"
#include "include/utils.hpp"

using namespace std;

namespace {

constexpr float kDetectConfThreshold = 0.35f;
constexpr float kDetectPersonConfThreshold = 0.30f;
constexpr float kDetectPoseAssistPersonConfThreshold = 0.24f;
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
constexpr int kDetectNumClasses = 7;
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

bool HasIntrusionDetection(const std::vector<ObjectDetection>& detections) {
    // 入侵事件仅由 mouse / snake 触发。
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

}  // namespace

bool g_exit_flag = false;
std::mutex g_mtx;
EventRecorder g_event_recorder;

void keyboard_listener() {
    std::string input;
    std::cout << "Keyboard listener started, input 'help' for commands, 'q' to quit..." << std::endl;

    while (std::getline(std::cin, input)) {
        bool should_exit = false;
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

int main() {
    using clock = std::chrono::steady_clock;

    const int img_width = 1920;
    const int img_height = 1080;
    const int crop_offset_x = 420;

    std::array<int, 2> img_shape = {img_width, img_height};
    std::array<int, 2> crop_shape = {1080, 1080};
    std::array<int, 2> detect_shape = {256, 256};
    // 模型运行尺寸按 [width, height] 保存。
    std::array<int, 2> pose_shape = {480, 320};

    std::string detect_model_path = "/app_demo/app_assets/models/yolov8nano.m1model";
    std::string pose_model_path = "/app_demo/app_assets/models/yolov8_pose.m1model";

    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
        return -1;
    }

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);

    YOLOV8NANO detect_detector;
    detect_detector.Initialize(detect_model_path, &crop_shape, &detect_shape, 300, kDetectNumClasses);

    YUNET pose_detector;
    pose_detector.Initialize(pose_model_path, &crop_shape, &pose_shape, true, 300);

    ObjectDetectionResult detect_result;
    FaceDetectionResult pose_result;

    VISUALIZER visualizer;
    visualizer.Initialize(img_shape, "shared_colorLUT.sscl");

    LOG_INFO("warmup sleep for 1 second before entering the main loop\n");
    sleep(1);

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
    ssne_tensor_t img_sensor = ssne_tensor_t{};
    std::vector<PoseDetection> cached_poses_original_coord;
    std::vector<PoseDetection> last_visual_poses_original_coord;
    std::vector<ObjectDetection> detections_original_coord;
    PoseRequest pose_request;
    float last_visual_pose_kpt_threshold = kPoseDrawKptConfThreshold;
    bool last_visual_pose_valid = false;
    uint64_t last_visual_pose_frame_index = 0;
    bool person_like_visual_active = false;

    std::thread listener_thread(keyboard_listener);

    while (!check_exit_flag()) {
        const auto capture_begin = clock::now();
        processor.GetImage(&img_sensor);
        const auto capture_end = clock::now();

        // Pose 不和 detect 串行绑定：只有存在人形目标、且达到强制兜底周期时，才在本帧让出 detect 去跑 pose。
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
            // 有合法主目标 ROI 时只把人体附近区域送入 pose；强制兜底帧才允许全局 pose。
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
                kDetectPersonConfThreshold;
            detect_detector.Predict(&img_sensor, &detect_result,
                                    kDetectConfThreshold,
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
        // FallJudge 只需要人体轨迹：本帧跑 pose 时使用 pose 框，否则使用 detect 框更新轻量追踪。
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
            // 只有“冷却合法 + ROI 合法 + 当前仍有人形目标”时才挂起下一次 pose 请求。
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

        // pose 骨架只在本次 pose 成功后刷新；连续超时后清空，避免旧骨架长期停留在画面上。
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
                // 复用 fire 的红色告警绘制通道，表示跌倒 suspect/confirmed 也需要红框提示。
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

    pose_detector.Release();
    detect_detector.Release();
    processor.Release();
    visualizer.Release();

    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }

    return 0;
}
