#include "../include/fallen_judge.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr int kLeftShoulderIndex = 5;
constexpr int kRightShoulderIndex = 6;
constexpr int kLeftElbowIndex = 7;
constexpr int kRightElbowIndex = 8;
constexpr int kLeftWristIndex = 9;
constexpr int kRightWristIndex = 10;
constexpr int kLeftHipIndex = 11;
constexpr int kRightHipIndex = 12;
constexpr int kLeftKneeIndex = 13;
constexpr int kRightKneeIndex = 14;
constexpr int kLeftAnkleIndex = 15;
constexpr int kRightAnkleIndex = 16;
constexpr float kPi = 3.14159265358979323846f;
// 卡尔曼滤波只平滑 bbox 中心和尺寸，不引入复杂运动模型，保证 CPU 负担可控。
constexpr float kKalmanProcessNoisePos = 4.0f;
constexpr float kKalmanProcessNoiseVel = 1.0f;
constexpr float kKalmanMeasurementNoise = 16.0f;

DetectionBox PoseToDetectionBox(const PoseDetection& pose) {
    DetectionBox det;
    det.box = pose.box;
    det.score = pose.score;
    det.class_id = pose.class_id;
    return det;
}

float ClampFloat(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

bool IsKeypointUsable(const PoseDetection& pose, int index, float min_conf) {
    // 关键点必须存在且置信度足够，后续跌倒逻辑才使用它。
    return index >= 0 &&
           index < static_cast<int>(pose.keypoints.size()) &&
           pose.keypoints[static_cast<size_t>(index)].conf >= min_conf;
}

bool AverageKeypoints(const PoseDetection& pose,
                      const std::vector<int>& indices,
                      float min_conf,
                      std::array<float, 2>* out) {
    if (out == nullptr) {
        return false;
    }

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    int count = 0;
    for (int index : indices) {
        if (!IsKeypointUsable(pose, index, min_conf)) {
            continue;
        }
        const PoseKeyPoint& kp = pose.keypoints[static_cast<size_t>(index)];
        sum_x += kp.x;
        sum_y += kp.y;
        ++count;
    }

    if (count <= 0) {
        return false;
    }

    (*out)[0] = sum_x / static_cast<float>(count);
    (*out)[1] = sum_y / static_cast<float>(count);
    return true;
}

float KeypointDistance(const PoseDetection& pose, int a, int b, float min_conf) {
    if (!IsKeypointUsable(pose, a, min_conf) || !IsKeypointUsable(pose, b, min_conf)) {
        return 0.0f;
    }

    const PoseKeyPoint& pa = pose.keypoints[static_cast<size_t>(a)];
    const PoseKeyPoint& pb = pose.keypoints[static_cast<size_t>(b)];
    const float dx = pa.x - pb.x;
    const float dy = pa.y - pb.y;
    return std::sqrt(dx * dx + dy * dy);
}

float MeanPositive(const std::vector<float>& values) {
    float sum = 0.0f;
    int count = 0;
    for (float value : values) {
        if (value <= 0.0f) {
            continue;
        }
        sum += value;
        ++count;
    }
    if (count <= 0) {
        return 0.0f;
    }
    return sum / static_cast<float>(count);
}

}  // namespace

FallJudge::FallJudge(const FallJudgeConfig& cfg) {
    Initialize(cfg);
}

void FallJudge::Initialize(const FallJudgeConfig& cfg) {
    m_cfg = cfg;
    Reset();
}

void FallJudge::Reset() {
    m_state = FallState::NORMAL;
    m_positive_count = 0;
    m_negative_count = 0;
    m_has_last_box = false;
    m_has_last_pose = false;
    m_has_tracked_box = false;
    m_has_tracked_pose = false;
    m_last_box = BoxFeature{};
    m_last_pose = PoseFeature{};
    m_tracked_box = {0.0f, 0.0f, 0.0f, 0.0f};
    m_tracked_pose = PoseDetection{};
    m_next_track_id = 1;
    m_primary_track_id = -1;
    m_tracks.clear();
    m_center_y_history.clear();
    m_standing_ref = StandingReference{};
}

FallState FallJudge::GetState() const {
    return m_state;
}

std::string FallJudge::GetStateString() const {
    switch (m_state) {
        case FallState::NORMAL: return "NORMAL";
        case FallState::SUSPECT: return "SUSPECT";
        case FallState::CONFIRMED: return "CONFIRMED";
        default: return "UNKNOWN";
    }
}

bool FallJudge::GetTrackedBox(std::array<float, 4>& out_box) const {
    if (!m_has_tracked_box) {
        return false;
    }
    out_box = m_tracked_box;
    return true;
}

bool FallJudge::GetTrackedPose(PoseDetection& out_pose) const {
    if (!m_has_tracked_pose) {
        return false;
    }
    out_pose = m_tracked_pose;
    return true;
}

bool FallJudge::GetAlertBoxes(std::vector<std::array<float, 4>>& out_boxes) const {
    out_boxes.clear();
    for (const auto& track : m_tracks) {
        if (track.fall_state == FallState::NORMAL) {
            continue;
        }
        const BoxFeature& box = track.updated_this_frame ? track.box : track.predicted_box;
        if (box.box[2] > box.box[0] && box.box[3] > box.box[1]) {
            out_boxes.push_back(box.box);
        }
    }
    return !out_boxes.empty();
}

bool FallJudge::GetTrackedPoses(std::vector<PoseDetection>& out_poses) const {
    out_poses.clear();
    for (const auto& track : m_tracks) {
        if (track.has_tracked_pose && track.misses <= m_cfg.track_max_age) {
            out_poses.push_back(track.tracked_pose);
        }
    }
    return !out_poses.empty();
}

FallJudge::BoxFeature FallJudge::BuildFeature(const DetectionBox& det) const {
    BoxFeature f;
    f.box = det.box;
    f.score = det.score;
    f.class_id = det.class_id;
    f.w = std::max(0.0f, det.box[2] - det.box[0]);
    f.h = std::max(0.0f, det.box[3] - det.box[1]);
    f.cx = 0.5f * (det.box[0] + det.box[2]);
    f.cy = 0.5f * (det.box[1] + det.box[3]);
    f.aspect = (f.h > 1e-6f) ? (f.w / f.h) : 0.0f;
    return f;
}

bool FallJudge::HasUsableTorso(const PoseDetection& pose) const {
    const PoseKeyPoint& left_shoulder = pose.keypoints[kLeftShoulderIndex];
    const PoseKeyPoint& right_shoulder = pose.keypoints[kRightShoulderIndex];
    const PoseKeyPoint& left_hip = pose.keypoints[kLeftHipIndex];
    const PoseKeyPoint& right_hip = pose.keypoints[kRightHipIndex];

    return left_shoulder.conf >= m_cfg.keypoint_min_conf &&
           right_shoulder.conf >= m_cfg.keypoint_min_conf &&
           left_hip.conf >= m_cfg.keypoint_min_conf &&
           right_hip.conf >= m_cfg.keypoint_min_conf;
}

FallJudge::PoseFeature FallJudge::BuildPoseFeature(const PoseDetection& pose) const {
    PoseFeature feature;
    feature.pose = pose;
    feature.box = BuildFeature(PoseToDetectionBox(pose));

    std::array<float, 2> head_center = {0.0f, 0.0f};
    feature.has_head = AverageKeypoints(
        pose, std::vector<int>{0, 1, 2, 3, 4}, m_cfg.keypoint_min_conf, &head_center);
    if (feature.has_head) {
        feature.head_y = head_center[1];
    }

    feature.has_shoulders = AverageKeypoints(
        pose, std::vector<int>{kLeftShoulderIndex, kRightShoulderIndex},
        m_cfg.keypoint_min_conf, &feature.shoulder_center);
    feature.has_wrists = AverageKeypoints(
        pose, std::vector<int>{kLeftWristIndex, kRightWristIndex},
        m_cfg.keypoint_min_conf, &feature.wrist_center);

    feature.has_ankles = AverageKeypoints(
        pose, std::vector<int>{kLeftAnkleIndex, kRightAnkleIndex}, m_cfg.keypoint_min_conf, &feature.ankle_center);

    feature.shoulder_elbow_len = MeanPositive({
        KeypointDistance(pose, kLeftShoulderIndex, kLeftElbowIndex, m_cfg.keypoint_min_conf),
        KeypointDistance(pose, kRightShoulderIndex, kRightElbowIndex, m_cfg.keypoint_min_conf)
    });
    feature.hip_knee_len = MeanPositive({
        KeypointDistance(pose, kLeftHipIndex, kLeftKneeIndex, m_cfg.keypoint_min_conf),
        KeypointDistance(pose, kRightHipIndex, kRightKneeIndex, m_cfg.keypoint_min_conf)
    });
    feature.knee_ankle_len = MeanPositive({
        KeypointDistance(pose, kLeftKneeIndex, kLeftAnkleIndex, m_cfg.keypoint_min_conf),
        KeypointDistance(pose, kRightKneeIndex, kRightAnkleIndex, m_cfg.keypoint_min_conf)
    });
    feature.has_front_chain =
        feature.has_head && feature.has_ankles &&
        feature.shoulder_elbow_len > 0.0f &&
        feature.hip_knee_len > 0.0f &&
        feature.knee_ankle_len > 0.0f;

    // 肩膀中点作为 Neck，髋部中点作为 Pelvis，用这条“脊柱线”判断侧向倒地。
    if (!HasUsableTorso(pose)) {
        feature.suspicion_score = (feature.box.aspect > m_cfg.horizontal_ratio) ? 0.2f : 0.0f;
        return feature;
    }

    const PoseKeyPoint& left_shoulder = pose.keypoints[kLeftShoulderIndex];
    const PoseKeyPoint& right_shoulder = pose.keypoints[kRightShoulderIndex];
    const PoseKeyPoint& left_hip = pose.keypoints[kLeftHipIndex];
    const PoseKeyPoint& right_hip = pose.keypoints[kRightHipIndex];

    feature.has_torso = true;
    feature.neck[0] = 0.5f * (left_shoulder.x + right_shoulder.x);
    feature.neck[1] = 0.5f * (left_shoulder.y + right_shoulder.y);
    feature.pelvis[0] = 0.5f * (left_hip.x + right_hip.x);
    feature.pelvis[1] = 0.5f * (left_hip.y + right_hip.y);
    feature.dx = feature.pelvis[0] - feature.neck[0];
    feature.dy = feature.pelvis[1] - feature.neck[1];
    feature.body_cx = 0.5f * (feature.neck[0] + feature.pelvis[0]);
    feature.body_cy = 0.5f * (feature.neck[1] + feature.pelvis[1]);

    const float abs_dx = std::fabs(feature.dx);
    const float abs_dy = std::fabs(feature.dy);
    feature.spine_angle_deg = std::atan2(abs_dy, std::max(abs_dx, 1e-6f)) * 180.0f / kPi;
    feature.is_horizontal =
        feature.spine_angle_deg <= m_cfg.horizontal_spine_angle_deg || abs_dy < abs_dx;

    const float upside_down_threshold =
        std::max(2.0f, m_cfg.upside_down_dy_ratio * static_cast<float>(std::max(1, m_cfg.image_height)));
    feature.is_upside_down = feature.dy < -upside_down_threshold;

    const float horizontal_score = std::max(0.0f, 1.0f - feature.spine_angle_deg / 90.0f);
    feature.suspicion_score = horizontal_score;
    if (feature.is_horizontal) {
        feature.suspicion_score += 1.0f;
    }
    if (feature.is_upside_down) {
        feature.suspicion_score += 0.75f;
    }
    if (feature.box.aspect > m_cfg.horizontal_ratio) {
        feature.suspicion_score += 0.2f;
    }

    return feature;
}

bool FallJudge::HasPersonDetection(const std::vector<DetectionBox>& detections) const {
    for (const auto& det : detections) {
        if (det.score >= m_cfg.min_score && det.class_id == m_cfg.person_class_id) {
            return true;
        }
    }
    return false;
}

float FallJudge::IoU(const std::array<float, 4>& a, const std::array<float, 4>& b) const {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);

    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;

    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    const float uni = area_a + area_b - inter;
    if (uni <= 1e-6f) {
        return 0.0f;
    }
    return inter / uni;
}

bool FallJudge::IsHorizontalPose(const BoxFeature& box) const {
    return box.aspect > m_cfg.horizontal_ratio;
}

bool FallJudge::IsSideFallWithPose(const PoseFeature& pose) const {
    return IsSideFallWithPose(pose, m_standing_ref);
}

bool FallJudge::IsSideFallWithPose(const PoseFeature& pose,
                                   const StandingReference& standing_ref) const {
    if (!pose.has_torso) {
        return false;
    }

    const float reference_angle =
        standing_ref.valid ? standing_ref.spine_angle_deg : 90.0f;
    const float delta = std::fabs(reference_angle - pose.spine_angle_deg);
    return delta >= m_cfg.side_spine_delta_deg &&
           pose.box.aspect >= m_cfg.side_box_ratio_with_pose;
}

bool FallJudge::IsSideFallWithoutPose(const BoxFeature& box) const {
    return box.aspect >= m_cfg.side_box_ratio_no_pose;
}

bool FallJudge::IsFrontHeadBelowLowerBody(const PoseFeature& pose) const {
    if (!pose.has_ankles || !pose.has_torso || m_cfg.image_height <= 0) {
        return false;
    }

    const float margin =
        m_cfg.front_head_below_margin_ratio * static_cast<float>(std::max(1, m_cfg.image_height));
    auto upper_body_below_lower_body = [&](float y) {
        return y > pose.pelvis[1] + margin &&
               y > pose.ankle_center[1] - margin;
    };

    // 正面摔倒时不只看头部，肩膀和手腕明显低于髋/踝也可以作为倒地证据。
    if (pose.has_head && upper_body_below_lower_body(pose.head_y)) {
        return true;
    }
    if (pose.has_shoulders && upper_body_below_lower_body(pose.shoulder_center[1])) {
        return true;
    }
    if (pose.has_wrists && upper_body_below_lower_body(pose.wrist_center[1])) {
        return true;
    }
    return false;
}

bool FallJudge::IsFrontCompressionFall(const PoseFeature& pose) const {
    if (!pose.has_front_chain || !m_standing_ref.valid || !HasDownwardMotion(pose) ||
        m_cfg.image_height <= 0) {
        return false;
    }

    const float ankle_dx = pose.ankle_center[0] - m_standing_ref.ankle_center[0];
    const float ankle_dy = pose.ankle_center[1] - m_standing_ref.ankle_center[1];
    const float ankle_motion_ratio =
        std::sqrt(ankle_dx * ankle_dx + ankle_dy * ankle_dy) /
        static_cast<float>(std::max(1, m_cfg.image_height));
    if (ankle_motion_ratio > m_cfg.front_ankle_stable_ratio) {
        return false;
    }

    std::vector<float> shrink_ratios;
    if (m_standing_ref.shoulder_elbow_len > 1e-6f) {
        shrink_ratios.push_back(pose.shoulder_elbow_len / m_standing_ref.shoulder_elbow_len);
    }
    if (m_standing_ref.hip_knee_len > 1e-6f) {
        shrink_ratios.push_back(pose.hip_knee_len / m_standing_ref.hip_knee_len);
    }
    if (m_standing_ref.knee_ankle_len > 1e-6f) {
        shrink_ratios.push_back(pose.knee_ankle_len / m_standing_ref.knee_ankle_len);
    }
    if (shrink_ratios.size() < 2) {
        return false;
    }

    float ratio_sum = 0.0f;
    for (float ratio : shrink_ratios) {
        ratio_sum += ratio;
    }
    const float mean_ratio = ratio_sum / static_cast<float>(shrink_ratios.size());
    if (mean_ratio >= m_cfg.front_segment_shrink_ratio) {
        return false;
    }

    float max_deviation = 0.0f;
    for (float ratio : shrink_ratios) {
        max_deviation = std::max(max_deviation, std::fabs(ratio - mean_ratio));
    }
    if (max_deviation > m_cfg.front_segment_consistency_ratio) {
        return false;
    }

    if (m_standing_ref.aspect <= 1e-6f) {
        return false;
    }
    return pose.box.aspect >=
           m_standing_ref.aspect * (1.0f + m_cfg.front_aspect_change_ratio);
}

bool FallJudge::IsFrontCompressionFall(const PoseFeature& pose, const Track& track) const {
    if (!pose.has_front_chain || !track.standing_ref.valid || !HasDownwardMotion(pose, track) ||
        m_cfg.image_height <= 0) {
        return false;
    }

    // 面向摄像头下蹲和跌倒都可能变矮，因此还要结合踝点稳定、肢段等比缩短和重心下移。
    const StandingReference& ref = track.standing_ref;
    const float ankle_dx = pose.ankle_center[0] - ref.ankle_center[0];
    const float ankle_dy = pose.ankle_center[1] - ref.ankle_center[1];
    const float ankle_motion_ratio =
        std::sqrt(ankle_dx * ankle_dx + ankle_dy * ankle_dy) /
        static_cast<float>(std::max(1, m_cfg.image_height));
    if (ankle_motion_ratio > m_cfg.front_ankle_stable_ratio) {
        return false;
    }

    std::vector<float> shrink_ratios;
    if (ref.shoulder_elbow_len > 1e-6f) {
        shrink_ratios.push_back(pose.shoulder_elbow_len / ref.shoulder_elbow_len);
    }
    if (ref.hip_knee_len > 1e-6f) {
        shrink_ratios.push_back(pose.hip_knee_len / ref.hip_knee_len);
    }
    if (ref.knee_ankle_len > 1e-6f) {
        shrink_ratios.push_back(pose.knee_ankle_len / ref.knee_ankle_len);
    }
    if (shrink_ratios.size() < 2) {
        return false;
    }

    float ratio_sum = 0.0f;
    for (float ratio : shrink_ratios) {
        ratio_sum += ratio;
    }
    const float mean_ratio = ratio_sum / static_cast<float>(shrink_ratios.size());
    if (mean_ratio >= m_cfg.front_segment_shrink_ratio) {
        return false;
    }

    float max_deviation = 0.0f;
    for (float ratio : shrink_ratios) {
        max_deviation = std::max(max_deviation, std::fabs(ratio - mean_ratio));
    }
    if (max_deviation > m_cfg.front_segment_consistency_ratio) {
        return false;
    }

    if (ref.aspect <= 1e-6f) {
        return false;
    }
    return pose.box.aspect >= ref.aspect * (1.0f + m_cfg.front_aspect_change_ratio);
}

bool FallJudge::HasDownwardMotion(const PoseFeature& pose) const {
    if (m_cfg.image_height <= 0) {
        return false;
    }

    const float current_y = pose.has_torso ? pose.body_cy : pose.box.cy;
    if (m_center_y_history.size() >= 2) {
        const size_t history_size = m_center_y_history.size();
        const size_t step_count = std::min<size_t>(3, history_size - 1);
        const float reference_y =
            m_center_y_history[history_size - 1 - step_count];
        const float dy = current_y - reference_y;
        const float avg_motion_ratio =
            (dy / static_cast<float>(step_count)) /
            static_cast<float>(std::max(1, m_cfg.image_height));
        if (avg_motion_ratio > m_cfg.downward_motion_ratio) {
            return true;
        }
    }

    if (!m_has_last_pose) {
        return false;
    }

    const float last_y = m_last_pose.has_torso ? m_last_pose.body_cy : m_last_pose.box.cy;
    const float dy = current_y - last_y;
    return (dy / static_cast<float>(m_cfg.image_height)) > m_cfg.downward_motion_ratio;
}

bool FallJudge::HasDownwardMotion(const PoseFeature& pose, const Track& track) const {
    if (m_cfg.image_height <= 0) {
        return false;
    }

    const float current_y = pose.has_torso ? pose.body_cy : pose.box.cy;
    if (track.center_y_history.size() >= 2) {
        const size_t history_size = track.center_y_history.size();
        const size_t step_count = std::min<size_t>(3, history_size - 1);
        const float reference_y =
            track.center_y_history[history_size - 1 - step_count];
        const float dy = current_y - reference_y;
        const float avg_motion_ratio =
            (dy / static_cast<float>(step_count)) /
            static_cast<float>(std::max(1, m_cfg.image_height));
        if (avg_motion_ratio > m_cfg.downward_motion_ratio) {
            return true;
        }
    }

    if (!track.has_last_pose) {
        return false;
    }

    const float last_y = track.last_pose.has_torso ? track.last_pose.body_cy : track.last_pose.box.cy;
    const float dy = current_y - last_y;
    return (dy / static_cast<float>(m_cfg.image_height)) > m_cfg.downward_motion_ratio;
}

bool FallJudge::IsStableAfterFall(const PoseFeature& pose) const {
    if (!m_has_last_pose || m_cfg.image_height <= 0) {
        return false;
    }

    const float current_y = pose.has_torso ? pose.body_cy : pose.box.cy;
    const float last_y = m_last_pose.has_torso ? m_last_pose.body_cy : m_last_pose.box.cy;
    const float dy = std::fabs(current_y - last_y);
    return (dy / static_cast<float>(m_cfg.image_height)) < m_cfg.stable_motion_ratio;
}

void FallJudge::UpdateStandingReference(const PoseFeature& pose) {
    UpdateStandingReference(pose, m_standing_ref);
}

void FallJudge::UpdateStandingReference(const PoseFeature& pose,
                                        StandingReference& standing_ref) const {
    if (!pose.has_torso || !pose.has_front_chain) {
        return;
    }

    // 站立参考采用慢速 EMA 更新，避免一次弯腰或遮挡把“正常姿态”污染掉。
    constexpr float kKeep = 0.80f;
    constexpr float kUpdate = 0.20f;

    if (!standing_ref.valid) {
        standing_ref.valid = true;
        standing_ref.spine_angle_deg = pose.spine_angle_deg;
        standing_ref.aspect = pose.box.aspect;
        standing_ref.shoulder_elbow_len = pose.shoulder_elbow_len;
        standing_ref.hip_knee_len = pose.hip_knee_len;
        standing_ref.knee_ankle_len = pose.knee_ankle_len;
        standing_ref.ankle_center = pose.ankle_center;
        return;
    }

    standing_ref.spine_angle_deg =
        standing_ref.spine_angle_deg * kKeep + pose.spine_angle_deg * kUpdate;
    standing_ref.aspect =
        standing_ref.aspect * kKeep + pose.box.aspect * kUpdate;
    standing_ref.shoulder_elbow_len =
        standing_ref.shoulder_elbow_len * kKeep + pose.shoulder_elbow_len * kUpdate;
    standing_ref.hip_knee_len =
        standing_ref.hip_knee_len * kKeep + pose.hip_knee_len * kUpdate;
    standing_ref.knee_ankle_len =
        standing_ref.knee_ankle_len * kKeep + pose.knee_ankle_len * kUpdate;
    standing_ref.ankle_center[0] =
        standing_ref.ankle_center[0] * kKeep + pose.ankle_center[0] * kUpdate;
    standing_ref.ankle_center[1] =
        standing_ref.ankle_center[1] * kKeep + pose.ankle_center[1] * kUpdate;
}

void FallJudge::InitializeKalman(Kalman1D& filter, float position) const {
    filter.position = position;
    filter.velocity = 0.0f;
    filter.p00 = 16.0f;
    filter.p01 = 0.0f;
    filter.p10 = 0.0f;
    filter.p11 = 16.0f;
}

void FallJudge::PredictKalman(Kalman1D& filter) const {
    filter.position += filter.velocity;

    const float p00 = filter.p00 + filter.p10 + filter.p01 + filter.p11 + kKalmanProcessNoisePos;
    const float p01 = filter.p01 + filter.p11;
    const float p10 = filter.p10 + filter.p11;
    const float p11 = filter.p11 + kKalmanProcessNoiseVel;

    filter.p00 = p00;
    filter.p01 = p01;
    filter.p10 = p10;
    filter.p11 = p11;
}

void FallJudge::UpdateKalman(Kalman1D& filter, float measurement) const {
    const float innovation = measurement - filter.position;
    const float s = filter.p00 + kKalmanMeasurementNoise;
    if (s <= 1e-6f) {
        return;
    }

    const float k0 = filter.p00 / s;
    const float k1 = filter.p10 / s;

    filter.position += k0 * innovation;
    filter.velocity += k1 * innovation;

    const float p00 = (1.0f - k0) * filter.p00;
    const float p01 = (1.0f - k0) * filter.p01;
    const float p10 = filter.p10 - k1 * filter.p00;
    const float p11 = filter.p11 - k1 * filter.p01;

    filter.p00 = p00;
    filter.p01 = p01;
    filter.p10 = p10;
    filter.p11 = p11;
}

FallJudge::BoxFeature FallJudge::BoxFromTrackState(const Track& track) const {
    BoxFeature feature;
    const float max_w = static_cast<float>(std::max(1, m_cfg.image_width));
    const float max_h = static_cast<float>(std::max(1, m_cfg.image_height));

    feature.w = ClampFloat(track.w_filter.position, 4.0f, max_w);
    feature.h = ClampFloat(track.h_filter.position, 4.0f, max_h);
    feature.cx = ClampFloat(track.cx_filter.position, 0.0f, max_w);
    feature.cy = ClampFloat(track.cy_filter.position, 0.0f, max_h);
    feature.box = {
        ClampFloat(feature.cx - 0.5f * feature.w, 0.0f, max_w),
        ClampFloat(feature.cy - 0.5f * feature.h, 0.0f, max_h),
        ClampFloat(feature.cx + 0.5f * feature.w, 0.0f, max_w),
        ClampFloat(feature.cy + 0.5f * feature.h, 0.0f, max_h)
    };
    feature.w = std::max(0.0f, feature.box[2] - feature.box[0]);
    feature.h = std::max(0.0f, feature.box[3] - feature.box[1]);
    feature.aspect = (feature.h > 1e-6f) ? (feature.w / feature.h) : 0.0f;
    feature.class_id = m_cfg.person_class_id;
    feature.score = track.box.score;
    return feature;
}

void FallJudge::InitializeTrack(Track& track, const BoxFeature& feature) {
    track.id = m_next_track_id++;
    track.box = feature;
    track.predicted_box = feature;
    track.age = 1;
    track.hits = 1;
    track.misses = 0;
    track.freeze_frames = 0;
    track.updated_this_frame = true;
    track.matched_high = true;
    track.fall_state = FallState::NORMAL;
    track.positive_count = 0;
    track.negative_count = 0;
    track.has_last_box = false;
    track.last_box = BoxFeature{};
    track.has_last_pose = false;
    track.last_pose = PoseFeature{};
    track.has_tracked_pose = false;
    track.tracked_pose = PoseDetection{};
    track.center_y_history.clear();
    track.standing_ref = StandingReference{};
    InitializeKalman(track.cx_filter, feature.cx);
    InitializeKalman(track.cy_filter, feature.cy);
    InitializeKalman(track.w_filter, feature.w);
    InitializeKalman(track.h_filter, feature.h);
}

void FallJudge::PredictTrack(Track& track) const {
    PredictKalman(track.cx_filter);
    PredictKalman(track.cy_filter);
    PredictKalman(track.w_filter);
    PredictKalman(track.h_filter);
    track.predicted_box = BoxFromTrackState(track);
    track.updated_this_frame = false;
    track.matched_high = false;
}

void FallJudge::UpdateTrack(Track& track, const BoxFeature& feature, bool matched_high) {
    UpdateKalman(track.cx_filter, feature.cx);
    UpdateKalman(track.cy_filter, feature.cy);
    UpdateKalman(track.w_filter, feature.w);
    UpdateKalman(track.h_filter, feature.h);
    track.box = BoxFromTrackState(track);
    track.box.score = feature.score;
    track.box.class_id = feature.class_id;
    track.predicted_box = track.box;
    track.age += 1;
    track.hits += 1;
    track.misses = 0;
    track.updated_this_frame = true;
    track.matched_high = matched_high;
}

float FallJudge::CenterDistanceRatio(const BoxFeature& a, const BoxFeature& b) const {
    const float dx = a.cx - b.cx;
    const float dy = a.cy - b.cy;
    const float distance = std::sqrt(dx * dx + dy * dy);
    const float norm = std::sqrt(
        static_cast<float>(std::max(1, m_cfg.image_width * m_cfg.image_width +
                                       m_cfg.image_height * m_cfg.image_height)));
    if (norm <= 1e-6f) {
        return 0.0f;
    }
    return distance / norm;
}

bool FallJudge::PassTrackGate(const Track& track,
                              const BoxFeature& detection,
                              float min_iou) const {
    // 关联门控同时看 IoU、中心距离和尺寸变化，降低多人场景中乱配的概率。
    const float iou = IoU(track.predicted_box.box, detection.box);
    if (iou < min_iou) {
        return false;
    }

    const float center_ratio = CenterDistanceRatio(track.predicted_box, detection);
    if (center_ratio > m_cfg.track_center_distance_ratio) {
        return false;
    }

    const float width_ratio =
        detection.w / std::max(1.0f, track.predicted_box.w);
    const float height_ratio =
        detection.h / std::max(1.0f, track.predicted_box.h);
    if (width_ratio < m_cfg.track_size_ratio_min || width_ratio > m_cfg.track_size_ratio_max) {
        return false;
    }
    if (height_ratio < m_cfg.track_size_ratio_min || height_ratio > m_cfg.track_size_ratio_max) {
        return false;
    }

    return true;
}

float FallJudge::ComputeAssociationScore(const Track& track,
                                         const BoxFeature& detection,
                                         bool low_score_stage) const {
    const float iou = IoU(track.predicted_box.box, detection.box);
    const float center_ratio = CenterDistanceRatio(track.predicted_box, detection);
    float score = iou * 2.5f + (1.0f - center_ratio) + detection.score * 0.3f;

    if (track.id == m_primary_track_id) {
        score += 0.4f;
    }
    if (track.freeze_frames > 0) {
        score += 0.3f;
    }
    if (IsHorizontalPose(detection)) {
        score += 0.1f;
    }
    if (low_score_stage) {
        score -= 0.05f;
    }
    return score;
}

void FallJudge::UpdateTracksFromDetections(const std::vector<DetectionBox>& detections) {
    // 先预测所有轨迹，再用高分框匹配，最后用低分框补救，属于轻量 ByteTrack 思路。
    for (auto& track : m_tracks) {
        PredictTrack(track);
        track.misses += 1;
        if (track.freeze_frames > 0) {
            track.freeze_frames -= 1;
        }
    }

    std::vector<BoxFeature> high_detections;
    std::vector<BoxFeature> low_detections;
    high_detections.reserve(detections.size());
    low_detections.reserve(detections.size());

    for (const auto& det : detections) {
        if (det.class_id != m_cfg.person_class_id || det.score < m_cfg.track_low_thresh) {
            continue;
        }

        const BoxFeature feature = BuildFeature(det);
        if (det.score >= m_cfg.track_high_thresh) {
            high_detections.push_back(feature);
        } else {
            low_detections.push_back(feature);
        }
    }

    std::vector<int> unmatched_tracks;
    unmatched_tracks.reserve(m_tracks.size());
    for (size_t i = 0; i < m_tracks.size(); ++i) {
        unmatched_tracks.push_back(static_cast<int>(i));
    }

    std::vector<char> high_det_used(high_detections.size(), 0);
    std::vector<char> low_det_used(low_detections.size(), 0);

    auto match_stage =
        [&](const std::vector<BoxFeature>& stage_detections,
            std::vector<char>& det_used,
            float min_iou,
            bool low_score_stage) {
            if (stage_detections.empty() || unmatched_tracks.empty()) {
                return;
            }

            while (true) {
                int best_track_pos = -1;
                int best_det_index = -1;
                float best_score = -std::numeric_limits<float>::infinity();

                for (size_t pos = 0; pos < unmatched_tracks.size(); ++pos) {
                    const int track_index = unmatched_tracks[pos];
                    const Track& track = m_tracks[static_cast<size_t>(track_index)];

                    for (size_t det_index = 0; det_index < stage_detections.size(); ++det_index) {
                        if (det_used[det_index] != 0) {
                            continue;
                        }

                        const BoxFeature& detection = stage_detections[det_index];
                        if (!PassTrackGate(track, detection, min_iou)) {
                            continue;
                        }

                        const float score = ComputeAssociationScore(track, detection, low_score_stage);
                        if (score > best_score) {
                            best_score = score;
                            best_track_pos = static_cast<int>(pos);
                            best_det_index = static_cast<int>(det_index);
                        }
                    }
                }

                if (best_track_pos < 0 || best_det_index < 0) {
                    break;
                }

                const int track_index = unmatched_tracks[static_cast<size_t>(best_track_pos)];
                UpdateTrack(m_tracks[static_cast<size_t>(track_index)],
                            stage_detections[static_cast<size_t>(best_det_index)],
                            !low_score_stage);
                det_used[static_cast<size_t>(best_det_index)] = 1;
                unmatched_tracks.erase(unmatched_tracks.begin() + best_track_pos);
            }
        };

    // 高分阶段保证精度，低分阶段只延续已有轨迹，不用来创建新轨迹。
    match_stage(high_detections, high_det_used, m_cfg.track_high_match_iou, false);
    match_stage(low_detections, low_det_used, m_cfg.track_low_match_iou, true);

    for (size_t i = 0; i < high_detections.size(); ++i) {
        if (high_det_used[i] != 0) {
            continue;
        }

        Track track;
        InitializeTrack(track, high_detections[i]);
        m_tracks.push_back(track);
    }

    m_tracks.erase(
        std::remove_if(m_tracks.begin(), m_tracks.end(),
                       [&](const Track& track) {
                           return track.misses > m_cfg.track_max_age;
                       }),
        m_tracks.end());

    if (FindTrackById(m_primary_track_id) == nullptr) {
        m_primary_track_id = -1;
    }
}

FallJudge::Track* FallJudge::FindTrackById(int track_id) {
    if (track_id < 0) {
        return nullptr;
    }
    for (auto& track : m_tracks) {
        if (track.id == track_id) {
            return &track;
        }
    }
    return nullptr;
}

const FallJudge::Track* FallJudge::FindTrackById(int track_id) const {
    if (track_id < 0) {
        return nullptr;
    }
    for (const auto& track : m_tracks) {
        if (track.id == track_id) {
            return &track;
        }
    }
    return nullptr;
}

bool FallJudge::SelectPrimaryTrack(BoxFeature& target, int& track_id) const {
    // 主目标不是简单最大框，而是综合命中次数、丢失帧、跌倒风险和 freeze 惯性后选择。
    const Track* primary_track = FindTrackById(m_primary_track_id);
    const bool primary_valid = primary_track != nullptr &&
                               primary_track->misses <= m_cfg.track_max_age &&
                               (primary_track->hits >= m_cfg.track_min_hits || primary_track->updated_this_frame);

    const Track* best_track = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();
    for (const auto& track : m_tracks) {
        if (track.misses > m_cfg.track_max_age) {
            continue;
        }
        if (track.hits < m_cfg.track_min_hits && !track.updated_this_frame) {
            continue;
        }

        const BoxFeature& track_box = track.updated_this_frame ? track.box : track.predicted_box;
        float score = track_box.score;
        score += std::min(track.hits, 8) * 0.06f;
        score -= track.misses * 0.08f;
        if (track.freeze_frames > 0) {
            score += 0.3f;
        }
        if (track.id == m_primary_track_id) {
            score += 0.4f;
        }
        if (IsHorizontalPose(track_box)) {
            score += 0.12f;
        }

        if (score > best_score) {
            best_score = score;
            best_track = &track;
        }
    }

    if (best_track == nullptr) {
        return false;
    }

    if (primary_valid) {
        const BoxFeature& primary_box = primary_track->updated_this_frame ?
                                        primary_track->box : primary_track->predicted_box;
        float primary_score = primary_box.score;
        primary_score += std::min(primary_track->hits, 8) * 0.06f;
        primary_score -= primary_track->misses * 0.08f;
        if (primary_track->freeze_frames > 0) {
            primary_score += 0.3f;
        }
        if (IsHorizontalPose(primary_box)) {
            primary_score += 0.12f;
        }

        if (primary_track->freeze_frames > 0 ||
            best_track->id == primary_track->id ||
            best_score < primary_score + m_cfg.primary_switch_margin) {
            target = primary_box;
            track_id = primary_track->id;
            return true;
        }
    }

    target = best_track->updated_this_frame ? best_track->box : best_track->predicted_box;
    track_id = best_track->id;
    return true;
}

bool FallJudge::SelectPrimaryPoseTarget(const std::vector<PoseDetection>& poses, PoseFeature& target) const {
    if (poses.empty()) {
        return false;
    }

    const Track* primary_track = FindTrackById(m_primary_track_id);
    int best_match_index = -1;
    float best_match_value = -std::numeric_limits<float>::infinity();
    int best_fallback_index = -1;
    float best_fallback_value = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < poses.size(); ++i) {
        if (poses[i].score < m_cfg.min_score) {
            continue;
        }

        const PoseFeature feature = BuildPoseFeature(poses[i]);
        float fallback_value = feature.suspicion_score + feature.pose.score * 0.2f;
        if (!feature.has_torso) {
            fallback_value -= 0.4f;
        }
        if (fallback_value > best_fallback_value) {
            best_fallback_value = fallback_value;
            best_fallback_index = static_cast<int>(i);
        }

        if (primary_track == nullptr) {
            continue;
        }

        const BoxFeature& primary_box = primary_track->updated_this_frame ?
                                        primary_track->box : primary_track->predicted_box;
        const float iou = IoU(feature.box.box, primary_box.box);
        const float center_ratio = CenterDistanceRatio(feature.box, primary_box);
        if (iou < m_cfg.track_iou_threshold &&
            center_ratio > (m_cfg.track_center_distance_ratio * 0.75f)) {
            continue;
        }

        float match_value = iou * 3.0f + (1.0f - center_ratio) +
                            feature.suspicion_score + feature.pose.score * 0.1f;
        if (primary_track->freeze_frames > 0) {
            match_value += 0.2f;
        }
        if (!feature.has_torso) {
            match_value -= 0.4f;
        }

        if (match_value > best_match_value) {
            best_match_value = match_value;
            best_match_index = static_cast<int>(i);
        }
    }

    const int selected_index = (best_match_index >= 0) ? best_match_index : best_fallback_index;
    if (selected_index < 0) {
        return false;
    }

    target = BuildPoseFeature(poses[static_cast<size_t>(selected_index)]);
    return true;
}

bool FallJudge::SelectPoseForTrack(const std::vector<PoseDetection>& poses,
                                   const Track& track,
                                   std::vector<char>& pose_used,
                                   PoseFeature& target) const {
    // pose 与 track 一对一消费，避免同一个 pose 结果被多个轨迹重复使用。
    int best_index = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    const BoxFeature& track_box = track.updated_this_frame ? track.box : track.predicted_box;

    for (size_t i = 0; i < poses.size(); ++i) {
        if (i < pose_used.size() && pose_used[i] != 0) {
            continue;
        }
        if (poses[i].score < m_cfg.min_score) {
            continue;
        }

        const PoseFeature feature = BuildPoseFeature(poses[i]);
        const float iou = IoU(feature.box.box, track_box.box);
        const float center_ratio = CenterDistanceRatio(feature.box, track_box);
        if (iou < m_cfg.track_iou_threshold &&
            center_ratio > (m_cfg.track_center_distance_ratio * 0.75f)) {
            continue;
        }

        float value = iou * 3.0f + (1.0f - center_ratio) +
                      feature.suspicion_score + poses[i].score * 0.1f;
        if (!feature.has_torso) {
            value -= 0.4f;
        }
        if (track.freeze_frames > 0) {
            value += 0.2f;
        }
        if (value > best_value) {
            best_value = value;
            best_index = static_cast<int>(i);
        }
    }

    if (best_index < 0) {
        return false;
    }

    target = BuildPoseFeature(poses[static_cast<size_t>(best_index)]);
    if (static_cast<size_t>(best_index) < pose_used.size()) {
        pose_used[static_cast<size_t>(best_index)] = 1;
    }
    return true;
}

float FallJudge::ComputeTrackFallRisk(const Track& track) const {
    const BoxFeature& box = track.updated_this_frame ? track.box : track.predicted_box;
    float risk = 0.0f;
    if (track.fall_state == FallState::CONFIRMED) {
        risk += 100.0f;
    } else if (track.fall_state == FallState::SUSPECT) {
        risk += 50.0f;
    }
    risk += static_cast<float>(track.positive_count) * 4.0f;
    risk -= static_cast<float>(track.negative_count) * 0.8f;
    risk -= static_cast<float>(track.misses) * 0.6f;
    if (track.updated_this_frame) {
        risk += 1.0f;
    }
    if (track.freeze_frames > 0) {
        risk += 5.0f;
    }
    if (IsSideFallWithoutPose(box)) {
        risk += 8.0f;
    } else if (box.aspect > 0.85f) {
        risk += 2.0f;
    }
    if (track.has_last_box && m_cfg.image_height > 0) {
        const float drop_ratio =
            (box.cy - track.last_box.cy) /
            static_cast<float>(std::max(1, m_cfg.image_height));
        const float aspect_gain = box.aspect - track.last_box.aspect;
        if (drop_ratio > (m_cfg.downward_motion_ratio * 0.35f)) {
            risk += 3.0f;
        }
        if (aspect_gain > 0.15f) {
            risk += 3.0f;
        }
    }
    return risk;
}

void FallJudge::UpdateTrackFallByBox(Track& track, bool detect_fresh) {
    // 没有新 pose 时只使用保守的框形态逻辑，主要兜底侧向倒地，避免坐姿/下蹲误报。
    if (!detect_fresh || !track.updated_this_frame || track.misses != 0 ||
        track.box.class_id != m_cfg.person_class_id ||
        track.box.score < m_cfg.min_score) {
        return;
    }

    bool box_only_suspicious = false;
    if (IsSideFallWithoutPose(track.box)) {
        bool has_transition = false;
        if (track.has_last_box && m_cfg.image_height > 0) {
            const float center_drop_ratio =
                (track.box.cy - track.last_box.cy) /
                static_cast<float>(std::max(1, m_cfg.image_height));
            const float aspect_gain = track.box.aspect - track.last_box.aspect;
            has_transition =
                center_drop_ratio > (m_cfg.downward_motion_ratio * 0.5f) ||
                aspect_gain > 0.20f;
        }
        box_only_suspicious = has_transition;
    }

    if (box_only_suspicious) {
        ++track.positive_count;
        track.negative_count = 0;
        track.freeze_frames =
            std::max(track.freeze_frames, m_cfg.primary_switch_hold_frames / 2);
        if (track.positive_count >= m_cfg.confirm_frames) {
            track.fall_state = FallState::CONFIRMED;
        } else if (track.positive_count >= m_cfg.suspect_frames) {
            track.fall_state = FallState::SUSPECT;
        }
    } else {
        ++track.negative_count;
        if (track.negative_count >= m_cfg.reset_frames) {
            track.fall_state = FallState::NORMAL;
            track.positive_count = 0;
        }
    }

    track.last_box = track.box;
    track.has_last_box = true;
}

void FallJudge::UpdateTrackFallByPose(Track& track, const PoseFeature& pose) {
    // pose 判定优先级高于纯框判定：侧倒、正面头/肩/腕下沉、肢段压缩都会增加 positive_count。
    BoxFeature pose_box = pose.box;
    pose_box.class_id = m_cfg.person_class_id;
    pose_box.score = std::max(pose_box.score, pose.pose.score);
    UpdateTrack(track, pose_box, true);

    track.has_tracked_pose = true;
    track.tracked_pose = pose.pose;

    track.center_y_history.push_back(pose.has_torso ? pose.body_cy : pose.box.cy);
    if (track.center_y_history.size() > 20) {
        track.center_y_history.pop_front();
    }

    const bool side_pose = IsSideFallWithPose(pose, track.standing_ref);
    const bool front_head_below = IsFrontHeadBelowLowerBody(pose);
    const bool front_compression = IsFrontCompressionFall(pose, track);
    const bool suspicious = side_pose || front_head_below || front_compression;

    if (suspicious) {
        ++track.positive_count;
        track.negative_count = 0;
        track.freeze_frames =
            std::max(track.freeze_frames, m_cfg.primary_switch_hold_frames);
        if (track.positive_count >= m_cfg.confirm_frames || front_head_below) {
            track.fall_state = FallState::CONFIRMED;
        } else if (track.positive_count >= m_cfg.suspect_frames ||
                   side_pose || front_compression) {
            track.fall_state = FallState::SUSPECT;
        }
    } else {
        ++track.negative_count;
        if (track.negative_count >= m_cfg.reset_frames) {
            track.fall_state = FallState::NORMAL;
            track.positive_count = 0;
        }
        UpdateStandingReference(pose, track.standing_ref);
    }

    track.last_box = pose.box;
    track.has_last_box = true;
    track.last_pose = pose;
    track.has_last_pose = true;
}

void FallJudge::RefreshAggregateState() {
    // 多轨迹内部独立判断，最终向主循环暴露全局最高状态和一个主目标 ROI。
    m_state = FallState::NORMAL;
    m_has_tracked_box = false;
    m_has_tracked_pose = false;
    m_has_last_box = false;
    m_has_last_pose = false;
    m_primary_track_id = -1;
    m_positive_count = 0;
    m_negative_count = 0;
    m_center_y_history.clear();
    m_standing_ref = StandingReference{};

    const Track* best_track = nullptr;
    float best_risk = -std::numeric_limits<float>::infinity();
    for (const auto& track : m_tracks) {
        if (track.misses > m_cfg.track_max_age) {
            continue;
        }
        const float risk = ComputeTrackFallRisk(track);
        if (risk > best_risk) {
            best_risk = risk;
            best_track = &track;
        }

        if (track.fall_state == FallState::CONFIRMED) {
            m_state = FallState::CONFIRMED;
        } else if (track.fall_state == FallState::SUSPECT &&
                   m_state == FallState::NORMAL) {
            m_state = FallState::SUSPECT;
        }
    }

    if (best_track != nullptr) {
        const BoxFeature& box = best_track->updated_this_frame ?
                                best_track->box : best_track->predicted_box;
        m_has_tracked_box = true;
        m_tracked_box = box.box;
        m_primary_track_id = best_track->id;
        m_positive_count = best_track->positive_count;
        m_negative_count = best_track->negative_count;
        m_last_box = best_track->has_last_box ? best_track->last_box : box;
        m_has_last_box = true;
        if (best_track->has_tracked_pose) {
            m_has_tracked_pose = true;
            m_tracked_pose = best_track->tracked_pose;
        }
        if (best_track->has_last_pose) {
            m_last_pose = best_track->last_pose;
            m_has_last_pose = true;
        }
        m_center_y_history = best_track->center_y_history;
        m_standing_ref = best_track->standing_ref;
    }
}

FallState FallJudge::Update(const std::vector<DetectionBox>& detections) {
    return Update(detections, std::vector<PoseDetection>{}, false);
}

FallState FallJudge::Update(const std::vector<DetectionBox>& detections,
                            const std::vector<PoseDetection>& poses,
                            bool pose_fresh) {
    // detect 帧负责更新轨迹；pose 帧负责修正对应轨迹的姿态和跌倒状态，两者不会同时重跑。
    const bool detect_fresh = !pose_fresh;
    if (detect_fresh) {
        UpdateTracksFromDetections(detections);
    } else {
        for (auto& track : m_tracks) {
            if (track.freeze_frames > 0) {
                track.freeze_frames -= 1;
            }
        }
    }

    if (detect_fresh) {
        for (auto& track : m_tracks) {
            UpdateTrackFallByBox(track, true);
        }
    }

    if (pose_fresh && !poses.empty()) {
        std::vector<char> pose_used(poses.size(), 0);
        for (auto& track : m_tracks) {
            if (track.misses > m_cfg.track_max_age) {
                continue;
            }
            PoseFeature pose_feature{};
            if (SelectPoseForTrack(poses, track, pose_used, pose_feature)) {
                UpdateTrackFallByPose(track, pose_feature);
            }
        }
    }

    RefreshAggregateState();
    return m_state;
}
