#ifndef FALL_JUDGE_HPP
#define FALL_JUDGE_HPP

#include <array>
#include <deque>
#include <string>
#include <vector>
#include "common.hpp"

struct DetectionBox {
    std::array<float, 4> box = {0.0f, 0.0f, 0.0f, 0.0f};
    float score = 0.0f;
    int class_id = -1;
};

enum class FallState {
    NORMAL = 0,
    SUSPECT = 1,
    CONFIRMED = 2
};

struct FallJudgeConfig {
    float min_score = 0.35f;
    float horizontal_ratio = 1.2f;
    float downward_motion_ratio = 0.08f;
    float stable_motion_ratio = 0.03f;

    float keypoint_min_conf = 0.35f;
    float horizontal_spine_angle_deg = 35.0f;
    float upside_down_dy_ratio = 0.02f;
    float track_iou_threshold = 0.10f;
    float side_spine_delta_deg = 28.0f;
    float side_box_ratio_with_pose = 1.10f;
    float side_box_ratio_no_pose = 1.25f;
    float front_head_below_margin_ratio = 0.02f;
    float front_ankle_stable_ratio = 0.04f;
    float front_segment_shrink_ratio = 0.82f;
    float front_segment_consistency_ratio = 0.18f;
    float front_aspect_change_ratio = 0.20f;

    float track_high_thresh = 0.45f;
    float track_low_thresh = 0.15f;
    float track_high_match_iou = 0.25f;
    float track_low_match_iou = 0.10f;
    float track_center_distance_ratio = 0.12f;
    float track_size_ratio_min = 0.55f;
    float track_size_ratio_max = 1.80f;
    float primary_switch_margin = 0.25f;
    int track_max_age = 8;
    int track_min_hits = 2;
    int primary_switch_hold_frames = 12;

    int suspect_frames = 2;
    int confirm_frames = 4;
    int reset_frames = 6;
    int person_class_id = 0;

    int image_width = 0;
    int image_height = 0;
};

class FallJudge {
public:
    FallJudge() = default;
    explicit FallJudge(const FallJudgeConfig& cfg);

    void Initialize(const FallJudgeConfig& cfg);
    void Reset();

    FallState Update(const std::vector<DetectionBox>& detections,
                     const std::vector<PoseDetection>& poses,
                     bool pose_fresh);
    FallState Update(const std::vector<DetectionBox>& detections);

    FallState GetState() const;
    std::string GetStateString() const;

    bool GetTrackedBox(std::array<float, 4>& out_box) const;
    bool GetTrackedPose(PoseDetection& out_pose) const;
    bool GetAlertBoxes(std::vector<std::array<float, 4>>& out_boxes) const;
    bool GetTrackedPoses(std::vector<PoseDetection>& out_poses) const;

private:
    struct BoxFeature {
        std::array<float, 4> box = {0.0f, 0.0f, 0.0f, 0.0f};
        float score = 0.0f;
        int class_id = -1;
        float w = 0.0f;
        float h = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        float aspect = 0.0f;
    };

    struct PoseFeature {
        PoseDetection pose;
        BoxFeature box;
        bool has_torso = false;
        bool has_head = false;
        bool has_shoulders = false;
        bool has_wrists = false;
        bool has_ankles = false;
        bool has_front_chain = false;
        std::array<float, 2> neck = {0.0f, 0.0f};
        std::array<float, 2> pelvis = {0.0f, 0.0f};
        std::array<float, 2> shoulder_center = {0.0f, 0.0f};
        std::array<float, 2> wrist_center = {0.0f, 0.0f};
        std::array<float, 2> ankle_center = {0.0f, 0.0f};
        float dx = 0.0f;
        float dy = 0.0f;
        float head_y = 0.0f;
        float spine_angle_deg = 90.0f;
        float body_cx = 0.0f;
        float body_cy = 0.0f;
        float shoulder_elbow_len = 0.0f;
        float hip_knee_len = 0.0f;
        float knee_ankle_len = 0.0f;
        bool is_horizontal = false;
        bool is_upside_down = false;
        float suspicion_score = 0.0f;
    };

    struct StandingReference {
        bool valid = false;
        float spine_angle_deg = 90.0f;
        float aspect = 0.0f;
        float shoulder_elbow_len = 0.0f;
        float hip_knee_len = 0.0f;
        float knee_ankle_len = 0.0f;
        std::array<float, 2> ankle_center = {0.0f, 0.0f};
    };

    struct Kalman1D {
        float position = 0.0f;
        float velocity = 0.0f;
        float p00 = 1.0f;
        float p01 = 0.0f;
        float p10 = 0.0f;
        float p11 = 1.0f;
    };

    struct Track {
        int id = -1;
        BoxFeature box;
        BoxFeature predicted_box;
        Kalman1D cx_filter;
        Kalman1D cy_filter;
        Kalman1D w_filter;
        Kalman1D h_filter;
        int age = 0;
        int hits = 0;
        int misses = 0;
        int freeze_frames = 0;
        bool updated_this_frame = false;
        bool matched_high = false;

        FallState fall_state = FallState::NORMAL;
        int positive_count = 0;
        int negative_count = 0;
        bool has_last_box = false;
        BoxFeature last_box;
        bool has_last_pose = false;
        PoseFeature last_pose;
        bool has_tracked_pose = false;
        PoseDetection tracked_pose;
        std::deque<float> center_y_history;
        StandingReference standing_ref;
    };

    BoxFeature BuildFeature(const DetectionBox& det) const;
    PoseFeature BuildPoseFeature(const PoseDetection& pose) const;
    bool HasUsableTorso(const PoseDetection& pose) const;

    bool HasPersonDetection(const std::vector<DetectionBox>& detections) const;
    float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b) const;
    bool IsHorizontalPose(const BoxFeature& box) const;
    bool IsSideFallWithPose(const PoseFeature& pose) const;
    bool IsSideFallWithPose(const PoseFeature& pose, const StandingReference& standing_ref) const;
    bool IsSideFallWithoutPose(const BoxFeature& box) const;
    bool IsFrontHeadBelowLowerBody(const PoseFeature& pose) const;
    bool IsFrontCompressionFall(const PoseFeature& pose) const;
    bool IsFrontCompressionFall(const PoseFeature& pose, const Track& track) const;
    bool HasDownwardMotion(const PoseFeature& pose) const;
    bool HasDownwardMotion(const PoseFeature& pose, const Track& track) const;
    bool IsStableAfterFall(const PoseFeature& pose) const;
    void UpdateStandingReference(const PoseFeature& pose);
    void UpdateStandingReference(const PoseFeature& pose, StandingReference& standing_ref) const;

    void InitializeKalman(Kalman1D& filter, float position) const;
    void PredictKalman(Kalman1D& filter) const;
    void UpdateKalman(Kalman1D& filter, float measurement) const;
    void InitializeTrack(Track& track, const BoxFeature& feature);
    void PredictTrack(Track& track) const;
    void UpdateTrack(Track& track, const BoxFeature& feature, bool matched_high);
    BoxFeature BoxFromTrackState(const Track& track) const;
    float CenterDistanceRatio(const BoxFeature& a, const BoxFeature& b) const;
    bool PassTrackGate(const Track& track, const BoxFeature& detection, float min_iou) const;
    float ComputeAssociationScore(const Track& track,
                                  const BoxFeature& detection,
                                  bool low_score_stage) const;
    void UpdateTracksFromDetections(const std::vector<DetectionBox>& detections);
    Track* FindTrackById(int track_id);
    const Track* FindTrackById(int track_id) const;
    bool SelectPrimaryTrack(BoxFeature& target, int& track_id) const;
    bool SelectPrimaryPoseTarget(const std::vector<PoseDetection>& poses, PoseFeature& target) const;
    bool SelectPoseForTrack(const std::vector<PoseDetection>& poses,
                            const Track& track,
                            std::vector<char>& pose_used,
                            PoseFeature& target) const;
    float ComputeTrackFallRisk(const Track& track) const;
    void UpdateTrackFallByBox(Track& track, bool detect_fresh);
    void UpdateTrackFallByPose(Track& track, const PoseFeature& pose);
    void RefreshAggregateState();

    FallJudgeConfig m_cfg{};
    FallState m_state = FallState::NORMAL;
    int m_positive_count = 0;
    int m_negative_count = 0;

    bool m_has_last_box = false;
    BoxFeature m_last_box{};

    bool m_has_last_pose = false;
    PoseFeature m_last_pose{};

    bool m_has_tracked_box = false;
    std::array<float, 4> m_tracked_box{0.0f, 0.0f, 0.0f, 0.0f};

    bool m_has_tracked_pose = false;
    PoseDetection m_tracked_pose{};

    int m_next_track_id = 1;
    int m_primary_track_id = -1;
    std::vector<Track> m_tracks;
    std::deque<float> m_center_y_history;
    StandingReference m_standing_ref{};
};

#endif
