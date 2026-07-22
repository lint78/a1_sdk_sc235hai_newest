#pragma once

#include <cstddef>
#include <cstdint>

constexpr int FACE_NET_W = 640;
constexpr int FACE_NET_H = 640;
constexpr int FACE_OUTPUT_COUNT = 12;
constexpr int FACE_MAX_DETECTIONS = 20;

enum class FaceTensorType : uint8_t {
    kInt8 = 0,
    kFloat32 = 1
};

struct FaceRawTensor {
    const void* data = nullptr;
    FaceTensorType type = FaceTensorType::kInt8;
    float scale = 1.0f;
    int channels = 0;
    int height = 0;
    int width = 0;
};

struct FacePoint {
    float x = 0.0f;
    float y = 0.0f;
};

enum class FaceIdentity : uint8_t {
    kUnknown = 0,
    kKnown = 1,
    kStranger = 2
};

struct FaceDetection {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float score = 0.0f;
    FaceIdentity identity = FaceIdentity::kUnknown;
    FacePoint landmarks[5];
};

struct FaceResult {
    uint16_t frame_id = 0;
    int count = 0;
    FaceDetection faces[FACE_MAX_DETECTIONS];
};

struct FaceConfig {
    float score_threshold = 0.50f;
    float nms_iou_threshold = 0.45f;
    int max_detections = FACE_MAX_DETECTIONS;
    int decode_level_begin = 0;
    int decode_level_end = 2;
};

float face_preprocess_rgb(const uint8_t* src_rgb,
                          int src_w,
                          int src_h,
                          int src_stride,
                          uint8_t* output_rgb);

int face_decode(const FaceRawTensor raw[FACE_OUTPUT_COUNT],
                int source_w,
                int source_h,
                float resize_scale,
                const FaceConfig& config,
                FaceResult* result);

using FaceSerialWrite = int (*)(void* user, const uint8_t* data, size_t bytes);

int face_send_uart(const FaceResult& result, FaceSerialWrite write, void* user);
