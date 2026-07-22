#include "face_business.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

struct Candidate {
    FaceDetection face;
};

float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

float obj_hinge_prob(float obj_hinge) {
    return std::max(0.0f, std::min(obj_hinge / 6.0f, 1.0f));
}

float fused_score(float cls_logit, float obj_hinge) {
    const float cls_score = sigmoid(cls_logit);
    const float obj_score = obj_hinge_prob(obj_hinge);
    return cls_score * obj_score;
}

float read_value(const FaceRawTensor& t, int c, int y, int x) {
    const size_t index =
        (static_cast<size_t>(c) * static_cast<size_t>(t.height) +
         static_cast<size_t>(y)) *
            static_cast<size_t>(t.width) +
        static_cast<size_t>(x);
    if (t.type == FaceTensorType::kInt8) {
        return static_cast<float>(static_cast<const int8_t*>(t.data)[index]) * t.scale;
    }
    return static_cast<const float*>(t.data)[index];
}

float clampf(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

float iou(const FaceDetection& a, const FaceDetection& b) {
    const float left = std::max(a.x, b.x);
    const float top = std::max(a.y, b.y);
    const float right = std::min(a.x + a.w, b.x + b.w);
    const float bottom = std::min(a.y + a.h, b.y + b.h);
    const float inter =
        std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    return inter / std::max(1e-6f, a.w * a.h + b.w * b.h - inter);
}

void put_u16(std::vector<uint8_t>* out, uint16_t value) {
    out->push_back(static_cast<uint8_t>(value & 0xFFU));
    out->push_back(static_cast<uint8_t>((value >> 8) & 0xFFU));
}

uint16_t pixel(float value) {
    return static_cast<uint16_t>(clampf(std::round(value), 0.0f, 65535.0f));
}

}  // namespace

float face_preprocess_rgb(const uint8_t* src_rgb,
                          int src_w,
                          int src_h,
                          int src_stride,
                          uint8_t* output_rgb) {
    if (src_rgb == nullptr || output_rgb == nullptr ||
        src_w <= 0 || src_h <= 0 || src_stride < src_w * 3) {
        return 0.0f;
    }

    const float scale = std::min(static_cast<float>(FACE_NET_W) / static_cast<float>(src_w),
                                 static_cast<float>(FACE_NET_H) / static_cast<float>(src_h));
    const int resize_w = std::max(1, static_cast<int>(std::round(src_w * scale)));
    const int resize_h = std::max(1, static_cast<int>(std::round(src_h * scale)));

    std::memset(output_rgb, 0, FACE_NET_W * FACE_NET_H * 3);
    for (int y = 0; y < resize_h; ++y) {
        const int src_y =
            std::min(src_h - 1, static_cast<int>(static_cast<float>(y) / scale));
        for (int x = 0; x < resize_w; ++x) {
            const int src_x =
                std::min(src_w - 1, static_cast<int>(static_cast<float>(x) / scale));
            std::memcpy(output_rgb + (y * FACE_NET_W + x) * 3,
                        src_rgb + src_y * src_stride + src_x * 3,
                        3);
        }
    }

    return scale;
}

int face_decode(const FaceRawTensor raw[FACE_OUTPUT_COUNT],
                int source_w,
                int source_h,
                float resize_scale,
                const FaceConfig& config,
                FaceResult* result) {
    if (raw == nullptr || result == nullptr ||
        source_w <= 0 || source_h <= 0 || resize_scale <= 0.0f) {
        return -1;
    }

    const int strides[3] = {8, 16, 32};
    const int level_begin = std::max(0, std::min(config.decode_level_begin, 2));
    const int level_end = std::max(level_begin, std::min(config.decode_level_end, 2));
    for (int level = level_begin; level <= level_end; ++level) {
        const FaceRawTensor& cls = raw[level];
        const FaceRawTensor& box = raw[3 + level];
        const FaceRawTensor& obj = raw[6 + level];
        const FaceRawTensor& kps = raw[9 + level];
        if (cls.data == nullptr || box.data == nullptr || obj.data == nullptr ||
            kps.data == nullptr || cls.channels != 1 || obj.channels != 1 ||
            box.channels != 4 || kps.channels != 10 ||
            cls.height != box.height || cls.width != box.width ||
            cls.height != obj.height || cls.width != obj.width ||
            cls.height != kps.height || cls.width != kps.width) {
            return -2;
        }
    }

    std::vector<Candidate> candidates;
    for (int level = level_begin; level <= level_end; ++level) {
        const FaceRawTensor& cls = raw[level];
        const FaceRawTensor& box = raw[3 + level];
        const FaceRawTensor& obj = raw[6 + level];
        const FaceRawTensor& kps = raw[9 + level];
        const float stride = static_cast<float>(strides[level]);

        for (int y = 0; y < cls.height; ++y) {
            for (int x = 0; x < cls.width; ++x) {
                const float score =
                    fused_score(read_value(cls, 0, y, x),
                                read_value(obj, 0, y, x));
                if (score < config.score_threshold) {
                    continue;
                }

                const float center_x = static_cast<float>(x) * stride +
                                       read_value(box, 0, y, x) * stride;
                const float center_y = static_cast<float>(y) * stride +
                                       read_value(box, 1, y, x) * stride;
                const float box_w =
                    std::exp(clampf(read_value(box, 2, y, x), -20.0f, 20.0f)) * stride;
                const float box_h =
                    std::exp(clampf(read_value(box, 3, y, x), -20.0f, 20.0f)) * stride;

                FaceDetection face;
                face.score = score;
                face.x = clampf((center_x - box_w * 0.5f) / resize_scale,
                                0.0f,
                                static_cast<float>(source_w));
                face.y = clampf((center_y - box_h * 0.5f) / resize_scale,
                                0.0f,
                                static_cast<float>(source_h));
                face.w = clampf(box_w / resize_scale,
                                0.0f,
                                static_cast<float>(source_w) - face.x);
                face.h = clampf(box_h / resize_scale,
                                0.0f,
                                static_cast<float>(source_h) - face.y);

                for (int p = 0; p < 5; ++p) {
                    face.landmarks[p].x =
                        clampf((static_cast<float>(x) * stride +
                                read_value(kps, p * 2, y, x) * stride) / resize_scale,
                               0.0f,
                               static_cast<float>(source_w));
                    face.landmarks[p].y =
                        clampf((static_cast<float>(y) * stride +
                                read_value(kps, p * 2 + 1, y, x) * stride) / resize_scale,
                               0.0f,
                               static_cast<float>(source_h));
                }
                candidates.push_back(Candidate{face});
            }
        }
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.face.score > b.face.score;
              });

    result->count = 0;
    const int max_count = std::min(config.max_detections, FACE_MAX_DETECTIONS);
    for (const Candidate& candidate : candidates) {
        bool keep = true;
        for (int i = 0; i < result->count; ++i) {
            if (iou(candidate.face, result->faces[i]) > config.nms_iou_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            result->faces[result->count++] = candidate.face;
        }
        if (result->count >= max_count) {
            break;
        }
    }
    return 0;
}

int face_send_uart(const FaceResult& result, FaceSerialWrite write, void* user) {
    if (write == nullptr || result.count < 0 || result.count > FACE_MAX_DETECTIONS) {
        return -1;
    }

    std::vector<uint8_t> frame;
    frame.reserve(8U + static_cast<size_t>(result.count) * 32U);
    frame.push_back(0xAA);
    frame.push_back(0x55);
    frame.push_back(0x01);
    frame.push_back(0x00);
    frame.push_back(0x00);
    put_u16(&frame, result.frame_id);
    frame.push_back(static_cast<uint8_t>(result.count));

    for (int i = 0; i < result.count; ++i) {
        const FaceDetection& face = result.faces[i];
        put_u16(&frame, pixel(face.x));
        put_u16(&frame, pixel(face.y));
        put_u16(&frame, pixel(face.w));
        put_u16(&frame, pixel(face.h));
        frame.push_back(static_cast<uint8_t>(
            clampf(std::round(face.score * 255.0f), 0.0f, 255.0f)));
        frame.push_back(static_cast<uint8_t>(face.identity));
        for (int p = 0; p < 5; ++p) {
            put_u16(&frame, pixel(face.landmarks[p].x));
            put_u16(&frame, pixel(face.landmarks[p].y));
        }
    }

    const uint16_t payload_len = static_cast<uint16_t>(frame.size() - 5U);
    frame[3] = static_cast<uint8_t>(payload_len & 0xFFU);
    frame[4] = static_cast<uint8_t>((payload_len >> 8) & 0xFFU);

    uint8_t checksum = 0;
    for (uint8_t byte : frame) {
        checksum ^= byte;
    }
    frame.push_back(checksum);

    for (size_t offset = 0; offset < frame.size(); offset += 32U) {
        const size_t size = std::min<size_t>(32U, frame.size() - offset);
        if (write(user, frame.data() + offset, size) != 0) {
            return -2;
        }
    }
    return 0;
}
