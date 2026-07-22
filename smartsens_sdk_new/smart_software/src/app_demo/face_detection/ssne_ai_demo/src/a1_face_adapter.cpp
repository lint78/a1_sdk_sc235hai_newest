#include "a1_face_adapter.hpp"

int FacePipeline::process_rgb(const uint8_t* rgb,
                              int width,
                              int height,
                              int stride,
                              uint16_t frame_id,
                              FaceResult* result) {
    if (result == nullptr) {
        return -1;
    }

    const float scale = face_preprocess_rgb(rgb, width, height, stride, input_);
    if (scale <= 0.0f) {
        return -2;
    }

    FaceRawTensor outputs[FACE_OUTPUT_COUNT] = {};
    const int ret = adapter_.infer_rgb224(input_, outputs);
    if (ret != 0) {
        return ret;
    }

    result->frame_id = frame_id;
    return face_decode(outputs, width, height, scale, cfg_, result);
}
