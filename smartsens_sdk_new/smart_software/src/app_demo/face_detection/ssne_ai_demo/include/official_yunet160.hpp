#pragma once

#include <array>
#include <string>
#include <vector>

#include "smartsoc/ssne_api.h"

struct OfficialYuNet160Detection {
    std::array<float, 4> box = {0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 10> landmarks = {0.0f};
    float score = 0.0f;
};

class OfficialYuNet160FaceDetector {
 public:
    void Initialize(const std::string& model_path,
                    const std::array<int, 2>& img_shape,
                    const std::array<int, 2>& det_shape);
    void Release();
    bool IsInitialized() const;
    bool Predict(ssne_tensor_t* img,
                 std::vector<OfficialYuNet160Detection>* detections,
                 std::string* error);

 private:
    void GeneratePriors();
    void RestoreOutputsFromHeads(const float* head0,
                                 const float* head1,
                                 const float* head2,
                                 const float* head3,
                                 int h0,
                                 int w0,
                                 int h1,
                                 int w1,
                                 int h2,
                                 int w2,
                                 int h3,
                                 int w3,
                                 std::vector<float>* loc,
                                 std::vector<float>* conf,
                                 std::vector<float>* iou);
    void Decode(const std::vector<float>& loc,
                const std::vector<float>& conf,
                const std::vector<float>& iou,
                std::vector<OfficialYuNet160Detection>* decoded);
    void Nms(std::vector<OfficialYuNet160Detection>* decoded);

    bool initialized_ = false;
    std::string model_path_;
    std::array<int, 2> img_shape_ = {0, 0};
    std::array<int, 2> det_shape_ = {0, 0};
    float w_scale_ = 1.0f;
    float h_scale_ = 1.0f;
    uint16_t model_id_ = 0;
    ssne_tensor_t inputs_[1] = {};
    ssne_tensor_t outputs_[4] = {};
    AiPreprocessPipe pipe_offline_ = AiPreprocessPipe{};
    bool pipe_initialized_ = false;
    bool input_created_ = false;
    bool output_ready_[4] = {false, false, false, false};
    std::vector<std::vector<int>> min_sizes_;
    std::vector<int> steps_;
    std::vector<float> variance_;
    std::vector<std::array<float, 4>> priors_;
};
