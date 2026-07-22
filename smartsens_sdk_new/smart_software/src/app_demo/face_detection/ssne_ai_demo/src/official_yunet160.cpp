#include "official_yunet160.hpp"

#include <algorithm>
#include <cmath>

#include "log.hpp"

namespace {

constexpr float kConfThreshold = 0.60f;
constexpr int kKeepTopK = 30;
constexpr float kNmsThreshold = 0.30f;

bool IsValidTensor(ssne_tensor_t tensor) {
    return get_data(tensor) != nullptr &&
           get_width(tensor) > 0 &&
           get_height(tensor) > 0 &&
           get_mem_size(tensor) > 0;
}

float ClampFloat(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

float IoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float left = std::max(a[0], b[0]);
    const float top = std::max(a[1], b[1]);
    const float right = std::min(a[2], b[2]);
    const float bottom = std::min(a[3], b[3]);
    const float inter =
        std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    const float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    const float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    return inter / std::max(1e-6f, area_a + area_b - inter);
}

}  // namespace

void OfficialYuNet160FaceDetector::Initialize(const std::string& model_path,
                                              const std::array<int, 2>& img_shape,
                                              const std::array<int, 2>& det_shape) {
    Release();
    LOG_INFO("official yunet160 init begin model=%s\n", model_path.c_str());

    model_path_ = model_path;
    img_shape_ = img_shape;
    det_shape_ = det_shape;
    w_scale_ = static_cast<float>(img_shape_[0]) / static_cast<float>(det_shape_[0]);
    h_scale_ = static_cast<float>(img_shape_[1]) / static_cast<float>(det_shape_[1]);

    min_sizes_ = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
    steps_ = {8, 16, 32, 64};
    variance_ = {0.1f, 0.2f};
    GeneratePriors();

    LOG_INFO("official yunet160 acquiring preprocess pipe\n");
    pipe_offline_ = GetAIPreprocessPipe();
    pipe_initialized_ = true;

    LOG_INFO("official yunet160 loading model\n");
    char* model_path_char = const_cast<char*>(model_path_.c_str());
    model_id_ = ssne_loadmodel(model_path_char, SSNE_STATIC_ALLOC);
    if (model_id_ == 0) {
        LOG_ERROR("official yunet160 load model failed: %s\n", model_path_.c_str());
        Release();
        return;
    }

    LOG_INFO("official yunet160 creating input tensor\n");
    inputs_[0] = create_tensor(static_cast<uint32_t>(det_shape_[0]),
                               static_cast<uint32_t>(det_shape_[1]),
                               SSNE_RGB,
                               SSNE_BUF_AI);
    input_created_ = true;
    if (!IsValidTensor(inputs_[0])) {
        LOG_ERROR("official yunet160 input tensor allocation failed\n");
        Release();
        return;
    }

    initialized_ = true;
    LOG_INFO("official yunet160 initialized model=%s det=[%d,%d]\n",
             model_path_.c_str(),
             det_shape_[0],
             det_shape_[1]);
}

void OfficialYuNet160FaceDetector::Release() {
    initialized_ = false;
    if (input_created_) {
        release_tensor(inputs_[0]);
    }
    inputs_[0] = ssne_tensor_t{};
    input_created_ = false;

    for (int i = 0; i < 4; ++i) {
        if (output_ready_[i]) {
            release_tensor(outputs_[i]);
        }
        outputs_[i] = ssne_tensor_t{};
        output_ready_[i] = false;
    }

    if (pipe_initialized_) {
        ReleaseAIPreprocessPipe(pipe_offline_);
    }
    pipe_offline_ = AiPreprocessPipe{};
    pipe_initialized_ = false;
    model_id_ = 0;
    priors_.clear();
}

bool OfficialYuNet160FaceDetector::IsInitialized() const {
    return initialized_;
}

bool OfficialYuNet160FaceDetector::Predict(ssne_tensor_t* img,
                                           std::vector<OfficialYuNet160Detection>* detections,
                                           std::string* error) {
    if (detections == nullptr) {
        return false;
    }
    detections->clear();

    if (!initialized_) {
        if (error != nullptr) {
            *error = "official yunet160 detector not initialized";
        }
        return false;
    }
    if (img == nullptr || !IsValidTensor(*img)) {
        if (error != nullptr) {
            *error = "official yunet160 invalid input tensor";
        }
        return false;
    }
    if (!input_created_ || !IsValidTensor(inputs_[0])) {
        if (error != nullptr) {
            *error = "official yunet160 invalid input buffer";
        }
        return false;
    }

    int ret = RunAiPreprocessPipe(pipe_offline_, *img, inputs_[0]);
    if (ret != 0) {
        if (error != nullptr) {
            *error = "official yunet160 preprocess failed";
        }
        LOG_ERROR("official yunet160 preprocess failed ret=%d\n", ret);
        return false;
    }

    ret = ssne_inference(model_id_, 1, inputs_);
    if (ret != 0) {
        if (error != nullptr) {
            *error = "official yunet160 inference failed";
        }
        LOG_ERROR("official yunet160 inference failed ret=%d\n", ret);
        return false;
    }

    ret = ssne_getoutput(model_id_, 4, outputs_);
    if (ret != 0) {
        if (error != nullptr) {
            *error = "official yunet160 getoutput failed";
        }
        LOG_ERROR("official yunet160 getoutput failed ret=%d\n", ret);
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        output_ready_[i] = true;
        if (!IsValidTensor(outputs_[i])) {
            if (error != nullptr) {
                *error = "official yunet160 output tensor invalid";
            }
            return false;
        }
    }

    std::vector<float> loc;
    std::vector<float> conf;
    std::vector<float> iou;
    RestoreOutputsFromHeads(
        static_cast<const float*>(get_data(outputs_[0])),
        static_cast<const float*>(get_data(outputs_[1])),
        static_cast<const float*>(get_data(outputs_[2])),
        static_cast<const float*>(get_data(outputs_[3])),
        static_cast<int>(get_height(outputs_[0])),
        static_cast<int>(get_width(outputs_[0])),
        static_cast<int>(get_height(outputs_[1])),
        static_cast<int>(get_width(outputs_[1])),
        static_cast<int>(get_height(outputs_[2])),
        static_cast<int>(get_width(outputs_[2])),
        static_cast<int>(get_height(outputs_[3])),
        static_cast<int>(get_width(outputs_[3])),
        &loc,
        &conf,
        &iou);

    std::vector<OfficialYuNet160Detection> decoded;
    Decode(loc, conf, iou, &decoded);
    Nms(&decoded);

    if (decoded.size() > static_cast<size_t>(kKeepTopK)) {
        decoded.resize(kKeepTopK);
    }
    *detections = decoded;
    return true;
}

void OfficialYuNet160FaceDetector::GeneratePriors() {
    const int w = det_shape_[0];
    const int h = det_shape_[1];
    const int feature_map_2th_h = ((h + 1) / 2) / 2;
    const int feature_map_2th_w = ((w + 1) / 2) / 2;
    const int feature_map_3th_h = feature_map_2th_h / 2;
    const int feature_map_3th_w = feature_map_2th_w / 2;
    const int feature_map_4th_h = feature_map_3th_h / 2;
    const int feature_map_4th_w = feature_map_3th_w / 2;
    const int feature_map_5th_h = feature_map_4th_h / 2;
    const int feature_map_5th_w = feature_map_4th_w / 2;
    const int feature_map_6th_h = feature_map_5th_h / 2;
    const int feature_map_6th_w = feature_map_5th_w / 2;

    const std::vector<std::array<int, 2>> feature_maps = {
        {feature_map_3th_h, feature_map_3th_w},
        {feature_map_4th_h, feature_map_4th_w},
        {feature_map_5th_h, feature_map_5th_w},
        {feature_map_6th_h, feature_map_6th_w}
    };

    priors_.clear();
    for (size_t k = 0; k < feature_maps.size(); ++k) {
        const int fh = feature_maps[k][0];
        const int fw = feature_maps[k][1];
        for (int i = 0; i < fh; ++i) {
            for (int j = 0; j < fw; ++j) {
                for (size_t m = 0; m < min_sizes_[k].size(); ++m) {
                    const int min_size = min_sizes_[k][m];
                    const float s_kx = static_cast<float>(min_size) / static_cast<float>(w);
                    const float s_ky = static_cast<float>(min_size) / static_cast<float>(h);
                    const float cx =
                        (static_cast<float>(j) + 0.5f) * static_cast<float>(steps_[k]) /
                        static_cast<float>(w);
                    const float cy =
                        (static_cast<float>(i) + 0.5f) * static_cast<float>(steps_[k]) /
                        static_cast<float>(h);
                    priors_.push_back({cx, cy, s_kx, s_ky});
                }
            }
        }
    }
}

void OfficialYuNet160FaceDetector::RestoreOutputsFromHeads(
    const float* head0,
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
    std::vector<float>* iou) {
    constexpr int kDimPerAnchor = 17;
    constexpr int kLocDim = 14;
    const int c0 = 51;
    const int c1 = 34;
    const int c2 = 34;
    const int c3 = 51;

    std::vector<float> flat;
    flat.reserve(h0 * w0 * c0 + h1 * w1 * c1 + h2 * w2 * c2 + h3 * w3 * c3);

    auto flatten_head = [&flat](const float* head, int height, int width, int channels) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int base = y * width * channels + x * channels;
                for (int c = 0; c < channels; ++c) {
                    flat.push_back(head[base + c]);
                }
            }
        }
    };

    flatten_head(head0, h0, w0, c0);
    flatten_head(head1, h1, w1, c1);
    flatten_head(head2, h2, w2, c2);
    flatten_head(head3, h3, w3, c3);

    if (flat.size() % kDimPerAnchor != 0U) {
        return;
    }

    const int num_anchors = static_cast<int>(flat.size() / kDimPerAnchor);
    loc->resize(static_cast<size_t>(num_anchors) * static_cast<size_t>(kLocDim));
    conf->resize(static_cast<size_t>(num_anchors) * 2U);
    iou->resize(static_cast<size_t>(num_anchors));

    for (int k = 0; k < num_anchors; ++k) {
        const float* p = &flat[static_cast<size_t>(k) * static_cast<size_t>(kDimPerAnchor)];
        for (int i = 0; i < kLocDim; ++i) {
            (*loc)[static_cast<size_t>(k) * static_cast<size_t>(kLocDim) +
                   static_cast<size_t>(i)] = p[i];
        }

        const float maxv = std::max(p[14], p[15]);
        const float e0 = std::exp(p[14] - maxv);
        const float e1 = std::exp(p[15] - maxv);
        const float sum = e0 + e1;
        (*conf)[static_cast<size_t>(k) * 2U] = e0 / sum;
        (*conf)[static_cast<size_t>(k) * 2U + 1U] = e1 / sum;
        (*iou)[static_cast<size_t>(k)] = p[16];
    }
}

void OfficialYuNet160FaceDetector::Decode(const std::vector<float>& loc,
                                          const std::vector<float>& conf,
                                          const std::vector<float>& iou,
                                          std::vector<OfficialYuNet160Detection>* decoded) {
    decoded->clear();
    const int num_priors = static_cast<int>(priors_.size());
    for (int i = 0; i < num_priors; ++i) {
        const float cls_score = conf[static_cast<size_t>(i) * 2U + 1U];
        const float iou_score = ClampFloat(iou[static_cast<size_t>(i)], 0.0f, 1.0f);
        const float score = std::sqrt(cls_score * iou_score);
        if (score <= kConfThreshold) {
            continue;
        }

        const std::array<float, 4>& prior = priors_[static_cast<size_t>(i)];
        const float cx = prior[0];
        const float cy = prior[1];
        const float s_kx = prior[2];
        const float s_ky = prior[3];
        const float* loc_ptr =
            &loc[static_cast<size_t>(i) * static_cast<size_t>(14)];

        const float pred_cx = cx + loc_ptr[0] * variance_[0] * s_kx;
        const float pred_cy = cy + loc_ptr[1] * variance_[0] * s_ky;
        const float pred_w = s_kx * std::exp(loc_ptr[2] * variance_[1]);
        const float pred_h = s_ky * std::exp(loc_ptr[3] * variance_[1]);

        OfficialYuNet160Detection face;
        face.score = score;
        face.box[0] =
            ClampFloat((pred_cx - pred_w * 0.5f) * static_cast<float>(det_shape_[0]),
                       0.0f,
                       static_cast<float>(det_shape_[0])) * w_scale_;
        face.box[1] =
            ClampFloat((pred_cy - pred_h * 0.5f) * static_cast<float>(det_shape_[1]),
                       0.0f,
                       static_cast<float>(det_shape_[1])) * h_scale_;
        face.box[2] =
            ClampFloat((pred_cx + pred_w * 0.5f) * static_cast<float>(det_shape_[0]),
                       0.0f,
                       static_cast<float>(det_shape_[0])) * w_scale_;
        face.box[3] =
            ClampFloat((pred_cy + pred_h * 0.5f) * static_cast<float>(det_shape_[1]),
                       0.0f,
                       static_cast<float>(det_shape_[1])) * h_scale_;

        for (int j = 0; j < 5; ++j) {
            face.landmarks[static_cast<size_t>(j) * 2U] =
                (cx + loc_ptr[4 + j * 2] * variance_[0] * s_kx) *
                static_cast<float>(det_shape_[0]) * w_scale_;
            face.landmarks[static_cast<size_t>(j) * 2U + 1U] =
                (cy + loc_ptr[4 + j * 2 + 1] * variance_[0] * s_ky) *
                static_cast<float>(det_shape_[1]) * h_scale_;
        }
        decoded->push_back(face);
    }
}

void OfficialYuNet160FaceDetector::Nms(std::vector<OfficialYuNet160Detection>* decoded) {
    std::sort(decoded->begin(), decoded->end(),
              [](const OfficialYuNet160Detection& a, const OfficialYuNet160Detection& b) {
                  return a.score > b.score;
              });
    std::vector<OfficialYuNet160Detection> kept;
    kept.reserve(decoded->size());
    for (const auto& det : *decoded) {
        bool keep = true;
        for (const auto& prior : kept) {
            if (IoU(det.box, prior.box) > kNmsThreshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            kept.push_back(det);
        }
        if (kept.size() >= static_cast<size_t>(kKeepTopK)) {
            break;
        }
    }
    *decoded = kept;
}
