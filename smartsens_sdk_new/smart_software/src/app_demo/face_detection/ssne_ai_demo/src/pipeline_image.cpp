/*
 * @Filename: pipeline_image.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 * @Description: 图像输入与在线裁剪模块
*/
#include "../include/common.hpp"
#include "../include/log.hpp"
#include <iostream>
#include <unistd.h>

/**
 * @brief 图像处理器初始化函数
 * @param in_img_shape 输入原图尺寸 [宽度, 高度]
 *
 * 官方标准流程：
 * - 原图：1920x1080
 * - 在线中心裁剪为：1080x1080
 * - 后续 AI 预处理链再 resize 到 640x640
 *
 * 这样可保证：
 * - crop 图比例 = 1:1
 * - 模型输入比例 = 1:1
 * - 避免直接拉伸原图导致目标变形
 */
void IMAGEPROCESSOR::Initialize(std::array<int, 2>* in_img_shape)
{
    img_shape = *in_img_shape;

    const uint16_t img_width  = static_cast<uint16_t>(img_shape[0]);
    const uint16_t img_height = static_cast<uint16_t>(img_shape[1]);

    // 在线图像格式
    format_online = SSNE_YUV422_16;

    // 1920x1080 -> 中心裁剪成 1080x1080
    // 左右各裁 420 像素
    const uint16_t crop_width   = 1080;
    const uint16_t crop_height  = 1080;
    const uint16_t crop_x_start = (img_width - crop_width) / 2;   // 1920 -> 1080 => 420
    const uint16_t crop_x_end   = crop_x_start + crop_width;      // 1500
    const uint16_t crop_y_start = 0;
    const uint16_t crop_y_end   = crop_height;

    std::cout << "[INFO] IMAGEPROCESSOR::Initialize" << std::endl;
    std::cout << "[INFO] original image      = [" << img_width << ", " << img_height << "]" << std::endl;
    std::cout << "[INFO] crop region        = [x:" << crop_x_start << " ~ " << crop_x_end
              << ", y:" << crop_y_start << " ~ " << crop_y_end << "]" << std::endl;
    std::cout << "[INFO] crop output image  = [" << crop_width << ", " << crop_height << "]" << std::endl;

    // pipe0：输出裁剪图；后续 detect/pose 坐标都以这个 1080x1080 crop 为基准再映射回原图。
    OnlineSetCrop(kPipeline0, crop_x_start, crop_x_end, crop_y_start, crop_y_end);
    OnlineSetOutputImage(kPipeline0, format_online, crop_width, crop_height);
    const int frame_drop_ret = OnlineSetFrameDrop(kPipeline0, 0, 0);
    if (frame_drop_ret != 0) {
        LOG_WARN("OnlineSetFrameDrop(kPipeline0, 0, 0) failed, ret=%d\n", frame_drop_ret);
    } else {
        LOG_INFO("OnlineSetFrameDrop(kPipeline0, 0, 0) applied\n");
    }

    int ret = OpenOnlinePipeline(kPipeline0);
    if (ret != 0) {
        std::cout << "[ERROR] Failed to open online pipeline!" << std::endl;
        std::cout << "[ERROR] ret = " << ret << std::endl;
        return;
    }

    std::cout << "[INFO] OpenOnlinePipeline(kPipeline0) success, ret = " << ret << std::endl;
}

/**
 * @brief 从 pipeline 获取裁剪图像数据
 * @param img_sensor 输出参数：存储从 pipe0 获取的裁剪图像
 *
 * @note 输出的是 crop 图（1080x1080），不是原图（1920x1080）
 */
void IMAGEPROCESSOR::GetImage(ssne_tensor_t* img_sensor)
{
    // 在线 pipeline 直接给出 YUV422 crop tensor，不在这里做 AI resize，resize 交给各模型离线预处理。
    int capture_code = GetImageData(img_sensor, kPipeline0, kSensor0, false);

    if (capture_code != 0) {
        std::cout << "[IMAGEPROCESSOR] Get invalid image from kPipeline0!" << std::endl;
        std::cout << "[IMAGEPROCESSOR] ret = " << capture_code << std::endl;
    }
}

/**
 * @brief 释放图像处理器资源，关闭 pipeline
 */
void IMAGEPROCESSOR::Release()
{
    // 关闭在线 pipeline 后，后续 tensor 生命周期由底层 SDK 释放管理。
    CloseOnlinePipeline(kPipeline0);
    std::cout << "[INFO] Online pipeline closed!" << std::endl;
}
