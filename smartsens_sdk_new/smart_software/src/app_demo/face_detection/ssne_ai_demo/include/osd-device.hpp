/*
 * @Author: Jingwen Bai
 * @Date: 2024-07-04 11:07:00
 * @Description: 
 * @Filename: osd-device.hpp
 */
#ifndef SST_OSD_DEVICE_HPP_
#define SST_OSD_DEVICE_HPP_

#include <vector>
#include <string>

#include "osd_lib_api.h"
#include "common.hpp"

#define BUFFER_TYPE_DMABUF  0x1
#define OSD_LAYER_SIZE 3  // 只使用3个图层：0(检测框), 1(固定正方形), 2(位图)

namespace sst{
namespace device{
namespace osd{

typedef struct {
    std::array<float, 4> box;          // 待绘制矩形的坐标 [x1, y1, x2, y2]
    int border;                        // 边框厚度
    int layer_id;                      // 目标图层 ID
    fdevice::QUADRANGLETYPE type;      // 绘制类型：空心或实心
    fdevice::ALPHATYPE alpha;          // 透明度
    int color;                         // LUT 颜色索引
}OsdQuadRangle;

// OSD 底层封装模块。
// 主要职责是：
// 1. 打开 OSD 设备并创建图层；
// 2. 管理图层对应的 DMA buffer；
// 3. 提供矩形框、Cover、位图等几种绘制接口。
class OsdDevice {
public:
    OsdDevice();
    ~OsdDevice();

    // 初始化 OSD 设备与图层。
    // width/height 对应整张输出画面的尺寸。
    void Initialize(int width, int height, const char* bitmap_lut_path = nullptr);

    // 释放设备、图层和 DMA buffer。
    void Release();

    // 自动图层模式：绘制一组四边形，空输入时清空所有图层。
    void Draw(std::vector<OsdQuadRangle> &quad_rangle);

    // 直接输入 bbox 列表并绘制到指定图层。
    void Draw(std::vector<std::array<float, 4>>& boxes, int border, int layer_id, fdevice::QUADRANGLETYPE type, fdevice::ALPHATYPE alpha, int color);

    // 指定图层模式：绘制一组四边形，空输入时只清指定图层。
    void Draw(std::vector<OsdQuadRangle> &quad_rangle, int layer_id);

    // 直接下发底层 cover 数据，适合画骨架线、关键点块等非标准框图元。
    void DrawCovers(std::vector<fdevice::COVER_ATTR_S>& covers, int layer_id);
    /**
     * @brief 绘制位图到指定图层
     * @param bitmap_path 位图文件路径（.ssbmp格式）
     * @param lut_path LUT文件路径（.sscl格式），如果为空则使用默认LUT
     * @param layer_id 图层ID
     * @param pos_x 位图左上角X坐标（相对于画面，0为左上角）
     * @param pos_y 位图左上角Y坐标（相对于画面，0为左上角）
     * @param alpha 透明度
     * @description 在位图图层上绘制位图，位置在整个图像上
     */
    void DrawTexture(const char* bitmap_path, const char* lut_path, int layer_id, int pos_x = 0, int pos_y = 0, fdevice::ALPHATYPE alpha = fdevice::TYPE_ALPHA100);

private:
    // 读取 LUT 文件并缓存到内存，供 OSD 初始化使用。
    int LoadLutFile(const char* filename);

    // 根据输入 box 和边框厚度，生成 OSD 底层需要的内外四边形顶点。
    void GenQrangleBox(std::array<float, 4>& det, int border);

private:
    handle_t m_osd_handle;                                  // OSD 设备句柄
    std::string m_osd_lut_path = "/app_demo/app_assets/colorLUT.sscl";  // 默认颜色 LUT 路径
    // std::string m_texture_path = "/ai/imgs/test_24.ssbmp";
    uint8_t *m_pcolor_lut = nullptr;                        // LUT 文件内容缓存
    int m_file_size = 0;                                    // LUT 文件大小
    int m_height, m_width;                                  // 画面尺寸
    
    fdevice::DMA_BUFFER_ATTR_S m_layer_dma[OSD_LAYER_SIZE]; // 每个图层绑定的 DMA buffer
    fdevice::VERTEXS_S m_qrangle_out={0}, m_qrangle_in={0}; // 当前矩形对应的外轮廓和内轮廓顶点
};

} // namespace osd
} // namespace device
} // namespace sst

#endif // SST_OSD_DEVICE_HPP_
