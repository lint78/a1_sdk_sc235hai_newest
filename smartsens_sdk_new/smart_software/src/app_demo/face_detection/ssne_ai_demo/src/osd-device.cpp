/*
 * @Author: Jingwen Bai
 * @Date: 2024-07-04 11:07:00
 * @Description: osd device
 * @Filename: osd-device.cpp
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>

#include "../include/osd-device.hpp"
#include "log.hpp"

using namespace fdevice;

namespace sst {
namespace device {
namespace osd {

OsdDevice::OsdDevice()
    : m_height(0),
      m_width(0) {
}

OsdDevice::~OsdDevice() {
    std::cout << "OsdDevice Destructor" << std::endl;
}

void OsdDevice::Initialize(int width, int height, const char* bitmap_lut_path) {
    // 保存整张输出画面的尺寸，所有图层都按这个尺寸初始化。
    m_width = width;
    m_height = height;

    // 如果之前已经分配过 LUT，先释放，避免重复 Initialize 泄漏
    if (m_pcolor_lut != nullptr) {
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
        m_file_size = 0;
    }

    // load osd color lut
    // 如果提供了位图LUT路径，优先使用位图LUT；否则使用默认LUT
    if (bitmap_lut_path != nullptr && std::strlen(bitmap_lut_path) > 0) {
        if (LoadLutFile(bitmap_lut_path) == 0) {
            std::cout << "[OsdDevice] Using bitmap LUT: " << bitmap_lut_path << std::endl;
        } else {
            std::cerr << "[OsdDevice] Warning: Failed to load bitmap LUT, using default LUT" << std::endl;
            if (LoadLutFile(m_osd_lut_path.c_str()) != 0) {
                std::cerr << "[OsdDevice] ERROR: Failed to load default LUT too!" << std::endl;
                return;
            }
        }
    } else {
        if (LoadLutFile(m_osd_lut_path.c_str()) != 0) {
            std::cerr << "[OsdDevice] ERROR: Failed to load default LUT!" << std::endl;
            return;
        }
    }

    // 打开底层 OSD 设备。
    m_osd_handle = osd_open_device();
    if (m_osd_handle == 0) {
        std::cerr << "[OsdDevice] ERROR: osd_open_device failed!" << std::endl;
        return;
    }

    // init osd (必须在创建图层前调用)
    int ret = osd_init_device(m_osd_handle, OSD_LAYER_SIZE, (char*)m_pcolor_lut);
    if (ret != 0) {
        std::cerr << "[OsdDevice] ERROR: osd_init_device failed! ret=" << ret << std::endl;
        return;
    }

    // 0/1 号图层用于绘制图形类对象，比如矩形框、线段和 cover。
    int dma_size = 1024;  // 图形层使用较小的 DMA buffer
    for (int layer_index = 0; layer_index < 2; layer_index++) {
        ret = osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma, dma_size);
        if (ret != 0) {
            std::cerr << "[OsdDevice] ERROR: osd_alloc_buffer failed for layer "
                      << layer_index << " dma, ret=" << ret << std::endl;
            continue;
        }

        usleep(250000);  // 250ms，原来的 sleep(0.25) 实际不会等待

        ret = osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma_2, dma_size);
        if (ret != 0) {
            std::cerr << "[OsdDevice] ERROR: osd_alloc_buffer failed for layer "
                      << layer_index << " dma_2, ret=" << ret << std::endl;
            continue;
        }

        int dma_fd = osd_get_buffer_fd(m_osd_handle, m_layer_dma[layer_index].dma);  // 当前 DMA buffer 的 fd

        LAYER_ATTR_S osd_layer;
        std::memset(&osd_layer, 0, sizeof(osd_layer));
        osd_layer.codeTYPE = SS_TYPE_QUADRANGLE;                       // 图层编码类型：四边形/cover
        osd_layer.layer_data_QR.osd_buf.buf_type = BUFFER_TYPE_DMABUF; // 图层数据来自 DMA buffer
        osd_layer.layer_data_QR.osd_buf.buf.fd_dmabuf = dma_fd;        // 绑定 DMA fd
        osd_layer.layerStart.layer_start_x = 0;                        // 图层起点 x
        osd_layer.layerStart.layer_start_y = 0;                        // 图层起点 y
        osd_layer.layerSize.layer_width = m_width;                     // 图层宽度
        osd_layer.layerSize.layer_height = m_height;                   // 图层高度
        osd_layer.layer_rgn = {TYPE_GRAPHIC, {m_width, m_height}};     // 区域类型：图形层

        ret = osd_create_layer(m_osd_handle, (ssLAYER_HANDLE)layer_index, &osd_layer);
        if (ret != 0) {
            std::cerr << "[OsdDevice] ERROR: osd_create_layer failed! ret=" << ret
                      << ", layer_index=" << layer_index << std::endl;
            continue;
        }

        ret = osd_set_layer_buffer(m_osd_handle, (ssLAYER_HANDLE)layer_index, m_layer_dma[layer_index]);
        if (ret != 0) {
            std::cerr << "[OsdDevice] ERROR: osd_set_layer_buffer failed! ret=" << ret
                      << ", layer_index=" << layer_index << std::endl;
            continue;
        }

        std::cout << "[OsdDevice] Graphic layer " << layer_index << " initialized successfully" << std::endl;
    }

    // 2 号图层用于绘制位图贴图，单独创建 TYPE_IMAGE，避免和检测框图层互相污染。
    {
        int layer_index = 2;
        int texture_dma_size = 0x20000;  // 位图层通常需要更大的 buffer

        ret = osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma, texture_dma_size);
        if (ret != 0) {
            std::cerr << "[OsdDevice] ERROR: osd_alloc_buffer failed for bitmap layer dma, ret="
                      << ret << std::endl;
            return;
        }

        usleep(250000);  // 250ms

        ret = osd_alloc_buffer(m_osd_handle, m_layer_dma[layer_index].dma_2, texture_dma_size);
        if (ret != 0) {
            std::cerr << "[OsdDevice] ERROR: osd_alloc_buffer failed for bitmap layer dma_2, ret="
                      << ret << std::endl;
            return;
        }

        int dma_fd = osd_get_buffer_fd(m_osd_handle, m_layer_dma[layer_index].dma);

        LAYER_ATTR_S osd_layer;
        std::memset(&osd_layer, 0, sizeof(osd_layer));
        osd_layer.codeTYPE = SS_TYPE_RLE;                               // 图层编码类型：RLE 位图
        osd_layer.layer_data_RLE.osd_buf.buf_type = BUFFER_TYPE_DMABUF; // 图层数据来自 DMA buffer
        osd_layer.layer_data_RLE.osd_buf.buf.fd_dmabuf = dma_fd;        // 绑定 DMA fd
        osd_layer.layerStart.layer_start_x = 0;                         // 图层起点 x
        osd_layer.layerStart.layer_start_y = 0;                         // 图层起点 y
        osd_layer.layerSize.layer_width = m_width;                      // 图层宽度
        osd_layer.layerSize.layer_height = m_height;                    // 图层高度
        osd_layer.layer_rgn = {TYPE_IMAGE, {m_width, m_height}};        // 区域类型：图像层

        std::cout << "[OsdDevice] Creating layer " << layer_index << " with TYPE_IMAGE" << std::endl;
        std::cout << "[OsdDevice] Layer region type: " << (int)osd_layer.layer_rgn.enType
                  << " (0=TYPE_IMAGE, 1=TYPE_GRAPHIC)" << std::endl;
        std::cout << "[OsdDevice] Layer region size: " << osd_layer.layer_rgn.size_s.w
                  << "x" << osd_layer.layer_rgn.size_s.h << std::endl;

        ret = osd_create_layer(m_osd_handle, (ssLAYER_HANDLE)layer_index, &osd_layer);
        if (ret != 0) {
            std::cerr << "[OsdDevice] ERROR: osd_create_layer failed! ret=" << ret
                      << ", layer_index=" << layer_index << std::endl;
            return;
        } else {
            std::cout << "[OsdDevice] Layer " << layer_index << " created successfully" << std::endl;
        }

        ret = osd_set_layer_buffer(m_osd_handle, (ssLAYER_HANDLE)layer_index, m_layer_dma[layer_index]);
        if (ret != 0) {
            std::cerr << "[OsdDevice] ERROR: osd_set_layer_buffer failed! ret=" << ret
                      << ", layer_index=" << layer_index << std::endl;
            return;
        } else {
            std::cout << "[OsdDevice] Layer " << layer_index << " buffer set successfully" << std::endl;
        }
    }

    // 图层0-1用于quad-rangle，图层2用于位图
    // 图层3-4未使用，已删除以节省内存
}

void OsdDevice::Release() {
    std::cout << "OsdDevice Release" << std::endl;

    if (m_osd_handle != 0) {
        // 依次销毁图层，并释放每个图层关联的 DMA buffer。
        // destroy layer and delete dma buf
        for (int i = 0; i < OSD_LAYER_SIZE; i++) {
            osd_destroy_layer(m_osd_handle, (ssLAYER_HANDLE)i);

            if (m_layer_dma[i].dma != nullptr) {
                osd_delete_buffer(m_osd_handle, m_layer_dma[i].dma);
                m_layer_dma[i].dma = nullptr;
            }

            if (m_layer_dma[i].dma_2 != nullptr) {
                osd_delete_buffer(m_osd_handle, m_layer_dma[i].dma_2);
                m_layer_dma[i].dma_2 = nullptr;
            }
        }

        osd_close_device(m_osd_handle);
        m_osd_handle = 0;
    }

    if (m_pcolor_lut != nullptr) {
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
        m_file_size = 0;
    }
}

int OsdDevice::LoadLutFile(const char* filename) {
    std::cout << "[OsdDevice] Attempting to load LUT file: " << filename << std::endl;

    // 若重复加载，先释放旧 LUT，避免泄漏
    if (m_pcolor_lut != nullptr) {
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
        m_file_size = 0;
    }

    // 检查文件是否存在
    struct stat file_stat;
    if (stat(filename, &file_stat) != 0) {
        std::cerr << "[OsdDevice] ERROR: File does not exist or cannot access: " << filename << std::endl;
        std::cerr << "[OsdDevice] Error code: " << errno << " (" << strerror(errno) << ")" << std::endl;
        return -1;
    }

    // 检查文件大小
    if (file_stat.st_size <= 0) {
        std::cerr << "[OsdDevice] ERROR: Invalid file size: " << file_stat.st_size << " bytes" << std::endl;
        return -1;
    }

    std::cout << "[OsdDevice] File exists, size: " << file_stat.st_size << " bytes" << std::endl;

    // 检查文件权限
    if (access(filename, R_OK) != 0) {
        std::cerr << "[OsdDevice] ERROR: No read permission for file: " << filename << std::endl;
        std::cerr << "[OsdDevice] Error code: " << errno << " (" << strerror(errno) << ")" << std::endl;
        return -1;
    }

    // 打开文件
    // 以二进制形式打开 LUT 文件。
    std::ifstream file(filename, std::ios::binary | std::ios::in | std::ios::ate);
    if (!file) {
        std::cerr << "[OsdDevice] ERROR: Cannot open file: " << filename << std::endl;
        std::cerr << "[OsdDevice] Error code: " << errno << " (" << strerror(errno) << ")" << std::endl;
        return -1;
    }

    // 获取文件大小
    m_file_size = static_cast<int>(file.tellg());
    if (m_file_size <= 0) {
        std::cerr << "[OsdDevice] ERROR: Invalid file size from stream: " << m_file_size << " bytes" << std::endl;
        file.close();
        return -1;
    }

    // 分配内存并读取文件
    // 申请内存并把 LUT 整体读入缓存。
    m_pcolor_lut = new uint8_t[m_file_size];
    file.seekg(0, std::ios::beg);
    file.read((char*)m_pcolor_lut, m_file_size);

    // 检查是否读取成功
    if (static_cast<int>(file.gcount()) != m_file_size) {
        std::cerr << "[OsdDevice] ERROR: Failed to read complete file. Expected: " << m_file_size
                  << " bytes, Read: " << file.gcount() << " bytes" << std::endl;
        delete[] m_pcolor_lut;
        m_pcolor_lut = nullptr;
        m_file_size = 0;
        file.close();
        return -1;
    }

    file.close();

    std::cout << "[OsdDevice] Successfully loaded LUT file: " << filename
              << ", size: " << m_file_size << " bytes" << std::endl;
    return 0;
}

// draw mode: auto alloc layer
void OsdDevice::Draw(std::vector<OsdQuadRangle> &quad_rangle) {
    if (quad_rangle.size() == 0) {
        // 自动图层模式下，空输入表示清空所有图层。
        osd_clean_all_layer(m_osd_handle);
        return;
    }

    for (auto &q : quad_rangle) {
        // 先把 box 变成底层需要的内外轮廓，再提交给 OSD。
        GenQrangleBox(q.box, q.border);
        COVER_ATTR_S qrangle_attr = {q.color, q.type, q.alpha, m_qrangle_out, m_qrangle_in};
        osd_add_quad_rangle(m_osd_handle, &qrangle_attr);
    }

    osd_flush_quad_rangle(m_osd_handle);
}

// draw mode: manual alloc layer
void OsdDevice::Draw(std::vector<OsdQuadRangle> &quad_rangle, int layer_id) {
    if (quad_rangle.size() == 0) {
        // 指定图层模式下，空输入只清空当前图层。
        osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
        LOG_DEBUG("Draw --- osd_clean_layer\n");
        return;
    }

    int ret = 0;
    for (auto &q : quad_rangle) {
        LOG_DEBUG("Draw --- q.box: %f, %f, %f, %f\n", q.box[0], q.box[1], q.box[2], q.box[3]);
        // 将业务层的 box/border 转换成底层四边形描述。
        GenQrangleBox(q.box, q.border);
        COVER_ATTR_S qrangle_attr = {q.color, q.type, q.alpha, m_qrangle_out, m_qrangle_in};
        ret = osd_add_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &qrangle_attr);
        LOG_DEBUG("Draw --- osd_add_quad_rangle_layer ret: %d\n", ret);
    }

    osd_flush_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
}

// draw mode: manual alloc layer
void OsdDevice::Draw(std::vector<std::array<float, 4>>& boxes,
                     int border,
                     int layer_id,
                     tagQUADRANGLETYPE type,
                     tagALPHATYPE alpha,
                     int color) {
    if (boxes.size() == 0) {
        osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
        return;
    }

    int ret = 0;
    for (auto &box : boxes) {
        // 这个重载只接收 bbox 坐标，因此统一套用调用者传入的样式参数。
        GenQrangleBox(box, border);
        COVER_ATTR_S qrangle_attr = {color, type, alpha, m_qrangle_out, m_qrangle_in};
        ret = osd_add_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &qrangle_attr);
        (void)ret;
    }

    osd_flush_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
}

void OsdDevice::DrawCovers(std::vector<fdevice::COVER_ATTR_S>& covers, int layer_id) {
    if (covers.empty()) {
        // 本帧没有可视化对象时必须清层，否则上一帧框和骨架会残留。
        osd_clean_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
        return;
    }

    for (auto& cover : covers) {
        // covers 已经是底层图元结构，直接透传给图层。
        osd_add_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &cover);
    }

    osd_flush_quad_rangle_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
}

/**
 * @brief 绘制位图到指定图层
 * @note LUT应该在初始化时加载，osd_init_device必须在创建图层前调用
 *       如果在绘制时重新初始化，会破坏已创建的图层
 */
void OsdDevice::DrawTexture(const char* bitmap_path,
                            const char* lut_path,
                            int layer_id,
                            int pos_x,
                            int pos_y,
                            fdevice::ALPHATYPE alpha) {
    (void)lut_path; // LUT 已在 Initialize 时加载
    (void)alpha;    // 当前实现仍固定使用 TYPE_ALPHA100

    // 位图绘制使用 BITMAP_INFO_S 描述文件路径、位置和透明度。
    fdevice::BITMAP_INFO_S bm_info;
    bm_info.pSSbmpFile = bitmap_path;       // 位图文件路径
    bm_info.alpha = fdevice::TYPE_ALPHA100; // 当前实现固定使用全不透明
    bm_info.position.x = pos_x;             // 位图左上角 x 坐标
    bm_info.position.y = pos_y;             // 位图左上角 y 坐标

    LOG_DEBUG("[OsdDevice] Drawing texture: %s", bitmap_path);
    LOG_DEBUG(" at absolute position %d,%d", pos_x, pos_y);
    LOG_DEBUG(" layer_id=%d\n", layer_id);
    LOG_DEBUG("[OsdDevice] Bitmap file path: %s\n", bitmap_path ? bitmap_path : "NULL");
    LOG_DEBUG("[OsdDevice] Bitmap position: %d,%d\n", bm_info.position.x, bm_info.position.y);
    LOG_DEBUG("[OsdDevice] Bitmap alpha: %d\n", (int)bm_info.alpha);

    int ret = osd_add_texture_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id, &bm_info);
    if (ret != 0) {
        std::cerr << "[OsdDevice] ERROR: osd_add_texture_layer failed! ret=" << ret
                  << ", layer_id=" << layer_id << std::endl;
        if (ret == -1) {
            std::cerr << "[OsdDevice] Layer does not exist or type mismatch (should be TYPE_IMAGE)" << std::endl;
        } else if (ret == -2) {
            std::cerr << "[OsdDevice] Bitmap add failed (encoding data too large or invalid file)" << std::endl;
        }
        return;
    }
    LOG_DEBUG("[OsdDevice] osd_add_texture_layer succeeded\n");

    ret = osd_flush_texture_layer(m_osd_handle, (ssLAYER_HANDLE)layer_id);
    if (ret != 0) {
        std::cerr << "[OsdDevice] ERROR: osd_flush_texture_layer failed! ret=" << ret
                  << ", layer_id=" << layer_id << std::endl;
        std::cerr << "[OsdDevice] Possible causes:" << std::endl;
        std::cerr << "[OsdDevice]   1. Layer codeTYPE is not SS_TYPE_RLE" << std::endl;
        std::cerr << "[OsdDevice]   2. Layer region type mismatch" << std::endl;
        std::cerr << "[OsdDevice]   3. Layer DMA buffer not set correctly" << std::endl;
        std::cerr << "[OsdDevice]   4. Layer region object encoding failed" << std::endl;
        std::cerr << "[OsdDevice]   5. Layer not enabled" << std::endl;
    } else {
        LOG_DEBUG("[OsdDevice] Texture drawn successfully\n");
    }
}

void OsdDevice::GenQrangleBox(std::array<float, 4>& det, int border) {
    // box[0..7]  对应内轮廓 4 个点
    // box[8..15] 对应外轮廓 4 个点
    // 二者一起用于描述空心框的边界区域。
    std::array<int, 16> box;

    box[0]  = std::min(m_width,  std::max(0, int(det[0] + border)));
    box[1]  = std::min(m_height, std::max(0, int(det[1] + border)));
    box[2]  = std::min(m_width,  std::max(0, int(det[0] + border)));
    box[3]  = std::min(m_height, std::max(0, int(det[3] - border)));
    box[4]  = std::min(m_width,  std::max(0, int(det[2] - border)));
    box[5]  = std::min(m_height, std::max(0, int(det[3] - border)));
    box[6]  = std::min(m_width,  std::max(0, int(det[2] - border)));
    box[7]  = std::min(m_height, std::max(0, int(det[1] + border)));

    box[8]  = std::min(m_width,  std::max(0, int(det[0] - border)));
    box[9]  = std::min(m_height, std::max(0, int(det[1] - border)));
    box[10] = std::min(m_width,  std::max(0, int(det[0] - border)));
    box[11] = std::min(m_height, std::max(0, int(det[3] + border)));
    box[12] = std::min(m_width,  std::max(0, int(det[2] + border)));
    box[13] = std::min(m_height, std::max(0, int(det[3] + border)));
    box[14] = std::min(m_width,  std::max(0, int(det[2] + border)));
    box[15] = std::min(m_height, std::max(0, int(det[1] - border)));

    m_qrangle_in.points[0]  = {box[0],  box[1]};
    m_qrangle_in.points[1]  = {box[2],  box[3]};
    m_qrangle_in.points[2]  = {box[4],  box[5]};
    m_qrangle_in.points[3]  = {box[6],  box[7]};
    m_qrangle_out.points[0] = {box[8],  box[9]};
    m_qrangle_out.points[1] = {box[10], box[11]};
    m_qrangle_out.points[2] = {box[12], box[13]};
    m_qrangle_out.points[3] = {box[14], box[15]};
}

} // namespace osd
} // namespace device
} // namespace sst
