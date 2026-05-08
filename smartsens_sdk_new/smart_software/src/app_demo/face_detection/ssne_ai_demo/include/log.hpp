/*
 * @Filename: log.hpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 */
#pragma once

#include <cstdio>

#ifndef ENABLE_DEBUG_LOG
#define ENABLE_DEBUG_LOG 0
#endif

#if ENABLE_DEBUG_LOG
#define LOG_DEBUG(fmt_str, ...) \
    std::printf("[DEBUG] " fmt_str, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt_str, ...) ((void)0)
#endif

#define LOG_INFO(fmt_str, ...) \
    std::printf("[INFO] " fmt_str, ##__VA_ARGS__)

#define LOG_WARN(fmt_str, ...) \
    std::printf("[WARN] " fmt_str, ##__VA_ARGS__)

#define LOG_ERROR(fmt_str, ...) \
    std::fprintf(stderr, "[ERROR] " fmt_str, ##__VA_ARGS__)
