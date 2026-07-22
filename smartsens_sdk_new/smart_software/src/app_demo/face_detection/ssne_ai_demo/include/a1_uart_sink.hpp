#pragma once

#include "face_business.h"

#ifdef SSNE_AI_DEMO_ENABLE_A1_UART
#include <smartsoc/uart_api.h>
#endif

struct A1UartSink {
#ifdef SSNE_AI_DEMO_ENABLE_A1_UART
    uart_handle_t handle = nullptr;
#else
    void* handle = nullptr;
#endif
};

int a1_uart_open(A1UartSink* sink, uint32_t baudrate = 115200);
void a1_uart_close(A1UartSink* sink);
int a1_uart_write(void* user, const uint8_t* data, size_t bytes);
