#include "a1_uart_sink.hpp"

int a1_uart_open(A1UartSink* sink, uint32_t baudrate) {
#ifdef SSNE_AI_DEMO_ENABLE_A1_UART
    if (sink == nullptr) {
        return -1;
    }

    sink->handle = uart_init();
    if (sink->handle == nullptr) {
        return -2;
    }

    if (uart_set_baudrate(sink->handle, UART_TX0, baudrate) != UART_SUCCESS) {
        uart_close(sink->handle);
        sink->handle = nullptr;
        return -3;
    }
    return 0;
#else
    (void)sink;
    (void)baudrate;
    return -100;
#endif
}

void a1_uart_close(A1UartSink* sink) {
#ifdef SSNE_AI_DEMO_ENABLE_A1_UART
    if (sink != nullptr && sink->handle != nullptr) {
        uart_close(sink->handle);
        sink->handle = nullptr;
    }
#else
    (void)sink;
#endif
}

int a1_uart_write(void* user, const uint8_t* data, size_t bytes) {
#ifdef SSNE_AI_DEMO_ENABLE_A1_UART
    A1UartSink* sink = static_cast<A1UartSink*>(user);
    if (sink == nullptr || sink->handle == nullptr || data == nullptr || bytes > 32U) {
        return -1;
    }
    return uart_send_data(sink->handle,
                          UART_TX0,
                          data,
                          static_cast<uint32_t>(bytes)) == UART_SUCCESS ? 0 : -2;
#else
    (void)user;
    (void)data;
    (void)bytes;
    return -100;
#endif
}
