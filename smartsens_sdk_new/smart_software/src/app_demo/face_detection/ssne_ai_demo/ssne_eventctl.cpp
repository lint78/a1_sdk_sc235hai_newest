#include "event_recorder.hpp"

#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    EventRecorder recorder;
    std::ostringstream command;
    if (argc <= 1) {
        command << "help";
    } else {
        // 将 argv 重新拼成一条命令，避免日期和时间参数被拆散。
        for (int i = 1; i < argc; ++i) {
            if (i > 1) {
                command << ' ';
            }
            command << argv[i];
        }
    }

    bool should_exit = false;
    if (!recorder.HandleCommand(command.str(), &should_exit)) {
        std::cerr << "Unknown command. Run: ssne_eventctl help" << std::endl;
        return 1;
    }

    return 0;
}
