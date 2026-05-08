#include "event_recorder.hpp"

#include <cstdio>
#include <cstdlib>

namespace {

constexpr size_t kMaxEventRecords = 256;
// 使用绝对路径，保证 demo 进程和独立命令读写同一份状态。
constexpr const char* kEventRecordsPath = "/tmp/ssne_ai_demo_events.log";
constexpr const char* kEventTimePath = "/tmp/ssne_ai_demo_event_time.cfg";

}  // namespace

// 事件记录器只关心告警状态从 false 变 true 的瞬间，避免持续告警反复刷记录。
void EventRecorder::Update(bool fall_active,
                           bool intrusion_active,
                           bool fire_active,
                           uint64_t frame_index) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const std::array<bool, static_cast<size_t>(EventType::COUNT)> next_active{{
        fall_active,
        intrusion_active,
        fire_active
    }};

    // 只记录事件开始的上升沿，避免持续告警反复刷记录。
    for (size_t i = 0; i < next_active.size(); ++i) {
        if (next_active[i] && !m_active[i]) {
            RecordStartLocked(static_cast<EventType>(i), frame_index);
        }
        m_active[i] = next_active[i];
    }
}

bool EventRecorder::HandleCommand(const std::string& line, bool* should_exit) {
    if (should_exit != nullptr) {
        *should_exit = false;
    }

    const std::string cmd = Trim(line);
    if (cmd.empty()) {
        return true;
    }

    if (cmd == "q" || cmd == "Q" || cmd == "quit" || cmd == "exit") {
        if (should_exit != nullptr) {
            *should_exit = true;
        }
        std::printf("Quit signal received.\n");
        return true;
    }

    if (cmd == "help" || cmd == "h" || cmd == "?") {
        PrintHelp();
        return true;
    }

    if (cmd == "events" || cmd == "event" || cmd == "event list" ||
        cmd == "event view" || cmd == "events list" ||
        cmd == "e" || cmd == "ls") {
        std::lock_guard<std::mutex> lock(m_mutex);
        PrintEventsLocked();
        return true;
    }

    if (cmd == "event clear" || cmd == "events clear" ||
        cmd == "clear" || cmd == "c") {
        std::lock_guard<std::mutex> lock(m_mutex);
        ClearEventsLocked();
        return true;
    }

    if (cmd == "time" || cmd == "time show" || cmd == "t") {
        std::lock_guard<std::mutex> lock(m_mutex);
        PrintTimeLocked();
        return true;
    }

    std::string time_value;
    // 保留短命令，适配串口会截断长命令的场景。
    if (StartsWith(cmd, "time set ")) {
        time_value = Trim(cmd.substr(9));
    } else if (StartsWith(cmd, "t set ")) {
        time_value = Trim(cmd.substr(6));
    } else if (StartsWith(cmd, "ts ")) {
        time_value = Trim(cmd.substr(3));
    } else if (StartsWith(cmd, "cal ")) {
        time_value = Trim(cmd.substr(4));
    } else if (StartsWith(cmd, "time ")) {
        time_value = Trim(cmd.substr(5));
    } else if (StartsWith(cmd, "t ")) {
        time_value = Trim(cmd.substr(2));
    }

    if (!time_value.empty()) {
        std::time_t calibrated_time = 0;
        if (!ParseDateTime(time_value, &calibrated_time)) {
            std::printf("Invalid time. Use: time set YYYY-MM-DD HH:MM:SS\n");
            return true;
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        SetCalibratedTimeLocked(calibrated_time);
        PrintTimeLocked();
        return true;
    }

    return false;
}

std::time_t EventRecorder::NowLocked() const {
    LoadCalibrationLocked();
    const std::time_t now =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // 校准只记录事件时钟偏移，不修改系统时间，避免影响底层 SDK 和日志时间戳。
    return static_cast<std::time_t>(now + m_time_offset_seconds);
}

void EventRecorder::SetCalibratedTimeLocked(std::time_t calibrated_time) {
    const std::time_t now =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    m_time_calibrated = true;
    m_time_calibration_loaded = true;
    m_time_offset_seconds = static_cast<int64_t>(calibrated_time - now);
    SaveCalibrationLocked();
}

void EventRecorder::RecordStartLocked(EventType type, uint64_t frame_index) {
    LoadRecordsLocked();
    // 环形保留最近若干条记录，避免长期运行时事件文件无限增长。
    if (m_records.size() >= kMaxEventRecords) {
        m_records.erase(m_records.begin());
    }

    EventRecord record;
    record.type = type;
    record.start_time = NowLocked();
    record.frame_index = frame_index;
    record.calibrated = m_time_calibrated;
    m_records.push_back(record);
    SaveRecordsLocked();

    std::printf("[EVENT] type=%s start=%s frame=%llu clock=%s\n",
                TypeName(type),
                FormatTime(record.start_time).c_str(),
                static_cast<unsigned long long>(record.frame_index),
                record.calibrated ? "calibrated" : "system");
}

void EventRecorder::PrintHelp() const {
    std::printf("Commands:\n");
    std::printf("  q | quit                              quit demo\n");
    std::printf("  time                                  show event clock\n");
    std::printf("  time set YYYY-MM-DD HH[:MM[:SS]]      calibrate event clock only\n");
    std::printf("  t YYYYMMDDHHMMSS                      short calibrate command\n");
    std::printf("  events | event view                    show recorded event start times\n");
    std::printf("  event clear                            clear event records\n");
    std::printf("Shell command:\n");
    std::printf("  ssne_eventctl time\n");
    std::printf("  ssne_eventctl time set 2026-05-03 14:00:00\n");
    std::printf("  ev t 20260503140000\n");
    std::printf("  ssne_eventctl events\n");
}

void EventRecorder::PrintTimeLocked() const {
    LoadCalibrationLocked();
    std::printf("Event clock: %s (%s)\n",
                FormatTime(NowLocked()).c_str(),
                m_time_calibrated ? "calibrated" : "system");
}

void EventRecorder::PrintEventsLocked() const {
    LoadRecordsLocked();
    std::printf("Event records: %u\n", static_cast<unsigned>(m_records.size()));
    if (m_records.empty()) {
        return;
    }

    for (size_t i = 0; i < m_records.size(); ++i) {
        const EventRecord& record = m_records[i];
        std::printf("  #%u type=%s start=%s frame=%llu clock=%s\n",
                    static_cast<unsigned>(i + 1),
                    TypeName(record.type),
                    FormatTime(record.start_time).c_str(),
                    static_cast<unsigned long long>(record.frame_index),
                    record.calibrated ? "calibrated" : "system");
    }
}

void EventRecorder::ClearEventsLocked() {
    m_records.clear();
    m_records_loaded = true;
    SaveRecordsLocked();
    std::printf("Event records cleared.\n");
}

const char* EventRecorder::TypeName(EventType type) {
    switch (type) {
        case EventType::FALL: return "fall";
        case EventType::INTRUSION: return "intrusion";
        case EventType::FIRE: return "fire";
        default: return "unknown";
    }
}

bool EventRecorder::TypeFromName(const std::string& value, EventType* out_type) {
    if (out_type == nullptr) {
        return false;
    }
    if (value == "fall") {
        *out_type = EventType::FALL;
        return true;
    }
    if (value == "intrusion") {
        *out_type = EventType::INTRUSION;
        return true;
    }
    if (value == "fire") {
        *out_type = EventType::FIRE;
        return true;
    }
    return false;
}

std::string EventRecorder::Trim(const std::string& value) {
    const char* whitespace = " \t\r\n";
    const size_t begin = value.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = value.find_last_not_of(whitespace);
    return value.substr(begin, end - begin + 1);
}

bool EventRecorder::StartsWith(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

bool EventRecorder::ParseDateTime(const std::string& value, std::time_t* out_time) {
    if (out_time == nullptr) {
        return false;
    }

    const std::string trimmed = Trim(value);
    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
    char tail = '\0';
    bool parsed_ok = false;
    // 同时支持可读格式和紧凑格式，例如 "2026-05-03 14:23:45"
    // 以及 "20260503142345"。
    if (std::sscanf(trimmed.c_str(),
                    "%d-%d-%d %d:%d:%d%c",
                    &year,
                    &month,
                    &day,
                    &hour,
                    &minute,
                    &second,
                    &tail) == 6) {
        parsed_ok = true;
    } else if (std::sscanf(trimmed.c_str(),
                           "%d-%d-%d %d:%d%c",
                           &year,
                           &month,
                           &day,
                           &hour,
                           &minute,
                           &tail) == 5) {
        second = 0;
        parsed_ok = true;
    } else if (std::sscanf(trimmed.c_str(),
                           "%d-%d-%d %d%c",
                           &year,
                           &month,
                           &day,
                           &hour,
                           &tail) == 4) {
        minute = 0;
        second = 0;
        parsed_ok = true;
    } else if (std::sscanf(trimmed.c_str(),
                           "%d-%d-%d%c",
                           &year,
                           &month,
                           &day,
                           &tail) == 3) {
        hour = 0;
        minute = 0;
        second = 0;
        parsed_ok = true;
    }

    if (!parsed_ok &&
        (trimmed.size() == 8 || trimmed.size() == 10 ||
         trimmed.size() == 12 || trimmed.size() == 14) &&
        trimmed.find_first_not_of("0123456789") == std::string::npos) {
        year = std::atoi(trimmed.substr(0, 4).c_str());
        month = std::atoi(trimmed.substr(4, 2).c_str());
        day = std::atoi(trimmed.substr(6, 2).c_str());
        hour = trimmed.size() >= 10 ? std::atoi(trimmed.substr(8, 2).c_str()) : 0;
        minute = trimmed.size() >= 12 ? std::atoi(trimmed.substr(10, 2).c_str()) : 0;
        second = trimmed.size() >= 14 ? std::atoi(trimmed.substr(12, 2).c_str()) : 0;
        parsed_ok = true;
    }

    if (!parsed_ok) {
        return false;
    }

    if (year < 1970 || month < 1 || month > 12 || day < 1 || day > 31 ||
        hour < 0 || hour > 23 || minute < 0 || minute > 59 ||
        second < 0 || second > 60) {
        return false;
    }

    std::tm tm_value = {};
    tm_value.tm_year = year - 1900;
    tm_value.tm_mon = month - 1;
    tm_value.tm_mday = day;
    tm_value.tm_hour = hour;
    tm_value.tm_min = minute;
    tm_value.tm_sec = second;
    tm_value.tm_isdst = -1;

    const std::time_t result = std::mktime(&tm_value);
    if (result == static_cast<std::time_t>(-1)) {
        return false;
    }

    *out_time = result;
    return true;
}

std::string EventRecorder::FormatTime(std::time_t value) {
    std::tm tm_value = {};
#if defined(_WIN32)
    localtime_s(&tm_value, &value);
#else
    localtime_r(&value, &tm_value);
#endif

    char buffer[32] = {};
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_value) == 0) {
        return "invalid-time";
    }
    return std::string(buffer);
}

void EventRecorder::LoadCalibrationLocked() const {
    m_time_calibration_loaded = true;
    m_time_calibrated = false;
    m_time_offset_seconds = 0;

    // 每次查询都重新加载，确保长期运行的 demo 能看到外部命令的校准结果。
    FILE* file = std::fopen(kEventTimePath, "r");
    if (file == nullptr) {
        return;
    }

    long long offset = 0;
    int calibrated = 0;
    if (std::fscanf(file, "%lld %d", &offset, &calibrated) == 2 && calibrated != 0) {
        m_time_offset_seconds = static_cast<int64_t>(offset);
        m_time_calibrated = true;
    }
    std::fclose(file);
}

void EventRecorder::SaveCalibrationLocked() const {
    FILE* file = std::fopen(kEventTimePath, "w");
    if (file == nullptr) {
        std::printf("Failed to write event time file: %s\n", kEventTimePath);
        return;
    }

    std::fprintf(file,
                 "%lld %d\n",
                 static_cast<long long>(m_time_offset_seconds),
                 m_time_calibrated ? 1 : 0);
    std::fclose(file);
}

void EventRecorder::LoadRecordsLocked() const {
    m_records_loaded = true;
    m_records.clear();

    // 文件格式：type|epoch_seconds|frame_index|calibrated。
    FILE* file = std::fopen(kEventRecordsPath, "r");
    if (file == nullptr) {
        return;
    }

    char line[256] = {};
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        long long start_time = 0;
        unsigned long long frame_index = 0;
        char type_name[32] = {};
        int calibrated = 0;
        if (std::sscanf(line,
                        "%31[^|]|%lld|%llu|%d",
                        type_name,
                        &start_time,
                        &frame_index,
                        &calibrated) != 4) {
            continue;
        }

        EventType type = EventType::FALL;
        if (!TypeFromName(type_name, &type)) {
            continue;
        }

        EventRecord record;
        record.type = type;
        record.start_time = static_cast<std::time_t>(start_time);
        record.frame_index = static_cast<uint64_t>(frame_index);
        record.calibrated = calibrated != 0;
        m_records.push_back(record);
        if (m_records.size() > kMaxEventRecords) {
            m_records.erase(m_records.begin());
        }
    }

    std::fclose(file);
}

void EventRecorder::SaveRecordsLocked() const {
    FILE* file = std::fopen(kEventRecordsPath, "w");
    if (file == nullptr) {
        std::printf("Failed to write event record file: %s\n", kEventRecordsPath);
        return;
    }

    for (const auto& record : m_records) {
        std::fprintf(file,
                     "%s|%lld|%llu|%d\n",
                     TypeName(record.type),
                     static_cast<long long>(record.start_time),
                     static_cast<unsigned long long>(record.frame_index),
                     record.calibrated ? 1 : 0);
    }
    std::fclose(file);
}
