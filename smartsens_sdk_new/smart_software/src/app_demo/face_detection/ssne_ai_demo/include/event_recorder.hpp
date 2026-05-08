#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

enum class EventType {
    FALL = 0,
    INTRUSION = 1,
    FIRE = 2,
    COUNT = 3
};

struct EventRecord {
    EventType type = EventType::FALL;
    std::time_t start_time = 0;
    uint64_t frame_index = 0;
    bool calibrated = false;
};

class EventRecorder {
public:
    void Update(bool fall_active,
                bool intrusion_active,
                bool fire_active,
                uint64_t frame_index);

    bool HandleCommand(const std::string& line, bool* should_exit);

private:
    std::time_t NowLocked() const;
    void SetCalibratedTimeLocked(std::time_t calibrated_time);
    void RecordStartLocked(EventType type, uint64_t frame_index);
    void PrintHelp() const;
    void PrintTimeLocked() const;
    void PrintEventsLocked() const;
    void ClearEventsLocked();

    static const char* TypeName(EventType type);
    static bool TypeFromName(const std::string& value, EventType* out_type);
    static std::string Trim(const std::string& value);
    static bool StartsWith(const std::string& value, const std::string& prefix);
    static bool ParseDateTime(const std::string& value, std::time_t* out_time);
    static std::string FormatTime(std::time_t value);
    void LoadCalibrationLocked() const;
    void SaveCalibrationLocked() const;
    void LoadRecordsLocked() const;
    void SaveRecordsLocked() const;

    mutable std::mutex m_mutex;
    std::array<bool, static_cast<size_t>(EventType::COUNT)> m_active{{false, false, false}};
    mutable std::vector<EventRecord> m_records;
    mutable bool m_records_loaded = false;
    mutable bool m_time_calibration_loaded = false;
    mutable bool m_time_calibrated = false;
    mutable int64_t m_time_offset_seconds = 0;
};
