#pragma once

#include "common.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>

enum class GestureCommand {
    NONE = 0,
    TU = 1,
    TD = 2,
    TL = 3,
    TR = 4
};

struct GestureResult {
    GestureCommand command = GestureCommand::NONE;
    bool valid = false;
    float confidence = 0.0f;
    std::array<float, 5> probabilities = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 5> logits = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
};

class GestureClassifier {
  public:
    std::string ModelName() const { return "gesture_mobilenetv1"; }

    void Initialize(std::string& model_path,
                    std::array<int, 2>* in_img_shape,
                    std::array<int, 2>* in_det_shape,
                    bool use_normalize = true,
                    uint8_t input_format = SSNE_RGB);
    void Predict(ssne_tensor_t* img, GestureResult* result, float conf_threshold = 0.55f);
    void SetFocusBox(const std::array<float, 4>* focus_box);
    void Release();

    std::array<int, 2> img_shape = {0, 0};
    std::array<int, 2> det_shape = {0, 0};
    float w_scale = 1.0f;
    float h_scale = 1.0f;
    bool normalize_enabled = true;
    uint8_t input_format = SSNE_RGB;

  private:
    uint16_t model_id = 0;
    ssne_tensor_t inputs[1] = {};
    ssne_tensor_t outputs[1] = {};
    AiPreprocessPipe pipe_offline = GetAIPreprocessPipe();
    bool focus_valid = false;
    std::array<float, 4> focus_box = {0.0f, 0.0f, 0.0f, 0.0f};
};

class GestureTemporalFilter {
  public:
    GestureTemporalFilter(int history_size = 4,
                          int required_hits = 2,
                          int emit_cooldown_frames = 3);

    GestureCommand Push(const GestureResult& result);
    void Reset();

  private:
    int history_size = 4;
    int required_hits = 2;
    int emit_cooldown_frames = 3;
    int cooldown_left = 0;
    std::deque<GestureCommand> history;
    GestureCommand last_emitted = GestureCommand::NONE;
};

enum class SnakeDirection {
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3
};

struct SnakeCell {
    SnakeCell() = default;
    SnakeCell(int x_in, int y_in) : x(x_in), y(y_in) {}

    int x = 0;
    int y = 0;
};

struct SnakeRenderData {
    int board_cols = 0;
    int board_rows = 0;
    int score = 0;
    int best_score = 0;
    bool paused = false;
    bool game_over = false;
    GestureCommand last_command = GestureCommand::NONE;
    float last_command_confidence = 0.0f;
    std::vector<SnakeCell> snake;
    SnakeCell food;
    bool has_food = false;
};

class SnakeGame {
  public:
    SnakeGame();

    void Initialize(int cols, int rows, uint32_t seed = 0U);
    void Reset();
    void SetPaused(bool paused);
    bool IsPaused() const { return paused; }
    bool IsGameOver() const { return game_over; }
    int Score() const { return score; }
    int TickIntervalMs() const;
    SnakeDirection Direction() const { return direction; }
    void SetDirection(SnakeDirection next_direction);
    bool Tick();
    SnakeRenderData BuildRenderData() const;

  private:
    bool IsOpposite(SnakeDirection a, SnakeDirection b) const;
    bool IsOccupied(int x, int y) const;
    void SpawnFood();

  private:
    int cols = 0;
    int rows = 0;
    bool initialized = false;
    bool paused = false;
    bool game_over = false;
    int score = 0;
    int best_score = 0;
    int grow_pending = 0;
    SnakeDirection direction = SnakeDirection::RIGHT;
    SnakeDirection pending_direction = SnakeDirection::RIGHT;
    std::deque<SnakeCell> body;
    SnakeCell food;
    bool has_food = false;
    std::mt19937 rng;
};

const char* GestureCommandName(GestureCommand command);
const char* SnakeDirectionName(SnakeDirection direction);
SnakeDirection GestureToSnakeDirection(GestureCommand command, SnakeDirection fallback);
GestureCommand SnakeDirectionToGesture(SnakeDirection direction);
