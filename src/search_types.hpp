#pragma once
#include "base_state.hpp"
#include "search_params.hpp"
#include <array>
#include <vector>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>

class State;
class TranspositionTable;

struct MoveKey {
    size_t from_row;
    size_t from_col;
    size_t to_row;
    size_t to_col;

    bool operator==(const MoveKey& other) const noexcept {
        return from_row == other.from_row
            && from_col == other.from_col
            && to_row == other.to_row
            && to_col == other.to_col;
    }
};

struct MoveKeyHash {
    size_t operator()(const MoveKey& key) const noexcept {
        size_t seed = std::hash<size_t>{}(key.from_row);
        seed ^= std::hash<size_t>{}(key.from_col) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        seed ^= std::hash<size_t>{}(key.to_row) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        seed ^= std::hash<size_t>{}(key.to_col) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }
};

inline MoveKey make_move_key(const Move& move){
    return MoveKey{
        move.first.first,
        move.first.second,
        move.second.first,
        move.second.second,
    };
}

struct RootUpdate {
    Move best_move;
    int score;
    int depth;
    int move_number;
    int total_moves;
};

struct SearchContext {
    uint64_t nodes = 0;
    uint64_t beta_cutoffs = 0;  // Alpha-beta pruning statistics
    uint64_t tt_hits = 0;       // Transposition table hits
    int seldepth = 0;
    bool stop = false;
    ParamMap params;
    std::shared_ptr<TranspositionTable> tt;  // Transposition table
    std::vector<std::array<Move, 2>> killer_moves;  // [ply][0..1]
    std::unordered_map<MoveKey, int, MoveKeyHash> history_moves;
    std::function<void(const RootUpdate&)> on_root_update;

    void reset(){
        nodes = 0;
        beta_cutoffs = 0;
        tt_hits = 0;
        seldepth = 0;
        killer_moves.clear();
        history_moves.clear();
    }

    void prepare_killer_moves(size_t max_depth){
        killer_moves.assign(max_depth + 1, {Move{}, Move{}});
    }

    void record_killer(size_t ply, const Move& move){
        if(ply >= killer_moves.size()){
            return;
        }
        auto& slots = killer_moves[ply];
        if(slots[0] == move){
            return;
        }
        slots[1] = slots[0];
        slots[0] = move;
    }
};

struct SearchResult {
    Move best_move;
    int score = 0;
    int depth = 0;
    int seldepth = 0;
    uint64_t nodes = 0;
    double time_ms = 0;
    std::vector<Move> pv;
};
