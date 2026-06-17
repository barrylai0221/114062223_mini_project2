#include <utility>
#include <memory>
#include "state.hpp"
#include "pvs.hpp"
#include "transposition_table.hpp"
#include <algorithm>
#include <array>

// 將 Move 編碼成 32-bit 整數
static uint32_t encode_move(const Move& move) {
    return (move.first.first << 24) | (move.first.second << 16) | 
           (move.second.first << 8) | move.second.second;
}

// 比較 Move 和編碼後的 uint32_t 是否相同
static bool is_same_move(const Move& move, uint32_t encoded_move) {
    return encode_move(move) == encoded_move;
}

static bool is_capture_move(const State* state, const Move& move) {
    return state->piece_at(
        1 - state->player,
        static_cast<int>(move.second.first),
        static_cast<int>(move.second.second)
    ) > 0;
}

// MVV-LVA scoring
static int score_move(
    const State* state,
    const Move& move,
    uint32_t tt_best_move_data,
    const std::array<Move, 2>* killer_slots,
    bool use_killers,
    const SearchContext* ctx,
    bool use_history
) {
    // 1. TT Best Move 優先權最高
    if (tt_best_move_data != 0 && is_same_move(move, tt_best_move_data)) {
        return 1000000;
    }

    if (use_killers && killer_slots != nullptr &&
        (move == (*killer_slots)[0] || move == (*killer_slots)[1])) {
        return 500000;
    }

    if(use_history && ctx != nullptr){
        const auto it = ctx->history_moves.find(make_move_key(move));
        if(it != ctx->history_moves.end()){
            return 1000 + it->second;
        }
    }

    if (is_capture_move(state, move)) {
        int aggressor_type = state->piece_at(
            state->player,
            static_cast<int>(move.first.first),
            static_cast<int>(move.first.second)
        );
        int victim_type = state->piece_at(
            1 - state->player,
            static_cast<int>(move.second.first),
            static_cast<int>(move.second.second)
        );
        static const int piece_values[7] = {0, 10, 50, 30, 30, 90, 900}; 
        return 100000 + piece_values[victim_type] * 10 - piece_values[aggressor_type];
    }
    
    return 0;
}

static const std::array<Move, 2>* killer_slots_for(SearchContext& ctx, int ply) {
    if (ply < 0 || static_cast<size_t>(ply) >= ctx.killer_moves.size()) {
        return nullptr;
    }
    return &ctx.killer_moves[ply];
}

/*============================================================
 * PVS — eval_ctx
 *============================================================*/
int PVS::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const PVSParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    // 【防護網 1】如果已經超時，立刻回傳，不浪費時間
    if(ctx.stop){
        return 0;
    }

    uint64_t zobrist = state->hash();
    int tt_score = 0;
    uint32_t tt_move = 0;
    int original_alpha = alpha;
    
    bool tt_hit = false;
    if(p.use_transposition_table && ctx.tt){
        tt_hit = ctx.tt->lookup(zobrist, depth, alpha, beta, tt_score, tt_move);
        if (depth > 0 && tt_hit) {
            ctx.tt_hits++;
            return tt_score;
        }
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(zobrist);

    const auto* killers = killer_slots_for(ctx, ply);

    if(state->legal_actions.size() > 1){
        std::sort(state->legal_actions.begin(), state->legal_actions.end(),
            [&](const Move& a, const Move& b){
                return score_move(state, a, tt_move, killers, p.enable_killer_moves, &ctx, p.enable_history_moves)
                     > score_move(state, b, tt_move, killers, p.enable_killer_moves, &ctx, p.enable_history_moves);
            });
    }

    if(depth <= 0){
        int score;
        if(p.use_quiescence_search){
            score = quiescence_search(state, 0, 4, history, ply, ctx, p, alpha, beta);
        } else {
            score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history); 
        }
        history.pop(zobrist);
        return score;
    }

    int best_score = -1000000;
    bool first_move = true;
    Move best_move;
    bool found_best_move = false;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int child_score;

        if (p.use_pvs && !first_move) {
            // Null-window search for subsequent moves
            child_score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -alpha - 1, -alpha);
        } else {
            // Full-window search for the first move or when PVS is disabled
            child_score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
        }

        int score = same ? child_score : -child_score;

        // If the null-window search failed high, we must re-search with the full window
        if (p.use_pvs && !first_move && score > alpha && score < beta) {
            child_score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
            score = same ? child_score : -child_score;
        }
        
        delete next;

        if(score > best_score){
            best_score = score;
            best_move = action;
            found_best_move = true;
        }

        if(best_score > alpha){
            alpha = best_score;
        }

        if(alpha >= beta){
            ctx.beta_cutoffs++;
            if (!is_capture_move(state, action)) {
                ctx.record_killer(static_cast<size_t>(ply), action);
            }
            if (!is_capture_move(state, action) && p.enable_history_moves) {
                auto& score = ctx.history_moves[make_move_key(action)];
                score += depth * depth;
                if(score > 100000){
                    score = 100000;
                }
            }
            break; 
        }
    }

    history.pop(zobrist);
    
    // 【防護網 3】如果超時了，代表這個 best_score 是不完整的垃圾分數，絕對不准存入 TT！
    if (ctx.stop) return 0;
    
    if(p.use_transposition_table && ctx.tt){
        BoundType bound_type = BOUND_EXACT;
        if (best_score <= original_alpha) {
            bound_type = BOUND_UPPER;
        } else if (best_score >= beta) {
            bound_type = BOUND_LOWER;
        }
        uint32_t best_move_encoded = 0; 
        if(found_best_move) {
            best_move_encoded = encode_move(best_move); 
        }

        ctx.tt->store(zobrist, best_score, depth, bound_type, best_move_encoded);
    }
    
    return best_score;
}

/*============================================================
 * PVS — quiescence_search
 *============================================================*/
int PVS::quiescence_search(
    State *state,
    int depth,
    int max_qd,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const PVSParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);

    if(stand_pat >= beta){
        return stand_pat;
    }
    if(stand_pat > alpha){
        alpha = stand_pat;
    }
    if(depth <= -max_qd){
        return stand_pat;
    }

    std::vector<Move> captures;
    state->get_capture_actions(captures);

    if(captures.empty()){
        return stand_pat;
    }

    if (p.use_mvv_lva) {
        std::sort(captures.begin(), captures.end(), 
            [&](const Move& a, const Move& b) {
                return score_move(state, a, 0, nullptr, false, nullptr, false) > score_move(state, b, 0, nullptr, false, nullptr, false);
            });
    }

    uint64_t zobrist = state->hash();
    history.push(zobrist);

    for(auto& action : captures){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int child_score = quiescence_search(next, depth - 1, max_qd, history, ply + 1, ctx, p, -beta, -alpha);

        int score = same ? child_score : -child_score;
        delete next;

        // 【防護網 4】QS 也一樣，超時就立刻中斷
        if (ctx.stop) {
            history.pop(zobrist);
            return 0;
        }

        if(score > stand_pat){
            stand_pat = score;
        }
        if(stand_pat > alpha){
            alpha = stand_pat;
        }
        if(alpha >= beta){
            ctx.beta_cutoffs++;
            break;
        }
    }

    history.pop(zobrist);
    return stand_pat;
}

/*============================================================
 * PVS — search (root level)
 *============================================================*/
SearchResult PVS::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    PVSParams p = PVSParams::from_map(ctx.params);
    
    if(p.use_transposition_table && !ctx.tt){
        ctx.tt = std::make_shared<TranspositionTable>(32);
    }
    if(ctx.tt){
        ctx.tt->new_search();
    }
    ctx.prepare_killer_moves(static_cast<size_t>(depth) + 2);
    
    SearchResult best_result_so_far;
    best_result_so_far.depth = 0;
    best_result_so_far.score = -1000000;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }
    if (state->legal_actions.empty()) return best_result_so_far;

    best_result_so_far.best_move = state->legal_actions[0];

    best_result_so_far.best_move = state->legal_actions[0];

    int previous_iteration_score = 0;
    bool have_previous_iteration_score = false;

    for (int current_depth = 1; current_depth <= depth; current_depth++) {
        int best_score = -1000000; 
        int alpha = -1000000;
        int beta = 1000000;
        if(have_previous_iteration_score){
            const int aspiration_window = 50;
            alpha = previous_iteration_score - aspiration_window;
            beta = previous_iteration_score + aspiration_window;
        }
        int total_moves = (int)state->legal_actions.size();
        
        Move current_depth_best_move = state->legal_actions[0];

        const auto* root_killers = killer_slots_for(ctx, 1);
        std::sort(state->legal_actions.begin(), state->legal_actions.end(), 
            [&](const Move& a, const Move& b) {
                if (best_result_so_far.depth > 0) {
                    bool a_is_best = is_same_move(a, encode_move(best_result_so_far.best_move));
                    bool b_is_best = is_same_move(b, encode_move(best_result_so_far.best_move));
                    if (a_is_best && !b_is_best) return true;
                    if (!a_is_best && b_is_best) return false;
                }
                return score_move(state, a, 0, root_killers, true, &ctx, p.enable_history_moves)
                     > score_move(state, b, 0, root_killers, true, &ctx, p.enable_history_moves);
            });

        auto search_root_once = [&](int search_alpha, int search_beta){
            int local_best_score = -1000000;
            Move local_best_move = state->legal_actions[0];
            bool local_first_move = true;
            int local_move_index = 0;

            for(auto& action : state->legal_actions){
                State* next = state->next_state(action);
                int child_score;

                if (p.use_pvs) {
                    if (local_first_move) {
                        child_score = eval_ctx(next, current_depth - 1, history, 1, ctx, p, -search_beta, -search_alpha);
                        local_first_move = false;
                    } else {
                        child_score = eval_ctx(next, current_depth - 1, history, 1, ctx, p, -search_alpha - 1, -search_alpha);

                        int tentative_score = next->same_player_as_parent() ? child_score : -child_score;
                        if (tentative_score > search_alpha && tentative_score < search_beta && !ctx.stop) {
                            child_score = eval_ctx(next, current_depth - 1, history, 1, ctx, p, -search_beta, -search_alpha);
                        }
                    }
                } else {
                    child_score = eval_ctx(next, current_depth - 1, history, 1, ctx, p, -search_beta, -search_alpha);
                }

                int score = next->same_player_as_parent() ? child_score : -child_score;
                delete next;

                if (ctx.stop) {
                    break;
                }

                if(score > local_best_score){
                    local_best_score = score;
                    local_best_move = action;

                    if(p.report_partial && ctx.on_root_update){
                       ctx.on_root_update({local_best_move, local_best_score, current_depth, local_move_index + 1, total_moves});
                    }
                }

                local_move_index++;
            }

            return std::pair<int, Move>{local_best_score, local_best_move};
        };

        auto [search_score, search_best_move] = search_root_once(alpha, beta);
        if(!ctx.stop && have_previous_iteration_score && search_score <= alpha){
            auto widened = search_root_once(-1000000, beta);
            search_score = widened.first;
            search_best_move = widened.second;
        } else if(!ctx.stop && have_previous_iteration_score && search_score >= beta){
            auto widened = search_root_once(alpha, 1000000);
            search_score = widened.first;
            search_best_move = widened.second;
        }

        if (ctx.stop) {
            break; // 被超時打斷，放棄這個深度的結果
        }

        best_score = search_score;
        current_depth_best_move = search_best_move;
        
        if (ctx.stop) {
            break; // 被超時打斷，放棄這個深度的結果
        }
        
        // 安全跑完，保留結果
        best_result_so_far.depth = current_depth;
        best_result_so_far.score = best_score;
        best_result_so_far.best_move = current_depth_best_move;
        previous_iteration_score = best_score;
        have_previous_iteration_score = true;
        
        if (best_score > 900000) break;
    }

    return best_result_so_far;
} 

ParamMap PVS::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
        {"UseAlphaBeta", "true"},
        {"UseTranspositionTable", "true"},
        {"UseQuiescenceSearch", "true"},
        {"UsePVS", "true"},
        {"UseMVVLVA", "true"}, 
        {"EnableKillerMoves", "true"},
        {"EnableHistoryHeuristic", "true"},
    };
}

std::vector<ParamDef> PVS::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
        {"UseAlphaBeta", ParamDef::CHECK, "true"},
        {"UseTranspositionTable", ParamDef::CHECK, "true"},
        {"UseQuiescenceSearch", ParamDef::CHECK, "true"},
        {"UsePVS", ParamDef::CHECK, "true"},
        {"UseMVVLVA", ParamDef::CHECK, "true"},
        {"EnableKillerMoves", ParamDef::CHECK, "true"},
        {"EnableHistoryHeuristic", ParamDef::CHECK, "true"},
    };
}