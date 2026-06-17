#include <utility>
#include <memory>
#include "state.hpp"
#include "pvs.hpp"
#include "transposition_table.hpp"
#include <algorithm>

// 將 Move 編碼成 32-bit 整數 (假設座標在 0-15 之間)
static uint32_t encode_move(const Move& move) {
    return (move.first.first << 24) | (move.first.second << 16) | 
           (move.second.first << 8) | move.second.second;
}

// 比較 Move 和編碼後的 uint32_t 是否相同
static bool is_same_move(const Move& move, uint32_t encoded_move) {
    return encode_move(move) == encoded_move;
}

// MVV-LVA scoring
// Scores captures based on the value of the victim and aggressor.
// Higher score is better.
// Score = (Victim Value * 10) - Aggressor Value
static int score_move(const State* state, const Move& move, uint32_t tt_best_move_data) {
    // 1. TT Best Move 優先權最高 (給予極大分數 1000000)
    if (tt_best_move_data != 0 && is_same_move(move, tt_best_move_data)) {
        return 1000000;
    }

    int aggressor_type = state->piece_at(state->player, move.first.first, move.first.second);
    int victim_type = state->piece_at(1 - state->player, move.second.first, move.second.second);

    if (victim_type > 0) {
        // This is a capture move
        static const int piece_values[7] = {0, 10, 50, 30, 30, 90, 900}; // P, R, N, B, Q, K
        return piece_values[victim_type] * 10 - piece_values[aggressor_type];
    }
    
    // For non-captures, return 0
    return 0;
}

/*============================================================
 * PVS — eval_ctx
 *
 * Negamax with Principal Variation Search.
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
    if(ctx.stop){
        return 0;
    }

    uint64_t zobrist = state->hash();
    int tt_score = 0;
    uint32_t tt_move = 0;
    
    // 記住原始的 alpha，用於稍後存入 TT 時判斷 UPPER BOUND
    int original_alpha = alpha;
    
    if(p.use_transposition_table && depth > 0 && ctx.tt){
        if(ctx.tt->lookup(zobrist, depth, alpha, beta, tt_score, tt_move)){
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

    if(depth <= 0){
        if(p.use_quiescence_search){
            int score = quiescence_search(state, 0, 4, history, ply, ctx, p, alpha, beta);
            history.pop(zobrist);
            return score;
        } else {
            int score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history); 
            history.pop(zobrist);
            return score;
        }
    }

    int best_score = -1000000; // 統一使用明確的負數，避免 M_MAX 常數被誤定義為正數
    bool first_move = true;
    Move best_move;
    bool found_best_move = false;

    // Sort moves using MVV-LVA (並且如果有 TT_move，應該讓它排第一)
    if (p.use_mvv_lva) {
        std::sort(state->legal_actions.begin(), state->legal_actions.end(), 
            [&](const Move& a, const Move& b) {
                // 傳入 tt_move 讓好步優先
                return score_move(state, a, tt_move) > score_move(state, b, tt_move);
            });
    }

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int child_score;

        if (p.use_pvs) {
            if (first_move) {
                // 第一步：使用完整的 Alpha-Beta 視窗搜尋
                child_score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
                first_move = false;
            } else {
                // 後續步驟：使用零視窗 (Zero Window) 測試
                child_score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -alpha - 1, -alpha);
                
                // 必須先轉換為當前視角的分數
                int tentative_score = same ? child_score : -child_score;
                
                // 如果零視窗測試失敗（代表這步棋比第一步好），且沒有好到引發對手剪枝
                // 則必須重新用完整的視窗 (-beta, -tentative_score) 搜尋
                if (tentative_score > alpha && tentative_score < beta) {
                    child_score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
                }
            }
        } else {
             // 如果不使用 PVS，就跑標準的 Alpha-Beta
            child_score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
        }

        // 轉換為當前玩家的分數
        int score = same ? child_score : -child_score;
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
            break; // 觸發 Beta 剪枝
        }
    }

    history.pop(zobrist);
    
    // 存入 Transposition Table
    if(p.use_transposition_table && ctx.tt){
        BoundType bound_type = BOUND_EXACT;
        
        // 判斷 UPPER BOUND 必須用剛進來這個函數時的 original_alpha
        // 如果連原本的 alpha 都沒超過，代表這是個爛盤面 (Fail-low)
        if (best_score <= original_alpha) {
            bound_type = BOUND_UPPER;
        } 
        // 如果分數 >= beta，代表引發了剪枝，這個分數只是個下限 (Fail-high)
        else if (best_score >= beta) {
            bound_type = BOUND_LOWER;
        }

        // 將 best_move 編碼後存入 TT
        uint32_t best_move_encoded = 0; 
        if(found_best_move) {
            best_move_encoded = encode_move(best_move); 
        }

        ctx.tt->store(zobrist, best_score, depth, bound_type, best_move_encoded);
    }
    
    return best_score;
}

/*============================================================
 * PVS — quiescence_search (identical to MiniMax)
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
        // No captures available, position is quiet
        return stand_pat;
    }

    // Sort captures using MVV-LVA
    if (p.use_mvv_lva) {
        std::sort(captures.begin(), captures.end(), 
            [&](const Move& a, const Move& b) {
                return score_move(state, a, 0) > score_move(state, b, 0);
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
    
    SearchResult best_result_so_far;
    best_result_so_far.depth = 0;
    best_result_so_far.score = -1000000;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }
    if (state->legal_actions.empty()) return best_result_so_far;

    // 預設一個備用步，防止意外
    best_result_so_far.best_move = state->legal_actions[0];

    // ========================================================
    // 實作 Iterative Deepening (迭代加深)
    // ========================================================
    for (int current_depth = 1; current_depth <= depth; current_depth++) {
        int best_score = -1000000; 
        int alpha = -1000000;
        int beta = 1000000;
        bool first_move = true;
        int move_index = 0;
        int total_moves = (int)state->legal_actions.size();
        
        Move current_depth_best_move = state->legal_actions[0];

        // 排序：將【上一次深度搜尋】拿到的最佳解排在第一位，讓 PVS 發揮威力
        if (p.use_mvv_lva) {
            std::sort(state->legal_actions.begin(), state->legal_actions.end(), 
                [&](const Move& a, const Move& b) {
                    if (best_result_so_far.depth > 0) {
                        bool a_is_best = is_same_move(a, encode_move(best_result_so_far.best_move));
                        bool b_is_best = is_same_move(b, encode_move(best_result_so_far.best_move));
                        if (a_is_best && !b_is_best) return true;
                        if (!a_is_best && b_is_best) return false;
                    }
                    // 否則照常使用 MVV-LVA
                    return score_move(state, a, 0) > score_move(state, b, 0);
                });
        }

        for(auto& action : state->legal_actions){
            State* next = state->next_state(action);
            
            // 【失憶症修正】：直接傳遞真實的 history，絕對不能在這裡 new 一個空的！
            int child_score;

            if (p.use_pvs) {
                if (first_move) {
                    child_score = eval_ctx(next, current_depth - 1, history, 1, ctx, p, -beta, -alpha);
                    first_move = false;
                } else {
                    child_score = eval_ctx(next, current_depth - 1, history, 1, ctx, p, -alpha - 1, -alpha);
                    
                    int tentative_score = next->same_player_as_parent() ? child_score : -child_score;
                    if (tentative_score > alpha && tentative_score < beta) {
                        child_score = eval_ctx(next, current_depth - 1, history, 1, ctx, p, -beta, -alpha);
                    }
                }
            } else {
                child_score = eval_ctx(next, current_depth - 1, history, 1, ctx, p, -beta, -alpha);
            }
            
            int score = next->same_player_as_parent() ? child_score : -child_score;
            delete next;

            // 救命防護網：如果超時被觸發了，中止這個深度的後續計算！
            if (ctx.stop) break;

            if(score > best_score){
                best_score = score;
                current_depth_best_move = action;

                if(p.report_partial && ctx.on_root_update){
                   ctx.on_root_update({current_depth_best_move, best_score, current_depth, move_index + 1, total_moves});
                }

                if(best_score > alpha){
                    alpha = best_score;
                }
            }
            move_index++;
        }
        
        // 救命防護網：如果迴圈是因為超時斷掉的，保留【上一層安全算完】的 best_result_so_far！
        if (ctx.stop) {
            break;
        }
        
        // 成功跑完這個深度！安全地更新到最終結果中
        best_result_so_far.depth = current_depth;
        best_result_so_far.score = best_score;
        best_result_so_far.best_move = current_depth_best_move;
        
        // 如果已經算到絕殺步 (Mate)，提早結束
        if (best_score > 900000) break;
    }

    return best_result_so_far;
} 

/*============================================================
 * PVS — default_params / param_defs
 *============================================================*/
ParamMap PVS::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
        {"UseAlphaBeta", "true"},
        {"UseTranspositionTable", "true"},
        {"UseQuiescenceSearch", "true"},
        {"UsePVS", "true"},
        {"UseMVVLVA", "true"}, // 記得把 MVVLVA 加進參數
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
    };
}