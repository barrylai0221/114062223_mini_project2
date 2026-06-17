#include <utility>
#include <memory>
#include <algorithm>
#include <array>
#include "state.hpp"
#include "minimax.hpp"
#include "transposition_table.hpp"

namespace {

uint32_t encode_move(const Move& move){
    return (static_cast<uint32_t>(move.first.first) << 24)
        | (static_cast<uint32_t>(move.first.second) << 16)
        | (static_cast<uint32_t>(move.second.first) << 8)
        | static_cast<uint32_t>(move.second.second);
}

bool is_same_move(const Move& move, uint32_t encoded_move){
    return encode_move(move) == encoded_move;
}

bool is_capture_move(const State* state, const Move& move){
    return state->piece_at(
        1 - state->player,
        static_cast<int>(move.second.first),
        static_cast<int>(move.second.second)
    ) > 0;
}

int move_order_score(
    const State* state,
    const Move& move,
    uint32_t tt_best_move,
    const std::array<Move, 2>* killer_slots,
    bool use_killers
){
    if(tt_best_move != 0 && is_same_move(move, tt_best_move)){
        return 1'000'000;
    }

    if(use_killers && killer_slots != nullptr &&
       (move == (*killer_slots)[0] || move == (*killer_slots)[1])){
        return 500'000;
    }

    if(is_capture_move(state, move)){
        static const int piece_values[7] = {0, 10, 50, 30, 30, 90, 900};
        const int aggressor_type = state->piece_at(
            state->player,
            static_cast<int>(move.first.first),
            static_cast<int>(move.first.second)
        );
        const int victim_type = state->piece_at(
            1 - state->player,
            static_cast<int>(move.second.first),
            static_cast<int>(move.second.second)
        );
        return 100'000 + piece_values[victim_type] * 10 - piece_values[aggressor_type];
    }

    return 0;
}

const std::array<Move, 2>* killer_slots_for(SearchContext& ctx, int ply){
    if(ply < 0 || static_cast<size_t>(ply) >= ctx.killer_moves.size()){
        return nullptr;
    }
    return &ctx.killer_moves[ply];
}

} // namespace


/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax with alpha-beta pruning and transposition table.
 * Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
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

    /* === Transposition table lookup === */
    uint64_t zobrist = state->hash();
    int tt_score = 0;
    uint32_t tt_move = 0;
    
    if(p.use_transposition_table && depth > 0 && ctx.tt){
        if(ctx.tt->lookup(zobrist, depth, alpha, beta, tt_score, tt_move)){
            ctx.tt_hits++;
            return tt_score;
        }
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // [ Hackathon TODO 3-1 ]
    // return the score for a winning terminal state
    // Hint: prefer faster wins by using ply.
    // ply is the number of moves from the root
    if(state->game_state == WIN){
        return P_MAX - ply;  // faster wins are better
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(zobrist);

    if(depth <= 0){
        // Call quiescence search to handle volatile positions
        if(p.use_quiescence_search){
            int score = quiescence_search(state, 0, 4, history, ply, ctx, p, alpha, beta);
            history.pop(zobrist);
            return score;
        } else {
            // Direct evaluation (legacy)
            int score = state->evaluate(
                p.use_kp_eval, p.use_eval_mobility, &history
            ); 
            history.pop(zobrist);
            return score;
        }
    }

    /* === Negamax loop with alpha-beta pruning === */
    int best_score = M_MAX;
    BoundType bound_type = BOUND_UPPER;  // Assume upper bound initially

    const auto* killers = killer_slots_for(ctx, ply);

    if(state->legal_actions.size() > 1){
        std::sort(state->legal_actions.begin(), state->legal_actions.end(),
            [&](const Move& a, const Move& b){
                return move_order_score(state, a, tt_move, killers, p.enable_killer_moves)
                     > move_order_score(state, b, tt_move, killers, p.enable_killer_moves);
            }
        );
    }

    for(auto& action : state->legal_actions){
        // [ Hackathon TODO 3-2 ]
        // create the child state after applying action
        State* next = state->next_state(action);

        bool same = next->same_player_as_parent();

        // [Hackathon TODO 3-3]
        // search the child one level deeper with alpha-beta window
        int child_score;
        if(p.use_alpha_beta){
            // Pass negated windows for opponent's perspective
            child_score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -beta, -alpha);
        } else {
            // Legacy mode without alpha-beta
            child_score = eval_ctx(next, depth - 1, history, ply + 1, ctx, p, -100000, 100000);
        }

        // [Hackathon TODO 3-4]
        // convert raw to the current player's perspective.
        int score;
        if(same){
            score = child_score;  // same player, don't negate
        } else {
            score = -child_score;  // opponent's gain is my loss, negate
        }

        delete next;

        // [ Hackathon TODO 3-5 ]
        // update best_score if this child is better.
        if(score > best_score){
            best_score = score;
            bound_type = BOUND_EXACT;
        }

        // Alpha-beta pruning: update alpha and check for cutoff
        if(p.use_alpha_beta){
            if(best_score > alpha){
                alpha = best_score;
            }
            if(alpha >= beta){
                ctx.beta_cutoffs++;  // Statistics
                bound_type = BOUND_LOWER;
                if(p.enable_killer_moves && !is_capture_move(state, action)){
                    ctx.record_killer(static_cast<size_t>(ply), action);
                }
                break;  // Beta cutoff
            }
        }
    }

    history.pop(zobrist);
    
    /* === Store in transposition table === */
    if(p.use_transposition_table && ctx.tt){
        ctx.tt->store(zobrist, best_score, depth, bound_type);
    }
    
    return best_score;
}


/*============================================================
 * MiniMax — quiescence_search
 *
 * Search only capture moves until position stabilizes.
 * Solves horizon effect by evaluating truly quiet positions.
 *============================================================*/
int MiniMax::quiescence_search(
    State *state,
    int depth,
    int max_qd,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
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

    /* === Terminal checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Standing pat: evaluate if no captures improve position === */
    int stand_pat = state->evaluate(
        p.use_kp_eval, p.use_eval_mobility, &history
    );

    // If position is good enough (alpha cutoff from evaluation alone)
    if(stand_pat >= beta){
        return stand_pat;
    }

    if(stand_pat > alpha){
        alpha = stand_pat;
    }

    /* === Quiescence depth limit === */
    if(depth <= -max_qd){
        // Stop quiescence search if too deep
        return stand_pat;
    }

    /* === Get capture moves only === */
    std::vector<Move> captures;
    state->get_capture_actions(captures);

    if(captures.empty()){
        // No captures available, position is quiet
        return stand_pat;
    }

    /* === Search captures with alpha-beta === */
    uint64_t zobrist = state->hash();
    history.push(zobrist);

    for(auto& action : captures){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int child_score = quiescence_search(
            next, depth - 1, max_qd, history, ply + 1, ctx, p, -beta, -alpha
        );

        int score;
        if(same){
            score = child_score;
        } else {
            score = -child_score;
        }

        delete next;

        if(score > stand_pat){
            stand_pat = score;
        }

        if(stand_pat > alpha){
            alpha = stand_pat;
        }

        if(alpha >= beta){
            ctx.beta_cutoffs++;
            break;  // Beta cutoff
        }
    }

    history.pop(zobrist);
    return stand_pat;
}


/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 * Initialize transposition table at search root.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    (void)history;
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    
    /* === Initialize transposition table === */
    if(p.use_transposition_table && !ctx.tt){
        ctx.tt = std::make_shared<TranspositionTable>(32);  // 32 MB
    }
    if(ctx.tt){
        ctx.tt->new_search();
    }
    ctx.prepare_killer_moves(static_cast<size_t>(depth) + 2);
    
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    const auto* root_killers = killer_slots_for(ctx, 1);
    if(state->legal_actions.size() > 1){
        std::sort(state->legal_actions.begin(), state->legal_actions.end(),
            [&](const Move& a, const Move& b){
                return move_order_score(state, a, 0, root_killers, p.enable_killer_moves)
                     > move_order_score(state, b, 0, root_killers, p.enable_killer_moves);
            }
        );
    }

    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    // Root node alpha-beta: start with full window
    int alpha = -100000;
    int beta = 100000;

    for(auto& action : state->legal_actions){
        /* [ Hackathon TODO 4-1 ]
         * search this move like TODO 3, but starting from the root */
        State* next = state->next_state(action);
        
        GameHistory root_history;
        int score;
        if(p.use_alpha_beta){
            score = eval_ctx(next, depth - 1, root_history, 1, ctx, p, -beta, -alpha);
        } else {
            score = eval_ctx(next, depth - 1, root_history, 1, ctx, p, -100000, 100000);
        }
        
        if(!next->same_player_as_parent()){
            score = -score;
        }
        
        delete next;

        if(score > best_score){
            // [ Hackathon TODO 4-2 ]
            // keep this move if it is the best so far
            best_score = score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
               ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }

            // Update alpha at root
            if(p.use_alpha_beta && best_score > alpha){
                alpha = best_score;
            }
        } else if(p.use_alpha_beta && score >= beta && p.enable_killer_moves){
            ctx.record_killer(1, action);
        }
        move_index++;
    }

    // [ Hackathon TODO 4-3 ]
    // update result and return
    result.score = best_score;

    return result;
} 


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
        {"UseAlphaBeta", "true"},
        {"EnableKillerMoves", "true"},
        {"UseTranspositionTable", "true"},
        {"UseQuiescenceSearch", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
        {"UseAlphaBeta", ParamDef::CHECK, "true"},
        {"EnableKillerMoves", ParamDef::CHECK, "true"},
        {"UseTranspositionTable", ParamDef::CHECK, "true"},
        {"UseQuiescenceSearch", ParamDef::CHECK, "true"},
    };
}
