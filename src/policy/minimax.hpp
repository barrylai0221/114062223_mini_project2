#pragma once
#include "search_types.hpp"
#include "game_history.hpp"
#include "transposition_table.hpp"

struct MMParams {
    bool use_kp_eval = true;
    bool use_eval_mobility = true;
    bool report_partial = true;
    bool use_alpha_beta = true;     // Enable alpha-beta pruning
    bool enable_killer_moves = true; // Enable killer move heuristic
    bool use_transposition_table = true; // Enable transposition table
    bool use_quiescence_search = true;   // Enable quiescence search

    static MMParams from_map(const ParamMap& m){
        MMParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval", true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.report_partial    = param_bool(m, "ReportPartial", true);
        p.use_alpha_beta    = param_bool(m, "UseAlphaBeta", true);
        p.enable_killer_moves = param_bool(m, "EnableKillerMoves", true);
        p.use_transposition_table = param_bool(m, "UseTranspositionTable", true);
        p.use_quiescence_search = param_bool(m, "UseQuiescenceSearch", true);
        return p;
    }
};

class MiniMax{
public:
    static int eval_ctx(
        State *state,
        int depth,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const MMParams& p,
        int alpha = -100000,
        int beta = 100000
    );
    
    // Quiescence search: search only captures until position is quiet
    static int quiescence_search(
        State *state,
        int depth,
        int max_qd,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const MMParams& p,
        int alpha = -100000,
        int beta = 100000
    );
    
    static SearchResult search(
        State *state,
        int depth,
        GameHistory& history,
        SearchContext& ctx
    );

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};
