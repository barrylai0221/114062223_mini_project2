#pragma once
#include "search_types.hpp"
#include "game_history.hpp"
#include "transposition_table.hpp"

// Use the same parameter structure, just add a PVS flag
struct PVSParams {
    bool use_kp_eval = true;
    bool use_eval_mobility = true;
    bool report_partial = true;
    bool use_alpha_beta = true;
    bool use_transposition_table = true;
    bool use_quiescence_search = true;
    bool use_pvs = true; // Enable Principal Variation Search
    bool use_mvv_lva = true; // Enable MVV-LVA move ordering

    static PVSParams from_map(const ParamMap& m){
        PVSParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval", true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.report_partial    = param_bool(m, "ReportPartial", true);
        p.use_alpha_beta    = param_bool(m, "UseAlphaBeta", true);
        p.use_transposition_table = param_bool(m, "UseTranspositionTable", true);
        p.use_quiescence_search = param_bool(m, "UseQuiescenceSearch", true);
        p.use_pvs           = param_bool(m, "UsePVS", true);
        p.use_mvv_lva       = param_bool(m, "UseMVVLVA", true);
        return p;
    }
};

class PVS {
public:
    static int eval_ctx(
        State *state,
        int depth,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const PVSParams& p,
        int alpha = -100000,
        int beta = 100000
    );
    
    static int quiescence_search(
        State *state,
        int depth,
        int max_qd,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const PVSParams& p,
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
