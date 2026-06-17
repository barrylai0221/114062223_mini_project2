#pragma once
#include <cstdint>
#include <vector>
#include <cstring>

/*============================================================
 * Transposition Table Entry
 * * Stores search result for a position with Zobrist hashing
 *============================================================*/

enum BoundType {
    BOUND_EXACT = 0,    // Exact score
    BOUND_LOWER = 1,    // Alpha cutoff (score >= lower bound)
    BOUND_UPPER = 2,    // Beta cutoff (score <= upper bound)
};

struct TTEntry {
    uint64_t zobrist;       // Position hash for verification
    int score;              // Evaluated score
    int depth;              // Search depth of this entry
    uint8_t bound_type;     // Type of bound (EXACT, LOWER, UPPER)
    uint32_t best_move_data;  // Encoded best move
    uint8_t generation;     // Generation counter for aging
    
    TTEntry() : zobrist(0), score(0), depth(-1), bound_type(0), 
                best_move_data(0), generation(0) {}
    
    bool is_valid() const {
        return depth >= 0;
    }
};

/*============================================================
 * Transposition Table
 * * Hash table using Zobrist hashing for storing search results
 * Uses linear probing with replacement strategy
 *============================================================*/

class TranspositionTable {
private:
    std::vector<TTEntry> table;
    uint64_t table_mask;
    uint64_t entries_count;
    uint8_t generation;
    
public:
    // Constructor: allocate table with given size (in MB)
    explicit TranspositionTable(size_t size_mb = 32) : generation(0) {
        // Round size to power of 2
        size_t bytes = size_mb * 1024 * 1024;
        size_t entry_count = bytes / sizeof(TTEntry);
        
        // Find next power of 2
        size_t power = 1;
        while(power < entry_count) power *= 2;
        entry_count = power;
        
        table.resize(entry_count);
        entries_count = entry_count;
        table_mask = entry_count - 1;
        
        clear();
    }
    
    // Clear all entries
    void clear() {
        for(auto& entry : table) {
            entry.zobrist = 0;
            entry.depth = -1;
            entry.generation = generation;
        }
    }
    
    // Increment generation (used for aging entries)
    void new_search() {
        generation++;
    }
    
    // Store entry in transposition table
    void store(uint64_t zobrist, int score, int depth, 
               BoundType bound, uint32_t best_move = 0) {
        uint64_t index = zobrist & table_mask;
        
        // 【防死當修正】：限制最多只探測 8 次，絕對不能無窮迴圈！
        uint64_t best_index = index;
        int min_depth = 10000;
        
        for (int probes = 0; probes < 8; ++probes) {
            // 如果找到空位，或者找到完全相同的盤面，就決定存在這裡
            if (!table[index].is_valid() || table[index].zobrist == zobrist) {
                best_index = index;
                break;
            }
            // 記錄這 8 個位置裡面，搜尋深度最淺 (價值最低) 的那個位置當作備胎
            if (table[index].depth < min_depth) {
                min_depth = table[index].depth;
                best_index = index;
            }
            index = (index + 1) & table_mask;
        }
        
        TTEntry& entry = table[best_index];
        
        // Replace if: empty, different hash, or shallower depth
        bool should_replace = !entry.is_valid() || 
                             entry.zobrist != zobrist || 
                             depth >= entry.depth;
        
        if(should_replace) {
            entry.zobrist = zobrist;
            entry.score = score;
            entry.depth = depth;
            entry.bound_type = bound;
            entry.best_move_data = best_move;
            entry.generation = generation;
        }
    }
    
    // Lookup entry in transposition table
    // Returns true if found, false otherwise
    bool lookup(uint64_t zobrist, int depth, int alpha, int beta,
                int& out_score, uint32_t& out_best_move) {
        uint64_t index = zobrist & table_mask;
        
        // Linear probing to find matching hash
        int probes = 0;
        const int MAX_PROBES = 8;  // Limit probes to avoid excessive searching
        
        while(probes < MAX_PROBES) {
            const TTEntry& entry = table[index];
            
            if(!entry.is_valid()) {
                break;  // Empty slot, entry not found
            }
            
            if(entry.zobrist == zobrist) {
                // 只要盤面相同，不管深度夠不夠，都把最佳步拿出來！
                out_best_move = entry.best_move_data;
                
                // 只有在快取深度 >= 當前要求深度時，才能直接使用分數剪枝
                if(entry.depth >= depth) {
                    // Return score based on bound type
                    if(entry.bound_type == BOUND_EXACT) {
                        out_score = entry.score;
                        return true;
                    } else if(entry.bound_type == BOUND_LOWER) {
                        // Score >= lower bound
                        if(entry.score >= beta) {
                            out_score = entry.score;
                            return true;  // Beta cutoff
                        }
                    } else if(entry.bound_type == BOUND_UPPER) {
                        // Score <= upper bound
                        if(entry.score <= alpha) {
                            out_score = entry.score;
                            return true;  // Alpha cutoff
                        }
                    }
                }
                // 【加速修正】：找到了盤面但深度不夠，就沒必要繼續往下找了，直接 return false
                return false; 
            }
            
            index = (index + 1) & table_mask;
            probes++;
        }
        
        return false;
    }
    
    // Get usage statistics
    size_t get_used_entries() const {
        size_t count = 0;
        for(const auto& entry : table) {
            if(entry.is_valid()) {
                count++;
            }
        }
        return count;
    }
    
    size_t get_total_entries() const {
        return entries_count;
    }
    
    double get_usage_percent() const {
        return 100.0 * get_used_entries() / get_total_entries();
    }
    
    // Resize table
    void resize(size_t size_mb) {
        TranspositionTable new_table(size_mb);
        *this = new_table;
    }
};