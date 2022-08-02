#ifndef TESTEMCTS1_H
#define TESTEMCTS1_H

#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../neuralnet/nneval.h"
#include "../search/search.h"

namespace EMCTS1Tests {
void runAllEMCTS1Tests();

// Checks that the models/const-policy-*-.bin.gz behave as expected
// (when using standard MCTS search).
void testConstPolicies();

// Test our modifications didn't break the original EMCTS.
void testMCTS();

// Checks one move's worth of MCTS search
void checkMCTSSearch(const Search& bot, const float win_prob,
                     const float loss_prob);

// Test EMCTS1
void testEMCTS1();

// Checks one move's worth of EMCTS1 search
void checkEMCTS1Search(const Search& bot, const float win_prob1,
                       const float loss_prob1, const float win_prob2,
                       const float loss_prob2);

// Checks how we select our move based on results of tree search.
void checkFinalMoveSelection(const Search& bot);

// Check playout logic (for either MCTS or EMCTS1)
// Our naively implemented check simulates the entire playout process and takes
// O(BP^3) time, where B is the size of the board and P is the number of
// playouts.
void checkPlayoutLogic(const Search& bot);

// Returns one sample of possible rules.
Rules parseRules(ConfigParser& cfg, Logger& logger);

// Helper functions

std::shared_ptr<NNEvaluator> getNNEval(std::string modelFile, ConfigParser& cfg,
                                       Logger& logger, uint64_t seed);

std::shared_ptr<NNResultBuf> evaluate(std::shared_ptr<NNEvaluator> nnEval,
                                      Board& board, BoardHistory& hist,
                                      Player nextPla, bool skipCache = true,
                                      bool includeOwnerMap = true);

void resetBot(Search& bot, int board_size, const Rules& rules);

// Helper struct for dealing with a search tree.
struct SearchTree {
  const SearchNode* const root;

  // DFS visit order.
  // So a child will always come after its parent in this list.
  std::vector<const SearchNode*> all_nodes;

  std::unordered_map<const SearchNode*, std::vector<const SearchNode*>>
      children;

  SearchTree(const Search& bot);

  std::vector<const SearchNode*> getSubtreeNodes(const SearchNode* node) const;

  std::vector<const SearchNode*> getPathToRoot(const SearchNode* node) const;
};

// Optional param: terminal_node_visits (specify if you want to override
// terminal node weights)
NodeStats averageStats(const Search& bot,
                       const std::vector<const SearchNode*>& nodes,
                       const std::unordered_map<const SearchNode*, int>*
                           terminal_node_visits = nullptr);

// Constants

const std::string EMCTS1_CONFIG_PATH = "cpp/tests/data/configs/test-emcts1.cfg";

// exp(1)
const float E = 2.718281828459045f;

// CONST_POLICY_1 will attempt to play in the lexicographically smallest square
// that is legal. If no such squares are possible, it will pass.
// Squares are ordered in row-major order, with the smallest square at the
// top left of the board.
// CONST_POLICY_1 has hardcoded win and lose probabilities for any input board.
const std::string CONST_POLICY_1_PATH =
    "cpp/tests/models/const-policy-1.bin.gz";
const float CP1_WIN_PROB = E / (1 + E);
const float CP1_LOSS_PROB = 1 - CP1_WIN_PROB;

// CONST_POLICY_2 will always try to pass.
// CONST_POLICY_2 has hardcoded win and lose probabilities for any input board.
const std::string CONST_POLICY_2_PATH =
    "cpp/tests/models/const-policy-2.bin.gz";
const float CP2_WIN_PROB = 1 / (1 + E * E);
const float CP2_LOSS_PROB = 1 - CP2_WIN_PROB;

}  // namespace EMCTS1Tests

#endif
