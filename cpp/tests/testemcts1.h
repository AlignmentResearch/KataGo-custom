#ifndef TESTEMCTS1_H
#define TESTEMCTS1_H

#include <memory>

#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../neuralnet/nneval.h"

namespace EMCTS1Tests {
void runAllEMCTS1Tests();

// Checks that the models/const-policy-*-.bin.gz behave as expected
// (when using standard MCTS search).
void testConstPolicies();

// Helper functions
std::shared_ptr<NNEvaluator> get_nneval(std::string modelFile,
                                        ConfigParser& cfg, Logger& logger,
                                        uint64_t seed);

std::shared_ptr<NNResultBuf> evaluate(std::shared_ptr<NNEvaluator> nnEval,
                                      Board& board, BoardHistory& hist,
                                      Player nextPla, bool skipCache = true,
                                      bool includeOwnerMap = true);

// Constants
const std::string CONST_POLICY_1_PATH =
    "cpp/tests/models/const-policy-1.bin.gz";
const std::string CONST_POLICY_2_PATH =
    "cpp/tests/models/const-policy-2.bin.gz";

const float E = 2.718281828459045f;

const float CP1_WIN_PROB = E / (1 + E);
const float CP1_LOSS_PROB = 1 - CP1_WIN_PROB;

const float CP2_WIN_PROB = 1 / (1 + E * E);
const float CP2_LOSS_PROB = 1 - CP2_WIN_PROB;

}  // namespace EMCTS1Tests

#endif
