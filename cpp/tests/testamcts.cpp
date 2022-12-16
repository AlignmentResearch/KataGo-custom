// #ifdef AMCTS_TESTS
#include "../tests/testamcts.h"

#include "../dataio/sgf.h"
#include "../program/play.h"
#include "../program/setup.h"
#include "../tests/tests.h"

using namespace std;

// Uncomment to enable debugging
#define DEBUG

//Tiny constant to add to numerator of puct formula to make it positive
//even when visits = 0.
static constexpr double POLICY_ILLEGAL_SELECTION_VALUE = -1e50;
static constexpr double TOTALCHILDWEIGHT_PUCT_OFFSET = 0.01;

static bool approxEqual(float x, float y) {
  float tolerance;
  tolerance = 0.0001f * std::max(std::abs(x), std::max(std::abs(y), 1.0f));
  return std::abs(x - y) < tolerance;
}

static bool approxEqual(double x, double y) {
  double tolerance;
  tolerance = 1e-5 * std::max(std::abs(x), std::max(std::abs(y), 1.0));
  return std::abs(x - y) < tolerance;
}

static bool approxEqual(const NodeStats& s1, const NodeStats& s2) {
  // When total weight is zero, we just check weights match.
  if (s1.weightSum == 0)
    return s2.weightSum == 0 && s1.weightSqSum == 0 && s2.weightSqSum == 0;

  return approxEqual(s1.winLossValueAvg, s2.winLossValueAvg) &&
         approxEqual(s1.noResultValueAvg, s2.noResultValueAvg) &&
         approxEqual(s1.scoreMeanAvg, s2.scoreMeanAvg) &&
         approxEqual(s1.scoreMeanSqAvg, s2.scoreMeanSqAvg) &&
         approxEqual(s1.leadAvg, s2.leadAvg) &&
         approxEqual(s1.utilityAvg, s2.utilityAvg) &&
         approxEqual(s1.utilitySqAvg, s2.utilitySqAvg) &&
         approxEqual(s1.weightSum, s2.weightSum) &&
         approxEqual(s1.weightSqSum, s2.weightSqSum);
}

static double cpuctExploration(double totalChildWeight, const SearchParams& searchParams) {
  return searchParams.cpuctExploration +
    searchParams.cpuctExplorationLog * log((totalChildWeight + searchParams.cpuctExplorationBase) / searchParams.cpuctExplorationBase);
}

double getExploreSelectionValue(
  double nnPolicyProb, double totalChildWeight, double childWeight,
  double childUtility, double parentUtilityStdevFactor, Player pla, const SearchParams& searchParams
) {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  double exploreComponent =
    cpuctExploration(totalChildWeight, searchParams)
    * parentUtilityStdevFactor
    * nnPolicyProb
    * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET)
    / (1.0 + childWeight);

  //At the last moment, adjust value to be from the player's perspective, so that players prefer values in their favor
  //rather than in white's favor
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;
  return exploreComponent + valueComponent;
}

double getExploreSelectionValueOfChild(
  const Search &bot, const SearchNode& parent, const float* parentPolicyProbs, const NodeStats& childStats,
  Loc moveLoc,
  double totalChildWeight, int64_t childEdgeVisits, double fpuValue,
  double parentWeightPerVisit, double parentUtilityStdevFactor,
  double maxChildWeight
) {
  int movePos =  NNPos::locToPos(moveLoc,bot.rootBoard.x_size,bot.nnXLen,bot.nnYLen);
  float nnPolicyProb = parentPolicyProbs[movePos];

  // int32_t childVirtualLosses = child->virtualLosses;
  int64_t childVisits = childStats.visits;
  double utilityAvg = childStats.utilityAvg;
  double scoreMeanAvg = childStats.scoreMeanAvg;
  double scoreMeanSqAvg = childStats.scoreMeanSqAvg;
  double childWeight = childStats.getChildWeight(childEdgeVisits);

  //It's possible that childVisits is actually 0 here with multithreading because we're visiting this node while a child has
  //been expanded but its thread not yet finished its first visit.
  //It's also possible that we observe childWeight <= 0 even though childVisits >= due to multithreading, the two could
  //be out of sync briefly since they are separate atomics.
  double childUtility;
  if(childVisits <= 0 || childWeight <= 0.0)
    childUtility = fpuValue;
  else {
    childUtility = utilityAvg;

    //Tiny adjustment for passing
    double endingScoreBonus = bot.getEndingWhiteScoreBonus(parent,moveLoc);
    if(endingScoreBonus != 0)
      childUtility += bot.getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);
  }

  //When multithreading, totalChildWeight could be out of sync with childWeight, so if they provably are, then fix that up
  if(totalChildWeight < childWeight)
    totalChildWeight = childWeight;

  //Virtual losses to direct threads down different paths
  /*if(childVirtualLosses > 0) {
    double virtualLossWeight = childVirtualLosses * searchParams.numVirtualLossesPerThread;

    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;
    double virtualLossUtility = (parent.nextPla == P_WHITE ? -utilityRadius : utilityRadius);
    double virtualLossWeightFrac = (double)virtualLossWeight / (virtualLossWeight + std::max(0.25,childWeight));
    childUtility = childUtility + (virtualLossUtility - childUtility) * virtualLossWeightFrac;
    childWeight += virtualLossWeight;
  }*/
  return getExploreSelectionValue(
    nnPolicyProb, totalChildWeight, childWeight, childUtility, parentUtilityStdevFactor, parent.nextPla, bot.searchParams
  );
}

double getFpuValueForChildrenAssumeVisited(
  const Search &bot, const SearchNode& node, const NodeStats stats, Player pla, bool isRoot, double policyProbMassVisited,
  double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
) {
  int64_t visits = stats.visits;
  double weightSum = stats.weightSum;
  double utilityAvg = stats.utilityAvg;
  double utilitySqAvg = stats.utilitySqAvg;

  assert(visits > 0);
  assert(weightSum >= 0.0);
  assert(weightSum > 0.0 || bot.searchParams.usingAdversarialAlgo());
  parentWeightPerVisit = weightSum / visits;
  parentUtility = utilityAvg;
  double variancePrior = bot.searchParams.cpuctUtilityStdevPrior * bot.searchParams.cpuctUtilityStdevPrior;
  double variancePriorWeight = bot.searchParams.cpuctUtilityStdevPriorWeight;
  double parentUtilityStdev;
  if(visits <= 0 || weightSum <= 1)
    parentUtilityStdev = bot.searchParams.cpuctUtilityStdevPrior;
  else {
    double utilitySq = parentUtility * parentUtility;
    //Make sure we're robust to numerical precision issues or threading desync of these values, so we don't observe negative variance
    if(utilitySqAvg < utilitySq)
      utilitySqAvg = utilitySq;
    parentUtilityStdev = sqrt(
      std::max(
        0.0,
        ((utilitySq + variancePrior) * variancePriorWeight + utilitySqAvg * weightSum)
        / (variancePriorWeight + weightSum - 1.0)
        - utilitySq
      )
    );
  }
  parentUtilityStdevFactor = 1.0 + bot.searchParams.cpuctUtilityStdevScale * (parentUtilityStdev / bot.searchParams.cpuctUtilityStdevPrior - 1.0);

  double parentUtilityForFPU = parentUtility;
  if(bot.searchParams.fpuParentWeightByVisitedPolicy) {
    double avgWeight = std::min(1.0, pow(policyProbMassVisited, bot.searchParams.fpuParentWeightByVisitedPolicyPow));
    parentUtilityForFPU = avgWeight * parentUtility + (1.0 - avgWeight) * bot.getUtilityFromNN(*(node.getNNOutput()));
  }
  else if(bot.searchParams.fpuParentWeight > 0.0) {
    parentUtilityForFPU = bot.searchParams.fpuParentWeight * bot.getUtilityFromNN(*(node.getNNOutput())) + (1.0 - bot.searchParams.fpuParentWeight) * parentUtility;
  }

  double fpuValue;
  {
    double fpuReductionMax = isRoot ? bot.searchParams.rootFpuReductionMax : bot.searchParams.fpuReductionMax;
    double fpuLossProp = isRoot ? bot.searchParams.rootFpuLossProp : bot.searchParams.fpuLossProp;
    double utilityRadius = bot.searchParams.winLossUtilityFactor + bot.searchParams.staticScoreUtilityFactor + bot.searchParams.dynamicScoreUtilityFactor;

    double reduction = fpuReductionMax * sqrt(policyProbMassVisited);
    fpuValue = pla == P_WHITE ? parentUtilityForFPU - reduction : parentUtilityForFPU + reduction;
    double lossValue = pla == P_WHITE ? -utilityRadius : utilityRadius;
    fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;
  }

  return fpuValue;
}

// Sets SearchParams in a such a way that makes checking (A)MCTS easy.
static void setSimpleSearchParams(SearchParams& params) {
  // Force bot to weight purely by visits for tests.
  // https://discord.com/channels/417022162348802048/583775968804732928/698893048049827870
  params.valueWeightExponent = 0;

  // We turn off subtree utility bias correction so backup is easier to check.
  // https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#subtree-value-bias-correction
  params.subtreeValueBiasFactor = 0;

  // Disable rootNoise so playouts are deterministic and can be checked.
  params.rootNoiseEnabled = false;

  // Disable rootEndingBonusPoints to remove complex hand-engineered scoring
  // adjustment.
  params.rootEndingBonusPoints = 0;

  // TODO(tony): Support this within the playout check.
  params.rootDesiredPerChildVisitsCoeff = 0;

  // This is not used in selfplay right now for backwards compatibility
  // reasons?!?
  params.useNonBuggyLcb = true;

  testAssert(params.cpuctExplorationLog == 0);
  testAssert(params.cpuctUtilityStdevScale == 0);
  testAssert(params.wideRootNoise == 0);
  testAssert(params.fpuParentWeight == 0);
  testAssert(!params.useNoisePruning);
  testAssert(!params.useUncertainty);
  testAssert(!params.antiMirror);
}


void AMCTSTests::runAllAMCTSTests(const int maxVisits,
                                    const int numMovesToSimulate) {
  // testConstPolicies();
  testMCTS(maxVisits, numMovesToSimulate);
  // testAMCTS(maxVisits, numMovesToSimulate);
}

void AMCTSTests::testConstPolicies() {
  cout << "Testing custom const policy nets..." << endl;

  ConfigParser cfg(AMCTS_CONFIG_PATH);
  Logger logger(&cfg, false);

  testAssert(parseRules(cfg, logger) == Rules::getTrompTaylorish());

  vector<SearchParams> searchParamss =
      Setup::loadParams(cfg, Setup::SETUP_FOR_OTHER);
  testAssert(searchParamss.size() == 2);

  SearchParams mctsParams = searchParamss[0];
  mctsParams.maxVisits = 1;
  {  // Remove all randomness from policy.
    mctsParams.chosenMoveTemperatureEarly = 0;
    mctsParams.chosenMoveTemperature = 0;
    mctsParams.rootNoiseEnabled = false;
  }

  auto nnEval1 = getNNEval(CONST_POLICY_1_PATH, cfg, logger, 42);
  auto nnEval2 = getNNEval(CONST_POLICY_2_PATH, cfg, logger, 42);

  {  // Check argmax-bot1 policy
    Search bot1(mctsParams, nnEval1.get(), &logger, "forty-two");

    for (int board_size : {5, 6, 7, 19}) {
      resetBot(bot1, board_size, Rules::getTrompTaylorish());
      Player pla = P_BLACK;
      for (int i = 0; i < 3 * board_size + 3; i++) {
        const Loc loc = bot1.runWholeSearchAndGetMove(pla);
        testAssert(loc != Board::PASS_LOC);

        if (i < bot1.rootBoard.x_size) {
          testAssert(Location::getX(loc, bot1.rootBoard.x_size) == i);
          testAssert(Location::getY(loc, bot1.rootBoard.x_size) == 0);
        }

        testAssert(bot1.makeMove(loc, pla));
        pla = getOpp(pla);
      }
    }
  }

  {  // Check argmax-bot1 and argmax-bot2 interaction.
    Search bot1(mctsParams, nnEval1.get(), &logger, "forty-two");
    Search bot2(mctsParams, nnEval2.get(), &logger, "forty-two");

    const int BOARD_SIZE = 7;
    resetBot(bot1, BOARD_SIZE, Rules::getTrompTaylorish());
    resetBot(bot2, BOARD_SIZE, Rules::getTrompTaylorish());

    testAssert(bot1.rootHistory.rules.multiStoneSuicideLegal);
    testAssert(bot1.rootHistory.rules.koRule == Rules::KO_POSITIONAL);

    Player pla = P_BLACK;
    for (int i = 0; i < 2 * 7 * 7; i++) {
      const Loc loc = i % 2 == 0 ? bot1.runWholeSearchAndGetMove(pla)
                                 : bot2.runWholeSearchAndGetMove(pla);

      if (i % 2 == 0) {  // bot1 (black) move
        if (i / 2 < 7 * 7 - 1) {
          testAssert(Location::getX(loc, 7) == (i / 2) % 7);
          testAssert(Location::getY(loc, 7) == (i / 2) / 7);
        } else {
          testAssert(loc == Board::PASS_LOC);  // Pass due to superko
        }
      } else {  // bot2 (white) move
        testAssert(loc == Board::PASS_LOC);
      }

      {
        auto buf = evaluate(nnEval1, bot1.rootBoard, bot1.rootHistory, pla);
        testAssert(approxEqual(buf->result->whiteWinProb,
                               pla == P_WHITE ? CP1_WIN_PROB : CP1_LOSS_PROB));
        testAssert(approxEqual(buf->result->whiteLossProb,
                               pla == P_WHITE ? CP1_LOSS_PROB : CP1_WIN_PROB));
        testAssert(approxEqual(buf->result->whiteNoResultProb, 0));
      }

      {
        auto buf = evaluate(nnEval2, bot1.rootBoard, bot1.rootHistory, pla);
        testAssert(approxEqual(buf->result->whiteWinProb,
                               pla == P_WHITE ? CP2_WIN_PROB : CP2_LOSS_PROB));
        testAssert(approxEqual(buf->result->whiteLossProb,
                               pla == P_WHITE ? CP2_LOSS_PROB : CP2_WIN_PROB));
        testAssert(approxEqual(buf->result->whiteNoResultProb, 0));
      }

      testAssert(bot1.makeMove(loc, pla));
      testAssert(bot2.makeMove(loc, pla));
      pla = getOpp(pla);
    }

    testAssert(bot1.rootHistory.isGameFinished);
    testAssert(bot2.rootHistory.isGameFinished);
  }
}

void AMCTSTests::testMCTS(const int maxVisits, const int numMovesToSimulate) {
  cout << "Testing MCTS..." << endl;

  ConfigParser cfg(AMCTS_CONFIG_PATH);
  Logger logger(&cfg, false);

  vector<SearchParams> searchParamss =
      Setup::loadParams(cfg, Setup::SETUP_FOR_OTHER);
  testAssert(searchParamss.size() == 2);

  const SearchParams mctsParams = [&]() {
    // First search params is MCTS, second is AMCTS; copy the first.
    SearchParams ret = searchParamss[0];
    setSimpleSearchParams(ret);
    return ret;
  }();

  auto nnEval1 = getNNEval(CONST_POLICY_1_PATH, cfg, logger, 42);
  auto nnEval2 = getNNEval(CONST_POLICY_2_PATH, cfg, logger, 42);
  Search bot1(mctsParams, nnEval1.get(), &logger, "forty-two");
  Search bot2(mctsParams, nnEval2.get(), &logger, "forty-two");

  for (auto bot_ptr : {&bot1, &bot2}) {
    Search& bot = *bot_ptr;

    const int BOARD_SIZE = 9;
    resetBot(bot, BOARD_SIZE, Rules::getTrompTaylorish());

    // The initial board we perform tests on.
    // It has 8 placed stones that are at the top left corner that look like
    // this:
    //    BBBB.....
    //    .WWWW....
    //    .........
    // Here, dots are empty spaces. It is black's turn to move.
    const unique_ptr<CompactSgf> initSgf(
        CompactSgf::parse("(;FF[4]KM[7.5]SZ[19];B[aa];W[bb];B[ba];W[cb];B[ca];"
                          "W[db];B[da];W[eb])"));
    for (auto& m : initSgf->moves) {
      bot.makeMove(m.loc, m.pla);
    }

    Player curPla = P_BLACK;
    for (int midx = 0; midx < numMovesToSimulate; midx++) {
      // Change up visit count to make tests more varied
      bot.searchParams.maxVisits = maxVisits + midx;

      bot.clearSearch();
      const Loc loc = bot.runWholeSearchAndGetMove(curPla);

      checkMCTSSearch(bot, (&bot == &bot1) ? CP1_WIN_PROB : CP2_WIN_PROB,
                      (&bot == &bot1) ? CP1_LOSS_PROB : CP2_LOSS_PROB);

      bot.makeMove(loc, curPla);
      curPla = getOpp(curPla);

      // Break if game is finished.
      if (bot.rootHistory.isGameFinished) break;
    }
  }
}

void AMCTSTests::testAMCTS(const int maxVisits,
                             const int numMovesToSimulate) {
  cout << "Testing AMCTS..." << endl;

  ConfigParser cfg(AMCTS_CONFIG_PATH);
  Logger logger(&cfg, false);

  vector<SearchParams> searchParamss =
      Setup::loadParams(cfg, Setup::SETUP_FOR_OTHER);
  testAssert(searchParamss.size() == 2);

  const SearchParams mctsParams = [&]() {
    SearchParams ret = searchParamss[0];
    setSimpleSearchParams(ret);

    // Make opponent MCTS deterministic for easy testing
    ret.chosenMoveTemperature = 0;
    ret.chosenMoveTemperatureEarly = 0;

    return ret;
  }();
  const SearchParams amcts_s_Params = [&]() {
    SearchParams ret = searchParamss[1];
    ret.searchAlgo = SearchParams::SearchAlgorithm::AMCTS_S;
    setSimpleSearchParams(ret);
    return ret;
  }();
  const SearchParams amcts_r_Params = [&]() {
    SearchParams ret = amcts_s_Params;
    ret.searchAlgo = SearchParams::SearchAlgorithm::AMCTS_R;
    return ret;
  }();

  auto nnEval1 =
      getNNEval(CONST_POLICY_1_PATH, cfg, logger, 42);  // move over pass
  auto nnEval2 =
      getNNEval(CONST_POLICY_2_PATH, cfg, logger, 42);  // pass over move
  Search bot11_s(amcts_s_Params, nnEval1.get(), &logger, "forty-two",
                  mctsParams, nnEval1.get());
  Search bot12_s(amcts_s_Params, nnEval1.get(), &logger, "forty-two",
                  mctsParams, nnEval2.get());

  Search bot11_r(amcts_r_Params, nnEval1.get(), &logger, "forty-two",
                  mctsParams, nnEval1.get());
  Search bot12_r(amcts_r_Params, nnEval1.get(), &logger, "forty-two",
                  mctsParams, nnEval2.get());

  for (auto bot_ptr : {&bot11_s, &bot12_s, &bot11_r, &bot12_r}) {
    Search& bot = *bot_ptr;

    const int BOARD_SIZE = 9;
    resetBot(bot, BOARD_SIZE, Rules::getTrompTaylorish());

    // The initial board we perform tests on.
    // It has 8 placed stones that are at the top left corner that look like
    // this:
    //    BBBB.....
    //    .WWWW....
    //    .........
    // Here, dots are empty spaces. It is black's turn to move.
    const unique_ptr<CompactSgf> initSgf(
        CompactSgf::parse("(;FF[4]KM[7.5]SZ[19];B[aa];W[bb];B[ba];W[cb];B[ca];"
                          "W[db];B[da];W[eb])"));
    for (auto& m : initSgf->moves) {
      bot.makeMove(m.loc, m.pla);
    }

    Player curPla = P_BLACK;
    for (int midx = 0; midx < numMovesToSimulate; midx++) {
      // Change up visit count to make tests more varied
      bot.searchParams.maxVisits =
          (maxVisits / bot.oppBot.get()->searchParams.maxVisits) + midx;

      bot.clearSearch();
      const Loc loc = bot.runWholeSearchAndGetMove(curPla);

      if (&bot == &bot11_s || &bot == &bot11_r) {
        checkAMCTSSearch(bot, CP1_WIN_PROB, CP1_LOSS_PROB, CP1_WIN_PROB,
                          CP1_LOSS_PROB);
      } else if (&bot == &bot12_s || &bot == &bot12_r) {
        checkAMCTSSearch(bot, CP1_WIN_PROB, CP1_LOSS_PROB, CP2_WIN_PROB,
                          CP2_LOSS_PROB);
      }

      bot.makeMove(loc, curPla);
      curPla = getOpp(curPla);

      // Make sure game hasn't been prematurely ended.
      testAssert(!bot.rootHistory.isGameFinished);
    }
  }
}

shared_ptr<NNEvaluator> AMCTSTests::getNNEval(string modelFile,
                                               ConfigParser& cfg,
                                               Logger& logger, uint64_t seed) {
  Setup::initializeSession(cfg);
  Rand seedRand(seed);
  int maxConcurrentEvals = 2;
  int expectedConcurrentEvals = 1;
  int defaultMaxBatchSize = 8;
  bool defaultRequireExactNNLen = false;
  string expectedSha256 = "";

  NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      modelFile, modelFile, expectedSha256, cfg, logger, seedRand,
      maxConcurrentEvals, expectedConcurrentEvals, NNPos::MAX_BOARD_LEN,
      NNPos::MAX_BOARD_LEN, defaultMaxBatchSize, defaultRequireExactNNLen,
      Setup::SETUP_FOR_OTHER);

  shared_ptr<NNEvaluator> ret(nnEval);
  return ret;
}

Rules AMCTSTests::parseRules(ConfigParser& cfg, Logger& logger) {
  GameInitializer gInit(cfg, logger);
  return gInit.createRules();
}

shared_ptr<NNResultBuf> AMCTSTests::evaluate(shared_ptr<NNEvaluator> nnEval,
                                              Board& board, BoardHistory& hist,
                                              Player nextPla, bool skipCache,
                                              bool includeOwnerMap) {
  MiscNNInputParams nnInputParams;
  NNResultBuf* buf = new NNResultBuf();
  nnEval->evaluate(board, hist, nextPla, nnInputParams, *buf, skipCache,
                   includeOwnerMap);
  shared_ptr<NNResultBuf> ret(buf);
  return ret;
}

void AMCTSTests::resetBot(Search& bot, int board_size, const Rules& rules) {
  Board board(board_size, board_size);
  BoardHistory hist(board, P_BLACK, rules, 0);
  bot.setPosition(P_BLACK, board, hist);
}

void AMCTSTests::checkMCTSSearch(const Search& bot, const float win_prob,
                                  const float loss_prob) {
  testAssert(bot.searchParams.searchAlgo ==
             SearchParams::SearchAlgorithm::MCTS);
  SearchTree tree(bot);

  // Not equality since sometimes we visit terminal nodes multiple times.
  testAssert(tree.all_nodes.size() <= bot.searchParams.maxPlayouts);

  // Test { nodes without nnOutputs } == { terminal nodes }
  for (auto node : tree.all_nodes) {
    if (node->getNNOutput() == nullptr) {
      assert(tree.getNodeHistory(node).isGameFinished);
    } else {
      assert(!tree.getNodeHistory(node).isGameFinished);
    }
  }

  // Test weights are as expected
  for (auto node : tree.all_nodes) {
    auto output = node->getNNOutput();
    if (output == nullptr) {
      // Terminal nodes don't have a nnoutput, so we directly check
      // weightSum. They might also be visited more than once.
      testAssert(NodeStats(node->stats).weightSum >= 1);
    } else {
      testAssert(bot.computeWeightFromNode(*node) == 1);
    }
  }

  // Test nnOutputs are as expected
  for (auto node : tree.all_nodes) {
    if (node->getNNOutput() == nullptr) continue;
    testAssert(approxEqual(node->getNNOutput()->whiteWinProb,
                           node->nextPla == P_WHITE ? win_prob : loss_prob));
    testAssert(approxEqual(node->getNNOutput()->whiteLossProb,
                           node->nextPla == P_WHITE ? loss_prob : win_prob));
    testAssert(approxEqual(node->getNNOutput()->whiteNoResultProb, 0));
  }

  // Test backup
  for (auto node : tree.all_nodes) {
    const NodeStats s1 = averageStats(bot, tree.getSubtreeNodes(node));
    const NodeStats s2(node->stats);
    testAssert(approxEqual(s1, s2));
  }

  checkFinalMoveSelection(bot);
}

void AMCTSTests::checkFinalMoveSelection(const Search& bot) {
  unordered_map<Loc, double> trueLocToPsv;
  {
    vector<double> playSelectionValues;
    vector<Loc> locs;
    bot.getPlaySelectionValues(locs, playSelectionValues, 0);
    for (size_t i = 0; i < playSelectionValues.size(); i++) {
      trueLocToPsv[locs[i]] = playSelectionValues[i];
    }

    testAssert(playSelectionValues.size() == locs.size());
    testAssert(trueLocToPsv.size() == locs.size());
  }

  unordered_map<const SearchNode *, double> childToPsv;
  SearchTree tree(bot);
  {
    testAssert(tree.outEdges.at(tree.root).size() > 0);

    for (const auto edge : tree.outEdges.at(tree.root)) {
      auto child = edge->getIfAllocated();
      const double child_weight =
          averageStats(bot, tree.getSubtreeNodes(child)).weightSum;
      childToPsv[child] = child_weight;
    }

    double totalChildWeight = 0;
    double maxChildWeight = 1e-50;
    const SearchNode *heaviestChild = nullptr;
    for (const auto edge : tree.outEdges.at(tree.root)) {
      auto child = edge->getIfAllocated();
      const double weight = childToPsv[child];
      totalChildWeight += weight;
      if (weight > maxChildWeight) {
        maxChildWeight = weight;
        heaviestChild = child;
      }
    }

    // Possibly reduce weight on outEdges that we spend too many visits on in
    // retrospect.
    // TODO(tony): Figure out what exactly is going on here and write it down on
    // overleaf.
    const float* policyProbs =
        tree.root->getNNOutput()->getPolicyProbsMaybeNoised();
    double bestChildExploreSelectionValue;
    {
      double parentUtility;
      double parentWeightPerVisit;
      double parentUtilityStdevFactor;
      double fpuValue = bot.getFpuValueForChildrenAssumeVisited(
        *tree.root, tree.root->nextPla, true, 1.0,
        parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
      );

      bestChildExploreSelectionValue = bot.getExploreSelectionValueOfChild(
          *tree.root, policyProbs, heaviestChild, tree.prevMoves.at(heaviestChild),
          totalChildWeight, heaviestChild->stats.visits.load(std::memory_order_acquire),
          fpuValue, parentUtility, parentWeightPerVisit, parentUtilityStdevFactor,
          false, false, maxChildWeight, NULL
      );
    }
    for (auto& [child, weight] : childToPsv) {
      if (child == heaviestChild) continue;
      const int64_t visits = child->stats.visits.load(std::memory_order_acquire);
      const double reduced = bot.getReducedPlaySelectionWeight(
          *tree.root, policyProbs, child, tree.prevMoves.at(child), totalChildWeight, visits, 1.0,
          bestChildExploreSelectionValue);
      weight = ceil(reduced);
    }

    // Adjust psvs with lcb values
    // TODO(tony): Figure out what exactly is going on here and write it down on
    // overleaf.
    testAssert(bot.searchParams.useLcbForSelection);
    testAssert(bot.searchParams.useNonBuggyLcb);
    {
      unordered_map<const SearchNode *, double> lcbs, radii;
      double bestLcb = -1e10;
      const SearchNode *bestLcbChild = nullptr;
      for (const auto edge : tree.outEdges.at(tree.root)) {
        auto child = edge->getIfAllocated();
        const Loc loc = tree.prevMoves.at(child);
        const int64_t visits = edge->getEdgeVisits(); // child->stats.visits.load(std::memory_order_acquire);
        bot.getSelfUtilityLCBAndRadius(*tree.root, child, visits, loc, lcbs[child],
                                       radii[child]);
        double weight = childToPsv[child];
        if (weight > 0 && weight >= bot.searchParams.minVisitPropForLCB * maxChildWeight) {
          if (lcbs[child] > bestLcb) {
            bestLcb = lcbs[child];
            bestLcbChild = child;
          }
        }
      }

      if (bestLcbChild != nullptr) {
        double best_bound = childToPsv[bestLcbChild];
        for (auto edge : tree.outEdges.at(tree.root)) {
          auto child = edge->getIfAllocated();
          if (child == bestLcbChild) continue;

          const double excessValue = bestLcb - lcbs[child];
          if (excessValue < 0) continue;

          const double radius = radii[child];
          const double radiusFactor =
              (radius + excessValue) / (radius + 0.20 * excessValue);

          double lbound = radiusFactor * radiusFactor * childToPsv[child];
          if (lbound > best_bound) best_bound = lbound;
        }
        childToPsv[bestLcbChild] = best_bound;
      }
    }

    // Prune
    testAssert(bot.searchParams.chosenMoveSubtract == 0);
    double maxPsv = -1e50;
    for (const auto& [_, psv] : childToPsv) maxPsv = max(maxPsv, psv);
    for (auto& [_, psv] : childToPsv) {
      if (psv < min(bot.searchParams.chosenMovePrune, maxPsv / 64)) {
        psv = 0;
      }
    }
  }

  testAssert(childToPsv.size() == trueLocToPsv.size());
  for (const auto [child, psv] : childToPsv) {
    const Loc loc = tree.prevMoves.at(child);
    testAssert(approxEqual(trueLocToPsv[loc], psv));
  }
}

void AMCTSTests::checkPlayoutLogic(const Search& bot) {
  if (bot.searchParams.usingAdversarialAlgo()) {
    // We need temperature to be zero for opponent to be determinstic.
    testAssert(bot.oppBot.get()->searchParams.chosenMoveTemperature == 0);
    testAssert(bot.oppBot.get()->searchParams.chosenMoveTemperatureEarly == 0);
  }

  SearchTree tree(bot);

  unordered_map<const SearchNode*, int> visits;
  auto filterEdgesToVisited = [&](const vector<const SearchChildPointer *>& edges) {
    vector<const SearchChildPointer *> ret;
    for (auto edge : edges) {
      const SearchNode *node = edge->getIfAllocated();
      if (visits[node] > 0) ret.push_back(edge);
    }
    return ret;
  };
  auto filterToVisited = [&](const vector<const SearchNode *>& nodes) {
    vector<const SearchNode *> ret;
    for (auto node : nodes) {
      if (visits[node] > 0) ret.push_back(node);
    }
    return ret;
  };

  auto edgeToPos = [&](const SearchChildPointer *edge) {
    return NNPos::locToPos(edge->getMoveLoc(), bot.rootBoard.x_size, bot.nnXLen,
                           bot.nnYLen);
  };

  auto averageVisSubtreeStats = [&](const SearchNode *node) {
    return averageStats(bot, filterToVisited(tree.getSubtreeNodes(node)),
                        &visits);
  };

  // Cache for which moves the opponent makes at opponent nodes.
  unordered_map<const SearchNode *, const SearchChildPointer *> oppMoveCache;

#ifdef DEBUG
  auto checkOnePlayout = [&](const bool is_last_playout) -> int {
#else
  auto checkOnePlayout = [&]() -> int {
#endif
    int numNodesAdded = 1;
    Board board = bot.rootBoard;
    BoardHistory history = bot.rootHistory;
    auto helper = [&](const SearchNode *node, const Loc prevMove, auto&& dfs) -> void {
      testAssert(node != nullptr);
      visits[node] += 1;

#ifdef DEBUG
      if (is_last_playout) {
        cout << "DFS Currently at node: " << node << endl;
        cout << "Children:";
        for (auto edge : tree.outEdges.at(node)) {
          const Loc loc =
              NNPos::posToLoc(edgeToPos(edge), bot.rootBoard.x_size,
                              bot.rootBoard.y_size, bot.nnXLen, bot.nnYLen);
          const string locString = Location::toString(loc, bot.rootBoard);
          cout << " (" << edge << ", " << locString << ", " << visits[edge->getIfAllocated()]
               << ")";
        }
        cout << endl;
      }
#endif

      if (!node->getNNOutput()) return;  // This is a terminal node
      if (visits[node] == 1) {  // First time visiting the node
        if (bot.searchParams.usingAdversarialAlgo()) {
          if (node->nextPla == bot.rootPla) return;

          numNodesAdded += 1;
          for (auto x : tree.getPathToRoot(node))
            visits[x] += 1;
        } else if (bot.searchParams.searchAlgo == SearchParams::SearchAlgorithm::MCTS) {
          return;
        } else {
          ASSERT_UNREACHABLE;
        }
      }

      if (node != tree.root) {
        assert(prevMove != Board::NULL_LOC);
        history.makeBoardMoveAssumeLegal(board, prevMove,
                                         getOpp(node->nextPla),
                                         bot.rootKoHashTable);
      } else {
        assert(prevMove == Board::NULL_LOC);
      }

      const float* policyProbs =
          node->getNNOutput()->getPolicyProbsMaybeNoised();

      const NodeStats nodeStats = averageVisSubtreeStats(node);
      const double totalChildWeight =
          nodeStats.weightSum - bot.computeWeightFromNode(*node);

      vector<const SearchChildPointer *> edgesByPos(bot.policySize, nullptr);
      for (auto &edge : tree.outEdges.at(node)) {
        edgesByPos[edgeToPos(edge)] = edge;
      }

      vector<bool> edgeVisitedMask(bot.policySize, false);
      double maxChildWeight = 0.0;
      double policyProbMassVisited = 0;
      for (auto edge : filterEdgesToVisited(tree.outEdges.at(node))) {
        edgeVisitedMask[edgeToPos(edge)] = true;
        policyProbMassVisited += policyProbs[edgeToPos(edge)];

        int64_t edgeVisits = visits[edge->getIfAllocated()];
        double childWeight = edge->getIfAllocated()->stats.getChildWeight(edgeVisits);
        if(childWeight > maxChildWeight)
          maxChildWeight = childWeight;
      }

      if (bot.searchParams.usingAdversarialAlgo() &&
          node->nextPla != bot.rootPla) {
        auto edge = oppMoveCache[node];
        if (edge == nullptr) {
          bot.oppBot.get()->setPosition(node->nextPla, board, history);
          const Loc loc =
              bot.oppBot.get()->runWholeSearchAndGetMove(node->nextPla);
          const int bestMovePos = NNPos::locToPos(loc, bot.rootBoard.x_size,
                                                  bot.nnXLen, bot.nnYLen);

          assert(loc != Board::NULL_LOC);
          assert(edgesByPos[bestMovePos] != nullptr);
          edge = edgesByPos[bestMovePos];
          oppMoveCache[node] = edge;
        }
        return dfs(edge->getIfAllocated(), edge->getMoveLoc(), dfs);
      }

#ifdef DEBUG
      vector<tuple<double, double, int>> vals_debug;
#endif

      // These track which child we will descend into.
      double maxSelectionValue = -1e50;
      Loc bestChildMoveLoc = Board::NULL_LOC;

      //First play urgency
      double parentUtility;
      double parentWeightPerVisit;
      double parentUtilityStdevFactor;
      double fpuValue = getFpuValueForChildrenAssumeVisited(
        bot, *node, averageVisSubtreeStats(node), node->nextPla, node == tree.root, policyProbMassVisited,
        parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
      );

      // Try all existing outEdges
      for (auto edge : filterEdgesToVisited(tree.outEdges.at(node))) {
        const SearchNode* child = edge->getIfAllocated();
        if(child == NULL)
          break;

        int64_t childEdgeVisits = edge->getEdgeVisits();
        Loc moveLoc = edge->getMoveLocRelaxed();
        bool isDuringSearch = true;
        double selectionValue = getExploreSelectionValueOfChild(
          bot,
          *node,policyProbs,averageVisSubtreeStats(child),
          moveLoc,
          totalChildWeight,childEdgeVisits,fpuValue,
          parentWeightPerVisit,parentUtilityStdevFactor,
          maxChildWeight
        );
        if(selectionValue > maxSelectionValue) {
          maxSelectionValue = selectionValue;
          bestChildMoveLoc = moveLoc;
        }
      }

      // Try unvisited outEdges
      //Try the new child with the best policy value
      Loc bestNewMoveLoc = Board::NULL_LOC;
      float bestNewNNPolicyProb = -1.0f;
      for(int movePos = 0; movePos<bot.policySize; movePos++) {
        bool alreadyTried = edgeVisitedMask[movePos];
        if(alreadyTried)
          continue;

        Loc moveLoc = NNPos::posToLoc(movePos,bot.rootBoard.x_size, bot.rootBoard.y_size,
                              bot.nnXLen, bot.nnYLen);
        if(moveLoc == Board::NULL_LOC)
          continue;

        //Special logic for the root
        if(node == tree.root) {
          // testAssert(bot.rootBoard.pos_hash == rootBoard.pos_hash);
          // testAssert(bot.rootPla == rootPla);
          if(!bot.isAllowedRootMove(moveLoc)) {
            continue;
          }
        }

        float nnPolicyProb = policyProbs[movePos];
        if(nnPolicyProb > bestNewNNPolicyProb) {
          bestNewNNPolicyProb = nnPolicyProb;
          bestNewMoveLoc = moveLoc;
        }
      }
      if(bestNewMoveLoc != Board::NULL_LOC) {
        double selectionValue = getExploreSelectionValue(
          bestNewNNPolicyProb,totalChildWeight,0.0,fpuValue,parentUtilityStdevFactor,node->nextPla, bot.searchParams
        );
        if(selectionValue > maxSelectionValue) {
          maxSelectionValue = selectionValue;
          bestChildMoveLoc = bestNewMoveLoc;
        }
      }

#ifdef DEBUG
      if (is_last_playout) {
        sort(vals_debug.begin(), vals_debug.end());
        for (auto v : vals_debug) {
          const int pos = get<2>(v);
          const Loc loc =
              NNPos::posToLoc(pos, bot.rootBoard.x_size, bot.rootBoard.y_size,
                              bot.nnXLen, bot.nnYLen);
          const string locString = Location::toString(loc, bot.rootBoard);
          cout << "(" << get<0>(v) << ", " << get<1>(v) << ", " << pos << ", "
               << loc << ", " << locString << ")";
        }
        cout << endl;
      }
#endif

      int bestPos = NNPos::locToPos(bestChildMoveLoc, bot.rootBoard.x_size, bot.nnXLen, bot.nnYLen);
      auto edge = edgesByPos[bestPos];
      if (!edge) {
        cout << "No edge for " << Location::toString(bestChildMoveLoc, bot.rootBoard) << endl;
        visits[node] -= 1;
      } else {
        cout << "Descending into " << Location::toString(bestChildMoveLoc, bot.rootBoard) << ", PSV: " << maxSelectionValue << endl;
        dfs(edge->getIfAllocated(), edge->getMoveLoc(), dfs);
      }
    };

      helper(tree.root, Board::NULL_LOC, helper);
      return numNodesAdded;
    };

  for (int i = 0; i < bot.searchParams.maxVisits;) {
#ifdef DEBUG
    cout << endl << "Checking playout #" << i << endl;
    i += checkOnePlayout(i + 4 >= bot.searchParams.maxVisits);
#else
    i += checkOnePlayout();
#endif
  }

  for (auto node : tree.all_nodes) {
    if (visits[node] != NodeStats(node->stats).visits)
      cout << "Node " << Location::toString(tree.prevMoves[node], bot.rootBoard) << " has " << visits[node] << " visits, expected" << NodeStats(node->stats).visits << endl;

    // testAssert(visits[node] > 0);
    // testAssert(visits[node] == NodeStats(node->stats).visits);
  }
  assert(false);
}

NodeStats AMCTSTests::averageStats(
    const Search& bot, const vector<const SearchNode*>& nodes,
    const unordered_map<const SearchNode*, int>* terminal_node_visits) {
  NodeStats stats;

  // During the following loop, stats will track sums and not averages!
  for (auto node : nodes) {
    const NNOutput* nnOutput = node->getNNOutput();

    if (nnOutput != nullptr) {
      // For a regular node with a nnOutput,
      // we get stats directly from the nnOutput.
      const double winProb = nnOutput->whiteWinProb;
      const double lossProb = nnOutput->whiteLossProb;
      const double noResultProb = nnOutput->whiteNoResultProb;
      const double scoreMean = nnOutput->whiteScoreMean;
      const double scoreMeanSq = nnOutput->whiteScoreMeanSq;
      const double lead = nnOutput->whiteLead;
      const double utility = bot.getUtilityFromNN(*nnOutput);

      const double w = bot.computeWeightFromNode(*node);

      stats.winLossValueAvg += w * (winProb - lossProb);
      stats.noResultValueAvg += w * noResultProb;
      stats.scoreMeanAvg += w * scoreMean;
      stats.scoreMeanSqAvg += w * scoreMeanSq;
      stats.leadAvg += w * lead;
      stats.utilityAvg += w * utility;
      stats.utilitySqAvg += w * utility * utility;

      stats.weightSum += w;
      stats.weightSqSum += w * w;
      stats.visits++;
    } else {
      // If nnOutput is null, this means the node is a terminal node.
      // In this case we need can only get the stats from node->stats.
      const NodeStats termStats(node->stats);
      const double w = (terminal_node_visits == nullptr)
                           ? termStats.weightSum
                           : terminal_node_visits->at(node);

      stats.winLossValueAvg += w * termStats.winLossValueAvg;
      stats.noResultValueAvg += w * termStats.noResultValueAvg;
      stats.scoreMeanAvg += w * termStats.scoreMeanAvg;
      stats.scoreMeanSqAvg += w * termStats.scoreMeanSqAvg;
      stats.leadAvg += w * termStats.leadAvg;
      stats.utilityAvg += w * termStats.utilityAvg;
      stats.utilitySqAvg += w * termStats.utilitySqAvg;

      stats.weightSum += w;
      stats.weightSqSum += termStats.weightSqSum;
    }
  }

  // We fix up the averages at the end.
  stats.winLossValueAvg /= stats.weightSum;
  stats.noResultValueAvg /= stats.weightSum;
  stats.scoreMeanAvg /= stats.weightSum;
  stats.scoreMeanSqAvg /= stats.weightSum;
  stats.leadAvg /= stats.weightSum;
  stats.utilityAvg /= stats.weightSum;
  stats.utilitySqAvg /= stats.weightSum;

  return stats;
}

AMCTSTests::SearchTree::SearchTree(const Search& bot)
    : root(bot.rootNode), rootHist(bot.rootHistory) {
  assert(!bot.searchParams.useGraphSearch);

  std::unordered_set<const SearchNode *> visited;
  auto build = [&](const SearchNode *node, auto&& dfs) -> void {
    assert(node != nullptr);
    auto [_, inserted] = visited.insert(node);
    assert(inserted); // No undirected cycles
    
    all_nodes.push_back(node);
    outEdges[node] = {};

    int __;  // Not used
    const auto arr = node->getChildren(__);
    const int numChildren = node->iterateAndCountChildren();

    auto edges = std::vector<const SearchChildPointer *>();
    for (int i = 0; i < numChildren; i++) {
      edges.push_back(&arr[i]);
    }
    outEdges[node] = edges;

    for (auto edge : outEdges[node]) {
      auto child = edge->getIfAllocated();
      prevMoves[child] = edge->getMoveLoc();
      parents[child] = node;
      dfs(child, dfs);
    }
  };

  build(root, build);
}

vector<const SearchNode*> AMCTSTests::SearchTree::getSubtreeNodes(
    const SearchNode* subtree_root) const {
  vector<const SearchNode*> subtree_nodes;

  auto walk = [this, &subtree_nodes](const SearchNode *node,
                                     auto&& dfs) -> void {
    subtree_nodes.push_back(node);
    for (auto edge : outEdges.at(node)) {
      dfs(edge->getIfAllocated(), dfs);
    }
  };

  walk(subtree_root, walk);
  return subtree_nodes;
}

std::vector<const SearchNode*> AMCTSTests::SearchTree::getPathToRoot(
    const SearchNode* node) const {
  vector<const SearchNode*> path = {node};
  while (*path.rbegin() != root) {
    path.push_back(parents.at(*path.rbegin()));
  }
  return path;
}

BoardHistory AMCTSTests::SearchTree::getNodeHistory(
    const SearchNode* node) const {
  const auto pathRootToNode = [&]() {
    auto path = getPathToRoot(node);
    std::reverse(path.begin(), path.end());
    return path;
  }();

  Board board = rootHist.getRecentBoard(0);
  BoardHistory hist = rootHist;
  for (auto& n : pathRootToNode) {
    if (n == root) continue;  // Skip root node
    hist.makeBoardMoveTolerant(board, prevMoves.at(n), getOpp(n->nextPla));
  }

  return hist;
}

void AMCTSTests::checkAMCTSSearch(const Search& bot, const float win_prob1,
                                    const float loss_prob1,
                                    const float win_prob2,
                                    const float loss_prob2) {
  testAssert(bot.searchParams.usingAdversarialAlgo());

  SearchTree tree(bot);

  // Not equality since sometimes we visit terminal nodes multiple times.
  testAssert(tree.all_nodes.size() <= bot.searchParams.maxPlayouts);

  // Test { nodes without nnOutputs } == { terminal nodes }
  for (auto node : tree.all_nodes) {
    if (node->getNNOutput() == nullptr) {
      testAssert(tree.getNodeHistory(node).isGameFinished);
    } else {
      testAssert(!tree.getNodeHistory(node).isGameFinished);
    }
  }

  // Test weights are as expected
  for (auto node : tree.all_nodes) {
    if (node->getNNOutput() == nullptr) {
      // Terminal nodes don't have a nnOutput, so we directly check
      // weightSum. They might also be visited more than once.
      testAssert(NodeStats(node->stats).weightSum >= 1);
    } else if (node->nextPla == bot.rootPla) {
      testAssert(bot.computeWeightFromNode(*node) == 1);
    } else {
      testAssert(bot.computeWeightFromNode(*node) == 0);
    }
  }

  // Test nnOutputs are as expected
  for (auto node : tree.all_nodes) {
    if (node->getNNOutput() == nullptr) continue;

    if (node->nextPla == bot.rootPla) {  // Adversary node
      const float win_prob =
          (node->nextPla == bot.rootPla) ? win_prob1 : win_prob2;
      const float loss_prob =
          (node->nextPla == bot.rootPla) ? loss_prob1 : loss_prob2;

      testAssert(approxEqual(node->getNNOutput()->whiteWinProb,
                             node->nextPla == P_WHITE ? win_prob : loss_prob));
      testAssert(approxEqual(node->getNNOutput()->whiteLossProb,
                             node->nextPla == P_WHITE ? loss_prob : win_prob));
      testAssert(approxEqual(node->getNNOutput()->whiteNoResultProb, 0));
    } else {  // Victim node
      testAssert(node->oppLocs.has_value());
      testAssert(node->oppPlaySelectionValues.has_value());
    }
  }

  // Test backup
  for (auto node : tree.all_nodes) {
    const NodeStats s1 = averageStats(bot, tree.getSubtreeNodes(node));
    const NodeStats s2(node->stats);
    testAssert(approxEqual(s1, s2));
  }

  checkFinalMoveSelection(bot);

  //checkPlayoutLogic(bot);
}