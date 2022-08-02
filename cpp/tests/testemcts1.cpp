#include "../tests/testemcts1.h"

#include "../dataio/sgf.h"
#include "../program/play.h"
#include "../program/setup.h"
#include "../tests/tests.h"

using namespace std;

// Uncomment to enable debugging
// #define DEBUG

void EMCTS1Tests::runAllEMCTS1Tests() {
  cout << "Running EMCTS1 tests" << endl;
  testConstPolicies();
  testMCTS();
  testEMCTS1();
}

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

// Sets SearchParams in a such a way that makes checking (E)MCTS easy.
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

  // TODO: Support this within the playout check.
  params.rootDesiredPerChildVisitsCoeff = 0;

  params.maxVisits = 1000;
  testAssert(params.cpuctExplorationLog == 0);
  testAssert(params.cpuctUtilityStdevScale == 0);
  testAssert(params.wideRootNoise == 0);
  testAssert(params.fpuParentWeight == 0);
  testAssert(!params.useNoisePruning);
  testAssert(!params.useUncertainty);
  testAssert(!params.antiMirror);
}

static double getFpuValue(const Search& bot, const SearchNode* node,
                          const double utilityAvg,
                          const double policyProbMassVisited) {
  const bool isRoot = node == bot.rootNode;
  const double fpuReductionMax = isRoot ? bot.searchParams.rootFpuReductionMax
                                        : bot.searchParams.fpuReductionMax;
  const double fpuLossProp =
      isRoot ? bot.searchParams.rootFpuLossProp : bot.searchParams.fpuLossProp;
  const double utilityRadius = bot.searchParams.winLossUtilityFactor +
                               bot.searchParams.staticScoreUtilityFactor +
                               bot.searchParams.dynamicScoreUtilityFactor;

  const double reduction = fpuReductionMax * sqrt(policyProbMassVisited);

  double fpuValue = node->nextPla == P_WHITE ? utilityAvg - reduction
                                             : utilityAvg + reduction;
  double lossValue = node->nextPla == P_WHITE ? -utilityRadius : utilityRadius;
  fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;

  return fpuValue;
}

void EMCTS1Tests::testConstPolicies() {
  ConfigParser cfg(EMCTS1_CONFIG_PATH);
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
    Search bot1(mctsParams, nnEval1.get(), &logger, "forty-two", nullptr);

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
    Search bot1(mctsParams, nnEval1.get(), &logger, "forty-two", nullptr);
    Search bot2(mctsParams, nnEval2.get(), &logger, "forty-two", nullptr);
    resetBot(bot1, 7, Rules::getTrompTaylorish());
    resetBot(bot2, 7, Rules::getTrompTaylorish());

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

void EMCTS1Tests::testMCTS() {
  ConfigParser cfg("cpp/tests/data/configs/test-emcts1.cfg");
  Logger logger(&cfg, false);

  vector<SearchParams> searchParamss =
      Setup::loadParams(cfg, Setup::SETUP_FOR_OTHER);
  testAssert(searchParamss.size() == 2);

  const SearchParams mctsParams = searchParamss[0];

  auto nnEval1 = getNNEval(CONST_POLICY_1_PATH, cfg, logger, 42);
  auto nnEval2 = getNNEval(CONST_POLICY_2_PATH, cfg, logger, 42);
  Search bot1(mctsParams, nnEval1.get(), &logger, "forty-two", nullptr);
  Search bot2(mctsParams, nnEval2.get(), &logger, "forty-two", nullptr);

  for (auto bot_ptr : {&bot1, &bot2}) {
    Search& bot = *bot_ptr;
    resetBot(bot, 9, Rules::getTrompTaylorish());
    setSimpleSearchParams(bot.searchParams);

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
    for (int midx = 0; midx < 4; midx++) {
      bot.searchParams.maxVisits += 1;  // Make tests more varied

      bot.clearSearch();
      const Loc loc = bot.runWholeSearchAndGetMove(curPla);

      checkMCTSSearch(bot, (&bot == &bot1) ? CP1_WIN_PROB : CP2_WIN_PROB,
                      (&bot == &bot1) ? CP1_LOSS_PROB : CP2_LOSS_PROB);

      bot.makeMove(loc, curPla);
      curPla = getOpp(curPla);
    }
  }
}

void EMCTS1Tests::testEMCTS1() {
  ConfigParser cfg("cpp/tests/data/configs/test-emcts1.cfg");
  Logger logger(&cfg, false);

  vector<SearchParams> searchParamss =
      Setup::loadParams(cfg, Setup::SETUP_FOR_OTHER);
  testAssert(searchParamss.size() == 2);

  const SearchParams mctsParams = searchParamss[0];
  const SearchParams emcts1Params = searchParamss[1];

  auto nnEval1 =
      getNNEval(CONST_POLICY_1_PATH, cfg, logger, 42);  // move over pass
  auto nnEval2 =
      getNNEval(CONST_POLICY_2_PATH, cfg, logger, 42);  // pass over move
  Search bot11(emcts1Params, nnEval1.get(), &logger, "forty-two",
               nnEval1.get());
  Search bot12(emcts1Params, nnEval1.get(), &logger, "forty-two",
               nnEval2.get());

  for (auto bot_ptr : {&bot11, &bot12}) {
    Search& bot = *bot_ptr;
    resetBot(bot, 9, Rules::getTrompTaylorish());
    setSimpleSearchParams(bot.searchParams);

    // Make EMCTS1 deterministic
    bot.searchParams.chosenMoveTemperature = 0;
    bot.searchParams.chosenMoveTemperatureEarly = 0;

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
    for (int midx = 0; midx < 4; midx++) {
      bot.searchParams.maxVisits += 1;  // Make tests more varied

      bot.clearSearch();
      const Loc loc = bot.runWholeSearchAndGetMove(curPla);

      if (&bot == &bot11) {
        checkEMCTS1Search(bot, CP1_WIN_PROB, CP1_LOSS_PROB, CP1_WIN_PROB,
                          CP1_LOSS_PROB);
      } else if (&bot == &bot12) {
        checkEMCTS1Search(bot, CP1_WIN_PROB, CP1_LOSS_PROB, CP2_WIN_PROB,
                          CP2_LOSS_PROB);
      }

      bot.makeMove(loc, curPla);
      curPla = getOpp(curPla);

      // Make sure game hasn't been prematurely ended.
      testAssert(!bot.rootHistory.isGameFinished);
    }
  }
}

void EMCTS1Tests::checkMCTSSearch(const Search& bot, const float win_prob,
                                  const float loss_prob) {
  testAssert(bot.searchParams.searchAlgo ==
             SearchParams::SearchAlgorithm::MCTS);
  SearchTree tree(bot);

  // Not equality since sometimes we visit terminal nodes multiple times.
  testAssert(tree.all_nodes.size() <= bot.searchParams.maxPlayouts);

  // Test weights are as expected
  for (auto node : tree.all_nodes) {
    if (node->getNNOutput() == nullptr) {
      // Terminal nodes don't have a nnoutput, so we directly check
      // weightSum. They might also be visited more than once.
      testAssert(NodeStats(node->stats).weightSum >= 1);
    } else {
      testAssert(bot.computeNodeWeight(*node) == 1);
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

  // TODO: Test final move selection

  // Test playout logic
  checkPlayoutLogic(bot);
}

void EMCTS1Tests::checkEMCTS1Search(const Search& bot, const float win_prob1,
                                    const float loss_prob1,
                                    const float win_prob2,
                                    const float loss_prob2) {
  testAssert(bot.searchParams.searchAlgo ==
             SearchParams::SearchAlgorithm::EMCTS1);

  SearchTree tree(bot);

  // Not equality since sometimes we visit terminal nodes multiple times.
  testAssert(tree.all_nodes.size() <= bot.searchParams.maxPlayouts);

  // Test weights are as expected
  for (auto node : tree.all_nodes) {
    if (node->getNNOutput() == nullptr) {
      // Terminal nodes don't have a nnoutput, so we directly check
      // weightSum. They might also be visited more than once.
      testAssert(NodeStats(node->stats).weightSum >= 1);
    } else if (node->nextPla == bot.rootPla) {
      testAssert(bot.computeNodeWeight(*node) == 1);
    } else {
      testAssert(bot.computeNodeWeight(*node) == 0);
    }
  }

  // Test nnOutputs are as expected
  for (auto node : tree.all_nodes) {
    if (node->getNNOutput() == nullptr) continue;

    const float win_prob =
        (node->nextPla == bot.rootPla) ? win_prob1 : win_prob2;
    const float loss_prob =
        (node->nextPla == bot.rootPla) ? loss_prob1 : loss_prob2;

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

  // TODO: Test final move selection

  checkPlayoutLogic(bot);
}

void EMCTS1Tests::checkPlayoutLogic(const Search& bot) {
  if (bot.searchParams.searchAlgo == SearchParams::SearchAlgorithm::EMCTS1) {
    // We need temperature to be zero for EMCTS1 to be determinstic.
    testAssert(bot.searchParams.chosenMoveTemperature == 0);
    testAssert(bot.searchParams.chosenMoveTemperatureEarly == 0);
  }

  SearchTree tree(bot);

  unordered_map<const SearchNode*, int> visits;
  auto filterToVisited = [&](const vector<const SearchNode*>& nodes) {
    vector<const SearchNode*> ret;
    for (auto node : nodes) {
      if (visits[node] > 0) ret.push_back(node);
    }
    return ret;
  };

  auto nodeToPos = [&](const SearchNode* node) {
    return NNPos::locToPos(node->prevMoveLoc, bot.rootBoard.x_size, bot.nnXLen,
                           bot.nnYLen);
  };

  auto averageVisSubtreeStats = [&](const SearchNode* node) {
    return averageStats(bot, filterToVisited(tree.getSubtreeNodes(node)),
                        &visits);
  };

#ifdef DEBUG
  auto checkOnePlayout = [&](const bool is_last_playout) -> int {
#else
  auto checkOnePlayout = [&]() -> int {
#endif
    int numNodesAdded = 1;
    Board board = bot.rootBoard;
    BoardHistory history = bot.rootHistory;
    auto helper = [&](const SearchNode* node, auto&& dfs) -> void {
      testAssert(node != nullptr);
      visits[node] += 1;

#ifdef DEBUG
      if (is_last_playout) {
        cout << "DFS Currently at node: " << node << endl;
        cout << "Children:";
        for (auto child : tree.children.at(node)) {
          const Loc loc =
              NNPos::posToLoc(nodeToPos(child), bot.rootBoard.x_size,
                              bot.rootBoard.y_size, bot.nnXLen, bot.nnYLen);
          const string locString = Location::toString(loc, bot.rootBoard);
          cout << " (" << child << ", " << locString << ", " << visits[child]
               << ")";
        }
        cout << endl;
      }
#endif

      if (node->getNNOutput() == nullptr) return;  // This is a terminal nodes
      if (visits[node] == 1) {  // First time visiting the node
        switch (bot.searchParams.searchAlgo) {
          case SearchParams::SearchAlgorithm::MCTS:
            return;
          case SearchParams::SearchAlgorithm::EMCTS1:
            if (node->nextPla == bot.rootPla) return;
            numNodesAdded += 1;
            for (auto x : tree.getPathToRoot(node)) visits[x] += 1;
            break;
          default:
            ASSERT_UNREACHABLE;
        }
      }

      if (node != bot.rootNode) {
        history.makeBoardMoveAssumeLegal(board, node->prevMoveLoc,
                                         getOpp(node->nextPla),
                                         bot.rootKoHashTable);
      }

      const float* policyProbs =
          node->getNNOutput()->getPolicyProbsMaybeNoised();

      const NodeStats nodeStats = averageVisSubtreeStats(node);
      const double totalChildWeight =
          nodeStats.weightSum - bot.computeNodeWeight(*node);

      vector<const SearchNode*> movePosNode(bot.policySize, nullptr);
      for (auto child : tree.children.at(node)) {
        movePosNode[nodeToPos(child)] = child;
      }

      vector<bool> movePosVis(bot.policySize, false);
      double policyProbMassVisited = 0;
      for (auto child : filterToVisited(tree.children.at(node))) {
        movePosVis[nodeToPos(child)] = true;
        policyProbMassVisited += policyProbs[nodeToPos(child)];
      }

      if (bot.searchParams.searchAlgo ==
              SearchParams::SearchAlgorithm::EMCTS1 &&
          node->nextPla != bot.rootPla) {
        int bestMovePos = -1;
        double maxPolicyProb = -1e50;
        for (int pos = 0; pos < bot.policySize; pos++) {
          const Loc loc =
              NNPos::posToLoc(pos, bot.rootBoard.x_size, bot.rootBoard.y_size,
                              bot.nnXLen, bot.nnYLen);

          if (loc == Board::NULL_LOC) continue;
          if (!history.isLegal(board, loc, node->nextPla)) continue;

          const double curPolicyProb = policyProbs[pos];
          if (curPolicyProb > maxPolicyProb) {
            maxPolicyProb = curPolicyProb;
            bestMovePos = pos;
          }
        }

        return dfs(movePosNode[bestMovePos], dfs);
      }

#ifdef DEBUG
      vector<tuple<double, double, int>> vals_debug;
#endif

      // These track which child we will descend into.
      int bestMovePos = -1;
      double maxSelectionValue = -1e50;
      auto considerMove = [&](const int pos, const double childWeight,
                              const double whiteUtility) {
        const double nnPolicyProb = policyProbs[pos];
        const double valueComponent =
            node->nextPla == P_WHITE ? whiteUtility : -whiteUtility;
        const double exploreComponent =
            bot.searchParams.cpuctExploration * nnPolicyProb *
            sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET) /
            (1.0 + childWeight);

        const double selectionValue = valueComponent + exploreComponent;
        if (selectionValue > maxSelectionValue) {
          maxSelectionValue = selectionValue;
          bestMovePos = pos;
        }

#ifdef DEBUG
        vals_debug.push_back({selectionValue, nnPolicyProb, pos});
#endif
      };

      // Try all existing children
      for (auto child : filterToVisited(tree.children.at(node))) {
        const NodeStats childStats = averageVisSubtreeStats(child);
        considerMove(nodeToPos(child), childStats.weightSum,
                     childStats.utilityAvg);
      }

      // Try unvisited children
      const double fpuValue =
          getFpuValue(bot, node, nodeStats.utilityAvg, policyProbMassVisited);
      for (int pos = 0; pos < bot.policySize; pos++) {
        if (movePosVis[pos]) continue;  // Skip moves that are visited

        // Only consider moves that are valid.
        {
          const Loc loc =
              NNPos::posToLoc(pos, bot.rootBoard.x_size, bot.rootBoard.y_size,
                              bot.nnXLen, bot.nnYLen);
          if (loc == Board::NULL_LOC) continue;
          if (node == bot.rootNode) {
            if (!bot.isAllowedRootMove(loc)) continue;
          }
        }

        considerMove(pos, 0, fpuValue);
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

      dfs(movePosNode[bestMovePos], dfs);
    };

    helper(bot.rootNode, helper);
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
    testAssert(visits[node] > 0);
    testAssert(visits[node] == NodeStats(node->stats).visits);
  }
}

Rules EMCTS1Tests::parseRules(ConfigParser& cfg, Logger& logger) {
  GameInitializer gInit(cfg, logger);
  return gInit.createRules();
}

shared_ptr<NNEvaluator> EMCTS1Tests::getNNEval(string modelFile,
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

shared_ptr<NNResultBuf> EMCTS1Tests::evaluate(shared_ptr<NNEvaluator> nnEval,
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

void EMCTS1Tests::resetBot(Search& bot, int board_size, const Rules& rules) {
  Board board(board_size, board_size);
  BoardHistory hist(board, P_BLACK, rules, 0);
  bot.setPosition(P_BLACK, board, hist);
}

EMCTS1Tests::SearchTree::SearchTree(const Search& bot)
    : tree_root(bot.rootNode) {
  auto build = [this](const SearchNode* node, auto&& dfs) -> void {
    all_nodes.push_back(node);
    children[node] = {};

    int _;  // Not used
    const auto arr = node->getChildren(_);
    const int numChildren = node->iterateAndCountChildren();
    for (size_t i = 0; i < numChildren; i++) {
      const SearchNode* child = arr[i].getIfAllocated();
      children[node].push_back(child);
      dfs(child, dfs);
    }
  };

  build(bot.rootNode, build);
}

vector<const SearchNode*> EMCTS1Tests::SearchTree::getSubtreeNodes(
    const SearchNode* subtree_root) const {
  vector<const SearchNode*> subtree_nodes;

  auto walk = [this, &subtree_nodes](const SearchNode* node,
                                     auto&& dfs) -> void {
    subtree_nodes.push_back(node);
    for (auto child : children.at(node)) {
      dfs(child, dfs);
    }
  };

  walk(subtree_root, walk);
  return subtree_nodes;
}

std::vector<const SearchNode*> EMCTS1Tests::SearchTree::getPathToRoot(
    const SearchNode* node) const {
  vector<const SearchNode*> path = {node};
  while (*path.rbegin() != tree_root) {
    path.push_back((*path.rbegin())->parent);
  }
  return path;
}

NodeStats EMCTS1Tests::averageStats(
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
      const double utility = bot.getResultUtilityFromNN(*nnOutput) +
                             bot.getScoreUtility(scoreMean, scoreMeanSq);

      const double w = bot.computeNodeWeight(*node);

      stats.winLossValueAvg += w * (winProb - lossProb);
      stats.noResultValueAvg += w * noResultProb;
      stats.scoreMeanAvg += w * scoreMean;
      stats.scoreMeanSqAvg += w * scoreMeanSq;
      stats.leadAvg += w * lead;
      stats.utilityAvg += w * utility;
      stats.utilitySqAvg += w * utility * utility;

      stats.weightSum += w;
      stats.weightSqSum += w * w;
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
