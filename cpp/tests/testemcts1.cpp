#include "../tests/testemcts1.h"

#include "../dataio/sgf.h"
#include "../program/play.h"
#include "../program/setup.h"
#include "../tests/tests.h"

using namespace std;

static bool approxEqual(float x, float y) {
  float tolerance;
  tolerance = 0.0001f * std::max(std::abs(x), std::max(std::abs(y), 1.0f));
  return std::abs(x - y) < tolerance;
}

static bool approxEqual(double x, double y) {
  // cout << x << " ==? " << y << endl;
  double tolerance;
  tolerance = 1e-5 * std::max(std::abs(x), std::max(std::abs(y), 1.0));
  return std::abs(x - y) < tolerance;
}

static bool approxEqual(const NodeStats& s1, const NodeStats& s2) {
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

void EMCTS1Tests::runAllEMCTS1Tests() {
  cout << "Running EMCTS1 tests" << endl;
  // testConstPolicies();
  testMCTSStats();
  // testEMCTS1Stats();
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

void EMCTS1Tests::testMCTSStats() {
  ConfigParser cfg("cpp/tests/data/configs/test-emcts1.cfg");
  Logger logger(&cfg, false);

  vector<SearchParams> searchParamss =
      Setup::loadParams(cfg, Setup::SETUP_FOR_OTHER);
  testAssert(searchParamss.size() == 2);

  const SearchParams mctsParams = searchParamss[0];

  auto nnEval = getNNEval(CONST_POLICY_1_PATH, cfg, logger, 42);
  Search bot(mctsParams, nnEval.get(), &logger, "forty-two", nnEval.get());
  resetBot(bot, 9, Rules::getTrompTaylorish());

  // The initial board we perform tests on.
  // It has 8 placed stones that are at the top left corner that look like this:
  //    BBBB.....
  //    .WWWW....
  //    .........
  // Here, dots are empty spaces. It is black's turn to move.
  const unique_ptr<CompactSgf> initSgf(CompactSgf::parse(
      "(;FF[4]KM[7.5]SZ[19];B[aa];W[bb];B[ba];W[cb];B[ca];W[db];B[da];W[eb])"));
  for (auto& m : initSgf->moves) {
    bot.makeMove(m.loc, m.pla);
  }

  // Force bot to weight purely by visits for tests.
  // https://discord.com/channels/417022162348802048/583775968804732928/698893048049827870
  bot.searchParams.valueWeightExponent = 0;

  // We turn off subtree utility bias correction so backup is easier to check.
  // https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#subtree-value-bias-correction
  bot.searchParams.subtreeValueBiasFactor = 0;

  // Disable rootNoise so playouts are deterministic and can be checked.
  bot.searchParams.rootNoiseEnabled = false;

  bot.searchParams.maxVisits = 37;
  testAssert(bot.searchParams.cpuctExplorationLog == 0);
  testAssert(bot.searchParams.cpuctUtilityStdevScale == 0);
  testAssert(bot.searchParams.wideRootNoise == 0);
  testAssert(bot.searchParams.fpuParentWeight == 0);
  // testAssert(bot.searchParams.rootNumSymmetriesToSample == 1);
  testAssert(!bot.searchParams.useNoisePruning);
  testAssert(!bot.searchParams.useUncertainty);
  testAssert(!bot.searchParams.antiMirror);

  Player curPla = P_BLACK;
  for (int midx = 0; midx < 4; midx++) {
    cout << "==============================================" << endl;
    const Loc loc = bot.runWholeSearchAndGetMove(curPla);

    SearchTree tree(bot);

    // Test all weights are 1.
    for (auto node : tree.all_nodes) {
      testAssert(bot.computeNodeWeight(*node) == 1);
    }

    // Test nnOutputs are as expected
    for (auto node : tree.all_nodes) {
      testAssert(
          approxEqual(node->getNNOutput()->whiteWinProb,
                      node->nextPla == P_WHITE ? CP1_WIN_PROB : CP1_LOSS_PROB));
      testAssert(
          approxEqual(node->getNNOutput()->whiteLossProb,
                      node->nextPla == P_WHITE ? CP1_LOSS_PROB : CP1_WIN_PROB));
      testAssert(approxEqual(node->getNNOutput()->whiteNoResultProb, 0));
    }

    // Test backup
    for (auto node : tree.all_nodes) {
      const NodeStats s1 = averageStats(bot, tree.getSubtreeNodes(node));
      const NodeStats s2(node->stats);
      testAssert(approxEqual(s1, s2));
    }

    // Test playout logic
    // Essentially a test of selectBestChildToDescend()
    // Our naively implemented test takes O(BP^3) time,
    // where B is the size of the board and P is the number of playouts.
    {
      unordered_map<const SearchNode*, bool> vis;
      auto filterToVisNodes = [&vis](const vector<const SearchNode*>& nodes) {
        vector<const SearchNode*> ret;
        for (auto node : nodes) {
          if (vis[node]) {
            ret.push_back(node);
          }
        }
        return ret;
      };

      auto checkPlayout = [&bot, &tree, &vis, &curPla, &filterToVisNodes](
                              const SearchNode* node, auto&& dfs) -> void {
        testAssert(node != nullptr);
        if (!vis[node]) {
          vis[node] = true;
          return;
        }

        const float* policyProbs =
            node->getNNOutput()->getPolicyProbsMaybeNoised();

        const NodeStats nodeStats =
            averageStats(bot, filterToVisNodes(tree.getSubtreeNodes(node)));
        cout << node << " weight: " << nodeStats.weightSum << endl;

        vector<const SearchNode*> movePosNode(bot.policySize, nullptr);
        vector<bool> movePosVis(bot.policySize, false);
        double policyProbMassVisited = 0;
        for (auto child : tree.adj.at(node)) {
          const int pos =
              NNPos::locToPos(child->prevMoveLoc, bot.rootBoard.x_size,
                              bot.nnXLen, bot.rootBoard.y_size);
          movePosNode[pos] = child;

          if (vis[child]) {
            movePosVis[pos] = true;
            policyProbMassVisited += policyProbs[pos];
          }
        }

        // These track which child we will descend into.
        int bestMovePos = -1;
        double maxSelectionValue = -1e50;
        vector<pair<double, double>> vals_debug;

        // Try all existing children
        for (auto child : tree.adj.at(node)) {
          if (!vis[child]) continue;
          const int pos =
              NNPos::locToPos(child->prevMoveLoc, bot.rootBoard.x_size,
                              bot.nnXLen, bot.rootBoard.y_size);
          const double nnPolicyProb = policyProbs[pos];

          const NodeStats childStats =
              averageStats(bot, filterToVisNodes(tree.getSubtreeNodes(child)));
          const double childWeight = childStats.weightSum;
          const double whiteUtility = childStats.utilityAvg;

          const double valueComponent =
              node->nextPla == P_WHITE ? whiteUtility : -whiteUtility;
          const float exploreComponent =
              bot.searchParams.cpuctExploration * nnPolicyProb *
              sqrt(nodeStats.weightSum + 0.01) / (1.0 + childWeight);

          const double selectionValue = valueComponent + exploreComponent;
          if (selectionValue > maxSelectionValue) {
            maxSelectionValue = selectionValue;
            bestMovePos = pos;
          }
          vals_debug.push_back({selectionValue, nnPolicyProb});
        }

        // Try unvisited children
        {
          double fpuValue;
          {
            const bool isRoot = node == bot.rootNode;
            const double fpuReductionMax =
                isRoot ? bot.searchParams.rootFpuReductionMax
                       : bot.searchParams.fpuReductionMax;
            const double fpuLossProp = isRoot ? bot.searchParams.rootFpuLossProp
                                              : bot.searchParams.fpuLossProp;
            const double utilityRadius =
                bot.searchParams.winLossUtilityFactor +
                bot.searchParams.staticScoreUtilityFactor +
                bot.searchParams.dynamicScoreUtilityFactor;
            const double parentUtility = nodeStats.utilityAvg;

            const double reduction =
                fpuReductionMax * sqrt(policyProbMassVisited);
            fpuValue = curPla == P_WHITE ? parentUtility - reduction
                                         : parentUtility + reduction;
            double lossValue =
                curPla == P_WHITE ? -utilityRadius : utilityRadius;
            fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;
          }

          for (int pos = 0; pos < bot.policySize; pos++) {
            // Only consider moves that are valid.
            {
              const Loc moveLoc =
                  NNPos::posToLoc(pos, bot.rootBoard.x_size,
                                  bot.rootBoard.y_size, bot.nnXLen, bot.nnYLen);
              if (moveLoc == Board::NULL_LOC) {
                continue;
              }
            }

            if (movePosVis[pos]) continue;

            const float nnPolicyProb = policyProbs[pos];
            const double childWeight = 0;
            const double whiteUtility = fpuValue;

            const double valueComponent =
                node->nextPla == P_WHITE ? whiteUtility : -whiteUtility;
            const float exploreComponent =
                bot.searchParams.cpuctExploration * nnPolicyProb *
                sqrt(nodeStats.weightSum + 0.01) / (1.0 + childWeight);

            const double selectionValue = valueComponent + exploreComponent;
            if (selectionValue > maxSelectionValue) {
              maxSelectionValue = selectionValue;
              bestMovePos = pos;
            }
            vals_debug.push_back({selectionValue, nnPolicyProb});
          }
        }

        sort(vals_debug.begin(), vals_debug.end());
        for (auto v : vals_debug) {
          cout << "(" << v.first << ", " << v.second << ") ";
        }
        cout << endl;

        dfs(movePosNode[bestMovePos], dfs);
      };

      for (int i = 0; i < bot.searchParams.maxVisits; i++) {
        cout << endl;
        cout << "Testing playout #" << i << endl;
        cout << "Total nodes: " << tree.all_nodes.size() << endl;
        cout << "Policy size: " << bot.policySize << endl;
        cout << bot.rootBoard.x_size << " " << bot.rootBoard.y_size << endl;
        cout << bot.nnXLen << " " << bot.nnYLen << endl;
        checkPlayout(bot.rootNode, checkPlayout);
      }
      for (auto node : tree.all_nodes) {
        testAssert(vis[node]);
      }
    }

    bot.makeMove(loc, curPla);
    curPla = getOpp(curPla);
  }
}

void EMCTS1Tests::testEMCTS1Stats() {
  ConfigParser cfg("cpp/tests/data/configs/test-emcts1.cfg");
  Logger logger(&cfg, false);

  vector<SearchParams> searchParamss =
      Setup::loadParams(cfg, Setup::SETUP_FOR_OTHER);
  testAssert(searchParamss.size() == 2);

  SearchParams mctsParams = searchParamss[0];
  mctsParams.maxVisits = 1000;
  {  // Remove all randomness from policy.
    mctsParams.chosenMoveTemperatureEarly = 0;
    mctsParams.chosenMoveTemperature = 0;
    mctsParams.rootNoiseEnabled = false;
  }

  SearchParams emcts1Params = searchParamss[1];
  emcts1Params.maxVisits = 10;
  {  // Remove all randomness from policy.
    emcts1Params.chosenMoveTemperatureEarly = 0;
    emcts1Params.chosenMoveTemperature = 0;
    emcts1Params.rootNoiseEnabled = false;
  }

  auto nnEval1 = getNNEval(CONST_POLICY_1_PATH, cfg, logger, 42);
  auto nnEval2 = getNNEval(CONST_POLICY_2_PATH, cfg, logger, 42);

  Search bot1_mcts(mctsParams, nnEval1.get(), &logger, "forty-two",
                   nnEval1.get());
  Search bot2_mcts(mctsParams, nnEval2.get(), &logger, "forty-two",
                   nnEval1.get());

  Search bot12_emcts1(emcts1Params, nnEval1.get(), &logger, "forty-two",
                      nnEval2.get());
  Search bot21_emcts1(emcts1Params, nnEval2.get(), &logger, "forty-two",
                      nnEval1.get());

  // The initial board we perform tests on.
  // It has 8 placed stones that are at the top left corner that look like this:
  //    BBBB...
  //    .WWWW..
  //    ...................
  // Here, dots are empty spaces. It is black's turn to move.
  unique_ptr<CompactSgf> init_sgf(CompactSgf::parse(
      "(;FF[4]KM[7.5]SZ[19];B[aa];W[bb];B[ba];W[cb];B[ca];W[db];B[da];W[eb])"));
  for (auto& m : init_sgf->moves) {
    bot1_mcts.makeMove(m.loc, m.pla);
    bot2_mcts.makeMove(m.loc, m.pla);
    bot12_emcts1.makeMove(m.loc, m.pla);
    bot21_emcts1.makeMove(m.loc, m.pla);
  }

  {
    bot1_mcts.searchParams.maxVisits = 30;
    const Loc loc = bot1_mcts.runWholeSearchAndGetMove(P_BLACK);

    const int numRootChildren = bot1_mcts.rootNode->iterateAndCountChildren();
    cout << numRootChildren << endl;
    cout << bot1_mcts.rootNode->stats.weightSum << endl;
    cout << bot1_mcts.rootNode->stats.winLossValueAvg << endl;

    testAssert(Location::toString(loc, bot1_mcts.rootBoard) ==
               "E19");  // TODO: Check if this is actually correct?
  }

  // TODO:
  // Run a full search, then check the values in the tree are what is expected.
  // For both MCTS, and EMCTS1.
  // Luckily, we store the nnOutputs at every node!
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

EMCTS1Tests::SearchTree::SearchTree(const Search& bot) {
  auto build = [this](const SearchNode* node, auto&& dfs) -> void {
    all_nodes.push_back(node);
    adj[node] = {};

    int _;  // Not used
    const auto arr = node->getChildren(_);
    const int numChildren = node->iterateAndCountChildren();
    for (size_t i = 0; i < numChildren; i++) {
      const SearchNode* child = arr[i].getIfAllocated();
      adj[node].push_back(child);
      dfs(child, dfs);
    }
  };

  build(bot.rootNode, build);
}

vector<const SearchNode*> EMCTS1Tests::SearchTree::getSubtreeNodes(
    const SearchNode* root) const {
  vector<const SearchNode*> subtree_nodes;

  auto walk = [this, &subtree_nodes](const SearchNode* node,
                                     auto&& dfs) -> void {
    subtree_nodes.push_back(node);
    for (auto child : adj.at(node)) {
      dfs(child, dfs);
    }
  };

  walk(root, walk);
  return subtree_nodes;
}

NodeStats EMCTS1Tests::averageStats(const Search& bot,
                                    const vector<const SearchNode*>& nodes) {
  NodeStats stats;
  for (auto node : nodes) {
    const NNOutput* nnOutput = node->getNNOutput();

    const double winProb = nnOutput->whiteWinProb;
    const double lossProb = nnOutput->whiteLossProb;
    const double noResultProb = nnOutput->whiteNoResultProb;
    const double scoreMean = nnOutput->whiteScoreMean;
    const double scoreMeanSq = nnOutput->whiteScoreMeanSq;
    const double lead = nnOutput->whiteLead;
    const double utility = bot.getResultUtilityFromNN(*nnOutput) +
                           bot.getScoreUtility(scoreMean, scoreMeanSq);
    // const double utility =
    //     bot.getResultUtility(winProb - lossProb, noResultProb) +
    //     bot.getScoreUtility(scoreMean, scoreMeanSq);

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
  }

  stats.winLossValueAvg /= stats.weightSum;
  stats.noResultValueAvg /= stats.weightSum;
  stats.scoreMeanAvg /= stats.weightSum;
  stats.scoreMeanSqAvg /= stats.weightSum;
  stats.leadAvg /= stats.weightSum;
  stats.utilityAvg /= stats.weightSum;
  stats.utilitySqAvg /= stats.weightSum;

  return stats;
}
