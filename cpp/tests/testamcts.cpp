// #ifdef AMCTS_TESTS
#include "../tests/testamcts.h"

#include "../dataio/sgf.h"
#include "../program/play.h"
#include "../program/setup.h"
#include "../tests/tests.h"

using namespace std;

// Uncomment to enable debugging
#define DEBUG

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
  testConstPolicies();
  testMCTS(maxVisits, numMovesToSimulate);
  testAMCTS(maxVisits, numMovesToSimulate);
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
      Board::printBoard(cout, bot.rootBoard, Board::NULL_LOC, &bot.rootHistory.moveHistory);

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

      if (bot.rootHistory.isGameFinished) {
        cout << "Game finished after " << midx + 1 << " moves" << endl;
        break;
      }
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
        const int64_t visits = edge->getEdgeVisits();
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
      if (tree.getNodeHistory(node).isGameFinished)
        cout << "omg" << endl;
      //testAssert(!tree.getNodeHistory(node).isGameFinished);
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
}