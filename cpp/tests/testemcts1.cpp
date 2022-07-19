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

void EMCTS1Tests::runAllEMCTS1Tests() {
  cout << "Running EMCTS1 tests" << endl;
  testConstPolicies();
  testEMCTSStats();
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

  auto nnEval1 = get_nneval(CONST_POLICY_1_PATH, cfg, logger, 42);
  auto nnEval2 = get_nneval(CONST_POLICY_2_PATH, cfg, logger, 42);

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

void EMCTS1Tests::testEMCTSStats() {
  ConfigParser cfg("cpp/tests/data/configs/test-emcts1.cfg");
  Logger logger(&cfg, false);

  vector<SearchParams> searchParamss =
      Setup::loadParams(cfg, Setup::SETUP_FOR_OTHER);
  testAssert(searchParamss.size() == 2);

  const SearchParams mctsParams = searchParamss[0];

  auto nnEval = get_nneval(CONST_POLICY_1_PATH, cfg, logger, 42);
  Search bot(mctsParams, nnEval.get(), &logger, "forty-two", nnEval.get());
  resetBot(bot, 9, Rules::getTrompTaylorish());

  // The initial board we perform tests on.
  // It has 8 placed stones that are at the top left corner that look like this:
  //    BBBB.....
  //    .WWWW....
  //    .........
  // Here, dots are empty spaces. It is black's turn to move.
  const unique_ptr<CompactSgf> init_sgf(CompactSgf::parse(
      "(;FF[4]KM[7.5]SZ[19];B[aa];W[bb];B[ba];W[cb];B[ca];W[db];B[da];W[eb])"));
  for (auto& m : init_sgf->moves) {
    bot.makeMove(m.loc, m.pla);
  }

  bot.searchParams.maxVisits = 222;
  {
    // Force bot to weight purely by visits.
    // See
    // https://discord.com/channels/417022162348802048/583775968804732928/698893048049827870
    bot.searchParams.valueWeightExponent = 0;
  }
  bot.runWholeSearchAndGetMove(P_BLACK);

  vector<const SearchNode*> nodes;
  unordered_map<const SearchNode*, vector<const SearchNode*>> adj;
  {  // parse search tree into nicer format
    auto buildtree = [&nodes, &adj](const SearchNode* node,
                                    auto&& dfs) -> void {
      nodes.push_back(node);

      int childrenCap_;  // Not used
      const auto arr = node->getChildren(childrenCap_);
      for (size_t i = 0; i < node->iterateAndCountChildren(); i++) {
        const SearchNode* child = arr[i].getIfAllocated();
        adj[node].push_back(child);
        dfs(child, dfs);
      }
    };
    buildtree(bot.rootNode, buildtree);
  }

  // Test we don't use uncertainty and all weights are 1.
  testAssert(!bot.searchParams.useUncertainty);
  for (auto node : nodes) {
    testAssert(bot.computeNodeWeight(*node) == 1);
  }

  {  // test nnOutputs are as expected
    for (auto node : nodes) {
      testAssert(
          approxEqual(node->getNNOutput()->whiteWinProb,
                      node->nextPla == P_WHITE ? CP1_WIN_PROB : CP1_LOSS_PROB));
      testAssert(
          approxEqual(node->getNNOutput()->whiteLossProb,
                      node->nextPla == P_WHITE ? CP1_LOSS_PROB : CP1_WIN_PROB));
      testAssert(approxEqual(node->getNNOutput()->whiteNoResultProb, 0));
    }
  }

  testAssert(!bot.searchParams.useNoisePruning);
  {  // test backup
    auto check_backup = [&adj](const SearchNode* node,
                               auto&& dfs) -> pair<double, double> {
      double totWinLossDiff = node->getNNOutput()->whiteWinProb -
                              node->getNNOutput()->whiteLossProb;
      double totWeight = 1;
      for (auto child : adj[node]) {
        auto child_tots = dfs(child, dfs);
        totWinLossDiff += child_tots.first;
        totWeight += child_tots.second;
      }

      /*
      cout << '('
           << node->getNNOutput()->whiteWinProb -
                  node->getNNOutput()->whiteLossProb
           << ", " << totWeight << ") ";
      cout << ' ' << node << ": ";
      for (auto child : adj[node]) cout << ' ' << child;
      cout << endl;*/
      const NodeStats stats(node->stats);
      testAssert(totWeight == stats.weightSum);
      testAssert(approxEqual(0, stats.noResultValueAvg));
      testAssert(
          approxEqual(totWinLossDiff / totWeight, stats.winLossValueAvg));

      return {totWinLossDiff, totWeight};
    };
    check_backup(bot.rootNode, check_backup);
  };
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

  auto nnEval1 = get_nneval(CONST_POLICY_1_PATH, cfg, logger, 42);
  auto nnEval2 = get_nneval(CONST_POLICY_2_PATH, cfg, logger, 42);

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

shared_ptr<NNEvaluator> EMCTS1Tests::get_nneval(string modelFile,
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

std::shared_ptr<NNResultBuf> EMCTS1Tests::evaluate(
    shared_ptr<NNEvaluator> nnEval, Board& board, BoardHistory& hist,
    Player nextPla, bool skipCache, bool includeOwnerMap) {
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
