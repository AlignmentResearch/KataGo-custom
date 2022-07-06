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

void EMCTS1Tests::runAllEMCTS1Tests() {
  cout << "Running EMCTS1 tests" << endl;
  testConstPolicies();
}

void EMCTS1Tests::testConstPolicies() {
  ConfigParser cfg("cpp/tests/data/configs/test-emcts1.cfg");
  Logger logger(&cfg, false);

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

  // argmax-bot1 will attempt to play in the lexicographically smallest square
  // that is legal. If no such squares are possible, it will pass.
  auto nnEval1 = get_nneval(CONST_POLICY_1_PATH, cfg, logger, 42);

  // argmax-bot2 will always pass.
  auto nnEval2 = get_nneval(CONST_POLICY_2_PATH, cfg, logger, 42);

  {  // Check argmax-bot1 policy
    Search bot1(mctsParams, nnEval1.get(), &logger, "forty-two", nullptr);
    Player pla = P_BLACK;
    for (int i = 0; i < 30; i++) {
      const Loc loc = bot1.runWholeSearchAndGetMove(pla);
      testAssert(loc != Board::PASS_LOC);

      if (i < 19) {
        testAssert(Location::getX(loc, 19) == i);
        testAssert(Location::getY(loc, 19) == 0);
      }

      testAssert(bot1.makeMove(loc, pla));
      pla = getOpp(pla);
    }
  }

  {  // Check argmax-bot1 and argmax-bot2 interaction.
    Search bot1(mctsParams, nnEval1.get(), &logger, "forty-two", nullptr);
    Search bot2(mctsParams, nnEval2.get(), &logger, "forty-two", nullptr);
    testAssert(bot1.rootHistory.rules.multiStoneSuicideLegal);
    testAssert(bot1.rootHistory.rules.koRule == Rules::KO_POSITIONAL);

    Player pla = P_BLACK;
    for (int i = 0; i < 2 * 19 * 19; i++) {
      const Loc loc = i % 2 == 0 ? bot1.runWholeSearchAndGetMove(pla)
                                 : bot2.runWholeSearchAndGetMove(pla);

      if (i % 2 == 0) {  // bot1 (black) move
        if (i / 2 < 19 * 19 - 1) {
          testAssert(Location::getX(loc, 19) == (i / 2) % 19);
          testAssert(Location::getY(loc, 19) == (i / 2) / 19);
        } else {
          testAssert(loc == Board::PASS_LOC);  // Pass due to superko
        }
      } else {  // bot2 (white) move
        testAssert(loc == Board::PASS_LOC);
      }

      {
        auto buf = evaluate(nnEval1, bot1.rootBoard, bot1.rootHistory, pla);
        testAssert(approxEqual(buf->result->whiteWinProb,
                               pla == P_BLACK ? CP1_LOSS_PROB : CP1_WIN_PROB));
        testAssert(approxEqual(buf->result->whiteLossProb,
                               pla == P_BLACK ? CP1_WIN_PROB : CP1_LOSS_PROB));
        testAssert(approxEqual(buf->result->whiteNoResultProb, 0));
      }

      {
        auto buf = evaluate(nnEval2, bot1.rootBoard, bot1.rootHistory, pla);
        testAssert(approxEqual(buf->result->whiteWinProb,
                               pla == P_BLACK ? CP2_LOSS_PROB : CP2_WIN_PROB));
        testAssert(approxEqual(buf->result->whiteLossProb,
                               pla == P_BLACK ? CP2_WIN_PROB : CP2_LOSS_PROB));
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

  unique_ptr<NNEvaluator> ret(nnEval);
  return ret;
}

std::shared_ptr<NNResultBuf> EMCTS1Tests::evaluate(
    shared_ptr<NNEvaluator> nnEval, Board& board, BoardHistory& hist,
    Player nextPla, bool skipCache, bool includeOwnerMap) {
  MiscNNInputParams nnInputParams;
  NNResultBuf* buf = new NNResultBuf();
  nnEval->evaluate(board, hist, nextPla, nnInputParams, *buf, skipCache,
                   includeOwnerMap);
  unique_ptr<NNResultBuf> ret(buf);
  return ret;
}
