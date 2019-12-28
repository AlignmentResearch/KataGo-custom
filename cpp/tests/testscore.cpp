#include "../tests/tests.h"

#include "../neuralnet/nninputs.h"

using namespace std;
using namespace TestCommon;

void Tests::runScoreTests() {
  cout << "Running score and utility tests" << endl;
  ostringstream out;

  auto printScoreStats = [&out](const Board& board, const BoardHistory& hist) {
    out << "Black self komi: " << hist.currentSelfKomi(P_BLACK) << endl;
    out << "White self komi: " << hist.currentSelfKomi(P_WHITE) << endl;

    out << "Winner: " << PlayerIO::colorToChar(hist.winner) << endl;
    double score = hist.finalWhiteMinusBlackScore;
    out << "Final score: " << score << endl;

    double stdev = sqrt(std::max(0.0,ScoreValue::whiteScoreMeanSqOfScoreGridded(score) - score * score));
    double expectedScoreValue = ScoreValue::expectedWhiteScoreValue(score, stdev, 0.0, 2.0, board);
    out << "Score Stdev" << ": " << stdev << endl;
    out << "Score Util Smooth " << ": " << ScoreValue::whiteScoreValueOfScoreSmooth(score, 0.0, 2.0, board) << endl;
    out << "Score Util Gridded" << ": " << expectedScoreValue << endl;
    out << "Score Util GridInv" << ": " << ScoreValue::approxWhiteScoreOfScoreValueSmooth(expectedScoreValue,0.0,2.0,board) << endl;
  };

  {
    const char* name = "On-board even 9x9, komi 7.5";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
.........
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }

  {
    const char* name = "On-board even 9x9, komi 7";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
.........
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }

  {
    const char* name = "On-board black ahead 7 9x9, komi 7";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
xxxxxxx..
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }


  {
    const char* name = "On-board even 5x5, komi 7";

    Board board = Board::parseBoard(5,5,R"%%(
.....
ooooo
.....
xxxxx
.....
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);
    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }

}
