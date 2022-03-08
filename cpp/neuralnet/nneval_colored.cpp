#include "../neuralnet/nneval_colored.h"
#include <string>

using namespace std;

NNEvaluatorColored::NNEvaluatorColored(NNEvaluator* b_nnEval, NNEvaluator* w_nnEval)
  : black_nnEval(b_nnEval), white_nnEval(w_nnEval) {}

NNEvaluatorColored::~NNEvaluatorColored() {
  // We are not responsible for deleting black_nnEval and white_nnEval.
}

string NNEvaluatorColored::getModelName() const {
  return black_nnEval->getModelName() + "__" + white_nnEval->getModelName();
}
string NNEvaluatorColored::getModelFileName() const {
  return black_nnEval->getModelFileName() + "__" + white_nnEval->getModelFileName();
}
string NNEvaluatorColored::getInternalModelName() const {
  return black_nnEval->getInternalModelName() + "__" + white_nnEval->getInternalModelName();
}
Logger* NNEvaluatorColored::getLogger() {
  assert(black_nnEval->getLogger() == white_nnEval->getLogger());
  return black_nnEval->getLogger();
}
bool NNEvaluatorColored::isNeuralNetLess() const {
  return black_nnEval->isNeuralNetLess() || white_nnEval->isNeuralNetLess();
}
int NNEvaluatorColored::getMaxBatchSize() const {
  assert(black_nnEval->getMaxBatchSize() == white_nnEval->getMaxBatchSize());
  return black_nnEval->getMaxBatchSize();
}
int NNEvaluatorColored::getNumGpus() const {
  assert(black_nnEval->getNumGpus() == white_nnEval->getNumGpus());
  return black_nnEval->getNumGpus();
}
int NNEvaluatorColored::getNumServerThreads() const {
  assert(black_nnEval->getNumServerThreads() == white_nnEval->getNumServerThreads());
  return black_nnEval->getNumServerThreads();
}
std::set<int> NNEvaluatorColored::getGpuIdxs() const {
  assert(black_nnEval->getGpuIdxs() == white_nnEval->getGpuIdxs());
  return black_nnEval->getGpuIdxs();
}
int NNEvaluatorColored::getNNXLen() const {
  assert(black_nnEval->getNNXLen() == white_nnEval->getNNXLen());
  return black_nnEval->getNNXLen();
}
int NNEvaluatorColored::getNNYLen() const {
  assert(black_nnEval->getNNYLen() == white_nnEval->getNNYLen());
  return black_nnEval->getNNYLen();
}
enabled_t NNEvaluatorColored::getUsingFP16Mode() const {
  assert(black_nnEval->getUsingFP16Mode() == white_nnEval->getUsingFP16Mode());
  return black_nnEval->getUsingFP16Mode();
}
enabled_t NNEvaluatorColored::getUsingNHWCMode() const {
  assert(black_nnEval->getUsingNHWCMode() == white_nnEval->getUsingNHWCMode());
  return black_nnEval->getUsingNHWCMode();
}
bool NNEvaluatorColored::supportsShorttermError() const {
  assert(black_nnEval->supportsShorttermError() == white_nnEval->supportsShorttermError());
  return black_nnEval->supportsShorttermError();
}
bool NNEvaluatorColored::getDoRandomize() const {
  assert(black_nnEval->getDoRandomize() == white_nnEval->getDoRandomize());
  return black_nnEval->getDoRandomize();
}
int NNEvaluatorColored::getDefaultSymmetry() const {
  assert(black_nnEval->getDefaultSymmetry() == white_nnEval->getDefaultSymmetry());
  return black_nnEval->getDefaultSymmetry();
}

void NNEvaluatorColored::setDoRandomize(bool b) {
  black_nnEval->setDoRandomize(b);
  white_nnEval->setDoRandomize(b);
}
void NNEvaluatorColored::setDefaultSymmetry(int s) {
  black_nnEval->setDefaultSymmetry(s);
  white_nnEval->setDefaultSymmetry(s);
}

Rules NNEvaluatorColored::getSupportedRules(const Rules& desiredRules, bool& supported) {
  assert(
    black_nnEval->getSupportedRules(desiredRules, supported) ==
    white_nnEval->getSupportedRules(desiredRules, supported));
  return black_nnEval->getSupportedRules(desiredRules, supported);
}

uint64_t NNEvaluatorColored::numRowsProcessed() const {
  return black_nnEval->numRowsProcessed() + white_nnEval->numRowsProcessed();
}
uint64_t NNEvaluatorColored::numBatchesProcessed() const {
  return black_nnEval->numBatchesProcessed() + white_nnEval->numBatchesProcessed();
}
double NNEvaluatorColored::averageProcessedBatchSize() const {
  return (double)numRowsProcessed() / (double)numBatchesProcessed();
}

void NNEvaluatorColored::clearStats() {
  black_nnEval->clearStats();
  white_nnEval->clearStats();
}
void NNEvaluatorColored::clearCache() {
  black_nnEval->clearCache();
  white_nnEval->clearCache();
}

void NNEvaluatorColored::setNumThreads(const vector<int>& gpuIdxByServerThr) {
  black_nnEval->setNumThreads(gpuIdxByServerThr);
  white_nnEval->setNumThreads(gpuIdxByServerThr);
}

void NNEvaluatorColored::spawnServerThreads() {
  black_nnEval->spawnServerThreads();
  white_nnEval->spawnServerThreads();
}

void NNEvaluatorColored::killServerThreads() {
  black_nnEval->killServerThreads();
  white_nnEval->killServerThreads();
}

void NNEvaluatorColored::waitForNextNNEvalIfAny() {
  black_nnEval->waitForNextNNEvalIfAny();
  white_nnEval->waitForNextNNEvalIfAny();
}

void NNEvaluatorColored::evaluate(
  Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap) {
  NNEvaluator* next_nnEval = (nextPlayer == P_BLACK) ? black_nnEval : white_nnEval;
  next_nnEval->evaluate(board, history, nextPlayer, nnInputParams, buf, skipCache, includeOwnerMap);
}
