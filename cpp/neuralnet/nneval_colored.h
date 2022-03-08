#ifndef NEURALNET_NNEVAL_COLORED_H_
#define NEURALNET_NNEVAL_COLORED_H_

#include "../neuralnet/nneval.h"

// Like NNEvaluator, but has different logic depending on color.
class NNEvaluatorColored : public NNEvaluator {
 public:
  NNEvaluator* black_nnEval;
  NNEvaluator* white_nnEval;

  NNEvaluatorColored(NNEvaluator* black_nnEval, NNEvaluator* white_nnEval);
  ~NNEvaluatorColored() override;

  NNEvaluatorColored(const NNEvaluatorColored& other) = delete;
  NNEvaluatorColored& operator=(const NNEvaluatorColored& other) = delete;

  std::string getModelName() const override;
  std::string getModelFileName() const override;
  std::string getInternalModelName() const override;
  Logger* getLogger() override;
  bool isNeuralNetLess() const override;
  int getMaxBatchSize() const override;
  int getNumGpus() const override;
  int getNumServerThreads() const override;
  std::set<int> getGpuIdxs() const override;
  int getNNXLen() const override;
  int getNNYLen() const override;
  enabled_t getUsingFP16Mode() const override;
  enabled_t getUsingNHWCMode() const override;

  bool supportsShorttermError() const override;
  Rules getSupportedRules(const Rules& desiredRules, bool& supported) override;

  void clearCache() override;
  void evaluate(
    Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const MiscNNInputParams& nnInputParams,
    NNResultBuf& buf,
    bool skipCache,
    bool includeOwnerMap) override;

  void waitForNextNNEvalIfAny() override;
  void spawnServerThreads() override;
  void killServerThreads() override;
  void setNumThreads(const std::vector<int>& gpuIdxByServerThr) override;

  bool getDoRandomize() const override;
  int getDefaultSymmetry() const override;
  void setDoRandomize(bool b) override;
  void setDefaultSymmetry(int s) override;

  uint64_t numRowsProcessed() const override;
  uint64_t numBatchesProcessed() const override;
  double averageProcessedBatchSize() const override;

  void clearStats() override;
};

#endif  // NEURALNET_NNEVAL_COLORED_H_
