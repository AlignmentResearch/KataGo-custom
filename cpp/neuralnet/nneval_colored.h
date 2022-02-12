#ifndef NEURALNET_NNEVAL_COLORED_H_
#define NEURALNET_NNEVAL_COLORED_H_

#include "../neuralnet/nneval.h"

// Like NNEvaluator, but has different logic depending on color.
class NNEvaluatorColored : public NNEvaluator {
 public:
  NNEvaluator* black_nnEval;
  NNEvaluator* white_nnEval;

  NNEvaluatorColored(NNEvaluator* black_nnEval, NNEvaluator* white_nnEval);
  ~NNEvaluatorColored();

  virtual std::string getModelName() const;
  std::string getModelFileName() const;
  std::string getInternalModelName() const;
  Logger* getLogger();
  bool isNeuralNetLess() const;
  int getMaxBatchSize() const;
  int getNumGpus() const;
  int getNumServerThreads() const;
  std::set<int> getGpuIdxs() const;
  int getNNXLen() const;
  int getNNYLen() const;
  enabled_t getUsingFP16Mode() const;
  enabled_t getUsingNHWCMode() const;
  bool supportsShorttermError() const;
  Rules getSupportedRules(const Rules& desiredRules, bool& supported);

  void clearCache();
  void evaluate(
    Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const MiscNNInputParams& nnInputParams,
    NNResultBuf& buf,
    bool skipCache,
    bool includeOwnerMap);

  void waitForNextNNEvalIfAny();
  void spawnServerThreads();
  void killServerThreads();
  void setNumThreads(const std::vector<int>& gpuIdxByServerThr);

  bool getDoRandomize() const;
  int getDefaultSymmetry() const;
  void setDoRandomize(bool b);
  void setDefaultSymmetry(int s);

  uint64_t numRowsProcessed() const;
  uint64_t numBatchesProcessed() const;
  double averageProcessedBatchSize() const;

  void clearStats();

  void serve(NNServerBuf& buf, Rand& rand, int gpuIdxForThisThread, int serverThreadIdx);
};

#endif  // NEURALNET_NNEVAL_COLORED_H_
