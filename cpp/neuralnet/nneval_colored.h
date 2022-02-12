#ifndef NEURALNET_NNEVAL_COLORED_H_
#define NEURALNET_NNEVAL_COLORED_H_

#include "../neuralnet/nneval.h"

// Like NNEvaluator, but has different logic depending on color.
class NNEvaluatorColored : public NNEvaluator {
 public:
  NNEvaluator* black_nnEval;
  NNEvaluator* white_nnEval;

  NNEvaluatorColored(NNEvaluator& black_nnEval, NNEvaluator& white_nnEval);
  NNEvaluatorColored(const NNEvaluatorColored& other) = delete;

  NNEvaluatorColored(const NNEvaluatorColored& other) = delete;
  NNEvaluatorColored& operator=(const NNEvaluatorColored& other) = delete;

  std::string getModelName() const;
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

  // Check if the loaded neural net supports shorttermError fields
  bool supportsShorttermError() const;

  // Return the "nearest" supported ruleset to desiredRules by this model.
  // Fills supported with true if desiredRules itself was exactly supported, false if some modifications had to be made.
  Rules getSupportedRules(const Rules& desiredRules, bool& supported);

  // Clear all entires cached in the table
  void clearCache();

  // Queue a position for the next neural net batch evaluation and wait for it. Upon evaluation, result
  // will be supplied in NNResultBuf& buf, the shared_ptr there can grabbed via std::move if desired.
  // logStream is for some error logging, can be NULL.
  // This function is threadsafe.
  void evaluate(
    Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const MiscNNInputParams& nnInputParams,
    NNResultBuf& buf,
    bool skipCache,
    bool includeOwnerMap);

  // If there is at least one evaluate ongoing, wait until at least one finishes.
  // Returns immediately if there isn't one ongoing right now.
  void waitForNextNNEvalIfAny();

  // Actually spawn threads to handle evaluations.
  // If doRandomize, uses randSeed as a seed, further randomized per-thread
  // If not doRandomize, uses defaultSymmetry for all nn evaluations, unless a symmetry is requested in
  // MiscNNInputParams. This function itself is not threadsafe.
  void spawnServerThreads();

  // Kill spawned server threads and join and free them. This function is not threadsafe, and along with
  // spawnServerThreads should have calls to it and spawnServerThreads singlethreaded.
  void killServerThreads();

  // Set the number of threads and what gpus they use. Only call this if threads are not spawned yet, or have been
  // killed.
  void setNumThreads(const std::vector<int>& gpuIdxByServerThr);

  // These are thread-safe. Setting them in the middle of operation might only affect future
  // neural net evals, rather than any in-flight.
  bool getDoRandomize() const;
  int getDefaultSymmetry() const;
  void setDoRandomize(bool b);
  void setDefaultSymmetry(int s);

  // Some stats
  uint64_t numRowsProcessed() const;
  uint64_t numBatchesProcessed() const;
  double averageProcessedBatchSize() const;

  void clearStats();

  // Helper, for internal use only
  void serve(NNServerBuf& buf, Rand& rand, int gpuIdxForThisThread, int serverThreadIdx);
};

#endif  // NEURALNET_NNEVAL_COLORED_H_
