// Implements part of nninterface.h using PyTorch's C++ API.
#ifndef NEURALNET_PYTORCHBACKEND_H_
#define NEURALNET_PYTORCHBACKEND_H_

#include <string>
#include <vector>

#include <torch/script.h>

#include  "../neuralnet/nneval.h"
#include  "../neuralnet/nninputs.h"

// TODO(tomtseng): Refactor this PyTorch interface to have less code duplication
// and combine less intrusively with other backends.
//
// nninterface.h is a namespace with global functions. Backends like
// cudabackend.cpp and trtbackend.cpp implement these global functions. It's
// the responsibility of the Makefile to only compile one these implementations
// into the executable. We can't have two backends compiled into one executable, but this
// is fine because only one backend is needed at once.
//
// Adding this PyTorch backend changes the situation since we want to be able to
// combine the PyTorch backend with another backend (e.g., the CUDA
// backend)---we will want to play PyTorch models against older, non-PyTorch
// models (e.g., the cyclic adversary or b40-s1184m), and those non-PyTorch
// models will naturally require a non-PyTorch backend unless we write a
// conversion script to port them to PyTorch.
//
// Currently this file deals with this issue by duplicating nninterface.h in a
// different namespace TorchNeuralNet. The cleaner way to do this is to turn
// nninterface.h into a class interface (let's say the interface is called
// NeuralNet for the purpose of this discussion) rather than a namespace with
// global functions, and then have the different backends subclass and implement
// this interface. This will have less code duplication, and hopefully it's easy
// for the caller the just around a NeuralNet, agnostic to the NeuralNet's
// subclass.
// One tricky part of changing nninterface.h into interface is that the
// interface creates these black-box objects LoadedModel, ComputeContext,
// ComputeHandle, and InputBuffers whose contents are backend-specific, and the
// caller needs to pass these objects into subsequent function calls. But we
// don't want the caller to be able to accidentally pass a CUDA ComputeContext
// into a PyTorch-backend NeuralNet function call. I can think of three ways to
// handle this example of having a ComputeContext of the wrong subclass:
// (1) The brain-dead code-smelly way is to have each backend dynamic_cast its
// ComputeContext to its expected subclass, and we'll get a runtime error if the
// ComputeContext is wrong.
// (2) The clean way is to refactor nninterface.h so that the caller doesn't need
// to pass these black-box objects around and it's instead NeuralNet's
// responsibility to manage them behind the scenes.
//   - If FAR AI were the sole maintainers of this codebase, we should do this.
//   But since we may want to merge upstream changes from lightvector, we may
//   want to opt for an option that minimizes merge conflicts by minimzing
//   changes to nninterface.h.
// (3) Make NeuralNet take the black-box objects as template arguments, and
// subclass implementations would fill in the template arguments with their
// versions of the black-box objects. This would give compile-time type checks
// (vs. (1)'s runtime checks) and might need less refactoring of nninterface.h
// than (2).
//   - I think we shouldn't do this since then every place we use NeuralNet
//   would become templated. This will cause a large code diff since templated
//   functions have their implementations moved from .cpp files to .h files. It
//   will be worse than (2) in terms of maintainability with upstream merges.
namespace TorchNeuralNet {

struct LoadedModel {
  torch::jit::script::Module model;

  LoadedModel(const std::string& fileName);
  LoadedModel(torch::jit::script::Module model);

};
LoadedModel* loadModelFile(const std::string& file, const std::string& expectedSha256);
void freeLoadedModel(LoadedModel* model);
int getModelVersion(const LoadedModel* model);

struct ComputeContext {
  const int nnXLen;
  const int nnYLen;
  const at::ScalarType dType;

  ComputeContext(int nnXLen, int nnYLen, enabled_t useFP16);
};
ComputeContext* createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const std::string& openCLTunerFile,
  const std::string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
);
void freeComputeContext(ComputeContext* context);

struct ComputeHandle {
  LoadedModel model;
  const torch::Device device;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  const at::ScalarType dType;

  ComputeHandle(
      const ComputeContext* context,
      const LoadedModel* model,
      int maxBatchSize,
      int gpuIdxForThisThread
  );
};
ComputeHandle* createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
);
void freeComputeHandle(ComputeHandle* gpuHandle);

struct InputBuffers {
  torch::Tensor hostSpatialInputs;
  torch::Tensor hostGlobalInputs;
  std::vector<torch::jit::IValue> modelInputs;

  InputBuffers(int maxBatchSize, int nnXLen, int nnYLen);
};
InputBuffers* createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen);
void freeInputBuffers(InputBuffers* inputBuffers);

void getOutput(
    ComputeHandle* gpuHandle,
    InputBuffers* inputBuffers,
    int numBatchEltsFilled,
    NNResultBuf** inputBufs,
    std::vector<NNOutput*>& outputs
);

}  // namespace TorchNeuralNet

#endif  // NEURALNET_PYTORCHBACKEND_H_
