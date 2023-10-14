#include "../neuralnet/pytorchbackend.h"

#include <cassert>
// #include <chrono> // for benchmarking
#include <iostream>

#include <ATen/autocast_mode.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>

#include "../core/global.h"
#include "../neuralnet/modelversion.h"

using namespace torch::indexing;

// #define TSTART do {  \
//   t1 = std::chrono::high_resolution_clock::now(); \
// } while (0)
// #define TEND(desc) do { \
//   const auto t2 = std::chrono::high_resolution_clock::now(); \
//   std::cout << desc << ' ' << std::chrono::duration<double, std::milli>(t2 - t1).count() << '\n'; \
// } while (0)
#define TSTART do {} while (0)
#define TEND(desc) do {} while (0)


namespace {

// TODO(tomtseng): We should write the model version and max model board size
// separately when exporting the model. For now we'll just hard code the
// values.
constexpr int MAX_BOARD_LEN = 19;
constexpr int MODEL_VERSION = 14;
const int NUM_SPATIAL_FEATURES = NNModelVersion::getNumSpatialFeatures(MODEL_VERSION);
const int NUM_GLOBAL_FEATURES = NNModelVersion::getNumGlobalFeatures(MODEL_VERSION);

}  // namespace

namespace TorchNeuralNet {

LoadedModel::LoadedModel(const std::string& fileName)
  : model(torch::jit::load(fileName)) {
  // We disable optimizations that make the TorchScript KataGo models crash.
  //
  // In particular, what I (tomtseng) saw was that with model b18-s7530m and
  // libtorch v2.0.1 + CUDA 11.8, the second time I executed the model with
  // batch size > 1 on CUDA, the model would crash (e.g., with an illegal CUDA
  // memory access) or return NaNs. The first few calls of a model are when
  // TorchScript performs profiling and optimization, hence why these calls are
  // slower and why this issue occurs on the second call of the model.
  //
  // This issue would also occur when I ran the TorchScript models in Python.
  // The fix in Python was to either turn off the default GPU fuser NVFuser
  // (https://pytorch.org/docs/stable/jit.html#fusion-backends) with
  // `torch._C._jit_set_nvfuser_enabled(False)` or to first run the model twice
  // on the CPU (maybe this makes the model optimize with the default CPU fuser
  // NNC instead of NVFuser, even after moving the model to the GPU?).
  torch::jit::fuser::cuda::setEnabled(false);
}

LoadedModel::LoadedModel(torch::jit::script::Module model_)
  : model(std::move(model_)) {}

LoadedModel* loadModelFile(const std::string& file, const std::string& expectedSha256) {
  if (expectedSha256.size() != 0) {
    throw StringError("Checking sha256 for PyTorch models is not yet implemented.\n");
  }
  return new LoadedModel(file);
}

void freeLoadedModel(LoadedModel* model) {
  delete model;
}

int getModelVersion(const LoadedModel*) {
  return MODEL_VERSION;
}

ComputeContext::ComputeContext(int nnXLen_, int nnYLen_, bool useFP16_)
  : nnXLen(nnXLen_)
  , nnYLen(nnYLen_)
  , useFP16(useFP16_) {
}

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
) {
  (void)gpuIdxs;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;
  if (useNHWCMode != enabled_t::False) {
    throw StringError("useNHWC is not yet implemented for PyTorch.");
  }
  const bool useFP16 = useFP16Mode  != enabled_t::False;
  if (useFP16) {
    // There doesn't seem to be a way to autocast without making it a global
    // setting.
    logger->write("Enabling useFP16 for all TorchScript models");
    at::autocast::set_enabled(true);
  }
  assert(nnXLen <= MAX_BOARD_LEN);
  assert(nnYLen <= MAX_BOARD_LEN);

  ComputeContext* context = new ComputeContext(nnXLen, nnYLen, useFP16);
  return context;
}

void freeComputeContext(ComputeContext* context) {
  delete context;
}

ComputeHandle::ComputeHandle(
    const ComputeContext* context,
    const LoadedModel* model_,
    int maxBatchSize_,
    int gpuIdxForThisThread
)
  : model(model_->model.clone())
  , device(torch::Device(at::kCUDA, gpuIdxForThisThread))
  , maxBatchSize(maxBatchSize_)
  , nnXLen(context->nnXLen)
  , nnYLen(context->nnYLen)
  , useFP16(context->useFP16) {
    model.model.eval();
    model.model.to(device);
  }

ComputeHandle* createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)logger;
  (void)requireExactNNLen;
  (void)serverThreadIdx;

  if (inputsUseNHWC) {
    throw StringError("inputsUseNHWC is not yet implemented for PyTorch.");
  }
  if (gpuIdxForThisThread == -1) {
    gpuIdxForThisThread = 0;
  }

  return new ComputeHandle(context, loadedModel, maxBatchSize, gpuIdxForThisThread);
}

void freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

InputBuffers::InputBuffers(int maxBatchSize)
  : hostSpatialInputs(torch::empty({maxBatchSize, NUM_SPATIAL_FEATURES, MAX_BOARD_LEN, MAX_BOARD_LEN}))
  , hostGlobalInputs(torch::empty({maxBatchSize, NUM_GLOBAL_FEATURES})) {
  const size_t NUM_INPUTS = 2;
  modelInputs.reserve(NUM_INPUTS);
}

InputBuffers* createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  (void)loadedModel;
  (void)nnXLen;
  (void)nnYLen;
  return new InputBuffers(maxBatchSize);
}

void freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

void getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  std::vector<NNOutput*>& outputs
) {
  // std::chrono::time_point tStart = std::chrono::high_resolution_clock::now();
  // std::chrono::time_point t1 = tStart;

  const int batchSize = numBatchEltsFilled;
  assert(batchSize <= gpuHandle->maxBatchSize);
  assert(batchSize > 0);
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  if (nnXLen != MAX_BOARD_LEN || nnYLen != MAX_BOARD_LEN) {
    // The PyTorch model assumes that smaller board sizes are
    // input as following example channel 0 spatial input (signifying which
    // locations are on the board) for a 5x5 input:
    //   1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   ...
    //   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //
    // If nnXLen and nnYLen are set to 5 instead of MAX_BOARD_LEN==19,
    // the inputs get populated instead as
    //   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    //   1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //   ...
    //   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    //
    // The other backends handle this but I haven't investigated how. For now
    // we'll just enforce that nnXLen and nnYLen are MAX_BOARD_LEN. If a user
    // wants to play on a 5x5 board, they should include MAX_BOARD_LEN==19 in
    // the bSizes config parameter, otherwise we throw an exception here.
    throw StringError(Global::strprintf("Board len not yet supported: %d x %d", nnXLen, nnYLen));
  }
  constexpr bool INPUTS_USE_NHWC = false;

  const auto& spatialInputs = inputBuffers->hostSpatialInputs;
  const auto& globalInputs = inputBuffers->hostGlobalInputs;
  for (int row = 0; row < batchSize; row++) {
    TSTART;
    SymmetryHelpers::copyInputsWithSymmetry(inputBufs[row]->rowSpatial, spatialInputs[row].data_ptr<float>(), 1, nnYLen, nnXLen, NUM_SPATIAL_FEATURES, INPUTS_USE_NHWC, inputBufs[row]->symmetry);
    TEND("copy spatial");
    const float* rowGlobal = inputBufs[row]->rowGlobal;
    TSTART;
    std::copy(rowGlobal, rowGlobal + NUM_GLOBAL_FEATURES, globalInputs[row].data_ptr<float>());
    TEND("copy global");
  }

  auto& modelInputs = inputBuffers->modelInputs;
  modelInputs.clear();
  TSTART;
  modelInputs.emplace_back(spatialInputs.index({Slice(0, batchSize)}).to(gpuHandle->device));
  modelInputs.emplace_back(globalInputs.index({Slice(0, batchSize)}).to(gpuHandle->device));
  TEND("slice+mv inputs");

  c10::IValue modelOutput;
  {
    torch::NoGradGuard no_grad;
    TSTART;
    modelOutput = gpuHandle->model.model.forward(modelInputs);
    TEND("forward");
  }
  TSTART;
  const auto& modelOutputs = modelOutput.toTupleRef().elements();
  at::Tensor policyOutputs = modelOutputs[0].toTensor();
  const at::Tensor& valueOutputs = modelOutputs[1].toTensor().to(at::kCPU);
  const at::Tensor& miscValueOutputs = modelOutputs[2].toTensor().to(at::kCPU);
  const at::Tensor& moreMiscValueOutputs = modelOutputs[3].toTensor().to(at::kCPU);
  at::Tensor ownershipOutputs;
  TEND("mv outputs");

  TSTART;
  // We're processing the optimistic policy head, which assumes model version
  // >= 12.
  // TODO(tomtseng): just case on policyOutputs.size(1) to see whether the
  // optimistic head channel exists, and ignore optimism otherwise.
  const bool has_optimistic_policy = policyOutputs.size(1) > 1;
  at::Tensor policies = policyOutputs.index({Slice(), 0});
  at::Tensor optimisticPolicyDiffs;
  if (has_optimistic_policy) {
    optimisticPolicyDiffs = policyOutputs.index({Slice(), 1}).sub_(policies);
  }
  for (int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    const int numPolicyValues = nnYLen * nnXLen + 1;
    at::Tensor policy = policies.index({row, Slice(0, numPolicyValues)});
    if (has_optimistic_policy) {
      const float policyOptimism = (float)inputBufs[row]->policyOptimism;
      // final policy = policy + (policy - optimisticPolicy) * policyoptimism
      policy.add_(optimisticPolicyDiffs.index({row, Slice(0, numPolicyValues)}), policyOptimism);
    }
    policy = policy.to(at::kCPU, torch::kFloat32).contiguous();
    SymmetryHelpers::copyOutputsWithSymmetry(policy.data_ptr<float>(), output->policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    // Copy the policy output for passing as well.
    output->policyProbs[nnYLen * nnXLen] = policy[nnYLen * nnXLen].item<float>();

    const auto& valueOutput = valueOutputs[row];
    output->whiteWinProb = valueOutput[0].item<float>();
    output->whiteLossProb = valueOutput[1].item<float>();
    output->whiteNoResultProb = valueOutput[2].item<float>();

    const auto& miscValueOutput = miscValueOutputs[row];
    output->whiteScoreMean = miscValueOutput[0].item<float>();
    output->whiteScoreMeanSq = miscValueOutput[1].item<float>();
    output->whiteLead = miscValueOutput[2].item<float>();
    output->varTimeLeft = miscValueOutput[3].item<float>();

    const auto& moreMiscValueOutput = moreMiscValueOutputs[row];
    output->shorttermWinlossError = moreMiscValueOutput[0].item<float>();
    output->shorttermScoreError = moreMiscValueOutput[1].item<float>();

    if (output->whiteOwnerMap != NULL) {
      if (!ownershipOutputs.defined()) {
        ownershipOutputs = modelOutputs[4].toTensor().to(at::kCPU);
      }
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipOutputs[row].data_ptr<float>(), output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }
  }
  TEND("process out");

  // std::chrono::time_point tEnd = std::chrono::high_resolution_clock::now();
  // std::cout << "torch getOutput: " << std::chrono::duration<double, std::milli>(tEnd - tStart).count() << "\n\n";
}

}  // namespace TorchNeuralNet
