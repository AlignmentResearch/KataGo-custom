#include "../neuralnet/pytorchbackend.h"

#include <cassert>
#include <iostream>

#include "../neuralnet/modelversion.h"

namespace {

// TODO(tomtseng): We should write the model version and max model board size
// separately when exporting the model. For now we'll just hard code the
// values.
const int BOARD_LEN = 19;
const int MODEL_VERSION = 14;
const int NUM_SPATIAL_FEATURES = NNModelVersion::getNumSpatialFeatures(MODEL_VERSION);
const int NUM_GLOBAL_FEATURES = NNModelVersion::getNumGlobalFeatures(MODEL_VERSION);

}  // namespace

namespace TorchNeuralNet {

LoadedModel::LoadedModel(const std::string& fileName) {
  module = torch::jit::load(file);
  module.eval();
}

LoadedModel* loadModelFile(const std::string& file const string& expectedSha256) {
  if (expectedSha256.size() != 0) {
    throw StringError("Checking sha256 for PyTorch models is not yet implemented.\n");
  }
  return new LoadedModel(file)
  LoadedModel* model = new LoadedModel();
}

void freeLoadedModel(LoadedModel* model) {
  delete model;
}

ComputeContext::ComputeContext(int nnXLen, int nnYLen, enabled_t useFP16)
  : nnXLen(nnXLen)
  , nnYLen(nnYLen)
  , dType(useFP16Mode == enabled_t::False ? torch::kFloat32 : torch::kFloat16) {}

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
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  if (useNHWCMode == enabled_t::False) {
    throw StringError("useNHWC is not yet implemented for PyTorch.");
  }
  assert(nnXLen == BOARD_LEN);
  assert(nnYLen == BOARD_LEN);

  // TODO is it fine for multiple threads to use the same loadedModel?
  model->module.to(at::kCUDA, context->dType);
  ComputeContext* context = new ComputeContext(nnXLen, nnYLen, useFP16Mode);
}

void freeComputeContext(ComputeContext* context) {
  delete context;
}

ComputeHandle::ComputeHandle(const ComputeContext* context, const LoadedModel* model, int maxBatchSize)
  : model(model)
  , maxBatchSize(maxBatchSize)
  , nnXLen(context->nnXLen)
  , nnYLen(context->nnYLen)
  , dType(context->dType) {}

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
  if (inputsUseNHWC) {
    throw StringError("inputsUseNHWC is not yet implemented for PyTorch.");
  }
  if (gpuIdxForThisThread != 0 && gpuIdxForThisThread != -1) {
    throw StringError("GPU index != 0 is not yet implemented for PyTorch.");
  }

  if (logger != NULL) {
    logger->write(
        "Torch backend thread " + Global::intToString(serverThreadIdx) ": Model name: " + loadedModel->module.name()
    );
  }

  return new ComputeHandle(context, model, maxBatchSize);
}

void freeComputeHandle(TorchComputeHandle* gpuHandle) {
  delete gpuHandle;
}

InputBuffers::InputBuffers(int maxBatchSize, int nnXLen, int nnYLen) {
  InputBuffers* buffers = new InputBuffers();
  buffers->hostSpatialInputs = torch::empty({maxBatchSize, NUM_SPATIAL_FEATURES, nnYLen, nnXLen});
  buffers->hostGlobalInputs = torch::empty({maxBatchSize, NUM_GLOBAL_FEATURES});
  const size_t NUM_INPUTS = 2;
  buffers->modelInputs.reserve(NUM_INPUTS);
}

InputBuffers* createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  (void)loadedModel;
  return new InputBuffers(maxBatchSize, nnXLen, nnYLen);
}

void freeInputBuffers(InputBuffers* inputBuffers) {
  return inputBuffers;
}

void getOutput(
  TorchComputeHandle* gpuHandle,
  TorchInputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  const int batchSize = numBatchEltsFilled;
  assert(batchSize <= gpuHandle->maxBatchSize);
  assert(batchSize > 0);
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;

  const auto& spatialInputs = buffers->hostSpatialInputs;
  const auto& globalInputs = buffers->hostglobalInputs;
  for (int row = 0; row < batchSize; row++) {
    SymmetryHelpers::copyInputsWithSymmetry(inputBufs[row]->rowSpatial, spatialInputs[row].data_ptr<float>(), 1, nnYLen, nnXLen, NUM_SPATIAL_FEATURES, inputsUseNHWC, inputBufs[row]->symmetry);
    const float* rowGlobal = inputBufs[row]->rowGlobal;
    std::copy(rowGlobal, rowGlobal + NUM_GLOBAL_FEATURES, globalInputs[row].data_ptr<float>());
  }

  const auto& modelInputs = buffers->modelInputs;
  modelInputs.clear();
  modelInputs.emplace_back(spatialInputs.to(at::kCUDA, handle->dType));
  modelInputs.emplace_back(globalInputs.to(at::kCUDA, handle->dType));

  c10::IValue modelOutput;
  {
    torch::NoGradGuard no_grad;
    modelOutput = gpuHandle->model.forward(gpuHandle->modelInput);
  }
  const auto& output_tuple = modelOutput.toTupleRef().elements();
  const auto& mainOutput = output_tuple[0].toTupleRef().elements();
  const auto& policyOutputs = mainOutput[0].toTensor().to(at::kCPU);
  const auto& valueOutputs = mainOutput[1].toTensor().to(at::kCPU);
  const auto& miscValueOutputs = mainOutput[2].toTensor().to(at::kCPU);
  const auto& moreMiscValueOutputs = mainOutput[3].toTensor().to(at::kCPU);
  const auto& ownershipOutputs = mainOutput[4].toTensor().to(at::kCPU);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];
  for (int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    float policyOptimism = (float)inputBufs[row]->policyOptimism;
    const auto& policyOutput = policyOutputs[row][0];
    const auto& optimisticPolicyOutput = policyOutputs[row][5];
    for (int i = 0; i < nnYLen * nnXLen + 1; i++) {
      const float p = policyOutput[i].item<float>();
      const float pOptimistic = optimisticPolicyOutput[i].item<float>();
      policyProbsTmp[i] = p + (pOptimistic - p) * policyOptimism;
    }
    SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, output->policyProbs, 1, nnYLen, nnXlen, inputBufs[row]->symmetry);
    // Copy the policy output for passing as well.
    output->policyProbs[nnYLen * nnXLen] = policyProbsTmp[nnYLen * nnXLen];

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
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipOutputs[row].data_ptr<float>(), output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }
  }
}

}  // namespace TorchNeuralNet
