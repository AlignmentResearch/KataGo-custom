#include "../neuralnet/pytorchbackend.h"

#include <cassert>
#include <iostream>

#include <torch/script.h>

#include "../neuralnet/modelversion.h"

void getTorchOutput(int numBatchEltsFilled, NNResultBuf** inputBufs, std::vector<NNOutput*>& outputs) {
  // obviously it's not correct to load the model here, we should do it at some
  // earlier point, but we're just doing basic debugging right now
  torch::jit::script::Module module;
  try {
    module = torch::jit::load("/nas/ucb/ttseng/go_attack/torch-script/traced-test-model.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading Torch Script model\n";
    throw e;
  }
  module.to(at::kCUDA);

  const int batchSize = numBatchEltsFilled;
  const int nnXLen = 19;
  const int nnYLen = 19;
  // need to assert on NHWC
  const bool inputsUseNHWC = false;
  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(14);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(14);
  std::vector<torch::jit::IValue> inputs;
  torch::Tensor spatialInputs = torch::empty({batchSize, numSpatialFeatures, nnYLen, nnXLen});
  torch::Tensor globalInputs = torch::empty({batchSize, numGlobalFeatures});
  for (int row = 0; row < batchSize; row++) {
    SymmetryHelpers::copyInputsWithSymmetry(inputBufs[row]->rowSpatial, spatialInputs[row].data_ptr<float>(), 1, nnYLen, nnXLen, numSpatialFeatures, inputsUseNHWC, inputBufs[row]->symmetry);
    const float* rowGlobal = inputBufs[row]->rowGlobal;
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, globalInputs[row].data_ptr<float>());
  }
  inputs.emplace_back(spatialInputs.to(at::kCUDA));
  inputs.emplace_back(globalInputs.to(at::kCUDA));

  c10::IValue modelOutput;
  {
    torch::NoGradGuard no_grad;
    modelOutput = module.forward(inputs);
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
    for (int i = 0; i < nnXLen * nnYLen + 1; i++) {
      const float p = policyOutput[i].item<float>();
      const float pOptimistic = optimisticPolicyOutput[i].item<float>();
      policyProbsTmp[i] = p + (pOptimistic - p) * policyOptimism;
    }
    SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, output->policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    // Copy the policy output for passing as well.
    output->policyProbs[nnXLen * nnYLen] = policyProbsTmp[nnXLen * nnYLen];

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
