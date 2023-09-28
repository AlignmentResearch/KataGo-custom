#include "../neuralnet/pytorchbackend.h"

#include <iostream>

#include <torch/script.h>

// exit()
// TODO remove this
#include <cstdlib>

void getTorchOutput() {
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
  std::cerr << "TORCH SCRIPT MODEL LOADED\n";

  module.to(at::kCUDA);

  std::cerr << "TORCH SCRIPT MODEL MOVED TO GPU\n";

  exit(-1);
}
