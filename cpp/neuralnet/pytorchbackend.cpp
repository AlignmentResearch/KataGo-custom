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

  std::vector<torch::jit::IValue> inputs;
  // try a dummy input
  inputs.push_back(torch::ones({1, 22, 19, 19}).to(at::kCUDA));
  inputs.push_back(torch::ones({1, 19}).to(at::kCUDA));
  std::cerr << "CREATED INPUT\n";
  const c10::IValue output = module.forward(inputs);
  std::cerr << "GOT OUTPUT\n";
  const auto& output_tuple = output.toTupleRef().elements();
  std::cerr << "GOT OUTPUT TUPLE\n";
  std::cerr << "len=" << output_tuple.size() << "\n";
  for (const auto& elem : output_tuple) {
    const auto& tup = elem.toTupleRef();
    std::cerr << "  len=" << tup.size() << "\n";
    for (const auto& e2 : tup.elements()) {
      std::cerr << e2.toTensor() << '\n';
    }
  }

  exit(0);
}
