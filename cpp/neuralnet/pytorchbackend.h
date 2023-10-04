// TODO comment
#ifndef NEURALNET_PYTORCHBACKEND_H_
#define NEURALNET_PYTORCHBACKEND_H_

#include <vector>

#include  "../neuralnet/nneval.h"
#include  "../neuralnet/nninputs.h"

void getTorchOutput(int numBatchEltsFilled, NNResultBuf** inputBufs, std::vector<NNOutput*>& outputs);

#endif  // NEURALNET_PYTORCHBACKEND_H_
