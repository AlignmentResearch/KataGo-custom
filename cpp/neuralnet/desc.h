/* Data descriptors shared between the backends. Supports I/O to simple text
   format generated by the python training. */

#ifndef DESC_H
#define DESC_H

#include <istream>
#include <string>
#include <vector>

#include "../game/rules.h"
#include "../neuralnet/activations.h"

struct ConvLayerDesc {
  std::string name;
  int convYSize;
  int convXSize;
  int inChannels;
  int outChannels;
  int dilationY;
  int dilationX;
  // outC x inC x H x W (col-major order - W has least stride, outC greatest)
  std::vector<float> weights;

  ConvLayerDesc();
  ConvLayerDesc(std::istream& in, bool binaryFloats);
  ConvLayerDesc(ConvLayerDesc&& other);

  ConvLayerDesc(const ConvLayerDesc&) = delete;
  ConvLayerDesc& operator=(const ConvLayerDesc&) = delete;

  ConvLayerDesc& operator=(ConvLayerDesc&& other);
};

struct BatchNormLayerDesc {
  std::string name;
  int numChannels;
  float epsilon;
  bool hasScale;
  bool hasBias;
  std::vector<float> mean;
  std::vector<float> variance;
  std::vector<float> scale;
  std::vector<float> bias;

  BatchNormLayerDesc();
  BatchNormLayerDesc(std::istream& in, bool binaryFloats);
  BatchNormLayerDesc(BatchNormLayerDesc&& other);

  BatchNormLayerDesc(const BatchNormLayerDesc&) = delete;
  BatchNormLayerDesc& operator=(const BatchNormLayerDesc&) = delete;

  BatchNormLayerDesc& operator=(BatchNormLayerDesc&& other);
};

struct ActivationLayerDesc {
  std::string name;
  int activation;

  ActivationLayerDesc();
  ActivationLayerDesc(std::istream& in, int version);
  ActivationLayerDesc(ActivationLayerDesc&& other);

  ActivationLayerDesc(const ActivationLayerDesc&) = delete;
  ActivationLayerDesc& operator=(const ActivationLayerDesc&) = delete;

  ActivationLayerDesc& operator=(ActivationLayerDesc&& other);
};

struct MatMulLayerDesc {
  std::string name;
  int inChannels;
  int outChannels;
  std::vector<float> weights;

  MatMulLayerDesc();
  MatMulLayerDesc(std::istream& in, bool binaryFloats);
  MatMulLayerDesc(MatMulLayerDesc&& other);

  MatMulLayerDesc(const MatMulLayerDesc&) = delete;
  MatMulLayerDesc& operator=(const MatMulLayerDesc&) = delete;

  MatMulLayerDesc& operator=(MatMulLayerDesc&& other);
};

struct MatBiasLayerDesc {
  std::string name;
  int numChannels;
  std::vector<float> weights;

  MatBiasLayerDesc();
  MatBiasLayerDesc(std::istream& in, bool binaryFloats);
  MatBiasLayerDesc(MatBiasLayerDesc&& other);

  MatBiasLayerDesc(const MatBiasLayerDesc&) = delete;
  MatBiasLayerDesc& operator=(const MatBiasLayerDesc&) = delete;

  MatBiasLayerDesc& operator=(MatBiasLayerDesc&& other);
};

struct ResidualBlockDesc {
  std::string name;
  BatchNormLayerDesc preBN;
  ActivationLayerDesc preActivation;
  ConvLayerDesc regularConv;
  BatchNormLayerDesc midBN;
  ActivationLayerDesc midActivation;
  ConvLayerDesc finalConv;

  ResidualBlockDesc();
  ResidualBlockDesc(std::istream& in, int version, bool binaryFloats);
  ResidualBlockDesc(ResidualBlockDesc&& other);

  ResidualBlockDesc(const ResidualBlockDesc&) = delete;
  ResidualBlockDesc& operator=(const ResidualBlockDesc&) = delete;

  ResidualBlockDesc& operator=(ResidualBlockDesc&& other);

  void iterConvLayers(std::function<void(const ConvLayerDesc& dest)> f) const;
};

struct GlobalPoolingResidualBlockDesc {
  std::string name;
  int version;
  BatchNormLayerDesc preBN;
  ActivationLayerDesc preActivation;
  ConvLayerDesc regularConv;
  ConvLayerDesc gpoolConv;
  BatchNormLayerDesc gpoolBN;
  ActivationLayerDesc gpoolActivation;
  MatMulLayerDesc gpoolToBiasMul;
  BatchNormLayerDesc midBN;
  ActivationLayerDesc midActivation;
  ConvLayerDesc finalConv;

  GlobalPoolingResidualBlockDesc();
  GlobalPoolingResidualBlockDesc(std::istream& in, int version, bool binaryFloats);
  GlobalPoolingResidualBlockDesc(GlobalPoolingResidualBlockDesc&& other);

  GlobalPoolingResidualBlockDesc(const GlobalPoolingResidualBlockDesc&) = delete;
  GlobalPoolingResidualBlockDesc& operator=(const GlobalPoolingResidualBlockDesc&) = delete;

  GlobalPoolingResidualBlockDesc& operator=(GlobalPoolingResidualBlockDesc&& other);

  void iterConvLayers(std::function<void(const ConvLayerDesc& dest)> f) const;
};

struct NestedBottleneckResidualBlockDesc {
  std::string name;
  int numBlocks;

  BatchNormLayerDesc preBN;
  ActivationLayerDesc preActivation;
  ConvLayerDesc preConv;

  std::vector<std::pair<int, unique_ptr_void>> blocks;

  BatchNormLayerDesc postBN;
  ActivationLayerDesc postActivation;
  ConvLayerDesc postConv;

  NestedBottleneckResidualBlockDesc();
  NestedBottleneckResidualBlockDesc(std::istream& in, int version, bool binaryFloats);
  NestedBottleneckResidualBlockDesc(NestedBottleneckResidualBlockDesc&& other);

  NestedBottleneckResidualBlockDesc(const NestedBottleneckResidualBlockDesc&) = delete;
  NestedBottleneckResidualBlockDesc& operator=(const NestedBottleneckResidualBlockDesc&) = delete;

  NestedBottleneckResidualBlockDesc& operator=(NestedBottleneckResidualBlockDesc&& other);

  void iterConvLayers(std::function<void(const ConvLayerDesc& dest)> f) const;
};

constexpr int ORDINARY_BLOCK_KIND = 0;
constexpr int GLOBAL_POOLING_BLOCK_KIND = 2;
constexpr int NESTED_BOTTLENECK_BLOCK_KIND = 3;

struct TrunkDesc {
  std::string name;
  int version;
  int numBlocks;
  int trunkNumChannels;
  int midNumChannels;      // Currently every plain residual block must have the same number of mid conv channels
  int regularNumChannels;  // Currently every gpool residual block must have the same number of regular conv hannels
  int gpoolNumChannels;    // Currently every gpooling residual block must have the same number of gpooling conv channels
  ConvLayerDesc initialConv;
  MatMulLayerDesc initialMatMul;
  std::vector<std::pair<int, unique_ptr_void>> blocks;
  BatchNormLayerDesc trunkTipBN;
  ActivationLayerDesc trunkTipActivation;

  TrunkDesc();
  ~TrunkDesc();
  TrunkDesc(std::istream& in, int version, bool binaryFloats);
  TrunkDesc(TrunkDesc&& other);

  TrunkDesc(const TrunkDesc&) = delete;
  TrunkDesc& operator=(const TrunkDesc&) = delete;

  TrunkDesc& operator=(TrunkDesc&& other);

  void iterConvLayers(std::function<void(const ConvLayerDesc& dest)> f) const;
};

struct PolicyHeadDesc {
  std::string name;
  int version;
  ConvLayerDesc p1Conv;
  ConvLayerDesc g1Conv;
  BatchNormLayerDesc g1BN;
  ActivationLayerDesc g1Activation;
  MatMulLayerDesc gpoolToBiasMul;
  BatchNormLayerDesc p1BN;
  ActivationLayerDesc p1Activation;
  ConvLayerDesc p2Conv;
  MatMulLayerDesc gpoolToPassMul;

  PolicyHeadDesc();
  ~PolicyHeadDesc();
  PolicyHeadDesc(std::istream& in, int version, bool binaryFloats);
  PolicyHeadDesc(PolicyHeadDesc&& other);

  PolicyHeadDesc(const PolicyHeadDesc&) = delete;
  PolicyHeadDesc& operator=(const PolicyHeadDesc&) = delete;

  PolicyHeadDesc& operator=(PolicyHeadDesc&& other);

  void iterConvLayers(std::function<void(const ConvLayerDesc& dest)> f) const;
};

struct ValueHeadDesc {
  std::string name;
  int version;
  ConvLayerDesc v1Conv;
  BatchNormLayerDesc v1BN;
  ActivationLayerDesc v1Activation;
  MatMulLayerDesc v2Mul;
  MatBiasLayerDesc v2Bias;
  ActivationLayerDesc v2Activation;
  MatMulLayerDesc v3Mul;
  MatBiasLayerDesc v3Bias;
  MatMulLayerDesc sv3Mul;
  MatBiasLayerDesc sv3Bias;
  ConvLayerDesc vOwnershipConv;

  ValueHeadDesc();
  ~ValueHeadDesc();
  ValueHeadDesc(std::istream& in, int version, bool binaryFloats);
  ValueHeadDesc(ValueHeadDesc&& other);

  ValueHeadDesc(const ValueHeadDesc&) = delete;
  ValueHeadDesc& operator=(const ValueHeadDesc&) = delete;

  ValueHeadDesc& operator=(ValueHeadDesc&& other);

  void iterConvLayers(std::function<void(const ConvLayerDesc& dest)> f) const;
};

struct ModelDesc {
  std::string name;
  int version;
  int numInputChannels;
  int numInputGlobalChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  TrunkDesc trunk;
  PolicyHeadDesc policyHead;
  ValueHeadDesc valueHead;

  ModelDesc();
  ~ModelDesc();
  ModelDesc(std::istream& in, bool binaryFloats);
  ModelDesc(ModelDesc&& other);

  ModelDesc(const ModelDesc&) = delete;
  ModelDesc& operator=(const ModelDesc&) = delete;

  ModelDesc& operator=(ModelDesc&& other);

  void iterConvLayers(std::function<void(const ConvLayerDesc& dest)> f) const;
  int maxConvChannels(int convXSize, int convYSize) const;

  //Loads a model from a file that may or may not be gzipped, storing it in descBuf
  //If expectedSha256 is nonempty, will also verify sha256 of the loaded data.
  static void loadFromFileMaybeGZipped(const std::string& fileName, ModelDesc& descBuf, const std::string& expectedSha256);

  //Return the "nearest" supported ruleset to desiredRules by this model.
  //Fills supported with true if desiredRules itself was exactly supported, false if some modifications had to be made.
  Rules getSupportedRules(const Rules& desiredRules, bool& supported) const;
};

#endif  // #ifndef DESC_H
