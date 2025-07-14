// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_LIGHTSAMPLERS_H
#define PBRT_LIGHTSAMPLERS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>  // LightBounds. Should that live elsewhere?
#include <pbrt/util/containers.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/lightsampler_util.h>
#include <pbrt/lightsamplers_constants.h>

#include <curand_kernel.h>


#include <algorithm>
#include <cstdint>
#include <string>
#include <sstream>
#include <queue>
#include <iostream>

namespace pbrt {

// UniformLightSampler Definition
class UniformLightSampler {
  public:
    // UniformLightSampler Public Methods
    UniformLightSampler(pstd::span<const Light> lights, Allocator alloc)
        : lights(lights.begin(), lights.end(), alloc) {}

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (lights.empty())
            return {};
        int lightIndex = std::min<int>(u * lights.size(), lights.size() - 1);
        return SampledLight{lights[lightIndex], 1.f / lights.size()};
    }

    PBRT_CPU_GPU
    Float PMF(Light light) const {
        if (lights.empty())
            return 0;
        return 1.f / lights.size();
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const {
        return Sample(u);
    }

    PBRT_CPU_GPU
    Float PMF(const LightSampleContext &ctx, Light light) const { return PMF(light); }

    PBRT_CPU_GPU inline Float HeuristicPMF(const LightSampleContext &ctx, Light light) const {return PMF(ctx, light);}

    std::string ToString() const { return "UniformLightSampler"; }

    
    PBRT_CPU_GPU
    inline bool isTraining(int pixelIndex, int depth) const { return false; }
    PBRT_CPU_GPU
    inline bool isNetworkEval(int pixelIndex, int depth) const { return false; }

    PBRT_CPU_GPU inline const LightBVHTree* getLightBVHTree() const { return nullptr; }
    PBRT_CPU_GPU inline int getOptBucket(Light light) const { return -1; }
    PBRT_CPU_GPU inline int getNeuralOutputDim() const { return -1; }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext &ctx, Light light, Half* importances) const {
      return false;
    }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext& ctx, Half* importances) const {
      return false;
    }
    PBRT_CPU_GPU inline void GetImportances(int pixelIndex, float* importances) const {}

    void EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size) {}
    Float TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size) {return 0.f;}
    
    void SaveModelWeights(const std::string &filename) const {}

  private:
    // UniformLightSampler Private Members
    pstd::vector<Light> lights;
};

// PowerLightSampler Definition
class PowerLightSampler {
  public:
    // PowerLightSampler Public Methods
    PowerLightSampler(pstd::span<const Light> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (!aliasTable.size())
            return {};
        Float pmf;
        int lightIndex = aliasTable.Sample(u, &pmf);
        return SampledLight{lights[lightIndex], pmf};
    }

    PBRT_CPU_GPU
    Float PMF(Light light) const {
        if (!aliasTable.size())
            return 0;
        return aliasTable.PMF(lightToIndex[light]);
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const {
        return Sample(u);
    }

    PBRT_CPU_GPU
    Float PMF(const LightSampleContext &ctx, Light light) const { return PMF(light); }

    PBRT_CPU_GPU inline Float HeuristicPMF(const LightSampleContext &ctx, Light light) const {return PMF(ctx, light);}

    std::string ToString() const;

    PBRT_CPU_GPU
    inline bool isTraining(int pixelIndex, int depth) const { return false; }
    PBRT_CPU_GPU
    inline bool isNetworkEval(int pixelIndex, int depth) const { return false; }

    PBRT_CPU_GPU inline const LightBVHTree* getLightBVHTree() const { return nullptr; }
    PBRT_CPU_GPU inline int getOptBucket(Light light) const { return -1; }
    PBRT_CPU_GPU inline int getNeuralOutputDim() const { return -1; }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext &ctx, Light light, Half* importances) const {
      return false;
    }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext& ctx, Half* importances) const {
      return false;
    }
    PBRT_CPU_GPU inline void GetImportances(int pixelIndex, float* importances) const {}

    void EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size) {}
    Float TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size) {return 0.f;}
    
    void SaveModelWeights(const std::string &filename) const {}

  private:
    // PowerLightSampler Private Members
    pstd::vector<Light> lights;
    HashMap<Light, size_t> lightToIndex;
    AliasTable aliasTable;
};

// LightCut Definitions
struct LightCutNode {
  unsigned int nodeIndex;
  unsigned int depth;
  uint32_t bitTrail; 
  // Float bsdfVal;
  // Float geometryTerm;

  PBRT_CPU_GPU
  LightCutNode(unsigned int nodeIndex, unsigned int depth, uint32_t bitTrail): 
    nodeIndex(nodeIndex), depth(depth), bitTrail(bitTrail) {}

  PBRT_CPU_GPU
  LightCutNode(): nodeIndex(0), depth(0), bitTrail(0) {}
};
template <uint32_t N>
struct LightCut {
  PBRT_CPU_GPU
  LightCut(): size(0) {};

  PBRT_CPU_GPU
  LightCut(const LightCut &other) : size(other.size) {
      for (uint32_t i = 0; i < size; ++i) {
          cut[i] = other.cut[i];
          cutWeights[i] = other.cutWeights[i];
      }
  }

  PBRT_CPU_GPU
  inline void Append(unsigned int nodeIndex, unsigned int depth, uint32_t bitTrail, Float weight){
    if (size >= N) return;
    cut[size] = LightCutNode(nodeIndex, depth, bitTrail);
    cutWeights[size] = weight;
    size++;
  }

  PBRT_CPU_GPU
  inline void Replace(unsigned int index, unsigned int nodeIndex, unsigned int depth, uint32_t bitTrail, Float weight){
    if (index >= size) return;
    cutWeights[index] = weight;
    cut[index] = LightCutNode(nodeIndex, depth, bitTrail);
  }

  PBRT_CPU_GPU
  inline void UpdateWeight(unsigned int index, Float weight){
    if (index >= size) return;
    cutWeights[index] = weight;
  }

  PBRT_CPU_GPU
  inline int Find(uint32_t bitTrail) const {
    for (uint32_t i = 0; i < size; ++i){
      // Check if last depth bits of bitTrail match the bitTrail of the cut node
      const uint32_t shiftedDepth = (1 << cut[i].depth) - 1;
      if ((cut[i].bitTrail & shiftedDepth) == (bitTrail & shiftedDepth))
        return i;
    }
    return -1;
  }

  PBRT_CPU_GPU
  inline LightCutNode operator[](uint32_t index) const {return cut[index];}

  PBRT_CPU_GPU
  inline Float Weight(uint32_t index) const {return cutWeights[index];}

  PBRT_CPU_GPU
  uint32_t Size() const {return size;}

  PBRT_CPU_GPU
  static uint32_t Capacity() {return N;}

  PBRT_CPU_GPU
  inline LightCutNode Sample(Float u, Float* pmf, Float* uRemapped) const {
    if (size == 0) return LightCutNode();
    int index = SampleDiscrete(cutWeights, u, pmf, uRemapped);
    return cut[index];
  }


    PBRT_CPU_GPU
    inline Float PMF(uint32_t bitTrail, LightCutNode* cutNode) const {
    for (uint32_t i = 0; i < size; ++i){
      // Check if last depth bits of bitTrail match the bitTrail of the cut node
      const uint32_t shiftedDepth = (1 << cut[i].depth) - 1;
      if ((cut[i].bitTrail & shiftedDepth) == (bitTrail & shiftedDepth)){
        if (cutNode) *cutNode = cut[i];
        return cutWeights[i];
      }
    }
    LOG_ERROR("BitTrail not found in LightCut");
    return -1.f;
  }

  PBRT_CPU_GPU inline void NormalizeWeights() {
    Float sumWeights = 0;
    for (uint32_t i = 0; i < size; ++i){
      sumWeights += cutWeights[i];
    }
    for (uint32_t i = 0; i < size; ++i){
      cutWeights[i] /= sumWeights;
    }
  }

  PBRT_CPU_GPU
  inline Float MaxWeight() const {
    Float maxWeight = 0;
    for (uint32_t i = 0; i < size; ++i){
      maxWeight = std::max(maxWeight, cutWeights[i]);
    }
    return maxWeight;
  }

  PBRT_CPU_GPU
  inline Float MeanWeight() const {
    Float meanWeight = 0;
    for (uint32_t i = 0; i < size; ++i){
      meanWeight += cutWeights[i];
    }
    return meanWeight / size;
  }

  PBRT_CPU_GPU
  LightCut &operator=(const LightCut &other) {
      if (this == &other) return *this;
      size = other.size;
      for (uint32_t i = 0; i < size; ++i) {
          cut[i] = other.cut[i];
          cutWeights[i] = other.cutWeights[i];
      }
      return *this;
  }
  

  PBRT_CPU_GPU
  inline void toChar(const LightBVHTree& lightTree, char* outChar) const {
    for (uint32_t i = 0; i < size; i++){
      LightCutNode node = cut[i];
      const LightBVHNode& lightBVHNode = lightTree.nodes[node.nodeIndex];
      outChar[i*6] = (char)node.depth + '0';
      outChar[i*6 + 1] = ' ';
      outChar[i*6 + 2] = (char)lightBVHNode.isLeaf + '0';
      outChar[i*6 + 3] = ' ';
      outChar[i*6 + 4] = '|';
      outChar[i*6 + 5] = ' ';
    }
    outChar[size*6 - 1] = '\0';
  }

  template <typename LightTreeType>
  inline std::string ToString(const LightTreeType& lightTree) const {
    std::stringstream ss;
    for (uint32_t i = 0; i < size; i++){
      LightCutNode node = cut[i];
      const LightBVHNode& lightBVHNode = lightTree.nodes[node.nodeIndex];
      ss << node.nodeIndex << " " << lightBVHNode.isLeaf << " | ";
      // ss << node.depth << " " << lightBVHNode.lightBounds.Phi() << " " << lightBVHNode.isLeaf << " | ";
      // ss << lightBVHNode.lightBounds.Phi() << " " << lightBVHNode.isLeaf << " | ";
    }
    return ss.str();
  }

  private:
    LightCutNode cut[N];
    Float cutWeights[N];
    uint32_t size;
};

template <int N>
class LightTreeSamplingResults {
  public:
    PBRT_CPU_GPU
    LightTreeSamplingResults() { ResetAll(); }

    PBRT_CPU_GPU
    inline void Append(uint32_t index, Float val, Float val2) {
        CHECK_LT(index, N);

        uint32_t t = sampleCount[index];
        Float delta = val - E[index];
        E[index] += delta / (Float(t + 1));
        Float delta2 = val - E[index];
        m2c[index] += delta * delta2;
        E2[index] = (t / Float(t + 1)) * (E2[index] - val2) + val2;

        ++sampleCount[index];
    }

    PBRT_CPU_GPU
    inline void Append(uint32_t index, const RGB& RGB, Float pdf) {
        CHECK_LT(index, N);

        uint32_t t = sampleCount[index];
        Float val = RGB.Average() / pdf;
        Float val2 = val * RGB.Average();
        Float delta = val - E[index];
        E[index] += delta / (Float(t + 1));
        Float delta2 = val - E[index];
        m2c[index] += delta * delta2;
        E2[index] = (t / Float(t + 1)) * (E2[index] - val2) + val2;
        ++sampleCount[index];
    }

    PBRT_CPU_GPU inline Float Exp(uint32_t index) const {
        CHECK_LT(index, N);
        return E[index];
    }

    PBRT_CPU_GPU inline Float Exp2(uint32_t index) const {
        CHECK_LT(index, N);
        return E2[index];
    }

    PBRT_CPU_GPU inline Float Var(uint32_t index) const {
        CHECK_LT(index, N);
        return m2c[index] / Float(sampleCount[index] + 1);
    }

    PBRT_CPU_GPU inline uint32_t SampleCount(uint32_t index) const {
        CHECK_LT(index, N);
        return sampleCount[index];
    }

    PBRT_CPU_GPU
    inline void Reset(uint32_t index) {
        CHECK_LT(index, N);
        E[index] = 0.f;
        E2[index] = 0.f;
        m2c[index] = 0.f;
        sampleCount[index] = 0u;
    }

    PBRT_CPU_GPU
    inline void ResetAll() {
        for (int i = 0; i < N; i++) {
            E[i] = 0.f;
            E2[i] = 0.f;
            m2c[i] = 0.f;
            sampleCount[i] = 0u;
        }
    }

    PBRT_CPU_GPU
    uint32_t MemoryCost() const {
        return N * (2 * sizeof(Float) + sizeof(uint32_t));
    }

  private:
    Float E[N], E2[N], m2c[N];
    uint32_t sampleCount[N];
};


// CompactLightBounds Definition
class CompactLightBounds {
  public:
    // CompactLightBounds Public Methods
    CompactLightBounds() = default;

    PBRT_CPU_GPU
    CompactLightBounds(const LightBounds &lb, const Bounds3f &allb)
        : w(Normalize(lb.w)),
          phi(lb.phi),
          qCosTheta_o(QuantizeCos(lb.cosTheta_o)),
          qCosTheta_e(QuantizeCos(lb.cosTheta_e)),
          twoSided(lb.twoSided) {
        // Quantize bounding box into _qb_
        for (int c = 0; c < 3; ++c) {
            qb[0][c] =
                pstd::floor(QuantizeBounds(lb.bounds[0][c], allb.pMin[c], allb.pMax[c]));
            qb[1][c] =
                pstd::ceil(QuantizeBounds(lb.bounds[1][c], allb.pMin[c], allb.pMax[c]));
        }
    }

    std::string ToString() const;
    std::string ToString(const Bounds3f &allBounds) const;

    PBRT_CPU_GPU
    inline bool isTraining(int pixelIndex, int depth) const { return false; }
    PBRT_CPU_GPU
    inline bool isNetworkEval(int pixelIndex, int depth) const { return false; }

    PBRT_CPU_GPU
    bool TwoSided() const { return twoSided; }
    PBRT_CPU_GPU
    Float CosTheta_o() const { return 2 * (qCosTheta_o / 32767.f) - 1; }
    PBRT_CPU_GPU
    Float CosTheta_e() const { return 2 * (qCosTheta_e / 32767.f) - 1; }
    PBRT_CPU_GPU
    OctahedralVector W() const { return w; }
    PBRT_CPU_GPU
    Float Phi() const { return phi; }

    PBRT_CPU_GPU
    Bounds3f Bounds(const Bounds3f &allb) const {
        return {Point3f(Lerp(qb[0][0] / 65535.f, allb.pMin.x, allb.pMax.x),
                        Lerp(qb[0][1] / 65535.f, allb.pMin.y, allb.pMax.y),
                        Lerp(qb[0][2] / 65535.f, allb.pMin.z, allb.pMax.z)),
                Point3f(Lerp(qb[1][0] / 65535.f, allb.pMin.x, allb.pMax.x),
                        Lerp(qb[1][1] / 65535.f, allb.pMin.y, allb.pMax.y),
                        Lerp(qb[1][2] / 65535.f, allb.pMin.z, allb.pMax.z))};
    }

  private:
    // CompactLightBounds Private Methods
    PBRT_CPU_GPU
    static unsigned int QuantizeCos(Float c) {
        CHECK(c >= -1 && c <= 1);
        return pstd::floor(32767.f * ((c + 1) / 2));
    }

    PBRT_CPU_GPU
    static Float QuantizeBounds(Float c, Float min, Float max) {
        CHECK(c >= min && c <= max);
        if (min == max)
            return 0;
        return 65535.f * Clamp((c - min) / (max - min), 0, 1);
    }

    // CompactLightBounds Private Members
    OctahedralVector w;
    Float phi = 0;
    struct {
        unsigned int qCosTheta_o : 15;
        unsigned int qCosTheta_e : 15;
        unsigned int twoSided : 1;
    };
    uint16_t qb[2][3];
};

// LightBVHNode Definition
struct alignas(32) LightBVHNode {
    // LightBVHNode Public Methods
    LightBVHNode() = default;

    PBRT_CPU_GPU
    static LightBVHNode MakeLeaf(unsigned int lightIndex, const CompactLightBounds &cb) {
        return LightBVHNode{cb, {lightIndex, 1}};
    }

    PBRT_CPU_GPU
    static LightBVHNode MakeInterior(unsigned int child1Index,
                                     const CompactLightBounds &cb) {
        return LightBVHNode{cb, {child1Index, 0}};
    }

    std::string ToString() const;

    // LightBVHNode Public Members
    CompactLightBounds lightBounds;
    struct {
        unsigned int childOrLightIndex : 31;
        unsigned int isLeaf : 1;
    };
};

class LightBVHTree {
  public:
    // LightTree Public Methods
    LightBVHTree(pstd::span<const Light> lights, Allocator alloc);
  
    pstd::vector<Light> bvhLights;
    pstd::vector<Light> infiniteLights;
    Bounds3f allLightBounds;
    pstd::vector<LightBVHNode> nodes;
    HashMap<Light, uint32_t> lightToBitTrail;
    HashMap<Light, uint32_t> lightToIndex;
    HashMap<Light, uint32_t> lightToNodeIndex;

    PBRT_CPU_GPU 
    inline bool bitTrail(int pixelIndex, Light light, uint32_t* bitTrail) const { 
      bool return_value = lightToBitTrail.HasKey(light);
      if (return_value && bitTrail)
        *bitTrail = lightToBitTrail[light];
      return return_value;
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (bvhLights.empty() && infiniteLights.empty())
            return {};
        size_t totalSize = bvhLights.size() + infiniteLights.size();
        int lightIndex = std::min<int>(u * totalSize, totalSize - 1);
        if (lightIndex < infiniteLights.size())
            return SampledLight{infiniteLights[lightIndex],
                                1.f / totalSize};
        return SampledLight{bvhLights[lightIndex], 1.f / totalSize};
    }

    PBRT_CPU_GPU
    Float PMF(Light light) const {
        if (bvhLights.empty() && infiniteLights.empty())
            return 0;
        size_t totalSize = bvhLights.size() + infiniteLights.size();
        return 1.f / totalSize;
    }

    PBRT_CPU_GPU
    bool RightIndices(int32_t* indices) const {
      if (nodes.empty())
        return false;
      
      for (int32_t i = 0; i < static_cast<int32_t>(nodes.size()); i++) {
        if (nodes[i].isLeaf) {
          indices[i] = -1;
        } else {
          indices[i] = static_cast<int32_t>(nodes[i].childOrLightIndex);
        }
      }
      return true;
    }

    PBRT_CPU_GPU
    bool BitTrails(int32_t* bitTrails) const {
      if (bvhLights.empty())
        return false;
      
      for (size_t i = 0; i < bvhLights.size(); i++) {
        if (!lightToBitTrail.HasKey(bvhLights[i])){
          LOG_VERBOSE("Light %u not found in lightToBitTrail. Returning false.", i);
          return false;
        }
          
        bitTrails[i] = static_cast<int32_t>(lightToBitTrail[bvhLights[i]]);
      }
      return true;
    }

    template <int N>
    LightCut<N> GenerateLightCut(Float thresholdP, int maxSize,
      std::function<Float(const LightBVHNode&)>&& weightingFunction) const {
      LightCut<N> lightCut;

      std::vector<LightCutNode> nodeList;
      nodeList.reserve(maxSize);

      auto compare = [&, this](const LightCutNode& a, const LightCutNode& b) {
        return weightingFunction(nodes[a.nodeIndex]) < weightingFunction(nodes[b.nodeIndex]);
      };
      std::priority_queue<LightCutNode, 
        std::vector<LightCutNode>, 
        decltype(compare)> Q(compare);
      
      Float weightSum = weightingFunction(nodes[0]);
      // Q.push(0);
      Q.push(LightCutNode(0, 0, 0));
      int currentSize = 1;
      while(!Q.empty() && currentSize < maxSize){
        LightCutNode currNode = Q.top();
        Q.pop();
        if (nodes[currNode.nodeIndex].isLeaf || weightingFunction(nodes[currNode.nodeIndex]) < thresholdP * weightSum){
          nodeList.push_back(currNode);
        } else{
          ++currentSize;
          const unsigned int secondChildOffset = nodes[currNode.nodeIndex].childOrLightIndex;
          Q.push(LightCutNode(currNode.nodeIndex + 1, currNode.depth + 1, currNode.bitTrail));
          Q.push(LightCutNode(secondChildOffset, currNode.depth + 1, currNode.bitTrail | (1 << currNode.depth)));
          weightSum = (weightSum - weightingFunction(nodes[currNode.nodeIndex])) + 
            weightingFunction(nodes[currNode.nodeIndex + 1]) + weightingFunction(nodes[secondChildOffset]);
        }
      }

      while(!Q.empty()){
        nodeList.push_back(Q.top());
        Q.pop();
      }

      int cutSize = nodeList.size();

      std::sort(nodeList.begin(), nodeList.end(), [](const LightCutNode& a, const LightCutNode& b){
        return a.nodeIndex < b.nodeIndex;
      });
      for (int i = 0; i < cutSize; ++i){
        LightCutNode currNode = nodeList[i];
        Float weight = weightingFunction(nodes[currNode.nodeIndex]);
        lightCut.Append(currNode.nodeIndex, currNode.depth, currNode.bitTrail, weight);
      }
      lightCut.NormalizeWeights();

      return lightCut;
    }

    std::string ToString() const {
      // Compute BFS print on the tree
      std::stringstream ss;
      
      // Initialize queue for BFS
      std::queue<std::pair<int, int>> q;
      q.push({0, 0});
      int depth = 0;
      int maxdepth = 0;
      while (!q.empty()) {
        auto [nodeIndex, nodeDepth] = q.front();
        q.pop();
        if (nodeDepth > depth) {
          ss << std::endl;
          depth = nodeDepth;
        }
        if (nodeDepth > maxdepth) {
          maxdepth = nodeDepth;
        }
        const LightBVHNode &node = nodes[nodeIndex];
        // ss << node.lightBounds.Phi() << " isLeaf: " << node.isLeaf << " |";
        if (node.isLeaf){
          const Light& light = bvhLights[node.childOrLightIndex];
          ss << nodeIndex << ": " << node.lightBounds.Phi() << " Leaf BT: " << lightToBitTrail[light] << " | ";
        }
        else{
          ss << nodeIndex << ": " << node.lightBounds.Phi() << " Not Leaf | ";
          q.push({nodeIndex + 1, nodeDepth + 1});
          q.push({node.childOrLightIndex, nodeDepth + 1});
        }
      }
      ss << "maxdepth: " << maxdepth << "\n";
      
      return ss.str();
    };

    PBRT_CPU_GPU inline Float pInfinite() const { return pInf; }

  private:
    std::pair<int, LightBounds> buildBVH(
      std::vector<std::pair<int, LightBounds>> &bvhLights, int start, int end,
      uint32_t bitTrail, int depth);

    Float EvaluateCost(const LightBounds &b, const Bounds3f &bounds, int dim) const {
        // Evaluate direction bounds measure for _LightBounds_
        Float theta_o = std::acos(b.cosTheta_o), theta_e = std::acos(b.cosTheta_e);
        Float theta_w = std::min(theta_o + theta_e, Pi);
        Float sinTheta_o = SafeSqrt(1 - Sqr(b.cosTheta_o));
        Float M_omega = 2 * Pi * (1 - b.cosTheta_o) +
                        Pi / 2 *
                            (2 * theta_w * sinTheta_o - std::cos(theta_o - 2 * theta_w) -
                             2 * theta_o * sinTheta_o + b.cosTheta_o);

        // Return complete cost estimate for _LightBounds_
        Float Kr = MaxComponentValue(bounds.Diagonal()) / bounds.Diagonal()[dim];
        return b.phi * M_omega * Kr * b.bounds.SurfaceArea();
    }

    Float pInf;
};

// BVHLightSampler Definition
class BVHLightSampler {
  public:
    // BVHLightSampler Public Methods
    BVHLightSampler(pstd::span<const Light> lights, Allocator alloc) : lightTree(lights, alloc) {}

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const {
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(lightTree.infiniteLights.size()) /
                          Float(lightTree.infiniteLights.size() + (lightTree.nodes.empty() ? 0 : 1));

        if (u < pInfinite) {
            // Sample infinite lights with uniform probability
            u /= pInfinite;
            int index =
                std::min<int>(u * lightTree.infiniteLights.size(), lightTree.infiniteLights.size() - 1);
            Float pmf = pInfinite / lightTree.infiniteLights.size();
            return SampledLight{lightTree.infiniteLights[index], pmf};

        } else {
            // Traverse light BVH to sample light
            if (lightTree.nodes.empty())
                return {};
            // Declare common variables for light BVH traversal
            Point3f p = ctx.p();
            Normal3f n = ctx.ns;
            u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);
            int nodeIndex = 0;
            Float pmf = 1 - pInfinite;

            while (true) {
                // Process light BVH node for light sampling
                LightBVHNode node = lightTree.nodes[nodeIndex];
                if (!node.isLeaf) {
                    // Compute light BVH child node importances
                    const LightBVHNode *children[2] = {&lightTree.nodes[nodeIndex + 1],
                                                       &lightTree.nodes[node.childOrLightIndex]};
                    Float ci[2] = {
                        Importance(p, n, lightTree.allLightBounds, children[0]->lightBounds),
                        Importance(p, n, lightTree.allLightBounds, children[1]->lightBounds)};
                    if (ci[0] == 0 && ci[1] == 0)
                        return {};

                    // Randomly sample light BVH child node
                    Float nodePMF;
                    int child = SampleDiscrete(ci, u, &nodePMF, &u);
                    pmf *= nodePMF;
                    nodeIndex = (child == 0) ? (nodeIndex + 1) : node.childOrLightIndex;

                } else {
                    // Confirm light has nonzero importance before returning light sample
                    if (nodeIndex > 0 ||
                        Importance(p, n, lightTree.allLightBounds, node.lightBounds) > 0)
                        return SampledLight{lightTree.bvhLights[node.childOrLightIndex], pmf};
                    
                    return {};
                }
            }
        }
    }

    PBRT_CPU_GPU
    Float PMF(const LightSampleContext &ctx, Light light) const {
        // Handle infinite _light_ PMF computation
        if (!lightTree.lightToBitTrail.HasKey(light))
            return 1.f / (lightTree.infiniteLights.size() + (lightTree.nodes.empty() ? 0 : 1));

        // Initialize local variables for BVH traversal for PMF computation
        uint32_t bitTrail = lightTree.lightToBitTrail[light];
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(lightTree.infiniteLights.size()) /
                          Float(lightTree.infiniteLights.size() + (lightTree.nodes.empty() ? 0 : 1));

        Float pmf = 1 - pInfinite;
        int nodeIndex = 0;

        // Compute light's PMF by walking down tree nodes to the light
        while (true) {
            const LightBVHNode *node = &lightTree.nodes[nodeIndex];
            if (node->isLeaf) {
                DCHECK_EQ(light, lightTree.bvhLights[node->childOrLightIndex]);
                return pmf;
            }
            // Compute child importances and update PMF for current node
            const LightBVHNode *child0 = &lightTree.nodes[nodeIndex + 1];
            const LightBVHNode *child1 = &lightTree.nodes[node->childOrLightIndex];
            Float ci[2] = {Importance(p, n, lightTree.allLightBounds, child0->lightBounds),
                           Importance(p, n, lightTree.allLightBounds, child1->lightBounds)};
            DCHECK_GT(ci[bitTrail & 1], 0);
            pmf *= ci[bitTrail & 1] / (ci[0] + ci[1]);

            // Use _bitTrail_ to find next node index and update its value
            nodeIndex = (bitTrail & 1) ? node->childOrLightIndex : (nodeIndex + 1);
            bitTrail >>= 1;
        }
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        return lightTree.Sample(u);
    }

    PBRT_CPU_GPU
    Float PMF(Light light) const {
        return lightTree.PMF(light);
    }

    PBRT_CPU_GPU inline Float HeuristicPMF(const LightSampleContext &ctx, Light light) const {return PMF(ctx, light);}

    std::string ToString() const;

    PBRT_CPU_GPU
    inline bool isTraining(int pixelIndex, int depth) const { return false; }
    PBRT_CPU_GPU
    inline bool isNetworkEval(int pixelIndex, int depth) const { return false; }

    PBRT_CPU_GPU inline const LightBVHTree* getLightBVHTree() const { return &lightTree; }
    PBRT_CPU_GPU inline int getOptBucket(Light light) const { return -1; }
    PBRT_CPU_GPU inline int getNeuralOutputDim() const { return -1; }

    PBRT_CPU_GPU
    static inline Float Importance(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb) {
        Bounds3f bounds = cb.Bounds(allb);
        Float cosTheta_o = cb.CosTheta_o(), cosTheta_e = cb.CosTheta_e();
        // Return importance for light bounds at reference point
        // Compute clamped squared distance to reference point
        Point3f pc = (bounds.pMin + bounds.pMax) / 2;
        Float d2 = DistanceSquared(p, pc);
        d2 = std::max(d2, Length(bounds.Diagonal()) / 2);

        // Define cosine and sine clamped subtraction lambdas
        auto cosSubClamped = [](Float sinTheta_a, Float cosTheta_a, Float sinTheta_b,
                                Float cosTheta_b) -> Float {
            if (cosTheta_a > cosTheta_b)
                return 1;
            return cosTheta_a * cosTheta_b + sinTheta_a * sinTheta_b;
        };

        auto sinSubClamped = [](Float sinTheta_a, Float cosTheta_a, Float sinTheta_b,
                                Float cosTheta_b) -> Float {
            if (cosTheta_a > cosTheta_b)
                return 0;
            return sinTheta_a * cosTheta_b - cosTheta_a * sinTheta_b;
        };

        // Compute sine and cosine of angle to vector _w_, $\theta_\roman{w}$
        Vector3f wi = Normalize(p - pc);
        Float cosTheta_w = Dot(Vector3f(cb.W()), wi);
        if (cb.TwoSided())
            cosTheta_w = std::abs(cosTheta_w);
        Float sinTheta_w = SafeSqrt(1 - Sqr(cosTheta_w));

        // Compute $\cos\,\theta_\roman{\+b}$ for reference point
        Float cosTheta_b = BoundSubtendedDirections(bounds, p).cosTheta;
        Float sinTheta_b = SafeSqrt(1 - Sqr(cosTheta_b));

        // Compute $\cos\,\theta'$ and test against $\cos\,\theta_\roman{e}$
        Float sinTheta_o = SafeSqrt(1 - Sqr(cosTheta_o));
        Float cosTheta_x = cosSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
        Float sinTheta_x = sinSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, cosTheta_o);
        Float cosThetap = cosSubClamped(sinTheta_x, cosTheta_x, sinTheta_b, cosTheta_b);
        if (cosThetap <= cosTheta_e)
            return 0;

        // Return final importance at reference point
        Float importance = cb.Phi() * cosThetap / d2;
        DCHECK_GE(importance, -1e-3);
        // Account for $\cos\theta_\roman{i}$ in importance at surfaces
        if (n != Normal3f(0, 0, 0)) {
            Float cosTheta_i = AbsDot(wi, n);
            Float sinTheta_i = SafeSqrt(1 - Sqr(cosTheta_i));
            Float cosThetap_i =
                cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
            importance *= cosThetap_i;
        }

        importance = std::max<Float>(importance, 0);
        return importance;
    }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext &ctx, Light light, Half* importances) const {
        if (!importances)
          return false;
      
      // Initialize local variables for BVH traversal for PMF computation
        uint32_t bitTrail = lightTree.lightToBitTrail[light];
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;

        int nodeIndex = 0;
        uint32_t index = 1;

        // Compute light's PMF by walking down tree nodes to the light
        while (true) {
            const LightBVHNode *node = &lightTree.nodes[nodeIndex];
            if (node->isLeaf) {
                DCHECK_EQ(light, lightTree.bvhLights[node->childOrLightIndex]);
                importances[0] = Half(static_cast<float>(index - 1));
                return true;
            }
            // Compute child importances and update PMF for current node
            const LightBVHNode *child0 = &lightTree.nodes[nodeIndex + 1];
            const LightBVHNode *child1 = &lightTree.nodes[node->childOrLightIndex];
            Float ci[2] = {Importance(p, n, lightTree.allLightBounds, child0->lightBounds),
                           Importance(p, n, lightTree.allLightBounds, child1->lightBounds)};
            DCHECK_GT(ci[bitTrail & 1], 0);
            const Float ci_w = ci[bitTrail & 1] / (ci[0] + ci[1]);
            importances[index++] = Half(ci_w);

            if (index >= MAX_CUT_SIZE){
              LOG_ERROR("Length of importances vec %d matches/exceeds MAX_CUT_SIZE %d", index, MAX_CUT_SIZE);
              return false;
            }

            // Use _bitTrail_ to find next node index and update its value
            nodeIndex = (bitTrail & 1) ? node->childOrLightIndex : (nodeIndex + 1);
            bitTrail >>= 1;
        }
    }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext& ctx, Half* importances) const {
      if (!importances || lightTree.nodes.empty())
        return false;
      
      Point3f p = ctx.p();
      Normal3f n = ctx.ns;

      for (size_t i = 0; i < lightTree.nodes.size(); i++){
        const LightBVHNode* node = &lightTree.nodes[i];
        if (node->isLeaf){
          importances[i] = Half(-1.f);
          continue;
        }

        // Compute child importances and update PMF for current node
        const LightBVHNode* child0 = &lightTree.nodes[i + 1];
        const LightBVHNode* child1 = &lightTree.nodes[node->childOrLightIndex];
        Float ci[2] = {Importance(p, n, lightTree.allLightBounds, child0->lightBounds),
                       Importance(p, n, lightTree.allLightBounds, child1->lightBounds)};
        const Float ciSum = ci[0] + ci[1];
        const Float ci_0 = ciSum > 0.0f ? ci[0] / ciSum : 0.0f;
        importances[i] = Half(ci_0);
      }

      return true;
    }
    PBRT_CPU_GPU inline void GetImportances(int pixelIndex, float* importances) const {}

    void EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size) {}
    Float TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size) {return 0.f;}
    
    void SaveModelWeights(const std::string &filename) const {}
  
  private:
    LightBVHTree lightTree;
};

// Importance Functions
struct UniformImportance {
    std::string ToString() const { return "UniformImportance"; }
    PBRT_CPU_GPU
    inline bool compute(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb0,  const CompactLightBounds &cb1, Float& prob0) const {
      prob0 = 0.5;
      return true;
    }

    PBRT_CPU_GPU
    inline Float compute(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb) const{
      return 0.5f;
    }
};

struct SLCImportance {
    std::string ToString() const { return "SLCImportance"; }
    PBRT_CPU_GPU
    inline bool compute(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb0,  const CompactLightBounds &cb1, Float& prob0) const {
      Bounds3f bounds0 = cb0.Bounds(allb);
      Bounds3f bounds1 = cb1.Bounds(allb);
      Float geom0 = GeomTermBound(p, Vector3f(n), bounds0.pMin, bounds0.pMax);
      Float geom1 = GeomTermBound(p, Vector3f(n), bounds1.pMin, bounds1.pMax);

      if (geom0 + geom1 == 0) return false;
      if (geom0 == 0){
        prob0 = 0.f;
        return true;
      } else if (geom1 == 0){
        prob0 = 1.f;
        return true;
      }

      Float intensGeom0 = cb0.Phi()*geom0;
      Float intensGeom1 = cb1.Phi()*geom1;
      Float l2_min0 = SquaredDistanceToClosestPoint(p, bounds0);
      Float l2_min1 = SquaredDistanceToClosestPoint(p, bounds1);

      if (l2_min0 > 0.01f * Length(bounds0.Diagonal()) && l2_min1 > 0.01f * Length(bounds1.Diagonal())){
        prob0 = normalizedWeights(l2_min0, l2_min1, intensGeom0, intensGeom1);
      } else {
        prob0 = intensGeom0 / (intensGeom0 + intensGeom1);
      }

      return true;
    }

    PBRT_CPU_GPU
    inline Float compute(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb) const{
      Bounds3f bounds = cb.Bounds(allb);
      Float geom = GeomTermBound(p, Vector3f(n), bounds.pMin, bounds.pMax);
      Float intensGeom = cb.Phi()*geom;
      Float l2_min = SquaredDistanceToClosestPoint(p, bounds);

      if (l2_min > 0.01f * Length(bounds.Diagonal()))
        return intensGeom / l2_min;
      else
        return intensGeom;
    }
};

struct SLCRTImportance {
    std::string ToString() const { return "SLCRTImportance"; }
    PBRT_CPU_GPU
    inline bool compute(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb0,  const CompactLightBounds &cb1, Float& prob0) const {
      Bounds3f bounds0 = cb0.Bounds(allb);
      Bounds3f bounds1 = cb1.Bounds(allb);
      Float geom0 = GeomTermBound(p, Vector3f(n), bounds0.pMin, bounds0.pMax);
      Float geom1 = GeomTermBound(p, Vector3f(n), bounds1.pMin, bounds1.pMax);

      if (geom0 + geom1 == 0) return false;
      if (geom0 == 0){
        prob0 = 0.f;
        return true;
      } else if (geom1 == 0){
        prob0 = 1.f;
        return true;
      }

      Float intensGeom0 = cb0.Phi()*geom0;
      Float intensGeom1 = cb1.Phi()*geom1;
      Float l2_min0 = SquaredDistanceToClosestPoint(p, bounds0);
      Float l2_min1 = SquaredDistanceToClosestPoint(p, bounds1);

      // Real-time SLC
      Float l2_max0 = SquaredDistanceToFarthestPoint(p, bounds0);
      Float l2_max1 = SquaredDistanceToFarthestPoint(p, bounds1);
      Float w_min0 = l2_min0 == 0.f && l2_min1 == 0.f ? intensGeom0 / (intensGeom0 + intensGeom1) : normalizedWeights(l2_min0, l2_min1, intensGeom0, intensGeom1);
      Float w_max0 = normalizedWeights(l2_max0, l2_max1, intensGeom0, intensGeom1);
      prob0 = 0.5f * (w_max0 + w_min0);

      return true;
    }

    PBRT_CPU_GPU
    inline Float compute(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb) const{
      Bounds3f bounds = cb.Bounds(allb);
      Float geom = GeomTermBound(p, Vector3f(n), bounds.pMin, bounds.pMax);
      Float intensGeom = cb.Phi()*geom;
      Float l2_min = SquaredDistanceToClosestPoint(p, bounds);
      Float l2_max = SquaredDistanceToFarthestPoint(p, bounds);
      Float w_min = l2_min == 0.f ? 1.f : intensGeom / l2_min;
      Float w_max = intensGeom / l2_max;
      return 0.5f * (w_max + w_min);
    }
};

struct ATSImportance {
  std::string ToString() const { return "ATSImportance"; }
  PBRT_CPU_GPU
  inline bool compute(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb0,  const CompactLightBounds &cb1, Float& prob0) const {
    auto w0 = BVHLightSampler::Importance(p, n, allb, cb0);
    auto w1 = BVHLightSampler::Importance(p, n, allb, cb1);
    prob0 = w0 / (w0 + w1);
    return (w0 + w1) > 0;
  }

  PBRT_CPU_GPU
  inline Float compute(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb) const {
    return BVHLightSampler::Importance(p, n, allb, cb);
  }
};

class ImportanceFunction : TaggedPointer<UniformImportance, SLCImportance, SLCRTImportance, ATSImportance> {
  public:
    using TaggedPointer::TaggedPointer;
    static ImportanceFunction Create(const std::string &name, Allocator alloc);
    std::string ToString() const;
    PBRT_CPU_GPU inline bool compute(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb0,  const CompactLightBounds &cb1, Float& prob0) const {
      auto res = [&](auto ptr) { return ptr->compute(p, n, allb, cb0, cb1, prob0); };
      return Dispatch(res); 
    }
    PBRT_CPU_GPU inline Float compute(Point3f p, Normal3f n, const Bounds3f &allb, const CompactLightBounds &cb) const {
      auto res = [&](auto ptr) { return ptr->compute(p, n, allb, cb); };
      return Dispatch(res); 
    }
};

// SLCLightTreeSampler Definition
class SLCLightTreeSampler {
  public:
    SLCLightTreeSampler(pstd::span<const Light> lights, Allocator alloc, const std::string& importanceFunctionName = "slc"): lightTree(lights, alloc){
      importanceFunction = ImportanceFunction::Create(importanceFunctionName, alloc);
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const {
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(lightTree.infiniteLights.size()) /
                          Float(lightTree.infiniteLights.size() + (lightTree.nodes.empty() ? 0 : 1));

        if (u < pInfinite) {
            // Sample infinite lights with uniform probability
            u /= pInfinite;
            int index =
                std::min<int>(u * lightTree.infiniteLights.size(), lightTree.infiniteLights.size() - 1);
            Float pmf = pInfinite / lightTree.infiniteLights.size();
            return SampledLight{lightTree.infiniteLights[index], pmf};

        } else {
            // Traverse light BVH to sample light
            if (lightTree.nodes.empty())
                return {};
            // Declare common variables for light BVH traversal
            Point3f p = ctx.p();
            Normal3f n = ctx.ns;
            u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);
            int nodeIndex = 0;
            Float pmf = 1 - pInfinite;

            while (true) {
                // Process light BVH node for light sampling
                LightBVHNode node = lightTree.nodes[nodeIndex];
                if (!node.isLeaf) {
                    // Compute light BVH child node importances
                    const LightBVHNode *children[2] = {&lightTree.nodes[nodeIndex + 1],
                                                       &lightTree.nodes[node.childOrLightIndex]};
                    Float prob0;
                    if (!importanceFunction.compute(p, n, lightTree.allLightBounds, children[0]->lightBounds, children[1]->lightBounds, prob0))
                      return {};
                    Float ci[2] = {prob0, 1 - prob0};

                    // Randomly sample light BVH child node
                    Float nodePMF;
                    int child = SampleDiscrete(ci, u, &nodePMF, &u);
                    pmf *= nodePMF;
                    nodeIndex = (child == 0) ? (nodeIndex + 1) : node.childOrLightIndex;

                } else {
                    if (nodeIndex > 0)
                        return SampledLight{lightTree.bvhLights[node.childOrLightIndex], pmf};
                    return {};
                }
            }
        }
    }

    PBRT_CPU_GPU
    Float PMF(const LightSampleContext &ctx, Light light) const {
        // Handle infinite _light_ PMF computation
        if (!lightTree.lightToBitTrail.HasKey(light))
            return 1.f / (lightTree.infiniteLights.size() + (lightTree.nodes.empty() ? 0 : 1));

        // Initialize local variables for BVH traversal for PMF computation
        uint32_t bitTrail = lightTree.lightToBitTrail[light];
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = Float(lightTree.infiniteLights.size()) /
                          Float(lightTree.infiniteLights.size() + (lightTree.nodes.empty() ? 0 : 1));

        Float pmf = 1 - pInfinite;
        int nodeIndex = 0;

        // Compute light's PMF by walking down tree nodes to the light
        while (true) {
            const LightBVHNode *node = &lightTree.nodes[nodeIndex];
            if (node->isLeaf) {
                DCHECK_EQ(light, lightTree.bvhLights[node->childOrLightIndex]);
                return pmf;
            }
            // Compute child importances and update PMF for current node
            const LightBVHNode *child0 = &lightTree.nodes[nodeIndex + 1];
            const LightBVHNode *child1 = &lightTree.nodes[node->childOrLightIndex];
            Float prob0;
            if (!importanceFunction.compute(p, n, lightTree.allLightBounds, child0->lightBounds, child1->lightBounds, prob0)){
              LOG_ERROR("Importance is 0");
              return 0.f;
            }
            Float ci[2] = {prob0, 1 - prob0};

            DCHECK_GT(ci[bitTrail & 1], 0);
            pmf *= ci[bitTrail & 1] / (ci[0] + ci[1]);

            // Use _bitTrail_ to find next node index and update its value
            nodeIndex = (bitTrail & 1) ? node->childOrLightIndex : (nodeIndex + 1);
            bitTrail >>= 1;
        }
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        return lightTree.Sample(u);
    }

    PBRT_CPU_GPU
    Float PMF(Light light) const {
        return lightTree.PMF(light);
    }

    PBRT_CPU_GPU inline Float HeuristicPMF(const LightSampleContext &ctx, Light light) const {return PMF(ctx, light);}

    std::string ToString() const { return std::string("SLCLightTreeSampler") + std::string(" ") + importanceFunction.ToString(); }

    PBRT_CPU_GPU
    inline bool isTraining(int pixelIndex, int depth) const { return false; }
    PBRT_CPU_GPU
    inline bool isNetworkEval(int pixelIndex, int depth) const { return false; }

    PBRT_CPU_GPU inline const LightBVHTree* getLightBVHTree() const { return &lightTree; }
    PBRT_CPU_GPU inline int getOptBucket(Light light) const { return -1; }
    PBRT_CPU_GPU inline int getNeuralOutputDim() const { return -1; }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext &ctx, Light light, Half* importances) const {
        if (!importances)
          return false;
      // Initialize local variables for BVH traversal for PMF computation
        uint32_t bitTrail = lightTree.lightToBitTrail[light];
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;

        int nodeIndex = 0;
        uint32_t index = 1;

        // Compute light's PMF by walking down tree nodes to the light
        while (true) {
            const LightBVHNode *node = &lightTree.nodes[nodeIndex];
            if (node->isLeaf) {
                DCHECK_EQ(light, lightTree.bvhLights[node->childOrLightIndex]);
                importances[0] = Half(static_cast<float>(index - 1));
                return true;
            }
            // Compute child importances and update PMF for current node
            const LightBVHNode *child0 = &lightTree.nodes[nodeIndex + 1];
            const LightBVHNode *child1 = &lightTree.nodes[node->childOrLightIndex];
            Float prob0;
            if (!importanceFunction.compute(p, n, lightTree.allLightBounds, child0->lightBounds, child1->lightBounds, prob0)){
              LOG_ERROR("Importance is 0");
              return false;
            }
            Float ci[2] = {prob0, 1 - prob0};
            DCHECK_GT(ci[bitTrail & 1], 0);
            const Float ci_w = ci[bitTrail & 1] / (ci[0] + ci[1]);
            importances[index++] = Half(ci_w);

            if (index >= MAX_CUT_SIZE){
              LOG_ERROR("Length of importances vec %d matches/exceeds MAX_CUT_SIZE %d", index, MAX_CUT_SIZE);
              return false;
            }

            // Use _bitTrail_ to find next node index and update its value
            nodeIndex = (bitTrail & 1) ? node->childOrLightIndex : (nodeIndex + 1);
            bitTrail >>= 1;
        }
    }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext& ctx, Half* importances) const {
      if (!importances || lightTree.nodes.empty())
        return false;

      Point3f p = ctx.p();
      Normal3f n = ctx.ns;

      for (size_t i = 0; i < lightTree.nodes.size(); i++){
        const LightBVHNode* node = &lightTree.nodes[i];
        if (node->isLeaf){
          importances[i] = Half(-1.f);
          continue;
        }

        // Compute child importances and update PMF for current node
        const LightBVHNode* child0 = &lightTree.nodes[i + 1];
        const LightBVHNode* child1 = &lightTree.nodes[node->childOrLightIndex];
        Float prob0;
        if (!importanceFunction.compute(p, n, lightTree.allLightBounds, child0->lightBounds, child1->lightBounds, prob0)){
          prob0 = 0.f;
        }
        importances[i] = Half(prob0);
      }

      return true;
    }
    PBRT_CPU_GPU inline void GetImportances(int pixelIndex, float* importances) const {}

    void EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size) {}
Float TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size) {return 0.f;}
    
    void SaveModelWeights(const std::string &filename) const {}

  private:
    LightBVHTree lightTree;
    ImportanceFunction importanceFunction;
};

// ExhaustiveLightSampler Definition
class ExhaustiveLightSampler {
  public:
    ExhaustiveLightSampler(pstd::span<const Light> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const;

    PBRT_CPU_GPU
    Float PMF(const LightSampleContext &ctx, Light light) const;

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (lights.empty())
            return {};

        int lightIndex = std::min<int>(u * lights.size(), lights.size() - 1);
        return SampledLight{lights[lightIndex], 1.f / lights.size()};
    }

    PBRT_CPU_GPU
    Float PMF(Light light) const {
        if (lights.empty())
            return 0;
        return 1.f / lights.size();
    }

    PBRT_CPU_GPU inline Float HeuristicPMF(const LightSampleContext &ctx, Light light) const {return PMF(ctx, light);}

    std::string ToString() const;

    PBRT_CPU_GPU
    inline bool isTraining(int pixelIndex, int depth) const { return false; }
    PBRT_CPU_GPU
    inline bool isNetworkEval(int pixelIndex, int depth) const { return false; }

    PBRT_CPU_GPU inline const LightBVHTree* getLightBVHTree() const { return nullptr; }
    PBRT_CPU_GPU inline int getOptBucket(Light light) const { return -1; }
    PBRT_CPU_GPU inline int getNeuralOutputDim() const { return -1; }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext &ctx, Light light, Half* importances) const {
      return false;
    }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext& ctx, Half* importances) const {
      return false;
    }

    PBRT_CPU_GPU inline void GetImportances(int pixelIndex, float* importances) const {}

    void EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size) {}
    Float TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size) {return 0.f;}
    
    void SaveModelWeights(const std::string &filename) const {}

  private:
    pstd::vector<Light> lights, boundedLights, infiniteLights;
    pstd::vector<LightBounds> lightBounds;
    HashMap<Light, size_t> lightToBoundedIndex;
};

struct ShadingPointBVHNode {
    // SceneBVHNode Public Methods
    ShadingPointBVHNode() = default;

    PBRT_CPU_GPU
    static ShadingPointBVHNode MakeLeaf(unsigned int clusterIndex, const Bounds3f &b) {
        return ShadingPointBVHNode{b, {clusterIndex, 1}};
    }

    PBRT_CPU_GPU
    static ShadingPointBVHNode MakeInterior(unsigned int child1Index,
                                     const Bounds3f &b) {
        return ShadingPointBVHNode{b, {child1Index, 0}};
    }

    // LightBVHNode Public Members
    Bounds3f bounds;
    struct {
        unsigned int childOrClusterIndex : 31;
        unsigned int isLeaf : 1;
    };
};

class ShadingPointBVHTree {
  public:
    ShadingPointBVHTree() = default;
    ShadingPointBVHTree(const LinearBVHNode *sceneNodes, uint32_t clusterSize, Allocator alloc);

    PBRT_CPU_GPU
    int ClusterIndex(const Point3f& p) const;

    PBRT_CPU_GPU 
    uint32_t ClusterSize() const { return clusterSize; }

    PBRT_CPU_GPU
    Bounds3f TotalBounds() const { return totalBounds; }

  private:
    std::pair<int, Bounds3f> buildBVH(
      std::vector<std::pair<int, Bounds3f>> &bvhSceneBounds, int start, int end);

    Float EvaluateCost(const Bounds3f &child, const Bounds3f &parent, int dim) {
      float parentArea = parent.SurfaceArea();
      if (parentArea == 0.f) return 0.f; // degenerate case
      float childArea  = child.SurfaceArea();
      return childArea / parentArea;
    }

    Bounds3f totalBounds;
    uint32_t clusterSize;
    pstd::vector<ShadingPointBVHNode> nodes;
    pstd::vector<Bounds3f> clusters;
};

struct HashTableEntry {
  LightCut<MAX_CUT_SIZE> lightCut;
  LightTreeSamplingResults<MAX_CUT_SIZE> samplingResults;
  uint32_t iteration;
  uint32_t curMaxQValueIndex;
  uint32_t noChangeIterations;

  HashTableEntry() = default;

  HashTableEntry(const LightCut<MAX_CUT_SIZE>& lightCut):
      lightCut(lightCut),
      samplingResults(),
      iteration(1),
      noChangeIterations(0),
      curMaxQValueIndex(0){}

  PBRT_CPU_GPU
  inline Float futureValue(const uint32_t strategy) const {
      switch (strategy){
      case 0: // qlearning
          return lightCut.MaxWeight();
          break;
      case 1: // mean
          return lightCut.MeanWeight();
          break;
      
      default:
          break;
      }
      return 0.0f;
  }
};

// VARLLightSampler Definition
class VARLLightSampler {
  public:
    VARLLightSampler(pstd::span<const Light> lights, int maxQueueSize, const Bounds3f& sceneBounds, Allocator alloc, const std::string& importanceFunctionName = "slc");

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const;

    PBRT_CPU_GPU
    Float PMF(const LightSampleContext &ctx, Light light) const;

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        return lightTree.Sample(u);
    }

    PBRT_CPU_GPU
    Float PMF(Light light) const {
        return lightTree.PMF(light);
    }

    PBRT_CPU_GPU inline Float HeuristicPMF(const LightSampleContext &ctx, Light light) const {return PMF(ctx, light);}

    std::string ToString() const { return std::string("VARLLightSampler") + std::string(" ") + importanceFunction.ToString(); }

    PBRT_CPU_GPU
    inline bool isTraining(int pixelIndex, int depth) const { return depth < MAX_TRAIN_DEPTH && pixelIndex % trainPixelStride == 0; }
    PBRT_CPU_GPU
    inline bool isNetworkEval(int pixelIndex, int depth) const { return true; }

    PBRT_CPU_GPU inline const LightBVHTree* getLightBVHTree() const { return &lightTree; }
    PBRT_CPU_GPU inline int getOptBucket(Light light) const { return -1; }
    PBRT_CPU_GPU inline int getNeuralOutputDim() const { return -1; }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext &ctx, Light light, Half* importances) const {
      return false;
    }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext& ctx, Half* importances) const {
      return false;
    }

    PBRT_CPU_GPU inline void GetImportances(int pixelIndex, float* importances) const {}

    void EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size);
    Float TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size);
    void SaveModelWeights(const std::string &filename) const {}

  private:
    PBRT_CPU_GPU inline uint32_t VARLLightSampler::UniformGridIndex(const Point3f& p, const Vector3f wo) const;

    const int maxQueueSize;
    const int trainPixelStride;
    const uint32_t directionalResolution;
    const Float directionalGridSize[2];
    const uint32_t initLightCutSize;
    const uint32_t clusterSize;
    const Float learningRate;
    const Float gamma;
    const uint32_t vStar;
    const uint32_t noChangeIterationLimit;
    const Bounds3f sceneBounds;

    LightBVHTree lightTree;
    LightCut<MAX_CUT_SIZE> globalLightCut;
    // ShadingPointBVHTree* spTree;
    // pstd::vector<HashTableEntry> hashMap;
    HashTableEntry* hashMap = nullptr;
    ImportanceFunction importanceFunction;
};


// Neural SLCLightSampler Definition
class NeuralSLCLightSampler {
public:
    NeuralSLCLightSampler(pstd::span<const Light> lights, int maxQueueSize, 
      Allocator alloc, const std::string& importanceFunctionName = "slc");

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const LightSampleContext &ctx, Float u) const;

    PBRT_CPU_GPU
    Float PMF(const LightSampleContext &ctx, Light light) const;

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        return lightTree.Sample(u);
    }

    PBRT_CPU_GPU
    Float PMF(Light light) const {
        return lightTree.PMF(light);
    }

    PBRT_CPU_GPU Float HeuristicPMF(const LightSampleContext &ctx, Light light) const;

    std::string ToString() const { return std::string("NeuralSLCLightSampler") + std::string(" ") + importanceFunction.ToString(); }

    PBRT_CPU_GPU
    inline bool isTraining(int pixelIndex, int depth) const { return depth < MAX_TRAIN_DEPTH && pixelIndex % trainPixelStride == 0; }
    PBRT_CPU_GPU
    inline bool isNetworkEval(int pixelIndex, int depth) const { return true; }

    PBRT_CPU_GPU inline const LightBVHTree* getLightBVHTree() const { return &lightTree; }
    PBRT_CPU_GPU inline int getOptBucket(Light light) const { return lightToOptBucket.HasKey(light) ? lightToOptBucket[light] : -1; }
    PBRT_CPU_GPU inline int getNeuralOutputDim() const { return neuralOutputDim; }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext &ctx, Light light, Half* importances) const {
      return false;
    }

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext& ctx, Half* importances) const {
      return false;
    }

    PBRT_CPU_GPU inline void GetImportances(int pixelIndex, float* importances) const {
      const uint32_t outputIndex = pixelIndexToOutputIndex[pixelIndex];
      for (uint32_t i = 0; i < neuralOutputDim; i++)
        importances[i] = inferenceOutputPtr[outputIndex * neuralOutputDim + i];
    }

    void EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size);
    Float TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size);
    
    void SaveModelWeights(const std::string &filename) const;

  private:
    // Base function that chooses between:
    // 1) Establishing direct mappings between network output and lights
    // 2) Establishing mappings between network output and light cluster nodes at a certain depth
    // If # lights <= NEURAL_OUTPUT_DIM_MAX, we use the first option, otherwise we use the second
    std::pair<int, int> buildOptBucketNodeIndexMappings(int nodeIndex, int optBucket, uint32_t depth, int n_collapsed_outputdims);
    // 1)
    int buildOptBucketNodeIndexMappings(int nodeIndex, int optBucket);
    // 2)
    std::pair<int, int> buildOptBucketNodeIndexMappingsHeightCapped(int nodeIndex, int optBucket, uint32_t depth, int n_collapsed_outputdims);

    void PopulateLightToOptBucket(int nodeIndex, int optBucket);

    template <typename T>
    PBRT_CPU_GPU inline void computeResidualPMF(Point3f p, Normal3f n, T* networkOutputsi) const;

    LightBVHTree lightTree;
    ImportanceFunction importanceFunction;
    int maxQueueSize;

    const int trainPixelStride;

    pstd::vector<uint32_t> optBucketToNodeIndex; // Mapping from nn's optimization index to (light) node index
    HashMap<Light, uint32_t> lightToOptBucket; // Mapping from light to nn's optimization index

    // Declaring these last since we need the mappings above to have been initialized first
    const uint32_t neuralOutputDim;
    const uint32_t neuralOutputDimPadded;
    
    uint32_t* pixelIndexToOutputIndex = nullptr;
    float* inferenceOutputPtr = nullptr;
};


PBRT_CPU_GPU inline pstd::optional<SampledLight> LightSampler::Sample(const LightSampleContext &ctx,
                                                         Float u) const {
    auto s = [&](auto ptr) { return ptr->Sample(ctx, u); };
    return Dispatch(s);
}

PBRT_CPU_GPU inline Float LightSampler::PMF(const LightSampleContext &ctx, Light light) const {
    auto pdf = [&](auto ptr) { return ptr->PMF(ctx, light); };
    return Dispatch(pdf);
}

PBRT_CPU_GPU inline pstd::optional<SampledLight> LightSampler::Sample(Float u) const {
    auto sample = [&](auto ptr) { return ptr->Sample(u); };
    return Dispatch(sample);
}

PBRT_CPU_GPU inline Float LightSampler::PMF(Light light) const {
    auto pdf = [&](auto ptr) { return ptr->PMF(light); };
    return Dispatch(pdf);
}

PBRT_CPU_GPU inline Float LightSampler::HeuristicPMF(const LightSampleContext &ctx, Light light) const {
    auto heuristic = [&](auto ptr) { return ptr->HeuristicPMF(ctx, light); };
    return Dispatch(heuristic);
}

PBRT_CPU_GPU inline bool LightSampler::isTraining(int pixelIndex, int depth) const {
    auto training = [&](auto ptr) { return ptr->isTraining(pixelIndex, depth); };
    return Dispatch(training);
}

inline bool LightSampler::isNetworkEval(int pixelIndex, int depth) const {
    auto training = [&](auto ptr) { return ptr->isNetworkEval(pixelIndex, depth); };
    return Dispatch(training);
}

inline const LightBVHTree* LightSampler::getLightBVHTree() const {
    auto getTree = [&](auto ptr) { return ptr->getLightBVHTree(); };
    return Dispatch(getTree);
}

PBRT_CPU_GPU inline int LightSampler::getOptBucket(Light light) const {
    auto getOptBucket = [&](auto ptr) { return ptr->getOptBucket(light); };
    return Dispatch(getOptBucket);
}

PBRT_CPU_GPU inline int LightSampler::getNeuralOutputDim() const {
    auto getNeuralOutputDim = [&](auto ptr) { return ptr->getNeuralOutputDim(); };
    return Dispatch(getNeuralOutputDim);
}

PBRT_CPU_GPU inline bool LightSampler::ImportancesArray(const LightSampleContext &ctx, Light light, Half* importances) const {
    auto importancesArray = [&](auto ptr) { return ptr->ImportancesArray(ctx, light, importances); };
    return Dispatch(importancesArray);
}

PBRT_CPU_GPU inline bool LightSampler::ImportancesArray(const LightSampleContext &ctx, Half* importances) const {
    auto ImportancesArray = [&](auto ptr) { return ptr->ImportancesArray(ctx, importances); };
    return Dispatch(ImportancesArray);
}

PBRT_CPU_GPU inline void LightSampler::GetImportances(int pixelIndex, float* importances) const {
    auto ImportancesArray = [&](auto ptr) { return ptr->GetImportances(pixelIndex, importances); };
    return Dispatch(ImportancesArray);
}

inline void LightSampler::EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size) {
    auto preprocess = [&](auto ptr) { ptr->EvalOrSample(inputBuffer, pixelIndexBuffer, residualInfoBuffer, size); };
    return Dispatch(preprocess);
}

inline Float LightSampler::TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size) {
    auto train = [&](auto ptr) { return ptr->TrainStep(inputs, radiances, lightIndices, residualInfoBuffer, radiancePDFs, size); };
    return Dispatch(train);
}

inline void LightSampler::SaveModelWeights(const std::string &filename) const {
    auto save = [&](auto ptr) { ptr->SaveModelWeights(filename); };
    return Dispatch(save);
}

}  // namespace pbrt

#endif  // PBRT_LIGHTSAMPLERS_H
