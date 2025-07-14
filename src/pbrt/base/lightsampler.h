// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_LIGHTSAMPLER_H
#define PBRT_BASE_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

#include <string>

#include <pbrt/base/light.h>
#include <pbrt/cpu/aggregates.h> // LinearBVHNode

namespace pbrt {

// SampledLight Definition
struct SampledLight {
    Light light;
    Float p = 0;
    std::string ToString() const;
};

class UniformLightSampler;
class PowerLightSampler;
class BVHLightSampler;
class SLCLightTreeSampler;
class ExhaustiveLightSampler;
class LightBVHTree;
class VARLLightSampler;
class NeuralSLCLightSampler;

class PixelSampleState;

// LightSampler Definition
class LightSampler : public TaggedPointer<UniformLightSampler, PowerLightSampler,
                                          ExhaustiveLightSampler, BVHLightSampler,
                                          SLCLightTreeSampler, VARLLightSampler,
                                          NeuralSLCLightSampler> {
  public:
    // LightSampler Interface
    using TaggedPointer::TaggedPointer;

    static LightSampler Create(const std::string &name, pstd::span<const Light> lights,
                               Allocator alloc);

    static LightSampler Create(const std::string &name, pstd::span<const Light> lights,
                               int maxQueueSize, const Bounds3f& sceneBounds, Vector2i resolution, Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(const LightSampleContext &ctx,
                                                            Float u) const;

    PBRT_CPU_GPU inline Float PMF(const LightSampleContext &ctx, Light light) const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(Float u) const;
    PBRT_CPU_GPU inline Float PMF(Light light) const;

    PBRT_CPU_GPU inline Float HeuristicPMF(const LightSampleContext &ctx, Light light) const;
    
    PBRT_CPU_GPU inline bool isTraining(int pixelIndex, int depth) const;
    PBRT_CPU_GPU inline bool isNetworkEval(int pixelIndex, int depth) const;

    PBRT_CPU_GPU inline const LightBVHTree* getLightBVHTree() const;
    PBRT_CPU_GPU inline int getOptBucket(Light light) const;
    PBRT_CPU_GPU inline int getNeuralOutputDim() const;

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext &ctx, Light light, Half* importances) const;

    PBRT_CPU_GPU inline bool ImportancesArray(const LightSampleContext& ctx, Half* importances) const;

    PBRT_CPU_GPU inline void GetImportances(int pixelIndex, float* importances) const;

    void EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size);

    Float TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size);

    void SaveModelWeights(const std::string &filename) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_LIGHTSAMPLER_H
