// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_WAVEFRONT_INTEGRATOR_H
#define PBRT_WAVEFRONT_INTEGRATOR_H

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/filter.h>
#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/base/sampler.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/options.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>
#include <pbrt/wavefront/workitems.h>
#include <pbrt/wavefront/workqueue.h>

#include <pbrt/wavefront/train.h>

namespace pbrt {

class BasicScene;
class GUI;

// WavefrontAggregate Definition
class WavefrontAggregate {
  public:
    // WavefrontAggregate Interface
    virtual ~WavefrontAggregate() = default;

    virtual Bounds3f Bounds() const = 0;

    virtual void IntersectClosest(int maxRays, const RayQueue *rayQ,
                                  EscapedRayQueue *escapedRayQ,
                                  HitAreaLightQueue *hitAreaLightQ,
                                  MaterialEvalQueue *basicMtlQ,
                                  MaterialEvalQueue *universalMtlQ,
                                  MediumSampleQueue *mediumSampleQ,
                                  RayQueue *nextRayQ) const = 0;

    virtual void IntersectShadow(int maxRays, ShadowRayQueue *shadowRayQueue,
                                 SOA<PixelSampleState> *pixelSampleState, 
                                 LightSamplerOptStateBuffer *lightSamplerOptStateBuffer,
                                 LightSampler lightSampler) const = 0;
    virtual void IntersectShadowTr(int maxRays, ShadowRayQueue *shadowRayQueue,
                                   SOA<PixelSampleState> *pixelSampleState,
                                   LightSamplerOptStateBuffer *lightSamplerOptStateBuffer,
                                   LightSampler lightSampler) const = 0;

    virtual void IntersectOneRandom(
        int maxRays, SubsurfaceScatterQueue *subsurfaceScatterQueue) const = 0;
};

// WavefrontPathIntegrator Definition
class WavefrontPathIntegrator {
  public:
    // WavefrontPathIntegrator Public Methods
    Float Render();

    void GenerateCameraRays(int y0, Transform movingFromcamera, int sampleIndex);
    template <typename Sampler>
    void GenerateCameraRays(int y0, Transform movingFromCamera, int sampleIndex);

    void GenerateRaySamples(int wavefrontDepth, int sampleIndex);
    template <typename Sampler>
    void GenerateRaySamples(int wavefrontDepth, int sampleIndex);

    void SampleLightSources(int sampleIndex, float CurrentTime);
    template <typename ConcreteMaterial>
    void SampleLightSources(int sampleIndex, float CurrentTime);
    template <typename ConcreteMaterial, typename TextureEvaluator>
    void SampleLightSources(LightSamplerQueue* lightSamplerQueue, int sampleIndex, float CurrentTime=0.0f);

    void TraceShadowRays(int wavefrontDepth);
    void SampleMediumInteraction(int wavefrontDepth);
    template <typename PhaseFunction>
    void SampleMediumScattering(int wavefrontDepth);
    void SampleSubsurface(int wavefrontDepth);

    void HandleEscapedRays();
    void HandleEmissiveIntersection(int sampleIndex);

    void EvaluateMaterialsAndBSDFs(int wavefrontDepth, int sampleIndex, Transform movingFromCamera);
    template <typename ConcreteMaterial>
    void EvaluateMaterialAndBSDF(int wavefrontDepth, int sampleIndex, Transform movingFromCamera);
    template <typename ConcreteMaterial, typename TextureEvaluator>
    void EvaluateMaterialAndBSDF(MaterialEvalQueue *evalQueue, LightSamplerQueue* lightSamplerQueue, 
                                 Transform movingFromCamera, int wavefrontDepth, int sampleIndex);

    void UpdateFilm();

    WavefrontPathIntegrator(pstd::pmr::memory_resource *memoryResource,
                            BasicScene &scene);

    template <typename F>
    void ParallelFor(const char *description, int nItems, F &&func) {
        if (Options->useGPU)
#ifdef PBRT_BUILD_GPU_RENDERER
            GPUParallelFor(description, nItems, func);
#else
            LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
        else
            pbrt::ParallelFor(0, nItems, func);
    }

    template <typename F>
    void Do(const char *description, F &&func) {
        if (Options->useGPU)
#ifdef PBRT_BUILD_GPU_RENDERER
            GPUParallelFor(description, 1, [=] PBRT_GPU(int) mutable { func(); });
#else
            LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
        else
            func();
    }

    RayQueue *CurrentRayQueue(int wavefrontDepth) {
        return rayQueues[wavefrontDepth & 1];
    }
    RayQueue *NextRayQueue(int wavefrontDepth) {
        return rayQueues[(wavefrontDepth + 1) & 1];
    }

#ifdef PBRT_BUILD_GPU_RENDERER
    void PrefetchGPUAllocations();
#endif  // PBRT_BUILD_GPU_RENDERER

    // --display-server methods
    void StartDisplayThread();
    void UpdateDisplayRGBFromFilm(Bounds2i pixelBounds);
    void StopDisplayThread();

    // intermediate data methods
    void AppendIntermediateData(Bounds2i pixelBounds, uint32_t index, float time, uint32_t spp);

    // --interactive support
    void UpdateFramebufferFromFilm(Bounds2i pixelBounds, Float exposure, RGB *rgb);

    // WavefrontPathIntegrator Member Variables
    bool initializeVisibleSurface;
    bool haveSubsurface;
    bool haveMedia;
    pstd::array<bool, Material::NumTags()> haveBasicEvalMaterial;
    pstd::array<bool, Material::NumTags()> haveUniversalEvalMaterial;

    struct Stats {
        Stats(int maxDepth, Allocator alloc);

        std::string Print() const;

        // Note: not atomics: tid 0 always updates them for everyone...
        uint64_t cameraRays = 0;
        pstd::vector<uint64_t> indirectRays, shadowRays;
    };
    Stats *stats;

    pstd::pmr::memory_resource *memoryResource;

    Filter filter;
    Film film;
    Sampler sampler;
    Camera camera;
    pstd::vector<Light> *infiniteLights;
    LightSampler lightSampler;
    #if VPL_USED
        LightSampler VPLDirectlightsampler;
    #endif
    LightSampler lightSamplerExploration;

    int maxDepth, samplesPerPixel;
    bool regularize;
    
    Float explorationSamplingRatio = 0.f;
    int nExplorationWarmupSamples = 0;
    Float trainingBudgetRatio = 0.f;

    int sppInterval = 0;
    float timeInterval = 0.0f;
    float timeLimit = 0.0f;

    int scanlinesPerPass, maxQueueSize;

    SOA<PixelSampleState> pixelSampleState;

    LightSamplerOptStateBuffer lightSamplerOptStateBuffer;

    RayQueue *rayQueues[2];

    WavefrontAggregate *aggregate = nullptr;
    Bounds3f sceneBounds;

    MediumSampleQueue *mediumSampleQueue = nullptr;
    MediumScatterQueue *mediumScatterQueue = nullptr;

    EscapedRayQueue *escapedRayQueue = nullptr;

    HitAreaLightQueue *hitAreaLightQueue = nullptr;

    MaterialEvalQueue *basicEvalMaterialQueue = nullptr;
    MaterialEvalQueue *universalEvalMaterialQueue = nullptr;

    LightSamplerQueue *basicLightSamplerQueue = nullptr;
    LightSamplerQueue *universalLightSamplerQueue = nullptr;

    InNxOutBuffer<LightSamplerTrainInput, RGB, int32_t, LightSamplerResidualInfo, Float> *lightSamplerTrainBuffer = nullptr;
    InNxOutBuffer<LightSamplerTrainInput, int32_t, LightSamplerResidualInfo> *lightSamplerEvalBuffer = nullptr;

    ShadowRayQueue *shadowRayQueue = nullptr;

    GetBSSRDFAndProbeRayQueue *bssrdfEvalQueue = nullptr;
    SubsurfaceScatterQueue *subsurfaceScatterQueue = nullptr;

    RGB *displayRGB = nullptr, *displayRGBHost = nullptr;
    std::atomic<bool> *exitCopyThread;
    std::thread *copyThread;
    
    // Stores rgb and time for intermediate data on device
    struct IntermediateAuxData {
        float time;
        uint32_t spp;
    };
    IntermediateAuxData *intermediateAuxDataBuffer = nullptr;
    RGB *intermediateRGBDataBuffer = nullptr;
    uint32_t nIntermediateDataPoints = 0;
};

}  // namespace pbrt

#endif  // PBRT_WAVEFRONT_INTEGRATOR_H
