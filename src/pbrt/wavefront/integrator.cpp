// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/wavefront/integrator.h>

#include <pbrt/base/medium.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/optix/aggregate.h>
#include <pbrt/gpu/memory.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/display.h>
#include <pbrt/util/file.h>
#include <pbrt/util/gui.h>
#include <pbrt/util/image.h>
#include <pbrt/util/log.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/string.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/wavefront/aggregate.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <map>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <cuda.h>
#include <cuda_runtime.h>

#endif  // PBRT_BUILD_GPU_RENDERER

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Wavefront integrator pixel state", pathIntegratorBytes);

static void updateMaterialNeeds(
    Material m, pstd::array<bool, Material::NumTags()> *haveBasicEvalMaterial,
    pstd::array<bool, Material::NumTags()> *haveUniversalEvalMaterial,
    bool *haveSubsurface, bool *haveMedia) {
    *haveMedia |= (m == nullptr);  // interface material
    if (!m)
        return;

    if (MixMaterial *mix = m.CastOrNullptr<MixMaterial>(); mix) {
        // This is a somewhat odd place for this check, but it's convenient...
        if (!m.CanEvaluateTextures(BasicTextureEvaluator()))
            ErrorExit("\"mix\" material has a texture that can't be evaluated with the "
                      "BasicTextureEvaluator, which is all that is currently supported "
                      "int the wavefront renderer--sorry! %s",
                      *mix);

        updateMaterialNeeds(mix->GetMaterial(0), haveBasicEvalMaterial,
                            haveUniversalEvalMaterial, haveSubsurface, haveMedia);
        updateMaterialNeeds(mix->GetMaterial(1), haveBasicEvalMaterial,
                            haveUniversalEvalMaterial, haveSubsurface, haveMedia);
        return;
    }

    *haveSubsurface |= m.HasSubsurfaceScattering();

    FloatTexture displace = m.GetDisplacement();
    if (m.CanEvaluateTextures(BasicTextureEvaluator()) &&
        (!displace || BasicTextureEvaluator().CanEvaluate({displace}, {})))
        (*haveBasicEvalMaterial)[m.Tag()] = true;
    else
        (*haveUniversalEvalMaterial)[m.Tag()] = true;
}

WavefrontPathIntegrator::WavefrontPathIntegrator(
    pstd::pmr::memory_resource *memoryResource, BasicScene &scene)
    : memoryResource(memoryResource), exitCopyThread(new std::atomic<bool>(false)) {
    ThreadLocal<Allocator> threadAllocators(
        [memoryResource]() { return Allocator(memoryResource); });

    Allocator alloc = threadAllocators.Get();

    // Allocate all of the data structures that represent the scene...
    std::map<std::string, Medium> media = scene.CreateMedia();

    // "haveMedia" is a bit of a misnomer in that determines both whether
    // queues are allocated for the medium sampling kernels and they are
    // launched as well as whether the ray marching shadow ray kernel is
    // launched... Thus, it will be true if there actually are no media,
    // but some "interface" materials are present in the scene.
    haveMedia = false;
    // Check the shapes and instance definitions...
    for (const auto &shape : scene.shapes)
        if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
            haveMedia = true;
    for (const auto &shape : scene.animatedShapes)
        if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
            haveMedia = true;
    for (const auto &instanceDefinition : scene.instanceDefinitions) {
        for (const auto &shape : instanceDefinition.second->shapes)
            if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
                haveMedia = true;
        for (const auto &shape : instanceDefinition.second->animatedShapes)
            if (!shape.insideMedium.empty() || !shape.outsideMedium.empty())
                haveMedia = true;
    }

    // Textures
    LOG_VERBOSE("Starting to create textures");
    NamedTextures textures = scene.CreateTextures();
    LOG_VERBOSE("Done creating textures");

    LOG_VERBOSE("Starting to create lights");
    pstd::vector<Light> allLights;
    pstd::vector<Light> allDirectLights;
    std::map<int, pstd::vector<Light> *> shapeIndexToAreaLights;

    infiniteLights = alloc.new_object<pstd::vector<Light>>(alloc);

    int count_direct = 0;
    int count_VPL = 0;

    for (Light l : scene.CreateLights(textures, &shapeIndexToAreaLights)) {
        if (l.Is<UniformInfiniteLight>() || l.Is<ImageInfiniteLight>() ||
            l.Is<PortalImageInfiniteLight>()) {
            infiniteLights->push_back(l);
        }

#if VPL_USED
        if (l.Is<VirtualPointLight>()) {
            allLights.push_back(l);
            count_VPL++;
        } else {
            allDirectLights.push_back(l);
            count_direct++;
        }
#else
    allLights.push_back(l);
#endif
    }
    
    LOG_VERBOSE("Done creating lights");

    LOG_VERBOSE("Starting to create materials");
    std::map<std::string, pbrt::Material> namedMaterials;
    std::vector<pbrt::Material> materials;
    scene.CreateMaterials(textures, &namedMaterials, &materials);

    haveBasicEvalMaterial.fill(false);
    haveUniversalEvalMaterial.fill(false);
    haveSubsurface = false;
    for (Material m : materials)
        updateMaterialNeeds(m, &haveBasicEvalMaterial, &haveUniversalEvalMaterial,
                            &haveSubsurface, &haveMedia);
    for (const auto &m : namedMaterials)
        updateMaterialNeeds(m.second, &haveBasicEvalMaterial, &haveUniversalEvalMaterial,
                            &haveSubsurface, &haveMedia);
    LOG_VERBOSE("Finished creating materials");

    // Retrieve these here so that the CPU isn't writing to managed memory
    // concurrently with the OptiX acceleration-structure construction work
    // that follows. (Verbotten on Windows.)
    camera = scene.GetCamera();
    film = camera.GetFilm();
    filter = film.GetFilter();
    sampler = scene.GetSampler();

    if (Options->useGPU) {
#ifdef PBRT_BUILD_GPU_RENDERER
        CUDATrackedMemoryResource *mr =
            dynamic_cast<CUDATrackedMemoryResource *>(memoryResource);
        CHECK(mr);
        aggregate = new OptiXAggregate(scene, mr, textures, shapeIndexToAreaLights, media,
                                       namedMaterials, materials);
#else
        LOG_FATAL("Options->useGPU was set without PBRT_BUILD_GPU_RENDERER enabled");
#endif
    } else
        aggregate = new CPUAggregate(scene, textures, shapeIndexToAreaLights, media,
                                     namedMaterials, materials);

    // Save sceneBounds
    sceneBounds = aggregate->Bounds();
    printf("Scene bounds: %s\n", sceneBounds.ToString().c_str());
    // exit(1);
    // Compute number of scanlines to render per pass
    Vector2i resolution = film.PixelBounds().Diagonal();
    // TODO: make this configurable. Base it on the amount of GPU memory?
    scanlinesPerPass = std::max(1, MAX_INFERENCE_NUM / resolution.x);
    int nPasses = (resolution.y + scanlinesPerPass - 1) / scanlinesPerPass;
    scanlinesPerPass = (resolution.y + nPasses - 1) / nPasses;
    maxQueueSize = resolution.x * scanlinesPerPass;

    LOG_VERBOSE("Will render in %d passes %d scanlines per pass\n", nPasses,
                scanlinesPerPass);

    // Preprocess the light sources
#if VPL_USED
    for (Light light: allDirectLights)
        light.Preprocess(sceneBounds);
#else
    for (Light light : allLights)
        light.Preprocess(sceneBounds);
#endif

    bool haveLights = !allLights.empty();
    for (const auto &m : media)
        haveLights |= m.second.IsEmissive();
    if (!haveLights)
        ErrorExit("No light sources specified");
    
    LOG_VERBOSE("Number of lights: %d", allLights.size() + allDirectLights.size());
    
    LOG_VERBOSE("Starting to create light sampler");
    std::string lightSamplerName =
        scene.integrator.parameters.GetOneString("lightsampler", "bvh");
    std::string directLightSamplerName =
        scene.integrator.parameters.GetOneString("directLightSamplerName", "uniform");
    if (allLights.size() == 1)
        lightSamplerName = "uniform";

    LOG_VERBOSE("Using light sampler: %s", lightSamplerName);
    lightSampler = LightSampler::Create(lightSamplerName, allLights, maxQueueSize, sceneBounds, resolution, alloc);
    #if VPL_USED
        VPLDirectlightsampler = LightSampler::Create(directLightSamplerName, allDirectLights,
                                                    maxQueueSize, sceneBounds, resolution, alloc);
    #endif                                            
    LOG_VERBOSE("Finished creating light sampler");

    if (scene.integrator.name != "path" && scene.integrator.name != "volpath")
        Warning(&scene.integrator.loc,
                "Ignoring specified integrator \"%s\": the wavefront integrator "
                "always uses a \"volpath\" integrator.",
                scene.integrator.name);

    // Integrator parameters
    regularize = scene.integrator.parameters.GetOneBool("regularize", false);
    maxDepth = scene.integrator.parameters.GetOneInt("maxdepth", 5);

    initializeVisibleSurface = film.UsesVisibleSurface();
    samplesPerPixel = sampler.SamplesPerPixel();

    // Warn about unsupported stuff...
    if (Options->forceDiffuse)
        ErrorExit("The wavefront integrator does not support --force-diffuse.");
    if (Options->writePartialImages)
        Warning("The wavefront integrator does not support --write-partial-images.");
    if (Options->recordPixelStatistics)
        ErrorExit("The wavefront integrator does not support --pixelstats.");
    if (!Options->mseReferenceImage.empty())
        ErrorExit("The wavefront integrator does not support --mse-reference-image.");
    if (!Options->mseReferenceOutput.empty())
        ErrorExit("The wavefront integrator does not support --mse-reference-out.");

        ///////////////////////////////////////////////////////////////////////////
        // Allocate storage for all of the queues/buffers...

#ifdef PBRT_BUILD_GPU_RENDERER
    size_t startSize = 0;
    if (Options->useGPU) {
        CUDATrackedMemoryResource *mr =
            dynamic_cast<CUDATrackedMemoryResource *>(memoryResource);
        CHECK(mr);
        startSize = mr->BytesAllocated();
    }
#endif  // PBRT_BUILD_GPU_RENDERER

    pixelSampleState = SOA<PixelSampleState>(maxQueueSize, alloc);

    lightSamplerOptStateBuffer = LightSamplerOptStateBuffer(maxQueueSize, alloc);

    rayQueues[0] = alloc.new_object<RayQueue>(maxQueueSize, alloc);
    rayQueues[1] = alloc.new_object<RayQueue>(maxQueueSize, alloc);

    #if VPL_USED
    shadowRayQueue = alloc.new_object<ShadowRayQueue>(maxQueueSize * 2, alloc);
    #else
    shadowRayQueue = alloc.new_object<ShadowRayQueue>(maxQueueSize, alloc);
    #endif
    //// std::cout << "maxQueueSize: " << maxQueueSize << std::endl;
    //// std::cout << "shadowRayQueue->Size(): " << shadowRayQueue->Size() << std::endl;

    if (haveSubsurface) {
        bssrdfEvalQueue =
            alloc.new_object<GetBSSRDFAndProbeRayQueue>(maxQueueSize, alloc);
        subsurfaceScatterQueue =
            alloc.new_object<SubsurfaceScatterQueue>(maxQueueSize, alloc);
    }

    if (infiniteLights->size())
        escapedRayQueue = alloc.new_object<EscapedRayQueue>(maxQueueSize, alloc);
    hitAreaLightQueue = alloc.new_object<HitAreaLightQueue>(maxQueueSize, alloc);

    basicEvalMaterialQueue = alloc.new_object<MaterialEvalQueue>(
        maxQueueSize, alloc,
        pstd::MakeConstSpan(&haveBasicEvalMaterial[1], haveBasicEvalMaterial.size() - 1));
    universalEvalMaterialQueue = alloc.new_object<MaterialEvalQueue>(
        maxQueueSize, alloc,
        pstd::MakeConstSpan(&haveUniversalEvalMaterial[1],
                            haveUniversalEvalMaterial.size() - 1));


    basicLightSamplerQueue = alloc.new_object<LightSamplerQueue>(
        maxQueueSize, alloc,
        pstd::MakeConstSpan(&haveBasicEvalMaterial[1], haveBasicEvalMaterial.size() - 1));
    universalLightSamplerQueue = alloc.new_object<LightSamplerQueue>(
        maxQueueSize, alloc,
        pstd::MakeConstSpan(&haveUniversalEvalMaterial[1],
                            haveUniversalEvalMaterial.size() - 1));

    explorationSamplingRatio = scene.integrator.parameters.GetOneFloat("explorationSamplingRatio", 0.0f);
    LOG_VERBOSE("Exploration Sampling Ratio: %f", explorationSamplingRatio);
    nExplorationWarmupSamples = scene.integrator.parameters.GetOneInt("nExplorationWarmupSamples", 0);
    LOG_VERBOSE("Number of Exploration Warmup Samples: %d", nExplorationWarmupSamples);
    trainingBudgetRatio = scene.integrator.parameters.GetOneFloat("trainingBudgetRatio", 0.0f);
    LOG_VERBOSE("Training Budget Ratio: %f", trainingBudgetRatio);

    sppInterval = scene.integrator.parameters.GetOneInt("sppInterval", 0);
    timeInterval = scene.integrator.parameters.GetOneFloat("timeInterval", 0.0f);

    if (timeInterval > 0.0f && sppInterval > 0)
        ErrorExit("Cannot specify both sppInterval and timeInterval");

    LOG_VERBOSE("Image saved every %f spp (default to 0, meaning we don't save according to spp)", sppInterval);
    LOG_VERBOSE(
        "Image saved every %f timeInterval (default to 0.0f, meaning we don't save according to timeInterval)",
        timeInterval);

    timeLimit = scene.integrator.parameters.GetOneFloat("timeLimit", 0.0f);
    LOG_VERBOSE("timeLimit, stop rendering when %f seconds is reached (default to 0.0f, and we will not consider it if it is not positive)", timeLimit);
    
    // Allocate intermediate data buffers
    nIntermediateDataPoints = 
        timeInterval > 0.f ? static_cast<uint32_t>(timeLimit / timeInterval) + 1
        : static_cast<uint32_t>(sppInterval > 0 ? (samplesPerPixel / sppInterval) + 1 : 0);
    
    if (timeLimit > 0.f)
        nIntermediateDataPoints += 1;
    
    if (nIntermediateDataPoints > 0) {
        // Initialize the intermediate data buffers
        CUDA_CHECK(cudaMalloc(&intermediateAuxDataBuffer, nIntermediateDataPoints * sizeof(IntermediateAuxData)));
        CUDA_CHECK(cudaMemset(intermediateAuxDataBuffer, 0, nIntermediateDataPoints * sizeof(IntermediateAuxData)));
        CUDA_CHECK(cudaMalloc(&intermediateRGBDataBuffer, nIntermediateDataPoints * resolution.x * resolution.y * sizeof(RGB)));
        CUDA_CHECK(cudaMemset(intermediateRGBDataBuffer, 0, nIntermediateDataPoints * resolution.x * resolution.y * sizeof(RGB)));
        LOG_VERBOSE("Allocated data for %d intermediate data points", nIntermediateDataPoints);
    }

    if (explorationSamplingRatio > 0.f || nExplorationWarmupSamples > 0){
        std::string explorationLightSamplerName = scene.integrator.parameters.GetOneString("explorationLightSampler", "bvh");
        lightSamplerExploration = LightSampler::Create(explorationLightSamplerName, allLights, maxQueueSize, sceneBounds, resolution, alloc);
        LOG_VERBOSE("Using exploration light sampler: %s", explorationLightSamplerName);
    }

    lightSamplerEvalBuffer = alloc.new_object<InNxOutBuffer<LightSamplerTrainInput, int32_t, LightSamplerResidualInfo>>(
        maxQueueSize, alloc);
    lightSamplerTrainBuffer = alloc.new_object<InNxOutBuffer<LightSamplerTrainInput, RGB, int32_t, LightSamplerResidualInfo, Float>>(
        maxQueueSize*MAX_TRAIN_DEPTH, alloc);

    if (haveMedia) {
        mediumSampleQueue = alloc.new_object<MediumSampleQueue>(maxQueueSize, alloc);

        // TODO: in the presence of multiple PhaseFunction implementations,
        // it could be worthwhile to see which are present in the scene and
        // then initialize havePhase accordingly...
        pstd::array<bool, PhaseFunction::NumTags()> havePhase;
        havePhase.fill(true);
        mediumScatterQueue =
            alloc.new_object<MediumScatterQueue>(maxQueueSize, alloc, havePhase);
    }

    stats = alloc.new_object<Stats>(maxDepth, alloc);

#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        CUDATrackedMemoryResource *mr =
            dynamic_cast<CUDATrackedMemoryResource *>(memoryResource);
        CHECK(mr);
        size_t endSize = mr->BytesAllocated();
        pathIntegratorBytes += endSize - startSize;
    }
#endif  // PBRT_BUILD_GPU_RENDERER
}

// WavefrontPathIntegrator Method Definitions
Float WavefrontPathIntegrator::Render() {
    if (haveMedia)
        ErrorExit("No support for scenes with media");
    
    if (haveSubsurface)
        ErrorExit("No support for subsurface scattering");

    uint32_t intermediateDataIndex = 0;
    Float nextTimeSave = timeInterval;
    std::string baseFilename = film.GetFilename();
    size_t extPos = baseFilename.find_last_of(".");
    std::string filenameWithoutExt = baseFilename.substr(0, extPos);
    std::string extension = baseFilename.substr(extPos);
    // Get the directory path and filename separately
    size_t lastSlash = baseFilename.find_last_of("/\\");
    std::string directory =
        (lastSlash != std::string::npos) ? baseFilename.substr(0, lastSlash + 1) : "";

    Bounds2i pixelBounds = film.PixelBounds();
    Vector2i resolution = pixelBounds.Diagonal();

    GUI *gui = nullptr;
    // FIXME: camera animation; whatever...
    Transform renderFromCamera =
        camera.GetCameraTransform().RenderFromCamera().startTransform;
    Transform cameraFromRender = Inverse(renderFromCamera);
    Transform cameraFromWorld =
        camera.GetCameraTransform().CameraFromWorld(camera.SampleTime(0.f));
    if (Options->interactive) {
        if (!Options->displayServer.empty())
            ErrorExit(
                "--interactive and --display-server cannot be used at the same time.");
        gui = new GUI(film.GetFilename(), resolution, sceneBounds);
    }

    // Prefetch allocations to GPU memory
#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU)
        PrefetchGPUAllocations();
#endif  // PBRT_BUILD_GPU_RENDERER

    // Launch thread to copy image for display server, if enabled
    if (!Options->displayServer.empty())
        StartDisplayThread();

    Timer timer;

    // Loop over sample indices and evaluate pixel samples
    int firstSampleIndex = 0, lastSampleIndex = samplesPerPixel;
    if (timeLimit > 0.0f) {
        lastSampleIndex = 10000; // Ignore spp limit by setting a large number on it
    }
    // Update sample index range based on debug start, if provided
    if (!Options->debugStart.empty()) {
        std::vector<int> values = SplitStringToInts(Options->debugStart, ',');
        if (values.size() != 1 && values.size() != 2)
            ErrorExit("Expected either one or two integer values for --debugstart.");

        firstSampleIndex = values[0];
        if (values.size() == 2)
            lastSampleIndex = firstSampleIndex + values[1];
        else
            lastSampleIndex = firstSampleIndex + 1;
    }

    ProgressReporter progress(lastSampleIndex - firstSampleIndex, "Rendering",
                              Options->quiet || Options->interactive, Options->useGPU);
    for (int sampleIndex = firstSampleIndex; sampleIndex < lastSampleIndex || gui;
         ++sampleIndex) {
        // Attempt to work around issue #145.
#if !(defined(PBRT_IS_WINDOWS) && defined(PBRT_BUILD_GPU_RENDERER) && \
      __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 1)
        CheckCallbackScope _([&]() {
            return StringPrintf("Wavefront rendering failed at sample %d. Debug with "
                                "\"--debugstart %d\"\n",
                                sampleIndex, sampleIndex);
        });
#endif
        // Keep running the outer for loop but don't take more samples if
        // the GUI is being used so that the user can move the camera, etc.
        if (sampleIndex < lastSampleIndex) {
            // Render image for sample _sampleIndex_
            LOG_VERBOSE("Starting to submit work for sample %d", sampleIndex);
            for (int y0 = pixelBounds.pMin.y; y0 < pixelBounds.pMax.y;
                 y0 += scanlinesPerPass) {
                // Generate camera rays for current scanline range
                RayQueue *cameraRayQueue = CurrentRayQueue(0);
                Do(
                   "Reset ray queue", PBRT_CPU_GPU_LAMBDA() {
                       PBRT_DBG("Starting scanlines at y0 = %d, sample %d / %d\n", y0,
                                sampleIndex, samplesPerPixel);
                       cameraRayQueue->Reset();
                   });

                Transform cameraMotion;
                if (gui)
                    cameraMotion =
                        renderFromCamera * gui->GetCameraTransform() * cameraFromRender;
                GenerateCameraRays(y0, cameraMotion, sampleIndex);
                Do(
                   "Update camera ray stats",
                   PBRT_CPU_GPU_LAMBDA() { stats->cameraRays += cameraRayQueue->Size(); });

                // Trace rays and estimate radiance up to maximum ray depth
                for (int wavefrontDepth = 0; true; ++wavefrontDepth) {
                    // Reset queues before tracing rays
                    RayQueue *nextQueue = NextRayQueue(wavefrontDepth);
                    Do(
                       "Reset queues before tracing rays", PBRT_CPU_GPU_LAMBDA() {
                           nextQueue->Reset();
                           // Reset queues before tracing next batch of rays
                           if (mediumSampleQueue)
                               mediumSampleQueue->Reset();
                           if (mediumScatterQueue)
                               mediumScatterQueue->Reset();

                            basicLightSamplerQueue->Reset();
                            universalLightSamplerQueue->Reset();

                           if (escapedRayQueue)
                               escapedRayQueue->Reset();
                           hitAreaLightQueue->Reset();

                           basicEvalMaterialQueue->Reset();
                           universalEvalMaterialQueue->Reset();

                           if (lightSamplerEvalBuffer)
                               lightSamplerEvalBuffer->Reset();

                           if (bssrdfEvalQueue)
                               bssrdfEvalQueue->Reset();
                           if (subsurfaceScatterQueue)
                               subsurfaceScatterQueue->Reset();
                       });

                    // Follow active ray paths and accumulate radiance estimates
                    GenerateRaySamples(wavefrontDepth, sampleIndex);

                    // Find closest intersections along active rays
                    aggregate->IntersectClosest(
                                                maxQueueSize, CurrentRayQueue(wavefrontDepth), escapedRayQueue,
                                                hitAreaLightQueue, basicEvalMaterialQueue, universalEvalMaterialQueue,
                                                mediumSampleQueue, NextRayQueue(wavefrontDepth));

                    if (wavefrontDepth > 0) {
                        // As above, with the indexing...
                        RayQueue *statsQueue = CurrentRayQueue(wavefrontDepth);
                        Do(
                           "Update indirect ray stats", PBRT_CPU_GPU_LAMBDA() {
                               stats->indirectRays[wavefrontDepth] += statsQueue->Size();
                           });
                    }

                    SampleMediumInteraction(wavefrontDepth);

                    HandleEscapedRays();
                    
                    HandleEmissiveIntersection(sampleIndex);
                    
                    #if (NEEONLY == 1)
                        // We extend the maximum depth by 1 to allow one extra bounce for pure specular surfaces (mirror and glass)
                        // When wavefrontDepth == maxDepth, we "gate" the extension by only pushing new rays if they are pure specular (done in EvaluateMaterialsAndBSDFs)
                        if (wavefrontDepth == maxDepth + 1) break;
                    #else
                        if (wavefrontDepth == maxDepth) break;
                    #endif

                    EvaluateMaterialsAndBSDFs(wavefrontDepth, sampleIndex, cameraMotion);

                    // Evaluate network, if applicable
                    // If lightSampler is VARL, we run EvalOrSample to initialize the lightCut weights for the first sample
                    if (lightSamplerEvalBuffer &&
                        (lightSampler.Is<VARLLightSampler>() ? sampleIndex == firstSampleIndex &&  wavefrontDepth == 0: true)){
                        GPUWait();
                        LOG_VERBOSE("Preprocessing light sampler eval buffer with size %d", lightSamplerEvalBuffer->Size());
                        lightSampler.EvalOrSample((const float*) lightSamplerEvalBuffer->Inputs(), lightSamplerEvalBuffer->Outputs<0>(), (const float*) lightSamplerEvalBuffer->Outputs<1>(), lightSamplerEvalBuffer->Size());
                    }

                    // Sample light sources, then enqueue shadow rays for tracing
                    SampleLightSources(sampleIndex, timer.ElapsedSeconds());

                    // Do immediately so that we have space for shadow rays for subsurface..
                    TraceShadowRays(wavefrontDepth);

                    SampleSubsurface(wavefrontDepth);
                }

                // CUDA_CHECK(cudaDeviceSynchronize());
                UpdateFilm();
            }

            // Copy updated film pixels to buffer for the display server.
            if (Options->useGPU && !Options->displayServer.empty())
                UpdateDisplayRGBFromFilm(pixelBounds);

            progress.Update();
        }

        if (gui) {
            RGB *rgb = gui->MapFramebuffer();
            UpdateFramebufferFromFilm(pixelBounds, gui->exposure, rgb);
            gui->UnmapFramebuffer();

            if (gui->printCameraTransform) {
                SquareMatrix<4> cfw =
                    (Inverse(gui->GetCameraTransform()) * cameraFromWorld).GetMatrix();
                Printf("Current camera transform:\nTransform [ ");
                for (int i = 0; i < 16; ++i)
                    Printf("%f ", cfw[i % 4][i / 4]);
                Printf("]\n");
                std::fflush(stdout);
                gui->printCameraTransform = false;
            }

            DisplayState state = gui->RefreshDisplay();
            if (state == DisplayState::EXIT)
                break;
            else if (state == DisplayState::RESET) {
                sampleIndex = firstSampleIndex - 1;
                ParallelFor(
                    "Reset pixels", resolution.x * resolution.y,
                    PBRT_CPU_GPU_LAMBDA(int i) {
                        int x = i % resolution.x, y = i / resolution.x;
                        film.ResetPixel(pixelBounds.pMin + Vector2i(x, y));
                    });
            }
        }

        const Float currentTime = timer.ElapsedSeconds();
        // Append intermediate data
        if (intermediateDataIndex < nIntermediateDataPoints) {
            if (sppInterval > 0 && 
                ((sampleIndex + 1) % sppInterval == 0 ||
                sampleIndex == lastSampleIndex - 1)) {
                    AppendIntermediateData(pixelBounds, intermediateDataIndex++, currentTime, sampleIndex + 1);
            }
            else if (timeInterval > 0.0f && currentTime > nextTimeSave) {
                nextTimeSave = currentTime + timeInterval;
                AppendIntermediateData(pixelBounds, intermediateDataIndex++, currentTime, sampleIndex + 1);
            }
        }
        
        // Break if time limit is reached (making sure we append the last intermediate data)
        if (timeLimit > 0 && currentTime > timeLimit) {
            AppendIntermediateData(pixelBounds, intermediateDataIndex++, currentTime, sampleIndex + 1);
            break;
        }

        // Populate training buffer before calling trainStep
        if (((timeLimit > 0.0f && currentTime < trainingBudgetRatio * timeLimit) ||
            (timeLimit == 0.0f && sampleIndex < trainingBudgetRatio * samplesPerPixel)) &&
            lightSamplerTrainBuffer) {
            ParallelFor("Populate Training Buffer", maxQueueSize, PBRT_CPU_GPU_LAMBDA(int pixelIndex) {
                if (pixelIndex >= maxQueueSize)
                    return;

                int depth = std::min(lightSamplerOptStateBuffer.curDepth[pixelIndex], MAX_TRAIN_DEPTH);

                const LightBVHTree* lightBVHTree = lightSampler.getLightBVHTree();
                if (!lightBVHTree) return;

                for (int i = 0; i < depth; i++){
                    const LightSampleRecordItem& record = lightSamplerOptStateBuffer.records[i][pixelIndex];
                    const Point3f ctxP = record.ctx.p();
                    const Normal3f ctxN = record.ctx.ns;
                    const Vector3f ctxWo = record.ctx.wo;
                    const LightSamplerResidualInfo residualInfo = {ctxP, ctxN, ctxWo};
                    const Vector3f pOffset = normalizePosition(ctxP, sceneBounds);
                    const Vector2f nSph = cartesianToSphericalNormalized(ctxN);

                    #if (LEARN_PRODUCT_SAMPLING == 0)
                        #if (NEURAL_GRID_DISCRETIZATION == 0)
                            const LightSamplerTrainInput in = {pOffset, nSph};
                        #else
                            const Vector3f pOffsetDiscretized = discretizeVec(pOffset, POSITIONAL_DISCRETIZATION_RESOLUTION);
                            const LightSamplerTrainInput in = {pOffsetDiscretized};
                        #endif
                    #else
                        const Vector3f woNormalized = normalizeDirection(record.ctx.wo);
                        #if (NEURAL_GRID_DISCRETIZATION == 0)
                            const LightSamplerTrainInput in = {pOffset, nSph, woNormalized};
                        #else
                            const Vector3f pOffsetDiscretized = discretizeVec(pOffset, POSITIONAL_DISCRETIZATION_RESOLUTION);
                            const Vector3f woNormalizedDiscretized = discretizeVec(woNormalized, DIRECTIONAL_DISCRETIZATION_RESOLUTION);
                            const LightSamplerTrainInput in = {pOffsetDiscretized, woNormalizedDiscretized};
                        #endif
                    #endif

                    RGB LRGB = film.ToOutputRGB(record.L, record.lambda);

                    // If the light sampler is NeuralSLCLightSampler, we can ignore zero radiance samples
                    if (lightSampler.Is<NeuralSLCLightSampler>()){
                        if (record.LPDF == 0.f || LRGB.Average() / record.LPDF == 0.f) continue;
                    }

                    uint32_t lightIndex = record.lightIndex;
                    lightSamplerTrainBuffer->Push(in, LRGB, lightIndex, residualInfo, record.LPDF);
                }

                // // Not resetting lightSamplerOptStateBuffer before next sample, since we want to accumulate up to max depth
                // Resetting depth to 0
                lightSamplerOptStateBuffer.Reset(pixelIndex);
            });
            GPUWait();
            LOG_VERBOSE("Populated training buffer for sample %d. Total size: %d", sampleIndex, lightSamplerTrainBuffer->Size());
            Float loss_i = lightSampler.TrainStep((const float*) lightSamplerTrainBuffer->Inputs(), lightSamplerTrainBuffer->Outputs<0>(), lightSamplerTrainBuffer->Outputs<1>(), (const float*) lightSamplerTrainBuffer->Outputs<2>(), (const float*) lightSamplerTrainBuffer->Outputs<3>(), lightSamplerTrainBuffer->Size());
            LOG_VERBOSE("Loss for sample %d: %f", sampleIndex, loss_i);
        } 
    }
    if (gui) {
        delete gui;
        gui = nullptr;
    }

    progress.Done();

#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU)
        GPUWait();
#endif  // PBRT_BUILD_GPU_RENDERER
    Float seconds = timer.ElapsedSeconds();

    // Shut down display server thread, if active
    StopDisplayThread();

    // Save intermediate data to disk
    if (intermediateDataIndex > 0) {
        // Move buffer to host
        IntermediateAuxData* intermediateAuxDataPtr = new IntermediateAuxData[nIntermediateDataPoints];
        CUDA_CHECK(cudaMemcpy(intermediateAuxDataPtr, intermediateAuxDataBuffer, nIntermediateDataPoints * sizeof(IntermediateAuxData), cudaMemcpyDeviceToHost));
        
        RGB* intermediateRGBDataPtr = new RGB[nIntermediateDataPoints * resolution.x * resolution.y];
        CUDA_CHECK(cudaMemcpy(intermediateRGBDataPtr, intermediateRGBDataBuffer, nIntermediateDataPoints * resolution.x * resolution.y * sizeof(RGB), cudaMemcpyDeviceToHost));
        
        for (uint32_t i = 0; i <= intermediateDataIndex; i++) {
            Image image(PixelFormat::Float, Point2i(pixelBounds.Diagonal()), {"R", "G", "B"});
            ParallelFor2D(pixelBounds, [&](Point2i p) {
                const uint32_t pixelIndex = p.x - pixelBounds.pMin.x + (p.y - pixelBounds.pMin.y) * resolution.x;
                const RGB rgb = intermediateRGBDataPtr[i * resolution.x * resolution.y + pixelIndex];
        
                Point2i pOffset(p.x - pixelBounds.pMin.x, p.y - pixelBounds.pMin.y);
                image.SetChannels(pOffset, {rgb[0], rgb[1], rgb[2]});
            });
            
            const float time_i = intermediateAuxDataPtr[i].time;
            const int spp_i = intermediateAuxDataPtr[i].spp;

            ImageMetadata metadata;
            metadata.renderTimeSeconds = time_i;
            metadata.samplesPerPixel = spp_i;
            metadata.pixelBounds = pixelBounds;

            std::string intermediateFilename =
                StringPrintf("%s/%d_%.1f%s", directory.c_str(), spp_i,
                time_i, extension.c_str());
            // Write intermediate image
            LOG_VERBOSE("Saving intermediate image at %d SPP: %s",
                spp_i, intermediateFilename.c_str());
                image.Write(intermediateFilename, metadata);
        }

        // Free intermediate data buffers
        CUDA_CHECK(cudaFree(intermediateAuxDataBuffer));
        CUDA_CHECK(cudaFree(intermediateRGBDataBuffer));
        delete[] intermediateAuxDataPtr;
        delete[] intermediateRGBDataPtr;
    }
    
    GPUWait();
    printf("WavefrontPathIntegrator::Render() done\n");
    return seconds;
}

void WavefrontPathIntegrator::HandleEscapedRays() {
    if (!escapedRayQueue)
        return;
    ForAllQueued(
        "Handle escaped rays", escapedRayQueue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const EscapedRayWorkItem w) {
            // Compute weighted radiance for escaped ray
            SampledSpectrum L(0.f);
            for (const auto &light : *infiniteLights) {
                if (SampledSpectrum Le = light.Le(Ray(w.rayo, w.rayd), w.lambda); Le) {
                    // Compute path radiance contribution from infinite light
                    PBRT_DBG("L %f %f %f %f beta %f %f %f %f Le %f %f %f %f\n", L[0], L[1],
                             L[2], L[3], w.beta[0], w.beta[1], w.beta[2], w.beta[3],
                             Le[0], Le[1], Le[2], Le[3]);
                    PBRT_DBG("depth %d specularBounce %d pdf uni %f %f %f %f "
                             "pdf nee %f %f %f %f\n",
                             w.depth, w.specularBounce,
                             w.r_u[0], w.r_u[1], w.r_u[2], w.r_u[3],
                             w.r_l[0], w.r_l[1], w.r_l[2], w.r_l[3]);

                    if (w.depth == 0 || w.specularBounce) {
                        L += w.beta * Le / w.r_u.Average();
                    } else {
                        // Compute MIS-weighted radiance contribution from infinite light
                        LightSampleContext ctx = w.prevIntrCtx;
                        LOG_VERBOSE("Computing MIS-weighted contribution from escaped ray");
                        Float lightChoicePDF = lightSampler.PMF(ctx, light);
                        SampledSpectrum r_l =
                            w.r_l * lightChoicePDF * light.PDF_Li(ctx, w.rayd, true);
                        L += w.beta * Le / (w.r_u + r_l).Average();
                    }
                }
            }

            // Update pixel radiance if ray's radiance is nonzero
            if (L) {
                PBRT_DBG("Added L %f %f %f %f for escaped ray pixel index %d\n", L[0],
                         L[1], L[2], L[3], w.pixelIndex);

                L += pixelSampleState.L[w.pixelIndex];
                pixelSampleState.L[w.pixelIndex] = L;
            }
        });
}

void WavefrontPathIntegrator::HandleEmissiveIntersection(int sampleIndex) {
    ForAllQueued(
    "Handle Emissive Intersection", hitAreaLightQueue, maxQueueSize,
    PBRT_CPU_GPU_LAMBDA(const HitAreaLightWorkItem w) {
        // Find emitted radiance from surface that ray hit
        SampledSpectrum Le = w.areaLight.L(w.p, w.n, w.uv, w.wo, w.lambda);
        if (!Le)
            return;
        
        // Compute area light's weighted radiance contribution to the path
        SampledSpectrum L(0.f);
        if (w.depth == 0 || w.specularBounce) {
            L = w.beta * Le / w.r_u.Average();
        } 
        #if (NEEONLY == 0)
            else {

            #if VPL_USED
                const LightSampler &lSampler = VPLDirectlightsampler;
            #else
                const LightSampler &lSampler = lightSampler;
            #endif

                // Decide between default or exploration lightSampler
                const float explorationSamplingRatio_i = sampleIndex < nExplorationWarmupSamples ? 1.f : explorationSamplingRatio;

                // Compute MIS-weighted radiance contribution from area light
                Vector3f wi = -w.wo;
                LightSampleContext ctx = w.prevIntrCtx;

                const Float lightChoiceExplorationPDF = explorationSamplingRatio_i > 0.f ? lightSamplerExploration.PMF(ctx, w.areaLight) : 0.f;
                DCHECK(!IsNaN(lightChoiceExplorationPDF));
                const Float lightChoiceGuidedPDF = explorationSamplingRatio_i < 1.f
                                                       ? lSampler.PMF(ctx, w.areaLight)
                                                       : 0.f;
                DCHECK(!IsNaN(lightChoiceGuidedPDF));
                const Float lightChoicePDF = (1.f - explorationSamplingRatio_i) * lightChoiceGuidedPDF + explorationSamplingRatio_i * lightChoiceExplorationPDF;
                const Float lightShapePDF = w.areaLight.PDF_Li(ctx, wi, true);
                const Float lightPDF = lightChoicePDF * lightShapePDF;

                SampledSpectrum r_u = w.r_u;
                SampledSpectrum r_l = w.r_l * lightPDF;
                Float denom = (r_u + r_l).Average();
                L = denom != 0.f ? w.beta * Le / denom : SampledSpectrum(0.f);

                #if (!VPL_USED)
                if (sampleIndex < trainingBudgetRatio * samplesPerPixel && lightSampler.isTraining(w.pixelIndex, w.depth)){
                    SampledSpectrum Ld_optim = Le;
                    SampledSpectrum r_u_optim = SampledSpectrum(0.f);
                    SampledSpectrum r_l_optim = r_l;
                    #if (LEARN_PRODUCT_SAMPLING == 1)
                        Ld_optim *= w.beta; // Multiply only current BSDF*cos term (not accumulated)
                        r_u_optim = w.r_u;
                    #endif
                    Float denom_optim = (r_u_optim + r_l_optim).Average();
                    // Ld_optim = denom_optim != 0.f ? Ld_optim / denom_optim : SampledSpectrum(0.f);

                    const LightBVHTree* lightTree = lightSampler.getLightBVHTree();
                    if (lightTree){
                        DCHECK(lightTree->lightToIndex.HasKey(w.areaLight));
                        uint32_t lightIndex = lightTree->lightToIndex[w.areaLight];
                        lightSamplerOptStateBuffer.incrementDepth(w.pixelIndex, ctx, w.lambda, Ld_optim, denom_optim, lightIndex);
                    }
                }
                #endif
            }
        #endif

        PBRT_DBG("Added L %f %f %f %f for pixel index %d\n", L[0], L[1], L[2], L[3],
                    w.pixelIndex);

        // Update _L_ in _PixelSampleState_ for area light's radiance
        L += pixelSampleState.L[w.pixelIndex];
        pixelSampleState.L[w.pixelIndex] = L;
    });
}


// EvaluateMaterialCallback Definition
struct EvaluateLightSourcesMaterialCallback {
    WavefrontPathIntegrator *integrator;
    int sampleIndex;
    float CurrentTime;
    // EvaluateMaterialCallback Public Methods
    template <typename ConcreteMaterial>
    void operator()() {
        if constexpr (!std::is_same_v<ConcreteMaterial, MixMaterial>)
            integrator->SampleLightSources<ConcreteMaterial>(sampleIndex, CurrentTime);
    }
};

void WavefrontPathIntegrator::SampleLightSources(int sampleIndex, float CurrentTime) {
    ForEachType(EvaluateLightSourcesMaterialCallback{this, sampleIndex, CurrentTime},
                Material::Types());
}

template <typename ConcreteMaterial>
void WavefrontPathIntegrator::SampleLightSources(int sampleIndex, float CurrentTime) {
    int index = Material::TypeIndex<ConcreteMaterial>();
    if (haveBasicEvalMaterial[index])
        SampleLightSources<ConcreteMaterial, BasicTextureEvaluator>(
            basicLightSamplerQueue, sampleIndex, CurrentTime);
    if (haveUniversalEvalMaterial[index])
        SampleLightSources<ConcreteMaterial, UniversalTextureEvaluator>(
            universalLightSamplerQueue, sampleIndex, CurrentTime);
}

template <typename ConcreteMaterial, typename TextureEvaluator>
void WavefrontPathIntegrator::SampleLightSources(LightSamplerQueue *lightSamplerQueue, int sampleIndex, float CurrentTime) {

    // Construct _desc_ for material/texture evaluation kernel
    std::string desc = StringPrintf(
        "LightSampler %s + Sample light sources (%s tex)", ConcreteMaterial::Name(),
        std::is_same_v<TextureEvaluator, BasicTextureEvaluator> ? "Basic" : "Universal");

    auto lqueue = lightSamplerQueue->Get<LightSamplerWorkItem<ConcreteMaterial>>();
    ForAllQueued(
        desc.c_str(), lqueue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const LightSamplerWorkItem<ConcreteMaterial> w) {
            TextureEvaluator texEval;

            // Get BSDF at intersection point
            SampledWavelengths lambda = w.lambda;
            MaterialEvalContext ctx = w.GetMaterialEvalContext();
            using ConcreteBxDF = typename ConcreteMaterial::BxDF;
            ConcreteBxDF bxdf = w.material->GetBxDF(texEval, ctx, lambda);
            BSDF bsdf(ctx.ns, ctx.dpdus, &bxdf);
            
            // Regularize BSDF, if appropriate
            if (regularize && w.anyNonSpecularBounces)
                bsdf.Regularize();
            
            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];

            // Sample light and enqueue shadow ray at intersection point
            BxDFFlags flags = bsdf.Flags();
            DCHECK(IsNonSpecular(flags));

            const Vector3f wo = w.wo;
            const Normal3f ns = w.ns;

            LightSampleContext ctx_ls(w.pi, w.n, ns, wo, w.pixelIndex);
            if (IsReflective(flags) && !IsTransmissive(flags))
                ctx_ls.pi = OffsetRayOrigin(ctx_ls.pi, w.n, wo);
            else if (IsTransmissive(flags) && IsReflective(flags))
                ctx_ls.pi = OffsetRayOrigin(ctx_ls.pi, w.n, -wo);

            // Decide between default or exploration lightSampler
            const float explorationSamplingRatio_i = sampleIndex < nExplorationWarmupSamples ? 1.f : explorationSamplingRatio;
            const bool isExploration = raySamples.direct.uer < explorationSamplingRatio_i;
            const LightSampler* lSamplers[2] = {&lightSampler, &lightSamplerExploration};
            const Float MISmultipliers[2] = {1.f - explorationSamplingRatio_i, explorationSamplingRatio_i};
            
            {
                pstd::optional<SampledLight> sampledLight =
                    isExploration
                        ? lightSamplerExploration.Sample(ctx_ls, raySamples.direct.uc)
                        : lightSampler.Sample(ctx_ls, raySamples.direct.uc);
                if (!sampledLight)
                    return;

                Light light = sampledLight->light;
                const Float lightChoiceSampledPDF = sampledLight->p;
                const Float lightChoiceOtherPDF =
                    MISmultipliers[!isExploration] > 0.f
                        ? lSamplers[!isExploration]->PMF(ctx_ls, light)
                        : 0.f;
                const Float lightChoicePDF =
                    MISmultipliers[isExploration] * lightChoiceSampledPDF +
                    MISmultipliers[!isExploration] * lightChoiceOtherPDF;

                // Sample light source and evaluate BSDF for direct lighting
                pstd::optional<LightLiSample> ls =
                    light.SampleLi(ctx_ls, raySamples.direct.u, lambda, true);
                if (!ls || !ls->L || ls->pdf == 0)
                    return;
                const Float lightShapePDF = ls->pdf;
                Vector3f wi = ls->wi;
                SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);

                // Compute path throughput and path PDFs for light sample
                SampledSpectrum beta = w.beta * f * AbsDot(wi, ns);
                const Float lightPDF = lightShapePDF * lightChoicePDF;
#if (NEEONLY == 1)
                const Float bsdfPDF = 0.f;
#else
                // This causes r_u to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                const Float bsdfPDF =
                    IsDeltaLight(light.Type()) ? 0.f : bsdf.PDF<ConcreteBxDF>(wo, wi);
#endif
                SampledSpectrum r_u = w.r_u * bsdfPDF;
                SampledSpectrum r_l = w.r_u * lightPDF;

                // Enqueue shadow ray with tentative radiance contribution
                SampledSpectrum Ld = beta * ls->L;
                Ray ray = SpawnRayTo(w.pi, w.n, w.time, ls->pLight.pi, ls->pLight.n);
                // Initialize _ray_ medium if media are present
                if (haveMedia)
                    ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                     : w.mediumInterface.inside;

                shadowRayQueue->Push(ShadowRayWorkItem{ray, 1 - ShadowEpsilon, lambda, Ld,
                                                       r_u, r_l, true, w.depth,
                                                       w.pixelIndex, true});

                if (((timeLimit > 0.0f && CurrentTime < trainingBudgetRatio * timeLimit) ||
                    (timeLimit == 0.0f && sampleIndex < trainingBudgetRatio * samplesPerPixel)) &&
                    lightSampler.isTraining(w.pixelIndex, w.depth)) {
                    SampledSpectrum Ld_optim = ls->L;
                    Float r_u_optim = 0.f;
                    Float r_l_optim = lightPDF;

                    #if (LEARN_PRODUCT_SAMPLING == 1)
                        Ld_optim *= f * AbsDot(wi, ns); // Multiply only current BSDF*cos term (not accumulated)
                    #endif
                    #if (NEEONLY == 0)
                        r_u_optim = bsdfPDF;
                    #endif
                    const Float Ld_optim_denom = r_u_optim + r_l_optim;
                    // Ld_optim = Ld_optim_denom > 0.f ? Ld_optim / Ld_optim_denom : SampledSpectrum(0.f);

                    const LightBVHTree* lightTree = lightSampler.getLightBVHTree();
                    if (lightTree){
                        uint32_t lightIndex = lightTree->lightToIndex[light];
                        lightSamplerOptStateBuffer.incrementDepth(w.pixelIndex, ctx_ls, w.lambda, Ld_optim, Ld_optim_denom, lightIndex);
                    }
                }
            }

            #if VPL_USED
            {
                const LightSampler *DirectlSamplers = &VPLDirectlightsampler;
                pstd::optional<SampledLight> sampledLight =
                    VPLDirectlightsampler.Sample(ctx_ls, raySamples.direct.uc);
                if (!sampledLight)
                    return;
            
                Light light = sampledLight->light;
                const Float lightChoicePDF = DirectlSamplers->PMF(ctx_ls, light);
            
                // Sample light source and evaluate BSDF for direct lighting
                pstd::optional<LightLiSample> ls =
                    light.SampleLi(ctx_ls, raySamples.direct.u, lambda,
                    true);
                if (!ls || ls->pdf == 0)
                    return;
                const Float lightShapePDF = ls->pdf;
                Vector3f wi = ls->wi;
                SampledSpectrum f = bsdf.f<ConcreteBxDF>(wo, wi);
            
                // Compute path throughput and path PDFs for light sample
                SampledSpectrum beta = w.beta * f * AbsDot(wi, ns);
                const Float lightPDF = lightShapePDF * lightChoicePDF;
    #if (NEEONLY == 1)
                const Float bsdfPDF = 0.f;
    #else
                // This causes r_u to be zero for the shadow ray, so that
                // part of MIS just becomes a no-op.
                const Float bsdfPDF =
                    IsDeltaLight(light.Type()) ? 0.f :
                    bsdf.PDF<ConcreteBxDF>(wo, wi);
    #endif
                SampledSpectrum r_u = w.r_u * bsdfPDF;
                SampledSpectrum r_l = w.r_u * lightPDF;
            
                // Enqueue shadow ray with tentative radiance contribution
                SampledSpectrum Ld = beta * ls->L;
                Ray ray = SpawnRayTo(w.pi, w.n, w.time, ls->pLight.pi,
                ls->pLight.n);
                // Initialize _ray_ medium if media are present
                if (haveMedia)
                    ray.medium = Dot(ray.d, w.n) > 0 ?
                    w.mediumInterface.outside: w.mediumInterface.inside;
            
                shadowRayQueue->Push(ShadowRayWorkItem{ray, 1 - ShadowEpsilon, lambda, Ld,
                                                        r_u, r_l, true,
                                                        w.depth,
                                                        w.pixelIndex, false});
            }
            #endif
        }
    );
}

// WavefrontPathIntegrator Surface Scattering Methods

void WavefrontPathIntegrator::TraceShadowRays(int wavefrontDepth) {
#if VPL_USED
    if (haveMedia)
        aggregate->IntersectShadowTr(maxQueueSize * 2, shadowRayQueue,
                                     &pixelSampleState, &lightSamplerOptStateBuffer,
                                     lightSampler);
    else
        aggregate->IntersectShadow(maxQueueSize * 2, shadowRayQueue,
                                   &pixelSampleState, &lightSamplerOptStateBuffer,
                                   lightSampler);
#else
    if (haveMedia)
        aggregate->IntersectShadowTr(maxQueueSize, shadowRayQueue, &pixelSampleState,
                                     &lightSamplerOptStateBuffer, lightSampler);
    else
        aggregate->IntersectShadow(maxQueueSize, shadowRayQueue, &pixelSampleState,
                                   &lightSamplerOptStateBuffer, lightSampler);
#endif
    // Reset shadow ray queue
    Do(
        "Reset shadowRayQueue", PBRT_CPU_GPU_LAMBDA() {
            stats->shadowRays[wavefrontDepth] += shadowRayQueue->Size();
            shadowRayQueue->Reset();
        });
}

WavefrontPathIntegrator::Stats::Stats(int maxDepth, Allocator alloc)
    : indirectRays(maxDepth + 1, alloc), shadowRays(maxDepth, alloc) {}

std::string WavefrontPathIntegrator::Stats::Print() const {
    std::string s;
    s += StringPrintf("    %-42s               %12" PRIu64 "\n", "Camera rays",
                      cameraRays);
    for (int i = 1; i < indirectRays.size(); ++i)
        s += StringPrintf("    %-42s               %12" PRIu64 "\n",
                          StringPrintf("Indirect rays, depth %-3d", i), indirectRays[i]);
    for (int i = 0; i < shadowRays.size(); ++i)
        s += StringPrintf("    %-42s               %12" PRIu64 "\n",
                          StringPrintf("Shadow rays, depth %-3d", i), shadowRays[i]);
    return s;
}

#ifdef PBRT_BUILD_GPU_RENDERER
void WavefrontPathIntegrator::PrefetchGPUAllocations() {
    int deviceIndex;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));
    int hasConcurrentManagedAccess;
    CUDA_CHECK(cudaDeviceGetAttribute(&hasConcurrentManagedAccess,
                                      cudaDevAttrConcurrentManagedAccess, deviceIndex));

    // Copy all of the scene data structures over to GPU memory.  This
    // ensures that there isn't a big performance hitch for the first batch
    // of rays as that stuff is copied over on demand.
    if (hasConcurrentManagedAccess) {
        // Set things up so that we can still have read from the
        // WavefrontPathIntegrator struct on the CPU without hurting
        // performance. (This makes it possible to use the values of things
        // like WavefrontPathIntegrator::haveSubsurface to conditionally launch
        // kernels according to what's in the scene...)
        CUDA_CHECK(cudaMemAdvise(this, sizeof(*this), cudaMemAdviseSetReadMostly,
                                 /* ignored argument */ 0));
        CUDA_CHECK(cudaMemAdvise(this, sizeof(*this), cudaMemAdviseSetPreferredLocation,
                                 deviceIndex));

        // Copy all of the scene data structures over to GPU memory.  This
        // ensures that there isn't a big performance hitch for the first batch
        // of rays as that stuff is copied over on demand.
        CUDATrackedMemoryResource *mr =
            dynamic_cast<CUDATrackedMemoryResource *>(memoryResource);
        CHECK(mr);
        mr->PrefetchToGPU();
    } else {
        // TODO: on systems with basic unified memory, just launching a
        // kernel should cause everything to be copied over. Is an empty
        // kernel sufficient?
    }
}
#endif  // PBRT_BUILD_GPU_RENDERER

void WavefrontPathIntegrator::StartDisplayThread() {
    Bounds2i pixelBounds = film.PixelBounds();
    Vector2i resolution = pixelBounds.Diagonal();

#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        // Allocate staging memory on the GPU to store the current WIP
        // image.
        CUDA_CHECK(cudaMalloc(&displayRGB, resolution.x * resolution.y * sizeof(RGB)));
        CUDA_CHECK(cudaMemset(displayRGB, 0, resolution.x * resolution.y * sizeof(RGB)));

        // Host-side memory for the WIP Image.  We'll just let this leak so
        // that the lambda passed to DisplayDynamic below doesn't access
        // freed memory after Render() returns...
        displayRGBHost = new RGB[resolution.x * resolution.y];

        // Note that we can't just capture |this| for the member variables
        // below because with managed memory on Windows, the CPU and GPU
        // can't be accessing the same memory concurrently...
        copyThread = new std::thread([exitCopyThread = this->exitCopyThread,
                                      displayRGBHost = this->displayRGBHost,
                                      displayRGB = this->displayRGB, resolution]() {
            GPURegisterThread("DISPLAY_SERVER_COPY_THREAD");

            // Copy back to the CPU using a separate stream so that we can
            // periodically but asynchronously pick up the latest results
            // from the GPU.
            cudaStream_t memcpyStream;
            CUDA_CHECK(cudaStreamCreate(&memcpyStream));
            GPUNameStream(memcpyStream, "DISPLAY_SERVER_COPY_STREAM");

            // Copy back to the host from the GPU buffer, without any
            // synthronization.
            while (!*exitCopyThread) {
                CUDA_CHECK(cudaMemcpyAsync(displayRGBHost, displayRGB,
                                           resolution.x * resolution.y * sizeof(RGB),
                                           cudaMemcpyDeviceToHost, memcpyStream));
                std::this_thread::sleep_for(std::chrono::milliseconds(50));

                CUDA_CHECK(cudaStreamSynchronize(memcpyStream));
            }

            // Copy one more time to get the final image before exiting.
            CUDA_CHECK(cudaMemcpy(displayRGBHost, displayRGB,
                                  resolution.x * resolution.y * sizeof(RGB),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
        });

        // Now on the CPU side, give the display system a lambda that
        // copies values from |displayRGBHost| into its buffers used for
        // sending messages to the display program (i.e., tev).
        DisplayDynamic(
            film.GetFilename(), {resolution.x, resolution.y}, {"R", "G", "B"},
            [resolution, this](Bounds2i b, pstd::span<pstd::span<float>> displayValue) {
                int index = 0;
                for (Point2i p : b) {
                    RGB rgb = displayRGBHost[p.x + p.y * resolution.x];
                    displayValue[0][index] = rgb.r;
                    displayValue[1][index] = rgb.g;
                    displayValue[2][index] = rgb.b;
                    ++index;
                }
            });
    } else
#endif  // PBRT_BUILD_GPU_RENDERER
        DisplayDynamic(
            film.GetFilename(), Point2i(pixelBounds.Diagonal()), {"R", "G", "B"},
            [pixelBounds, this](Bounds2i b, pstd::span<pstd::span<float>> displayValue) {
                int index = 0;
                for (Point2i p : b) {
                    RGB rgb =
                        film.GetPixelRGB(pixelBounds.pMin + p, 1.f /* splat scale */);
                    for (int c = 0; c < 3; ++c)
                        displayValue[c][index] = rgb[c];
                    ++index;
                }
            });
}

void WavefrontPathIntegrator::UpdateDisplayRGBFromFilm(Bounds2i pixelBounds) {
#ifdef PBRT_BUILD_GPU_RENDERER
    Vector2i resolution = pixelBounds.Diagonal();
    GPUParallelFor(
        "Update Display RGB Buffer", resolution.x * resolution.y,
        PBRT_CPU_GPU_LAMBDA(int index) {
            Point2i p(index % resolution.x, index / resolution.x);
            displayRGB[index] = film.GetPixelRGB(p + pixelBounds.pMin);
        });
#endif  //  PBRT_BUILD_GPU_RENDERER
}

void WavefrontPathIntegrator::StopDisplayThread() {
#ifdef PBRT_BUILD_GPU_RENDERER
    if (Options->useGPU) {
        // Wait until rendering is all done before we start to shut down the
        // display stuff..
        if (!Options->displayServer.empty()) {
            *exitCopyThread = true;
            copyThread->join();
            delete copyThread;
            copyThread = nullptr;
        }

        // Another synchronization to make sure no kernels are running on the
        // GPU so that we can safely access unified memory from the CPU.
        GPUWait();
    }
#endif  // PBRT_BUILD_GPU_RENDERER
}

void WavefrontPathIntegrator::UpdateFramebufferFromFilm(Bounds2i pixelBounds,
                                                        Float exposure, RGB *rgb) {
    Vector2i resolution = pixelBounds.Diagonal();
    ParallelFor(
        "Update framebuffer", resolution.x * resolution.y,
        PBRT_CPU_GPU_LAMBDA(int index) {
            Point2i p(index % resolution.x, index / resolution.x);
            rgb[index] = exposure * film.GetPixelRGB(p + film.PixelBounds().pMin);
        });
}

void WavefrontPathIntegrator::AppendIntermediateData(Bounds2i pixelBounds, uint32_t bufferIndex, float time, uint32_t spp) {
    #ifdef PBRT_BUILD_GPU_RENDERER
    Vector2i resolution = pixelBounds.Diagonal();
    GPUParallelFor(
        "Append to Intermediate Data RGB Buffer", resolution.x * resolution.y,
        PBRT_CPU_GPU_LAMBDA(int pixelIndex) {
            Point2i p(pixelIndex % resolution.x, pixelIndex / resolution.x);
            const RGB rgb = film.GetPixelRGB(p + pixelBounds.pMin);
            
            const uint32_t index = pixelIndex + bufferIndex * resolution.x * resolution.y;
            intermediateRGBDataBuffer[index] = rgb;
        });

    Do(
        "Append to Intermediate Data Time Buffer", PBRT_CPU_GPU_LAMBDA() {
            intermediateAuxDataBuffer[bufferIndex] = {time, spp};
        }
    );
    #endif  //  PBRT_BUILD_GPU_RENDERER

}

}  // namespace pbrt
