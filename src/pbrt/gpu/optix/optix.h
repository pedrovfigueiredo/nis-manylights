// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_GPU_OPTIX_OPTIX_H
#define PBRT_GPU_OPTIX_OPTIX_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/medium.h>
#include <pbrt/base/shape.h>
#include <pbrt/base/texture.h>
#include <pbrt/util/pstd.h>
#include <pbrt/wavefront/workitems.h>
#include <pbrt/wavefront/workqueue.h>

#include <optix.h>

namespace pbrt {

class TriangleMesh;
class BilinearPatchMesh;

struct TriangleMeshRecord {
    const TriangleMesh *mesh;
    Material material;
    FloatTexture alphaTexture;
    pstd::span<Light> areaLights;
    MediumInterface *mediumInterface;
};

struct BilinearMeshRecord {
    const BilinearPatchMesh *mesh;
    Material material;
    FloatTexture alphaTexture;
    pstd::span<Light> areaLights;
    MediumInterface *mediumInterface;
};

struct QuadricRecord {
    Shape shape;
    Material material;
    FloatTexture alphaTexture;
    Light areaLight;
    MediumInterface *mediumInterface;
};

struct RayIntersectParameters {
    OptixTraversableHandle traversable;

    const RayQueue *rayQueue;

    // closest hit
    RayQueue *nextRayQueue;
    EscapedRayQueue *escapedRayQueue;
    HitAreaLightQueue *hitAreaLightQueue;
    MaterialEvalQueue *basicEvalMaterialQueue, *universalEvalMaterialQueue;
    MediumSampleQueue *mediumSampleQueue;

    // shadow rays
    ShadowRayQueue *shadowRayQueue;
    SOA<PixelSampleState> pixelSampleState;

    // light sampling
    LightSamplerOptStateBuffer lightSamplerOptStateBuffer;

    // Subsurface scattering...
    SubsurfaceScatterQueue *subsurfaceScatterQueue;
    LightSampler lightSampler;
};

}  // namespace pbrt

#endif  // PBRT_GPU_OPTIX_OPTIX_H
