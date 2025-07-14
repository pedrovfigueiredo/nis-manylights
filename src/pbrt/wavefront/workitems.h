// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_WAVEFRONT_WORKITEMS_H
#define PBRT_WAVEFRONT_WORKITEMS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/sampler.h>
#include <pbrt/film.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/materials.h>
#include <pbrt/ray.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/soa.h>
#include <pbrt/wavefront/workqueue.h>

#include <pbrt/lightsamplers_constants.h>

namespace pbrt {

// RaySamples Definition
struct RaySamples {
    // RaySamples Public Members
    struct {
        Point2f u;
        Float uc;
        Float uer; // random number to sample exploration lightSampler
    } direct;
    struct {
        Float uc, rr;
        Point2f u;
    } indirect;
    struct {
        Float uc;
        Point2f u;
    } subsurface;
    bool haveSubsurface;
};

template <>
struct SOA<RaySamples> {
  public:
    SOA() = default;

    SOA(int size, Allocator alloc) {
        direct = alloc.allocate_object<Float4>(size);
        indirect = alloc.allocate_object<Float4>(size);
        subsurface = alloc.allocate_object<Float4>(size);
        mediaDist = alloc.allocate_object<Float>(size);
        mediaMode = alloc.allocate_object<Float>(size);
    }

    PBRT_CPU_GPU
    RaySamples operator[](int i) const {
        RaySamples rs;
        Float4 dir = Load4(direct + i);
        rs.direct.u = Point2f(dir.v[0], dir.v[1]);
        rs.direct.uc = dir.v[2];
        rs.direct.uer = dir.v[3];

        Float4 ind = Load4(indirect + i);
        rs.indirect.uc = ind.v[0];
        rs.indirect.rr = ind.v[1];
        rs.indirect.u = Point2f(ind.v[2], ind.v[3]);

        Float4 ss = Load4(subsurface + i);
        rs.subsurface.uc = ss.v[0];
        rs.subsurface.u = Point2f(ss.v[1], ss.v[2]);
        rs.haveSubsurface = int(ss.v[3]) & 1;

        return rs;
    }

    struct GetSetIndirector {
        PBRT_CPU_GPU
        operator RaySamples() const { return (*(const SOA *)soa)[index]; }

        PBRT_CPU_GPU
        void operator=(RaySamples rs) {
            
            soa->direct[index] =
                Float4{rs.direct.u[0], rs.direct.u[1], rs.direct.uc, rs.direct.uer};
            soa->indirect[index] = Float4{rs.indirect.uc, rs.indirect.rr,
                                          rs.indirect.u[0], rs.indirect.u[1]};
            int flags = rs.haveSubsurface ? 1 : 0;
            soa->subsurface[index] =
                Float4{rs.subsurface.uc, rs.subsurface.u.x, rs.subsurface.u.y, Float(flags)};
        }

        SOA *soa;
        int index;
    };

    PBRT_CPU_GPU
    GetSetIndirector operator[](int i) { return GetSetIndirector{this, i}; }

  private:
    Float4 *PBRT_RESTRICT direct;
    Float4 *PBRT_RESTRICT indirect;
    Float4 *PBRT_RESTRICT subsurface;
    Float *PBRT_RESTRICT mediaDist, *PBRT_RESTRICT mediaMode;
};

// PixelSampleState Definition
struct PixelSampleState {
    // PixelSampleState Public Members
    Point2i pPixel;
    SampledSpectrum L;
    SampledWavelengths lambda;
    Float filterWeight;
    VisibleSurface visibleSurface;
    SampledSpectrum cameraRayWeight;
    RaySamples samples;
};

// LightSampleRecordItem Definition
struct LightSampleRecordItem {
    // LightSampleRecordItem Public Members
    LightSampleContext ctx; // input
    SampledWavelengths lambda; // output wavelength
    SampledSpectrum L; // output radiance not divided by PDF
    Float LPDF; // output PDF
    uint32_t lightIndex; // selected light index
};

// LightSamplerOptState Definition
struct LightSamplerOptState {
    // LightSamplerOptState Public Members
    LightSampleRecordItem records[MAX_TRAIN_DEPTH];
    uint32_t curDepth{};     // current depth
};

// RayWorkItem Definition
struct RayWorkItem {
    // RayWorkItem Public Members
    Ray ray;
    int depth;
    SampledWavelengths lambda;
    int pixelIndex;
    SampledSpectrum beta, r_u, r_l;
    LightSampleContext prevIntrCtx;
    Float etaScale;
    int specularBounce;
    int anyNonSpecularBounces;
};

// EscapedRayWorkItem Definition
struct EscapedRayWorkItem {
    // EscapedRayWorkItem Public Members
    Point3f rayo;
    Vector3f rayd;
    int depth;
    SampledWavelengths lambda;
    int pixelIndex;
    SampledSpectrum beta;
    int specularBounce;
    SampledSpectrum r_u, r_l;
    LightSampleContext prevIntrCtx;
};

// HitAreaLightWorkItem Definition
struct HitAreaLightWorkItem {
    // HitAreaLightWorkItem Public Members
    Light areaLight;
    Point3f p;
    Normal3f n;
    Point2f uv;
    Vector3f wo;
    SampledWavelengths lambda;
    int depth;
    SampledSpectrum beta, r_u, r_l;
    LightSampleContext prevIntrCtx;
    int specularBounce;
    int pixelIndex;
};

// HitAreaLightQueue Definition
using HitAreaLightQueue = WorkQueue<HitAreaLightWorkItem>;

// ShadowRayWorkItem Definition
struct ShadowRayWorkItem {
    Ray ray;
    Float tMax;
    SampledWavelengths lambda;
    SampledSpectrum Ld, r_u, r_l;
    bool isBasicBSDF;
    int depth; // Used to query if training is enabled
    int pixelIndex;
    bool isVPL;
};

// GetBSSRDFAndProbeRayWorkItem Definition
struct GetBSSRDFAndProbeRayWorkItem {
    PBRT_CPU_GPU
    MaterialEvalContext GetMaterialEvalContext() const {
        MaterialEvalContext ctx;
        ctx.wo = wo;
        ctx.n = n;
        ctx.ns = ns;
        ctx.dpdus = dpdus;
        ctx.p = p;
        ctx.uv = uv;
        return ctx;
    }

    Material material;
    SampledWavelengths lambda;
    SampledSpectrum beta, r_u;
    Point3f p;
    Vector3f wo;
    Normal3f n, ns;
    Vector3f dpdus;
    Point2f uv;
    int depth;
    MediumInterface mediumInterface;
    Float etaScale;
    int pixelIndex;
};

// SubsurfaceScatterWorkItem Definition
struct SubsurfaceScatterWorkItem {
    Point3f p0, p1;
    int depth;
    Material material;
    TabulatedBSSRDF bssrdf;
    SampledWavelengths lambda;
    SampledSpectrum beta, r_u;
    Float reservoirPDF;
    Float uLight;
    SubsurfaceInteraction ssi;
    MediumInterface mediumInterface;
    Float etaScale;
    int pixelIndex;
};

// MediumSampleWorkItem Definition
struct MediumSampleWorkItem {
    // Both enqueue types (have mtl and no hit)
    Ray ray;
    int depth;
    Float tMax;
    SampledWavelengths lambda;
    SampledSpectrum beta;
    SampledSpectrum r_u;
    SampledSpectrum r_l;
    int pixelIndex;
    LightSampleContext prevIntrCtx;
    int specularBounce;
    int anyNonSpecularBounces;
    Float etaScale;

    // Have a hit material as well
    Light areaLight;
    Point3fi pi;
    Normal3f n;
    Vector3f dpdu, dpdv;
    Vector3f wo;
    Point2f uv;
    Material material;
    Normal3f ns;
    Vector3f dpdus, dpdvs;
    Normal3f dndus, dndvs;
    int faceIndex;
    MediumInterface mediumInterface;
};

// MediumScatterWorkItem Definition
template <typename PhaseFunction>
struct MediumScatterWorkItem {
    Point3f p;
    int depth;
    SampledWavelengths lambda;
    SampledSpectrum beta, r_u;
    const PhaseFunction *phase;
    Vector3f wo;
    Float time;
    Float etaScale;
    Medium medium;
    int pixelIndex;
};

// MaterialEvalWorkItem Definition
template <typename ConcreteMaterial>
struct MaterialEvalWorkItem {
    // MaterialEvalWorkItem Public Methods
    PBRT_CPU_GPU
    NormalBumpEvalContext GetNormalBumpEvalContext(Float dudx, Float dudy, Float dvdx,
                                                   Float dvdy) const {
        NormalBumpEvalContext ctx;
        ctx.p = Point3f(pi);
        ctx.uv = uv;
        ctx.dudx = dudx;
        ctx.dudy = dudy;
        ctx.dvdx = dvdx;
        ctx.dvdy = dvdy;
        ctx.shading.n = ns;
        ctx.shading.dpdu = dpdus;
        ctx.shading.dpdv = dpdvs;
        ctx.shading.dndu = dndus;
        ctx.shading.dndv = dndvs;
        ctx.faceIndex = faceIndex;
        return ctx;
    }

    PBRT_CPU_GPU
    MaterialEvalContext GetMaterialEvalContext(Float dudx, Float dudy, Float dvdx,
                                               Float dvdy, Normal3f ns,
                                               Vector3f dpdus) const {
        MaterialEvalContext ctx;
        ctx.wo = wo;
        ctx.n = n;
        ctx.ns = ns;
        ctx.dpdus = dpdus;
        ctx.p = Point3f(pi);
        ctx.uv = uv;
        ctx.dudx = dudx;
        ctx.dudy = dudy;
        ctx.dvdx = dvdx;
        ctx.dvdy = dvdy;
        ctx.faceIndex = faceIndex;
        return ctx;
    }

    // MaterialEvalWorkItem Public Members
    const ConcreteMaterial *material;
    Point3fi pi;
    Normal3f n;
    Vector3f dpdu, dpdv;
    Float time;
    int depth;
    Normal3f ns;
    Vector3f dpdus, dpdvs;
    Normal3f dndus, dndvs;
    Point2f uv;
    int faceIndex;
    SampledWavelengths lambda;
    int pixelIndex;
    int anyNonSpecularBounces;
    Vector3f wo;
    SampledSpectrum beta, r_u;
    Float etaScale;
    MediumInterface mediumInterface;
};

// LightSamplerWorkItem Definition
template <typename ConcreteMaterial>
struct LightSamplerWorkItem {

    PBRT_CPU_GPU
    MaterialEvalContext GetMaterialEvalContext() const {
        MaterialEvalContext ctx;
        ctx.wo = wo;
        ctx.n = n;
        ctx.ns = ns;
        ctx.dpdus = dpdus;
        ctx.p = Point3f(pi);
        ctx.uv = uv;
        ctx.dudx = dudx;
        ctx.dudy = dudy;
        ctx.dvdx = dvdx;
        ctx.dvdy = dvdy;
        ctx.faceIndex = faceIndex;
        return ctx;
    }

    // LightSamplerWorkItem Public Members
    const ConcreteMaterial *material;
    Point3fi pi;
    Vector3f wo;
    Normal3f ns;
    Normal3f n;
    SampledSpectrum beta, r_u;
    SampledWavelengths lambda;
    Float time;
    Float dudx, dudy, dvdx, dvdy;
    Vector3f dpdus;
    Point2f uv;
    int faceIndex;
    int depth;
    int anyNonSpecularBounces;
    MediumInterface mediumInterface;
    int pixelIndex;
};

#include "wavefront_workitems_soa.h"

// LightSamplerOptStateBuffer Definition
class LightSamplerOptStateBuffer : public SOA<LightSamplerOptState> {
public:
    // LightSamplerOptStateBuffer Public Methods
    LightSamplerOptStateBuffer() = default;
    LightSamplerOptStateBuffer(int n, Allocator alloc) : SOA<LightSamplerOptState>(n, alloc) {}
    
    PBRT_CPU_GPU
    inline void Reset(int pixelIndex) {
        this->curDepth[pixelIndex] = 0;
    }

    PBRT_CPU_GPU
    inline void incrementDepth(int pixelIndex, const LightSampleContext& ctx, SampledWavelengths lambda,
        const SampledSpectrum& L = SampledSpectrum(0.f), Float LPDF = 0.f, uint32_t lightIndex = 0) {
        const int depth = this->curDepth[pixelIndex];
        if (depth >= MAX_TRAIN_DEPTH){
            PBRT_DBG("Pixelindex: %d. LightSamplerOptStateBuffer::incrementDepth: depth (%u) >= MAX_TRAIN_DEPTH (%u)", pixelIndex, depth, MAX_TRAIN_DEPTH);
            return;
        }

        LightSampleRecordItem tmp;
        tmp.ctx = ctx;
        tmp.lambda = lambda;
        tmp.L = L;
        tmp.LPDF = LPDF;
        tmp.lightIndex = lightIndex;

        this->records[depth][pixelIndex] = tmp;
        this->curDepth[pixelIndex] = depth + 1;
    }

    PBRT_CPU_GPU
    inline void recordRadiance(int pixelIndex, const SampledSpectrum& L, Float LPDF){
        int depth = std::min(this->curDepth[pixelIndex], MAX_TRAIN_DEPTH) - 1;
        if (depth < 0){
            PBRT_DBG("Pixelindex: %d. LightSamplerOptStateBuffer::recordRadiance: depth (%d) < 0", pixelIndex, depth);
            return;
        }
        LightSampleRecordItem record = this->records[depth][pixelIndex];
        record.L = L;
        record.LPDF = LPDF;
        this->records[depth][pixelIndex] = record;
    }

    PBRT_CPU_GPU
    inline void recordLight(int pixelIndex, uint32_t lightIndex){
        int depth = std::min(this->curDepth[pixelIndex], MAX_TRAIN_DEPTH) - 1;
        if (depth < 0){
            PBRT_DBG("Pixelindex: %d. LightSamplerOptStateBuffer::recordLight: depth (%d) < 0", pixelIndex, depth);
            return;
        }
        LightSampleRecordItem record = this->records[depth][pixelIndex];
        record.lightIndex = lightIndex;
        this->records[depth][pixelIndex] = record;
    }
};

// RayQueue Definition
class RayQueue : public WorkQueue<RayWorkItem> {
  public:
    using WorkQueue::WorkQueue;
    // RayQueue Public Methods
    PBRT_CPU_GPU
    int PushCameraRay(const Ray &ray, const SampledWavelengths &lambda, int pixelIndex);

    PBRT_CPU_GPU
    int PushIndirectRay(const Ray &ray, int depth, const LightSampleContext &prevIntrCtx,
                        const SampledSpectrum &beta, const SampledSpectrum &r_u,
                        const SampledSpectrum &r_l, const SampledWavelengths &lambda,
                        Float etaScale, bool specularBounce, bool anyNonSpecularBounces,
                        int pixelIndex);
};

// RayQueue Inline Methods
PBRT_CPU_GPU inline int RayQueue::PushCameraRay(const Ray &ray, const SampledWavelengths &lambda,
                                   int pixelIndex) {
    int index = AllocateEntry();
    DCHECK(!ray.HasNaN());
    this->ray[index] = ray;
    this->depth[index] = 0;
    this->pixelIndex[index] = pixelIndex;
    this->lambda[index] = lambda;
    this->beta[index] = SampledSpectrum(1.f);
    this->etaScale[index] = 1.f;
    this->anyNonSpecularBounces[index] = false;
    this->r_u[index] = SampledSpectrum(1.f);
    this->r_l[index] = SampledSpectrum(1.f);
    this->specularBounce[index] = false;
    return index;
}

PBRT_CPU_GPU
inline int RayQueue::PushIndirectRay(
    const Ray &ray, int depth, const LightSampleContext &prevIntrCtx,
    const SampledSpectrum &beta, const SampledSpectrum &r_u,
    const SampledSpectrum &r_l, const SampledWavelengths &lambda, Float etaScale,
    bool specularBounce, bool anyNonSpecularBounces, int pixelIndex) {
    int index = AllocateEntry();
    DCHECK(!ray.HasNaN());
    this->ray[index] = ray;
    this->depth[index] = depth;
    this->pixelIndex[index] = pixelIndex;
    this->prevIntrCtx[index] = prevIntrCtx;
    this->beta[index] = beta;
    this->r_u[index] = r_u;
    this->r_l[index] = r_l;
    this->lambda[index] = lambda;
    this->anyNonSpecularBounces[index] = anyNonSpecularBounces;
    this->specularBounce[index] = specularBounce;
    this->etaScale[index] = etaScale;
    return index;
}

// ShadowRayQueue Definition
using ShadowRayQueue = WorkQueue<ShadowRayWorkItem>;

// EscapedRayQueue Definition
class EscapedRayQueue : public WorkQueue<EscapedRayWorkItem> {
  public:
    // EscapedRayQueue Public Methods
    PBRT_CPU_GPU
    int Push(RayWorkItem r);

    using WorkQueue::WorkQueue;

    using WorkQueue::Push;
};

PBRT_CPU_GPU inline int EscapedRayQueue::Push(RayWorkItem r) {
    return Push(EscapedRayWorkItem{r.ray.o, r.ray.d, r.depth, r.lambda, r.pixelIndex,
                                   r.beta, (int)r.specularBounce, r.r_u, r.r_l,
                                   r.prevIntrCtx});
}

// GetBSSRDFAndProbeRayQueue Definition
class GetBSSRDFAndProbeRayQueue : public WorkQueue<GetBSSRDFAndProbeRayWorkItem> {
  public:
    using WorkQueue::WorkQueue;

    PBRT_CPU_GPU
    int Push(Material material, SampledWavelengths lambda, SampledSpectrum beta,
             SampledSpectrum r_u, Point3f p, Vector3f wo, Normal3f n, Normal3f ns,
             Vector3f dpdus, Point2f uv, int depth, MediumInterface mediumInterface,
             Float etaScale, int pixelIndex) {
        int index = AllocateEntry();
        this->material[index] = material;
        this->lambda[index] = lambda;
        this->beta[index] = beta;
        this->r_u[index] = r_u;
        this->p[index] = p;
        this->wo[index] = wo;
        this->n[index] = n;
        this->ns[index] = ns;
        this->dpdus[index] = dpdus;
        this->uv[index] = uv;
        this->depth[index] = depth;
        this->mediumInterface[index] = mediumInterface;
        this->etaScale[index] = etaScale;
        this->pixelIndex[index] = pixelIndex;
        return index;
    }
};

// SubsurfaceScatterQueue Definition
class SubsurfaceScatterQueue : public WorkQueue<SubsurfaceScatterWorkItem> {
  public:
    using WorkQueue::WorkQueue;

    PBRT_CPU_GPU
    int Push(Point3f p0, Point3f p1, int depth, Material material, TabulatedBSSRDF bssrdf,
             SampledWavelengths lambda, SampledSpectrum beta, SampledSpectrum r_u,
             MediumInterface mediumInterface, Float etaScale, int pixelIndex) {
        int index = AllocateEntry();
        this->p0[index] = p0;
        this->p1[index] = p1;
        this->depth[index] = depth;
        this->material[index] = material;
        this->bssrdf[index] = bssrdf;
        this->lambda[index] = lambda;
        this->beta[index] = beta;
        this->r_u[index] = r_u;
        this->mediumInterface[index] = mediumInterface;
        this->etaScale[index] = etaScale;
        this->pixelIndex[index] = pixelIndex;
        return index;
    }
};

// MediumSampleQueue Definition
class MediumSampleQueue : public WorkQueue<MediumSampleWorkItem> {
  public:
    using WorkQueue::WorkQueue;

    using WorkQueue::Push;

    PBRT_CPU_GPU
    int Push(Ray ray, Float tMax, SampledWavelengths lambda, SampledSpectrum beta,
             SampledSpectrum r_u, SampledSpectrum r_l, int pixelIndex,
             LightSampleContext prevIntrCtx, int specularBounce,
             int anyNonSpecularBounces, Float etaScale) {
        int index = AllocateEntry();
        this->ray[index] = ray;
        this->tMax[index] = tMax;
        this->lambda[index] = lambda;
        this->beta[index] = beta;
        this->r_u[index] = r_u;
        this->r_l[index] = r_l;
        this->pixelIndex[index] = pixelIndex;
        this->prevIntrCtx[index] = prevIntrCtx;
        this->specularBounce[index] = specularBounce;
        this->anyNonSpecularBounces[index] = anyNonSpecularBounces;
        this->etaScale[index] = etaScale;
        return index;
    }

    PBRT_CPU_GPU
    int Push(RayWorkItem r, Float tMax) {
        return Push(r.ray, tMax, r.lambda, r.beta, r.r_u, r.r_l, r.pixelIndex,
                    r.prevIntrCtx, r.specularBounce, r.anyNonSpecularBounces, r.etaScale);
    }
};

// MediumScatterQueue Definition
using MediumScatterQueue = MultiWorkQueue<
    typename MapType<MediumScatterWorkItem, typename PhaseFunction::Types>::type>;

// MaterialEvalQueue Definition
using MaterialEvalQueue = MultiWorkQueue<
    typename MapType<MaterialEvalWorkItem, typename Material::Types>::type>;

using LightSamplerQueue = MultiWorkQueue<
    typename MapType<LightSamplerWorkItem, typename Material::Types>::type>;

}  // namespace pbrt

#endif  // PBRT_WAVEFRONT_WORKITEMS_H
