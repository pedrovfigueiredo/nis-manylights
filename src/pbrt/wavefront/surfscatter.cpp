// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/bxdfs.h>
#include <pbrt/cameras.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/wavefront/integrator.h>

#include <type_traits>

namespace pbrt {

// EvaluateMaterialCallback Definition
struct EvaluateMaterialCallback {
    int wavefrontDepth;
    int sampleIndex;
    WavefrontPathIntegrator *integrator;
    Transform movingFromCamera;
    // EvaluateMaterialCallback Public Methods
    template <typename ConcreteMaterial>
    void operator()() {
        if constexpr (!std::is_same_v<ConcreteMaterial, MixMaterial>)
            integrator->EvaluateMaterialAndBSDF<ConcreteMaterial>(wavefrontDepth, sampleIndex,
                                                                  movingFromCamera);
    }
};

// WavefrontPathIntegrator Surface Scattering Methods
void WavefrontPathIntegrator::EvaluateMaterialsAndBSDFs(int wavefrontDepth, int sampleIndex,
                                                        Transform movingFromCamera) {
    ForEachType(EvaluateMaterialCallback{wavefrontDepth, sampleIndex, this, movingFromCamera},
                Material::Types());
}

template <typename ConcreteMaterial>
void WavefrontPathIntegrator::EvaluateMaterialAndBSDF(int wavefrontDepth, int sampleIndex,
                                                      Transform movingFromCamera) {
    int index = Material::TypeIndex<ConcreteMaterial>();
    if (haveBasicEvalMaterial[index])
        EvaluateMaterialAndBSDF<ConcreteMaterial, BasicTextureEvaluator>(
            basicEvalMaterialQueue, basicLightSamplerQueue, movingFromCamera, wavefrontDepth, sampleIndex);
    if (haveUniversalEvalMaterial[index])
        EvaluateMaterialAndBSDF<ConcreteMaterial, UniversalTextureEvaluator>(
            universalEvalMaterialQueue, universalLightSamplerQueue, movingFromCamera, wavefrontDepth, sampleIndex);
}

template <typename ConcreteMaterial, typename TextureEvaluator>
void WavefrontPathIntegrator::EvaluateMaterialAndBSDF(MaterialEvalQueue *evalQueue,
                                                      LightSamplerQueue *lightSamplerQueue,
                                                      Transform movingFromCamera,
                                                      int wavefrontDepth,
                                                      int sampleIndex) {
    // Get BSDF for items in _evalQueue_ and sample illumination
    // Construct _desc_ for material/texture evaluation kernel
    std::string desc = StringPrintf(
        "%s + BxDF eval (%s tex)", ConcreteMaterial::Name(),
        std::is_same_v<TextureEvaluator, BasicTextureEvaluator> ? "Basic" : "Universal");

    RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);
    auto queue = evalQueue->Get<MaterialEvalWorkItem<ConcreteMaterial>>();
    auto lqueue = lightSamplerQueue->Get<LightSamplerWorkItem<ConcreteMaterial>>();
    ForAllQueued(
        desc.c_str(), queue, maxQueueSize,
        PBRT_CPU_GPU_LAMBDA(const MaterialEvalWorkItem<ConcreteMaterial> w) {
            // Evaluate material and BSDF for ray intersection
            TextureEvaluator texEval;
            // Compute differentials for position and $(u,v)$ at intersection point
            Vector3f dpdx, dpdy;
            Float dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
            if (!GetOptions().disableTextureFiltering) {
                Point3f pc = movingFromCamera.ApplyInverse(Point3f(w.pi));
                Normal3f nc = movingFromCamera.ApplyInverse(w.n);
                camera.Approximate_dp_dxy(pc, nc, w.time, samplesPerPixel, &dpdx, &dpdy);
                Vector3f dpdu = w.dpdu, dpdv = w.dpdv;
                // Estimate screen-space change in $(u,v)$
                // Compute $\transpose{\XFORM{A}} \XFORM{A}$ and its determinant
                Float ata00 = Dot(dpdu, dpdu), ata01 = Dot(dpdu, dpdv);
                Float ata11 = Dot(dpdv, dpdv);
                Float invDet = 1 / DifferenceOfProducts(ata00, ata11, ata01, ata01);
                invDet = IsFinite(invDet) ? invDet : 0.f;

                // Compute $\transpose{\XFORM{A}} \VEC{b}$ for $x$ and $y$
                Float atb0x = Dot(dpdu, dpdx), atb1x = Dot(dpdv, dpdx);
                Float atb0y = Dot(dpdu, dpdy), atb1y = Dot(dpdv, dpdy);

                // Compute $u$ and $v$ derivatives with respect to $x$ and $y$
                dudx = DifferenceOfProducts(ata11, atb0x, ata01, atb1x) * invDet;
                dvdx = DifferenceOfProducts(ata00, atb1x, ata01, atb0x) * invDet;
                dudy = DifferenceOfProducts(ata11, atb0y, ata01, atb1y) * invDet;
                dvdy = DifferenceOfProducts(ata00, atb1y, ata01, atb0y) * invDet;

                // Clamp derivatives of $u$ and $v$ to reasonable values
                dudx = IsFinite(dudx) ? Clamp(dudx, -1e8f, 1e8f) : 0.f;
                dvdx = IsFinite(dvdx) ? Clamp(dvdx, -1e8f, 1e8f) : 0.f;
                dudy = IsFinite(dudy) ? Clamp(dudy, -1e8f, 1e8f) : 0.f;
                dvdy = IsFinite(dvdy) ? Clamp(dvdy, -1e8f, 1e8f) : 0.f;
            }

            // Compute shading normal if bump or normal mapping is being used
            Normal3f ns = w.ns;
            Vector3f dpdus = w.dpdus;
            FloatTexture displacement = w.material->GetDisplacement();
            const Image *normalMap = w.material->GetNormalMap();
            if (normalMap) {
                // Call _NormalMap()_ to find shading geometry
                NormalBumpEvalContext bctx =
                    w.GetNormalBumpEvalContext(dudx, dudy, dvdx, dvdy);
                Vector3f dpdvs;
                NormalMap(*normalMap, bctx, &dpdus, &dpdvs);
                ns = Normal3f(Normalize(Cross(dpdus, dpdvs)));
                ns = FaceForward(ns, w.n);

            } else if (displacement) {
                // Call _BumpMap()_ to find shading geometry
                if (displacement)
                    DCHECK(texEval.CanEvaluate({displacement}, {}));
                NormalBumpEvalContext bctx =
                    w.GetNormalBumpEvalContext(dudx, dudy, dvdx, dvdy);
                Vector3f dpdvs;
                BumpMap(texEval, displacement, bctx, &dpdus, &dpdvs);
                ns = Normal3f(Normalize(Cross(dpdus, dpdvs)));
                ns = FaceForward(ns, w.n);
            }

            // Get BSDF at intersection point
            SampledWavelengths lambda = w.lambda;
            MaterialEvalContext ctx =
                w.GetMaterialEvalContext(dudx, dudy, dvdx, dvdy, ns, dpdus);
            using ConcreteBxDF = typename ConcreteMaterial::BxDF;
            ConcreteBxDF bxdf = w.material->GetBxDF(texEval, ctx, lambda);
            BSDF bsdf(ctx.ns, ctx.dpdus, &bxdf);
            // Handle terminated secondary wavelengths after BSDF creation
            if (lambda.SecondaryTerminated())
                pixelSampleState.lambda[w.pixelIndex] = lambda;

            // Regularize BSDF, if appropriate
            if (regularize && w.anyNonSpecularBounces)
                bsdf.Regularize();

            // Initialize _VisibleSurface_ at first intersection if necessary
            if (w.depth == 0 && initializeVisibleSurface) {
                SurfaceInteraction isect;
                isect.pi = w.pi;
                isect.n = w.n;
                isect.shading.n = ns;
                isect.uv = w.uv;
                isect.wo = w.wo;
                isect.time = w.time;
                isect.dpdx = dpdx;
                isect.dpdy = dpdy;

                // Estimate BSDF's albedo
                // Define sample arrays _ucRho_ and _uRho_ for reflectance estimate
                constexpr int nRhoSamples = 16;
                const Float ucRho[nRhoSamples] = {
                    0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
                    0.5990858,  0.09467703, 0.8578725, 0.45746812, 0.686759,  0.17708716,
                    0.9674518,  0.2995429,  0.5083201, 0.047338516};
                const Point2f uRho[nRhoSamples] = {
                    Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
                    Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
                    Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
                    Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
                    Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
                    Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
                    Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
                    Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};

                SampledSpectrum albedo = bsdf.rho(isect.wo, ucRho, uRho);

                pixelSampleState.visibleSurface[w.pixelIndex] =
                    VisibleSurface(isect, albedo, lambda);
            }

            // Sample BSDF and enqueue indirect ray at intersection point
            Vector3f wo = w.wo;
            BxDFFlags flags = bsdf.Flags();
            LightSampleContext lightctx(w.pi, w.n, ns, wo, w.pixelIndex);
            if (IsReflective(flags) && !IsTransmissive(flags))
                lightctx.pi = OffsetRayOrigin(lightctx.pi, w.n, wo);
            else if (IsTransmissive(flags) && IsReflective(flags))
                lightctx.pi = OffsetRayOrigin(lightctx.pi, w.n, -wo);
            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            
            #if (NEEONLY == 1)
                // We allow one extra bounce beyond maxDepth for specular surfaces
                const bool allowNextBounce = (wavefrontDepth < maxDepth) || 
                                            (wavefrontDepth == maxDepth && !w.anyNonSpecularBounces);
            #else
                const bool allowNextBounce = true;
            #endif

            if (allowNextBounce) {
                pstd::optional<BSDFSample> bsdfSample = bsdf.Sample_f<ConcreteBxDF>(
                    wo, raySamples.indirect.uc, raySamples.indirect.u);
                if (bsdfSample) {
                    // Compute updated path throughput and PDFs and enqueue indirect ray
                    Vector3f wi = bsdfSample->wi;
                    SampledSpectrum beta =
                        w.beta * bsdfSample->f * AbsDot(wi, ns) / bsdfSample->pdf;
                    SampledSpectrum r_u = w.r_u, r_l;

                    PBRT_DBG("%s f*cos[0] %f bsdfSample->pdf %f f*cos/pdf %f\n",
                            ConcreteBxDF::Name(), bsdfSample->f[0] * AbsDot(wi, ns),
                            bsdfSample->pdf,
                            bsdfSample->f[0] * AbsDot(wi, ns) / bsdfSample->pdf);

                    // Update _r_u_ based on BSDF sample PDF
                    if (bsdfSample->pdfIsProportional)
                        r_l = r_u / bsdf.PDF<ConcreteBxDF>(wo, bsdfSample->wi);
                    else
                        r_l = r_u / bsdfSample->pdf;

                    // Update _etaScale_ accounting for BSDF scattering
                    Float etaScale = w.etaScale;
                    if (bsdfSample->IsTransmission())
                        etaScale *= Sqr(bsdfSample->eta);

                    // Apply Russian roulette to indirect ray based on weighted path
                    // throughput
                    SampledSpectrum rrBeta = beta * etaScale / r_u.Average();
                    // Note: depth >= 1 here to match VolPathIntegrator (which increments
                    // depth earlier).
                    if (rrBeta.MaxComponentValue() < 1 && w.depth >= 1) {
                        Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                        if (raySamples.indirect.rr < q) {
                            beta = SampledSpectrum(0.f);
                            PBRT_DBG("Path terminated with RR\n");
                        } else
                            beta /= 1 - q;
                    }

                    if (beta) {
                        // Initialize spawned ray and enqueue for next ray depth
                        if (bsdfSample->IsTransmission() &&
                            w.material->HasSubsurfaceScattering()) {
                            bssrdfEvalQueue->Push(w.material, lambda, beta, r_u,
                                                Point3f(w.pi), wo, w.n, ns, dpdus, w.uv,
                                                w.depth, w.mediumInterface, etaScale,
                                                w.pixelIndex);
                        } else {
                            Ray ray = SpawnRay(w.pi, w.n, w.time, wi);
                            // Initialize _ray_ medium if media are present
                            if (haveMedia)
                                ray.medium = Dot(ray.d, w.n) > 0 ? w.mediumInterface.outside
                                                                : w.mediumInterface.inside;

                            bool anyNonSpecularBounces =
                                !bsdfSample->IsSpecular() || w.anyNonSpecularBounces;
                            // NOTE: slightly different than context below. Problem?
                            // LightSampleContext ctx(w.pi, w.n, ns, wo, w.pixelIndex);
                            // Using same light context for NEE and BSDF sampling
                            nextRayQueue->PushIndirectRay(
                                ray, w.depth + 1, lightctx, beta, r_u, r_l, lambda,
                                etaScale, bsdfSample->IsSpecular(), anyNonSpecularBounces,
                                w.pixelIndex);

                            PBRT_DBG(
                                "Spawned indirect ray at depth %d from w.index %d. "
                                "Specular %d beta %f %f %f %f r_u %f %f %f %f r_l %f "
                                "%f %f %f beta/r_u %f %f %f %f\n",
                                w.depth + 1, w.pixelIndex, int(bsdfSample->IsSpecular()),
                                beta[0], beta[1], beta[2], beta[3], r_u[0], r_u[1],
                                r_u[2], r_u[3], r_l[0], r_l[1], r_l[2],
                                r_l[3], SafeDiv(beta, r_u)[0],
                                SafeDiv(beta, r_u)[1], SafeDiv(beta, r_u)[2],
                                SafeDiv(beta, r_u)[3]);
                        }
                    }
                }
            }

            // Sample light and enqueue shadow ray at intersection point
            const bool allowNEE = allowNextBounce && IsNonSpecular(flags);
            if (allowNEE) {
                lqueue->Push(
                    LightSamplerWorkItem<ConcreteMaterial>{
                        w.material, w.pi, wo, ns, w.n, 
                        w.beta, w.r_u, lambda, w.time,
                        dudx, dudy, dvdx, dvdy,
                        dpdus, w.uv, w.faceIndex, 
                        w.depth, w.anyNonSpecularBounces,
                        w.mediumInterface, w.pixelIndex}
                );    

        #if (NEEONLY == 0)
            //  If not NEEONLY, we need to include specular bounces to the evalbuffer since they may be needed at the PMF eval in case they hit a light.
            }
        #endif
            {
                // Add to evaluation buffer if networkeval is enabled and is not infinite light
                RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
                Float u = raySamples.direct.uc;
                const LightBVHTree* tree = lightSampler.getLightBVHTree();

                // Only add to eval buffer if not infinite light AND exploration is not 100%
                const float explorationSamplingRatio_i = sampleIndex < nExplorationWarmupSamples ? 1.f : explorationSamplingRatio;
                if (lightSampler.isNetworkEval(w.pixelIndex, w.depth) && (tree ? u >= tree->pInfinite() : true) && explorationSamplingRatio_i < 1.f){
                    const Point3f lightCtxP = lightctx.p();
                    const Normal3f lightCtxNs = lightctx.ns; 
                    const Vector3f lightCtxWo = lightctx.wo;
                    const Vector3f pOffset = normalizePosition(lightCtxP, sceneBounds);
                    const Vector2f nSph = cartesianToSphericalNormalized(lightCtxNs);

                    #if (LEARN_PRODUCT_SAMPLING == 0)
                        #if (NEURAL_GRID_DISCRETIZATION == 0)
                            const LightSamplerTrainInput in = {pOffset, nSph};
                        #else
                            const Vector3f pOffsetDiscretized = discretizeVec(pOffset, POSITIONAL_DISCRETIZATION_RESOLUTION);
                            const LightSamplerTrainInput in = {pOffsetDiscretized};
                        #endif
                    #else
                        const Vector3f woNormalized = normalizeDirection(lightctx.wo);
                        #if (NEURAL_GRID_DISCRETIZATION == 0)
                            const LightSamplerTrainInput in = {pOffset, nSph, woNormalized};
                        #else
                            const Vector3f pOffsetDiscretized = discretizeVec(pOffset, POSITIONAL_DISCRETIZATION_RESOLUTION);
                            const Vector3f woNormalizedDiscretized = discretizeVec(woNormalized, DIRECTIONAL_DISCRETIZATION_RESOLUTION);
                            const LightSamplerTrainInput in = {pOffsetDiscretized, woNormalizedDiscretized};
                        #endif
                    #endif

                    lightSamplerEvalBuffer->Push(in, w.pixelIndex, {lightCtxP, lightCtxNs, lightCtxWo});
                }
            }
        #if (NEEONLY == 1)
            // If NEEONLY, we can ignore specular bounces since we never call PMF on the next bounce
            }
        #endif
        });
}

}  // namespace pbrt
