// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

// PhysLight code contributed by Anders Langlands and Luca Fascione
// Copyright (c) 2020, Weta Digital, Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <pbrt/lights.h>

#include <pbrt/cameras.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/memory.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/shapes.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/float.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <mutex>

namespace pbrt {

STAT_COUNTER("Scene/Lights", numLights);
STAT_COUNTER("Scene/AreaLights", numAreaLights);

// Light Method Definitions
std::string LightLiSample::ToString() const {
    return StringPrintf("[ LightLiSample L: %s wi: %s pdf: %f pLight: %s ]", L, wi, pdf,
                        pLight);
}

std::string LightLeSample::ToString() const {
    return StringPrintf("[ LightLeSample L: %s ray: %s intr: %s pdfPos: %f pdfDir: %f ]",
                        L, ray, intr, pdfPos, pdfDir);
}

std::string ToString(LightType lf) {
    switch (lf) {
    case LightType::VPL:
        return "VPL";
    case LightType::DeltaPosition:
        return "DeltaPosition";
    case LightType::DeltaDirection:
        return "DeltaDirection,";
    case LightType::Area:
        return "Area";
    case LightType::Infinite:
        return "Infinite";
    default:
        LOG_FATAL("Unhandled type");
        return "";
    }
}

// LightBase Method Definitions
LightBase::LightBase(LightType type, const Transform &renderFromLight,
                     const MediumInterface &mediumInterface)
    : type(type), mediumInterface(mediumInterface), renderFromLight(renderFromLight) {
    ++numLights;
}

std::string LightBase::BaseToString() const {
    return StringPrintf("type: %s mediumInterface: %s renderFromLight: %s", type,
                        mediumInterface, renderFromLight);
}

InternCache<DenselySampledSpectrum> *LightBase::spectrumCache;

const DenselySampledSpectrum *LightBase::LookupSpectrum(Spectrum s) {
    // Initialize _spectrumCache_ on first call
    static std::mutex mutex;
    mutex.lock();
    if (!spectrumCache)
        spectrumCache = new InternCache<DenselySampledSpectrum>(
#ifdef PBRT_BUILD_GPU_RENDERER
            Options->useGPU ? Allocator(&CUDATrackedMemoryResource::singleton) :
#endif
                            Allocator{});
    mutex.unlock();

    // Return unique _DenselySampledSpectrum_ from intern cache for _s_
    auto create = [](Allocator alloc, const DenselySampledSpectrum &s) {
        return alloc.new_object<DenselySampledSpectrum>(s, alloc);
    };
    return spectrumCache->Lookup(DenselySampledSpectrum(s), create);
}

std::string LightBounds::ToString() const {
    return StringPrintf("[ LightBounds bounds: %s w: %s phi: %f "
                        "cosTheta_o: %f cosTheta_e: %f twoSided: %s ]",
                        bounds, w, phi, cosTheta_o, cosTheta_e, twoSided);
}

// LightBounds Method Definitions
PBRT_CPU_GPU Float LightBounds::Importance(Point3f p, Normal3f n) const {
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
    Float cosTheta_w = Dot(Vector3f(w), wi);
    if (twoSided)
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
    Float importance = phi * cosThetap / d2;
    DCHECK_GE(importance, -1e-3);
    // Account for $\cos\theta_\roman{i}$ in importance at surfaces
    if (n != Normal3f(0, 0, 0)) {
        Float cosTheta_i = AbsDot(wi, n);
        Float sinTheta_i = SafeSqrt(1 - Sqr(cosTheta_i));
        Float cosThetap_i = cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);
        importance *= cosThetap_i;
    }

    importance = std::max<Float>(importance, 0);
    return importance;
}

// PointLight Method Definitions
SampledSpectrum PointLight::Phi(SampledWavelengths lambda) const {
    return 4 * Pi * scale * I->Sample(lambda);
}

pstd::optional<LightBounds> PointLight::Bounds() const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Float phi = 4 * Pi * scale * I->MaxValue();
    return LightBounds(Bounds3f(p, p), Vector3f(0, 0, 1), phi, std::cos(Pi),
                       std::cos(Pi / 2), false);
}

PBRT_CPU_GPU pstd::optional<LightLeSample> PointLight::SampleLe(Point2f u1, Point2f u2,
                                                   SampledWavelengths &lambda,
                                                   Float time) const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Ray ray(p, SampleUniformSphere(u1), time, mediumInterface.outside);
    return LightLeSample(scale * I->Sample(lambda), ray, 1, UniformSpherePDF());
}

PBRT_CPU_GPU void PointLight::PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0;
    *pdfDir = UniformSpherePDF();
}

std::string PointLight::ToString() const {
    return StringPrintf("[ PointLight %s I: %s scale: %f ]", BaseToString(), I, scale);
}

PointLight *PointLight::Create(const Transform &renderFromLight, Medium medium,
                               const ParameterDictionary &parameters,
                               const RGBColorSpace *colorSpace, const FileLoc *loc,
                               Allocator alloc) {
    Spectrum I = parameters.GetOneSpectrum("I", &colorSpace->illuminant,
                                           SpectrumType::Illuminant, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    sc /= SpectrumToPhotometric(I);

    Float phi_v = parameters.GetOneFloat("power", -1);
    if (phi_v > 0) {
        Float k_e = 4 * Pi;
        sc *= phi_v / k_e;
    }

    Point3f from = parameters.GetOnePoint3f("from", Point3f(0, 0, 0));
    Transform tf = Translate(Vector3f(from.x, from.y, from.z));
    Transform finalRenderFromLight(renderFromLight * tf);

    return alloc.new_object<PointLight>(finalRenderFromLight, medium, I, sc);
}

std::string VirtualPointLight::ToString() const {
    return StringPrintf("[ VirtualPointLight %s I: %s scale: %f ]", BaseToString(), I, scale);
}

VirtualPointLight *VirtualPointLight::Create(const Transform &renderFromLight,
                                             Medium medium,
                                             const ParameterDictionary &parameters,
                                             const RGBColorSpace *colorSpace,
                                             const FileLoc *loc, Allocator alloc) {
    Spectrum I = parameters.GetOneSpectrum("I", &colorSpace->illuminant,
                                           SpectrumType::Illuminant, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);
    sc /= SpectrumToPhotometric(I);
    Float phi_v = parameters.GetOneFloat("power", -1);
    if (phi_v > 0) {
        Float k_e = 4 * Pi;
        sc *= phi_v / k_e;
    }
    Point3f p = parameters.GetOnePoint3f("from", Point3f(0, 0, 0));
    Normal3f n = parameters.GetOneNormal3f("normal", Normal3f(0, 0, 1));
    Float reps = parameters.GetOneFloat("radius", 1e-4);

    // Apply transform to position and normal
    Transform finalRenderFromLight = renderFromLight * Translate(Vector3f(p));
    p = finalRenderFromLight(Point3f(0, 0, 0));
    n = Normalize(finalRenderFromLight(n));

    return alloc.new_object<VirtualPointLight>(finalRenderFromLight, medium, p, n, reps, I, sc);
}

// DistantLight Method Definitions
SampledSpectrum DistantLight::Phi(SampledWavelengths lambda) const {
    return scale * Lemit->Sample(lambda) * Pi * Sqr(sceneRadius);
}

PBRT_CPU_GPU pstd::optional<LightLeSample> DistantLight::SampleLe(Point2f u1, Point2f u2,
                                                     SampledWavelengths &lambda,
                                                     Float time) const {
    // Choose point on disk oriented toward infinite light direction
    Vector3f w = Normalize(renderFromLight(Vector3f(0, 0, 1)));
    Frame wFrame = Frame::FromZ(w);
    Point2f cd = SampleUniformDiskConcentric(u1);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));

    // Compute _DistantLight_ light ray
    Ray ray(pDisk + sceneRadius * w, -w, time);

    return LightLeSample(scale * Lemit->Sample(lambda), ray, 1 / (Pi * Sqr(sceneRadius)),
                         1);
}

PBRT_CPU_GPU void DistantLight::PDF_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 1 / (Pi * sceneRadius * sceneRadius);
    *pdfDir = 0;
}

std::string DistantLight::ToString() const {
    return StringPrintf("[ DistantLight %s Lemit: %s scale: %f ]", BaseToString(), Lemit,
                        scale);
}

DistantLight *DistantLight::Create(const Transform &renderFromLight,
                                   const ParameterDictionary &parameters,
                                   const RGBColorSpace *colorSpace, const FileLoc *loc,
                                   Allocator alloc) {
    Spectrum L = parameters.GetOneSpectrum("L", &colorSpace->illuminant,
                                           SpectrumType::Illuminant, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    Point3f from = parameters.GetOnePoint3f("from", Point3f(0, 0, 0));
    Point3f to = parameters.GetOnePoint3f("to", Point3f(0, 0, 1));

    Vector3f w = Normalize(from - to);
    Vector3f v1, v2;
    CoordinateSystem(w, &v1, &v2);
    Float m[4][4] = {v1.x, v2.x, w.x, 0, v1.y, v2.y, w.y, 0,
                     v1.z, v2.z, w.z, 0, 0,    0,    0,   1};
    Transform t(m);
    Transform finalRenderFromLight = renderFromLight * t;

    // Scale the light spectrum to be equivalent to 1 nit
    sc /= SpectrumToPhotometric(L);

    // Adjust scale to meet target illuminance value
    // Like for IBLs we measure illuminance as incident on an upward-facing
    // patch.
    Float E_v = parameters.GetOneFloat("illuminance", -1);
    if (E_v > 0)
        sc *= E_v;

    return alloc.new_object<DistantLight>(finalRenderFromLight, L, sc);
}

STAT_MEMORY_COUNTER("Memory/Light image and distributions", imageBytes);

// ProjectionLight Method Definitions
ProjectionLight::ProjectionLight(Transform renderFromLight,
                                 MediumInterface mediumInterface, Image im,
                                 const RGBColorSpace *imageColorSpace, Float scale,
                                 Float fov, Allocator alloc)
    : LightBase(LightType::DeltaPosition, renderFromLight, mediumInterface),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      scale(scale),
      distrib(alloc) {
    // _ProjectionLight_ constructor implementation
    // Initialize _ProjectionLight_ projection matrix
    Float aspect = Float(image.Resolution().x) / Float(image.Resolution().y);
    if (aspect > 1)
        screenBounds = Bounds2f(Point2f(-aspect, -1), Point2f(aspect, 1));
    else
        screenBounds = Bounds2f(Point2f(-1, -1 / aspect), Point2f(1, 1 / aspect));
    screenFromLight = Perspective(fov, hither, 1e30f /* yon */);
    lightFromScreen = Inverse(screenFromLight);

    // Compute projection image area _A_
    Float opposite = std::tan(Radians(fov) / 2);
    A = 4 * Sqr(opposite) * (aspect > 1 ? aspect : (1 / aspect));

    // Compute sampling distribution for _ProjectionLight_
    ImageChannelDesc channelDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("Image used for ProjectionLight does not have R, G, B channels.");
    CHECK_EQ(3, channelDesc.size());
    CHECK(channelDesc.IsIdentity());
    auto dwdA = [&](const Point2f &p) {
        Vector3f w = Vector3f(lightFromScreen(Point3f(p.x, p.y, 0)));
        return Pow<3>(CosTheta(Normalize(w)));
    };
    Array2D<Float> d = image.GetSamplingDistribution(dwdA, screenBounds);
    distrib = PiecewiseConstant2D(d, screenBounds);

    imageBytes += image.BytesUsed() + distrib.BytesUsed();
}

PBRT_CPU_GPU pstd::optional<LightLiSample> ProjectionLight::SampleLi(LightSampleContext ctx, Point2f u,
                                                        SampledWavelengths lambda,
                                                        bool allowIncompletePDF) const {
    // Return sample for incident radiance from _ProjectionLight_
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f wi = Normalize(p - ctx.p());
    Vector3f wl = renderFromLight.ApplyInverse(-wi);
    SampledSpectrum Li = I(wl, lambda) / DistanceSquared(p, ctx.p());
    if (!Li)
        return {};
    return LightLiSample(Li, wi, 1, Interaction(p, &mediumInterface));
}

PBRT_CPU_GPU Float ProjectionLight::PDF_Li(LightSampleContext, Vector3f,
                              bool allowIncompletePDF) const {
    return 0.f;
}

std::string ProjectionLight::ToString() const {
    return StringPrintf("[ ProjectionLight %s scale: %f A: %f ]", BaseToString(), scale,
                        A);
}

PBRT_CPU_GPU SampledSpectrum ProjectionLight::I(Vector3f w, const SampledWavelengths &lambda) const {
    // Discard directions behind projection light
    if (w.z < hither)
        return SampledSpectrum(0.f);

    // Project point onto projection plane and compute RGB
    Point3f ps = screenFromLight(Point3f(w));
    if (!Inside(Point2f(ps.x, ps.y), screenBounds))
        return SampledSpectrum(0.f);
    Point2f uv = Point2f(screenBounds.Offset(Point2f(ps.x, ps.y)));
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(uv, c);

    // Return scaled wavelength samples corresponding to RGB
    RGBIlluminantSpectrum s(*imageColorSpace, ClampZero(rgb));
    return scale * s.Sample(lambda);
}

SampledSpectrum ProjectionLight::Phi(SampledWavelengths lambda) const {
    SampledSpectrum sum(0.f);
    for (int y = 0; y < image.Resolution().y; ++y)
        for (int x = 0; x < image.Resolution().x; ++x) {
            // Compute change of variables factor _dwdA_ for projection light pixel
            Point2f ps = screenBounds.Lerp(Point2f((x + 0.5f) / image.Resolution().x,
                                                   (y + 0.5f) / image.Resolution().y));
            Vector3f w = Vector3f(lightFromScreen(Point3f(ps.x, ps.y, 0)));
            w = Normalize(w);
            Float dwdA = Pow<3>(CosTheta(w));

            // Update _sum_ for projection light pixel
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({x, y}, c);
            RGBIlluminantSpectrum s(*imageColorSpace, ClampZero(rgb));
            sum += s.Sample(lambda) * dwdA;
        }
    // Return final power for projection light
    return scale * A * sum / (image.Resolution().x * image.Resolution().y);
}

pstd::optional<LightBounds> ProjectionLight::Bounds() const {
    Float sum = 0;
    for (int v = 0; v < image.Resolution().y; ++v)
        for (int u = 0; u < image.Resolution().x; ++u)
            sum += std::max({image.GetChannel({u, v}, 0), image.GetChannel({u, v}, 1),
                             image.GetChannel({u, v}, 2)});
    Float phi = scale * sum / (image.Resolution().x * image.Resolution().y);

    Point3f pCorner(screenBounds.pMax.x, screenBounds.pMax.y, 0);
    Vector3f wCorner = Normalize(Vector3f(lightFromScreen(pCorner)));
    Float cosTotalWidth = CosTheta(wCorner);

    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f w = Normalize(renderFromLight(Vector3f(0, 0, 1)));
    return LightBounds(Bounds3f(p, p), w, phi, std::cos(0.f), cosTotalWidth, false);
}

PBRT_CPU_GPU pstd::optional<LightLeSample> ProjectionLight::SampleLe(Point2f u1, Point2f u2,
                                                        SampledWavelengths &lambda,
                                                        Float time) const {
    // Sample light space ray direction for projection light
    Float pdf;
    Point2f ps = distrib.Sample(u1, &pdf);
    if (pdf == 0)
        return {};
    Vector3f w = Vector3f(lightFromScreen(Point3f(ps.x, ps.y, 0)));

    // Compute PDF for sampled projection light direction
    Float cosTheta = CosTheta(Normalize(w));
    CHECK_GT(cosTheta, 0);
    Float pdfDir = pdf * screenBounds.Area() / (A * Pow<3>(cosTheta));

    // Compute radiance and return projection light sample
    Point2f p = Point2f(screenBounds.Offset(ps));
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(p, c);
    SampledSpectrum L =
        scale * RGBIlluminantSpectrum(*imageColorSpace, rgb).Sample(lambda);
    Ray ray = renderFromLight(
        Ray(Point3f(0, 0, 0), Normalize(w), time, mediumInterface.outside));
    return LightLeSample(L, ray, 1, pdfDir);
}

PBRT_CPU_GPU void ProjectionLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0;
    // Transform ray direction to light space and reject invalid ones
    Vector3f w = Normalize(renderFromLight.ApplyInverse(ray.d));
    if (w.z < hither) {
        *pdfDir = 0;
        return;
    }

    // Compute screen space coordinates for direction and test against bounds
    Point3f ps = screenFromLight(Point3f(w));
    if (!Inside(Point2f(ps.x, ps.y), screenBounds)) {
        *pdfDir = 0;
        return;
    }

    *pdfDir = distrib.PDF(Point2f(ps.x, ps.y)) * screenBounds.Area() /
              (A * Pow<3>(CosTheta(w)));
}

ProjectionLight *ProjectionLight::Create(const Transform &renderFromLight, Medium medium,
                                         const ParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc) {
    Float scale = parameters.GetOneFloat("scale", 1);
    Float power = parameters.GetOneFloat("power", -1);
    Float fov = parameters.GetOneFloat("fov", 90.);

    std::string texname = ResolveFilename(parameters.GetOneString("filename", ""));
    if (texname.empty())
        ErrorExit(loc, "Must provide \"filename\" to \"projection\" light source");

    ImageAndMetadata imageAndMetadata = Image::Read(texname, alloc);
    if (imageAndMetadata.image.HasAnyInfinitePixels())
        ErrorExit(
            loc, "%s: image has infinite pixel values and so is not suitable as a light.",
            texname);
    if (imageAndMetadata.image.HasAnyNaNPixels())
        ErrorExit(
            loc,
            "%s: image has not-a-number pixel values and so is not suitable as a light.",
            texname);

    const RGBColorSpace *colorSpace = imageAndMetadata.metadata.GetColorSpace();

    ImageChannelDesc channelDesc = imageAndMetadata.image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit(loc, "Image provided to \"projection\" light must have R, G, "
                       "and B channels.");
    Image image = imageAndMetadata.image.SelectChannels(channelDesc, alloc);

    scale /= SpectrumToPhotometric(&colorSpace->illuminant);
    if (power > 0) {
        Bounds2f screenBounds;
        Float A;
        Transform lightFromScreen, screenFromLight;
        Float hither = 1e-3f;
        // Initialize _ProjectionLight_ projection matrix
        Float aspect = Float(image.Resolution().x) / Float(image.Resolution().y);
        if (aspect > 1)
            screenBounds = Bounds2f(Point2f(-aspect, -1), Point2f(aspect, 1));
        else
            screenBounds = Bounds2f(Point2f(-1, -1 / aspect), Point2f(1, 1 / aspect));
        screenFromLight = Perspective(fov, hither, 1e30f /* yon */);
        lightFromScreen = Inverse(screenFromLight);

        // Compute projection image area _A_
        Float opposite = std::tan(Radians(fov) / 2);
        A = 4 * Sqr(opposite) * (aspect > 1 ? aspect : (1 / aspect));

        Float sum = 0;
        RGB luminance = colorSpace->LuminanceVector();
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                Point2f ps = screenBounds.Lerp(
                    {(x + .5f) / image.Resolution().x, (y + .5f) / image.Resolution().y});
                Vector3f w = Vector3f(lightFromScreen(Point3f(ps.x, ps.y, 0)));
                w = Normalize(w);
                Float dwdA = Pow<3>(w.z);

                for (int c = 0; c < 3; ++c)
                    sum += image.GetChannel({x, y}, c) * luminance[c] * dwdA;
            }
        scale *= power / (A * sum / (image.Resolution().x * image.Resolution().y));
    }

    Transform flip = Scale(1, -1, 1);
    Transform renderFromLightFlipY = renderFromLight * flip;

    return alloc.new_object<ProjectionLight>(
        renderFromLightFlipY, medium, std::move(image), colorSpace, scale, fov, alloc);
}

// GoniometricLight Method Definitions
GoniometricLight::GoniometricLight(const Transform &renderFromLight,
                                   const MediumInterface &mediumInterface, Spectrum Iemit,
                                   Float scale, Image im, Allocator alloc)
    : LightBase(LightType::DeltaPosition, renderFromLight, mediumInterface),
      Iemit(LookupSpectrum(Iemit)),
      scale(scale),
      image(std::move(im)),
      distrib(alloc) {
    CHECK_EQ(1, image.NChannels());
    CHECK_EQ(image.Resolution().x, image.Resolution().y);
    // Compute sampling distribution for _GoniometricLight_
    Array2D<Float> d = image.GetSamplingDistribution();
    distrib = PiecewiseConstant2D(d);

    imageBytes += image.BytesUsed() + distrib.BytesUsed();
}

PBRT_CPU_GPU pstd::optional<LightLiSample> GoniometricLight::SampleLi(LightSampleContext ctx,
                                                         Point2f u,
                                                         SampledWavelengths lambda,
                                                         bool allowIncompletePDF) const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f wi = Normalize(p - ctx.p());
    SampledSpectrum L =
        I(renderFromLight.ApplyInverse(-wi), lambda) / DistanceSquared(p, ctx.p());
    return LightLiSample(L, wi, 1, Interaction(p, &mediumInterface));
}

PBRT_CPU_GPU Float GoniometricLight::PDF_Li(LightSampleContext, Vector3f,
                               bool allowIncompletePDF) const {
    return 0.f;
}

SampledSpectrum GoniometricLight::Phi(SampledWavelengths lambda) const {
    Float sumY = 0;
    for (int y = 0; y < image.Resolution().y; ++y)
        for (int x = 0; x < image.Resolution().x; ++x)
            sumY += image.GetChannel({x, y}, 0);
    return scale * Iemit->Sample(lambda) * 4 * Pi * sumY /
           (image.Resolution().x * image.Resolution().y);
}

pstd::optional<LightBounds> GoniometricLight::Bounds() const {
    Float sumY = 0;
    for (int y = 0; y < image.Resolution().y; ++y)
        for (int x = 0; x < image.Resolution().x; ++x)
            sumY += image.GetChannel({x, y}, 0);
    Float phi = scale * Iemit->MaxValue() * 4 * Pi * sumY /
                (image.Resolution().x * image.Resolution().y);

    Point3f p = renderFromLight(Point3f(0, 0, 0));
    // Bound it as an isotropic point light.
    return LightBounds(Bounds3f(p, p), Vector3f(0, 0, 1), phi, std::cos(Pi),
                       std::cos(Pi / 2), false);
}

PBRT_CPU_GPU pstd::optional<LightLeSample> GoniometricLight::SampleLe(Point2f u1, Point2f u2,
                                                         SampledWavelengths &lambda,
                                                         Float time) const {
    // Sample direction and PDF for ray leaving goniometric light
    Float pdf;
    Point2f uv = distrib.Sample(u1, &pdf);
    Vector3f wLight = EqualAreaSquareToSphere(uv);
    Float pdfDir = pdf / (4 * Pi);

    Ray ray =
        renderFromLight(Ray(Point3f(0, 0, 0), wLight, time, mediumInterface.outside));
    return LightLeSample(I(wLight, lambda), ray, 1, pdfDir);
}

PBRT_CPU_GPU void GoniometricLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0.f;
    Vector3f wLight = Normalize(renderFromLight.ApplyInverse(ray.d));
    Point2f uv = EqualAreaSphereToSquare(wLight);
    *pdfDir = distrib.PDF(uv) / (4 * Pi);
}

std::string GoniometricLight::ToString() const {
    return StringPrintf("[ GoniometricLight %s Iemit: %s scale: %f ]", BaseToString(),
                        Iemit, scale);
}

GoniometricLight *GoniometricLight::Create(const Transform &renderFromLight,
                                           Medium medium,
                                           const ParameterDictionary &parameters,
                                           const RGBColorSpace *colorSpace,
                                           const FileLoc *loc, Allocator alloc) {
    Spectrum I = parameters.GetOneSpectrum("I", &colorSpace->illuminant,
                                           SpectrumType::Illuminant, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    Image image(alloc);

    std::string texname = ResolveFilename(parameters.GetOneString("filename", ""));
    if (texname.empty())
        Warning(loc, "No \"filename\" parameter provided for goniometric light.");
    else {
        ImageAndMetadata imageAndMetadata = Image::Read(texname, alloc);

        if (imageAndMetadata.image.HasAnyInfinitePixels())
            ErrorExit(
                loc,
                "%s: image has infinite pixel values and so is not suitable as a light.",
                texname);
        if (imageAndMetadata.image.HasAnyNaNPixels())
            ErrorExit(loc,
                      "%s: image has not-a-number pixel values and so is not suitable as "
                      "a light.",
                      texname);

        if (imageAndMetadata.image.Resolution().x !=
            imageAndMetadata.image.Resolution().y)
            ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely "
                      "this is an equal-area environment map.",
                      texname, imageAndMetadata.image.Resolution().x,
                      imageAndMetadata.image.Resolution().y);

        ImageChannelDesc rgbDesc = imageAndMetadata.image.GetChannelDesc({"R", "G", "B"});
        ImageChannelDesc yDesc = imageAndMetadata.image.GetChannelDesc({"Y"});

        if (rgbDesc) {
            if (yDesc)
                ErrorExit("%s: has both \"R\", \"G\", and \"B\" or \"Y\" "
                          "channels.",
                          texname);
            image = Image(imageAndMetadata.image.Format(),
                          imageAndMetadata.image.Resolution(), {"Y"},
                          imageAndMetadata.image.Encoding(), alloc);
            for (int y = 0; y < image.Resolution().y; ++y)
                for (int x = 0; x < image.Resolution().x; ++x)
                    image.SetChannel(
                        {x, y}, 0,
                        imageAndMetadata.image.GetChannels({x, y}, rgbDesc).Average());
        } else if (yDesc)
            image = imageAndMetadata.image;
        else
            ErrorExit(loc,
                      "%s: has neither \"R\", \"G\", and \"B\" or \"Y\" "
                      "channels.",
                      texname);
    }

    sc /= SpectrumToPhotometric(I);

    Float phi_v = parameters.GetOneFloat("power", -1);
    if (phi_v > 0) {
        Float sumY = 0;
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x)
                sumY += image.GetChannel({x, y}, 0);
        Float k_e = 4 * Pi * sumY / (image.Resolution().x * image.Resolution().y);
        sc *= phi_v / k_e;
    }

    const Float swapYZ[4][4] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1};
    Transform t(swapYZ);
    Transform finalRenderFromLight = renderFromLight * t;

    return alloc.new_object<GoniometricLight>(finalRenderFromLight, medium, I, sc,
                                              std::move(image), alloc);
}

// DiffuseAreaLight Method Definitions
DiffuseAreaLight::DiffuseAreaLight(const Transform &renderFromLight,
                                   const MediumInterface &mediumInterface, Spectrum Le,
                                   Float scale, const Shape shape, FloatTexture alpha,
                                   Image im, const RGBColorSpace *imageColorSpace,
                                   bool twoSided)
    : LightBase(
          [](FloatTexture alpha) {
              // Special case handling for area lights with constant zero-valued alpha
              // textures to allow invisible area lights: we will null out the alpha
              // texture below so that as far as the DiffuseAreaLight is concerned, there
              // is no alpha texture and the light is fully emissive. However, such lights
              // will never be intersected by rays (because their associated primitives
              // still have the alpha texture), so we mark them as DeltaPosition lights
              // here so that MIS isn't used for direct illumination. Thus, light sampling
              // is the only strategy used and we get an unbiased (if potentially high
              // variance) estimate.

              const FloatConstantTexture *fc =
                  alpha.CastOrNullptr<FloatConstantTexture>();
              if (fc && fc->Evaluate(TextureEvalContext()) == 0)
                  return LightType::DeltaPosition;
              return LightType::Area;
          }(alpha),
          renderFromLight, mediumInterface),
      shape(shape),
      alpha(type == LightType::Area ? alpha : nullptr),
      area(shape.Area()),
      twoSided(twoSided),
      Lemit(LookupSpectrum(Le)),
      scale(scale),
      image(std::move(im)),
      imageColorSpace(imageColorSpace) {
    ++numAreaLights;

    if (image) {
        ImageChannelDesc desc = image.GetChannelDesc({"R", "G", "B"});
        if (!desc)
            ErrorExit("Image used for DiffuseAreaLight doesn't have R, G, B "
                      "channels.");
        CHECK_EQ(3, desc.size());
        CHECK(desc.IsIdentity());
        CHECK(imageColorSpace);
    } else {
        CHECK(Le);
    }

    // Warn if light has transformation with non-uniform scale, though not
    // for Triangles or bilinear patches, since this doesn't matter for them.
    if (renderFromLight.HasScale() && !shape.Is<Triangle>() && !shape.Is<BilinearPatch>())
        Warning("Scaling detected in rendering to light space transformation! "
                "The system has numerous assumptions, implicit and explicit, "
                "that this transform will have no scale factors in it. "
                "Proceed at your own risk; your image may have errors.");
}

PBRT_CPU_GPU pstd::optional<LightLiSample> DiffuseAreaLight::SampleLi(LightSampleContext ctx,
                                                         Point2f u,
                                                         SampledWavelengths lambda,
                                                         bool allowIncompletePDF) const {
    // Sample point on shape for _DiffuseAreaLight_
    ShapeSampleContext shapeCtx(ctx.pi, ctx.n, ctx.ns, 0 /* time */);
    pstd::optional<ShapeSample> ss = shape.Sample(shapeCtx, u);
    if (!ss || ss->pdf == 0 || LengthSquared(ss->intr.p() - ctx.p()) == 0)
        return {};
    DCHECK(!IsNaN(ss->pdf));
    ss->intr.mediumInterface = &mediumInterface;

    // Check sampled point on shape against alpha texture, if present
    if (AlphaMasked(ss->intr))
        return {};

    // Return _LightLiSample_ for sampled point on shape
    Vector3f wi = Normalize(ss->intr.p() - ctx.p());
    SampledSpectrum Le = L(ss->intr.p(), ss->intr.n, ss->intr.uv, -wi, lambda);
    if (!Le)
        return {};
    return LightLiSample(Le, wi, ss->pdf, ss->intr);
}

PBRT_CPU_GPU Float DiffuseAreaLight::PDF_Li(LightSampleContext ctx, Vector3f wi,
                               bool allowIncompletePDF) const {
    ShapeSampleContext shapeCtx(ctx.pi, ctx.n, ctx.ns, 0 /* time */);
    return shape.PDF(shapeCtx, wi);
}

SampledSpectrum DiffuseAreaLight::Phi(SampledWavelengths lambda) const {
    SampledSpectrum L(0.f);
    if (image) {
        // Compute average light image emission
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                RGB rgb;
                for (int c = 0; c < 3; ++c)
                    rgb[c] = image.GetChannel({x, y}, c);
                L += RGBIlluminantSpectrum(*imageColorSpace, ClampZero(rgb))
                         .Sample(lambda);
            }
        L *= scale / (image.Resolution().x * image.Resolution().y);

    } else
        L = Lemit->Sample(lambda) * scale;
    return Pi * (twoSided ? 2 : 1) * area * L;
}

pstd::optional<LightBounds> DiffuseAreaLight::Bounds() const {
    // Compute _phi_ for diffuse area light bounds
    Float phi = 0;
    if (image) {
        // Compute average _DiffuseAreaLight_ image channel value
        // Assume no distortion in the mapping, FWIW...
        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x)
                for (int c = 0; c < 3; ++c)
                    phi += image.GetChannel({x, y}, c);
        phi /= 3 * image.Resolution().x * image.Resolution().y;

    } else
        phi = Lemit->MaxValue();
    phi *= scale * area * Pi;

    DirectionCone nb = shape.NormalBounds();
    return LightBounds(shape.Bounds(), nb.w, phi, nb.cosTheta, std::cos(Pi / 2),
                       twoSided);
}

PBRT_CPU_GPU pstd::optional<LightLeSample> DiffuseAreaLight::SampleLe(Point2f u1, Point2f u2,
                                                         SampledWavelengths &lambda,
                                                         Float time) const {
    // Sample a point on the area light's _Shape_
    pstd::optional<ShapeSample> ss = shape.Sample(u1);
    if (!ss)
        return {};
    ss->intr.time = time;
    ss->intr.mediumInterface = &mediumInterface;

    // Check sampled point on shape against alpha texture, if present
    if (AlphaMasked(ss->intr))
        return {};

    // Sample a cosine-weighted outgoing direction _w_ for area light
    Vector3f w;
    Float pdfDir;
    if (twoSided) {
        // Choose side of surface and sample cosine-weighted outgoing direction
        if (u2[0] < 0.5f) {
            u2[0] = std::min(u2[0] * 2, OneMinusEpsilon);
            w = SampleCosineHemisphere(u2);
        } else {
            u2[0] = std::min((u2[0] - 0.5f) * 2, OneMinusEpsilon);
            w = SampleCosineHemisphere(u2);
            w.z *= -1;
        }
        pdfDir = CosineHemispherePDF(std::abs(w.z)) / 2;

    } else {
        w = SampleCosineHemisphere(u2);
        pdfDir = CosineHemispherePDF(w.z);
    }
    if (pdfDir == 0)
        return {};

    // Return _LightLeSample_ for ray leaving area light
    const Interaction &intr = ss->intr;
    Frame nFrame = Frame::FromZ(intr.n);
    w = nFrame.FromLocal(w);
    SampledSpectrum Le = L(intr.p(), intr.n, intr.uv, w, lambda);
    return LightLeSample(Le, intr.SpawnRay(w), intr, ss->pdf, pdfDir);
}

PBRT_CPU_GPU void DiffuseAreaLight::PDF_Le(const Interaction &intr, Vector3f w, Float *pdfPos,
                              Float *pdfDir) const {
    CHECK_NE(intr.n, Normal3f(0, 0, 0));
    *pdfPos = shape.PDF(intr);
    *pdfDir = twoSided ? (CosineHemispherePDF(AbsDot(intr.n, w)) / 2)
                       : CosineHemispherePDF(Dot(intr.n, w));
}

std::string DiffuseAreaLight::ToString() const {
    return StringPrintf("[ DiffuseAreaLight %s Lemit: %s scale: %f shape: %s alpha: %s "
                        "twoSided: %s area: %f image: %s ]",
                        BaseToString(), Lemit, scale, shape, alpha,
                        twoSided ? "true" : "false", area, image);
}

DiffuseAreaLight *DiffuseAreaLight::Create(const Transform &renderFromLight,
                                           Medium medium,
                                           const ParameterDictionary &parameters,
                                           const RGBColorSpace *colorSpace,
                                           const FileLoc *loc, Allocator alloc,
                                           const Shape shape, FloatTexture alphaTex) {
    Spectrum L = parameters.GetOneSpectrum("L", nullptr, SpectrumType::Illuminant, alloc);
    Float scale = parameters.GetOneFloat("scale", 1);
    bool twoSided = parameters.GetOneBool("twosided", false);

    std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
    Image image(alloc);
    const RGBColorSpace *imageColorSpace = nullptr;
    if (!filename.empty()) {
        if (L)
            ErrorExit(loc, "Both \"L\" and \"filename\" specified for DiffuseAreaLight.");
        ImageAndMetadata im = Image::Read(filename, alloc);

        if (im.image.HasAnyInfinitePixels())
            ErrorExit(
                loc,
                "%s: image has infinite pixel values and so is not suitable as a light.",
                filename);
        if (im.image.HasAnyNaNPixels())
            ErrorExit(loc,
                      "%s: image has not-a-number pixel values and so is not suitable as "
                      "a light.",
                      filename);

        ImageChannelDesc channelDesc = im.image.GetChannelDesc({"R", "G", "B"});
        if (!channelDesc)
            ErrorExit(loc,
                      "%s: Image provided to \"diffuse\" area light must have "
                      "R, G, and B channels.",
                      filename);
        image = im.image.SelectChannels(channelDesc, alloc);

        imageColorSpace = im.metadata.GetColorSpace();
    } else if (!L)
        L = &colorSpace->illuminant;

    // scale so that radiance is equivalent to 1 nit
    scale /= SpectrumToPhotometric(L ? L : &colorSpace->illuminant);

    Float phi_v = parameters.GetOneFloat("power", -1.0f);
    if (phi_v > 0) {
        // k_e is the emissive power of the light as defined by the spectral
        // distribution and texture and is used to normalize the emitted
        // radiance such that the user-defined power will be the actual power
        // emitted by the light.
        Float k_e = 1;
        if (image) {
            // Get the appropriate luminance vector from the image colour space
            RGB lum = imageColorSpace->LuminanceVector();
            k_e = 0;
            // Assume no distortion in the mapping, FWIW...
            for (int y = 0; y < image.Resolution().y; ++y)
                for (int x = 0; x < image.Resolution().x; ++x) {
                    for (int c = 0; c < 3; ++c)
                        k_e += image.GetChannel({x, y}, c) * lum[c];
                }
            k_e /= image.Resolution().x * image.Resolution().y;
        }

        k_e *= (twoSided ? 2 : 1) * shape.Area() * Pi;

        // now multiply up scale to hit the target power
        scale *= phi_v / k_e;
    }

    return alloc.new_object<DiffuseAreaLight>(renderFromLight, medium, L, scale, shape,
                                              alphaTex, std::move(image), imageColorSpace,
                                              twoSided);
}

// UniformInfiniteLight Method Definitions
UniformInfiniteLight::UniformInfiniteLight(const Transform &renderFromLight,
                                           Spectrum Lemit, Float scale)
    : LightBase(LightType::Infinite, renderFromLight, MediumInterface()),
      Lemit(LookupSpectrum(Lemit)),
      scale(scale) {}

PBRT_CPU_GPU SampledSpectrum UniformInfiniteLight::Le(const Ray &ray,
                                         const SampledWavelengths &lambda) const {
    return scale * Lemit->Sample(lambda);
}

PBRT_CPU_GPU pstd::optional<LightLiSample> UniformInfiniteLight::SampleLi(
    LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
    bool allowIncompletePDF) const {
    if (allowIncompletePDF)
        return {};
    // Return uniform spherical sample for uniform infinite light
    Vector3f wi = SampleUniformSphere(u);
    Float pdf = UniformSpherePDF();
    return LightLiSample(scale * Lemit->Sample(lambda), wi, pdf,
                         Interaction(ctx.p() + wi * (2 * sceneRadius), &mediumInterface));
}

PBRT_CPU_GPU Float UniformInfiniteLight::PDF_Li(LightSampleContext ctx, Vector3f w,
                                   bool allowIncompletePDF) const {
    if (allowIncompletePDF)
        return 0;
    return UniformSpherePDF();
}

SampledSpectrum UniformInfiniteLight::Phi(SampledWavelengths lambda) const {
    return 4 * Pi * Pi * Sqr(sceneRadius) * scale * Lemit->Sample(lambda);
}

PBRT_CPU_GPU pstd::optional<LightLeSample> UniformInfiniteLight::SampleLe(Point2f u1, Point2f u2,
                                                             SampledWavelengths &lambda,
                                                             Float time) const {
    // Sample direction for uniform infinite light ray
    Vector3f w = SampleUniformSphere(u1);

    // Compute infinite light sample ray
    Frame wFrame = Frame::FromZ(-w);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w, time);

    // Compute probabilities for uniform infinite light
    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));
    Float pdfDir = UniformSpherePDF();

    return LightLeSample(scale * Lemit->Sample(lambda), ray, pdfPos, pdfDir);
}

PBRT_CPU_GPU void UniformInfiniteLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfDir = UniformSpherePDF();
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
}

std::string UniformInfiniteLight::ToString() const {
    return StringPrintf("[ UniformInfiniteLight %s Lemit: %s ]", BaseToString(), Lemit);
}

// ImageInfiniteLight Method Definitions
ImageInfiniteLight::ImageInfiniteLight(Transform renderFromLight, Image im,
                                       const RGBColorSpace *imageColorSpace, Float scale,
                                       std::string filename, Allocator alloc)
    : LightBase(LightType::Infinite, renderFromLight, MediumInterface()),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      scale(scale),
      distribution(alloc),
      compensatedDistribution(alloc) {
    // ImageInfiniteLight constructor implementation
    // Initialize sampling PDFs for image infinite area light
    ImageChannelDesc channelDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("%s: image used for ImageInfiniteLight doesn't have R, G, B "
                  "channels.",
                  filename);
    CHECK_EQ(3, channelDesc.size());
    CHECK(channelDesc.IsIdentity());
    if (image.Resolution().x != image.Resolution().y)
        ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely "
                  "this is an equal area environment map.",
                  filename, image.Resolution().x, image.Resolution().y);
    Array2D<Float> d = image.GetSamplingDistribution();
    Bounds2f domain = Bounds2f(Point2f(0, 0), Point2f(1, 1));
    distribution = PiecewiseConstant2D(d, domain, alloc);

    // Initialize compensated PDF for image infinite area light
    Float average = std::accumulate(d.begin(), d.end(), 0.) / d.size();
    for (Float &v : d)
        v = std::max<Float>(v - average, 0);
    if (std::all_of(d.begin(), d.end(), [](Float v) { return v == 0; }))
        std::fill(d.begin(), d.end(), Float(1));
    compensatedDistribution = PiecewiseConstant2D(d, domain, alloc);
}

PBRT_CPU_GPU Float ImageInfiniteLight::PDF_Li(LightSampleContext ctx, Vector3f w,
                                 bool allowIncompletePDF) const {
    Vector3f wLight = renderFromLight.ApplyInverse(w);
    Point2f uv = EqualAreaSphereToSquare(wLight);
    Float pdf = 0;
    if (allowIncompletePDF)
        pdf = compensatedDistribution.PDF(uv);
    else
        pdf = distribution.PDF(uv);
    return pdf / (4 * Pi);
}

SampledSpectrum ImageInfiniteLight::Phi(SampledWavelengths lambda) const {
    // We're computing fluence, then converting to power...
    SampledSpectrum sumL(0.);

    int width = image.Resolution().x, height = image.Resolution().y;
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({u, v}, c, WrapMode::OctahedralSphere);
            sumL +=
                RGBIlluminantSpectrum(*imageColorSpace, ClampZero(rgb)).Sample(lambda);
        }
    }
    // Integrating over the sphere, so 4pi for that.  Then one more for Pi
    // r^2 for the area of the disk receiving illumination...
    return 4 * Pi * Pi * Sqr(sceneRadius) * scale * sumL / (width * height);
}

PBRT_CPU_GPU pstd::optional<LightLeSample> ImageInfiniteLight::SampleLe(Point2f u1, Point2f u2,
                                                           SampledWavelengths &lambda,
                                                           Float time) const {
    // Sample infinite light image and compute ray direction _w_
    Float mapPDF;
    pstd::optional<Point2f> uv = distribution.Sample(u1, &mapPDF);
    if (!uv)
        return {};
    Vector3f wLight = EqualAreaSquareToSphere(*uv);
    Vector3f w = -renderFromLight(wLight);

    // Compute infinite light sample ray
    Frame wFrame = Frame::FromZ(-w);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w, time);

    // Compute _ImageInfiniteLight_ ray PDFs
    Float pdfDir = mapPDF / (4 * Pi);
    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));

    return LightLeSample(ImageLe(*uv, lambda), ray, pdfPos, pdfDir);
}

PBRT_CPU_GPU void ImageInfiniteLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    Vector3f wl = -renderFromLight.ApplyInverse(ray.d);
    Float mapPDF = distribution.PDF(EqualAreaSphereToSquare(wl));
    *pdfDir = mapPDF / (4 * Pi);
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
}

std::string ImageInfiniteLight::ToString() const {
    return StringPrintf("[ ImageInfiniteLight %s scale: %f ]", BaseToString(), scale);
}

// PortalImageInfiniteLight Method Definitions
PortalImageInfiniteLight::PortalImageInfiniteLight(
    const Transform &renderFromLight, Image equalAreaImage,
    const RGBColorSpace *imageColorSpace, Float scale, const std::string &filename,
    std::vector<Point3f> p, Allocator alloc)
    : LightBase(LightType::Infinite, renderFromLight, MediumInterface()),
      image(alloc),
      imageColorSpace(imageColorSpace),
      scale(scale),
      filename(filename),
      distribution(alloc) {
    ImageChannelDesc channelDesc = equalAreaImage.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("%s: image used for PortalImageInfiniteLight doesn't have R, "
                  "G, B channels.",
                  filename);
    CHECK_EQ(3, channelDesc.size());
    CHECK(channelDesc.IsIdentity());

    if (equalAreaImage.Resolution().x != equalAreaImage.Resolution().y)
        ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely "
                  "this is an equal area environment map.",
                  filename, equalAreaImage.Resolution().x, equalAreaImage.Resolution().y);

    if (p.size() != 4)
        ErrorExit("Expected 4 vertices for infinite light portal but given %d", p.size());
    for (int i = 0; i < 4; ++i)
        portal[i] = p[i];

    // PortalImageInfiniteLight constructor conclusion
    // Compute frame for portal coordinate system
    Vector3f p01 = Normalize(portal[1] - portal[0]);
    Vector3f p12 = Normalize(portal[2] - portal[1]);
    Vector3f p32 = Normalize(portal[2] - portal[3]);
    Vector3f p03 = Normalize(portal[3] - portal[0]);
    // Do opposite edges have the same direction?
    if (std::abs(Dot(p01, p32) - 1) > .001 || std::abs(Dot(p12, p03) - 1) > .001)
        Error("Infinite light portal isn't a planar quadrilateral");
    // Sides perpendicular?
    if (std::abs(Dot(p01, p12)) > .001 || std::abs(Dot(p12, p32)) > .001 ||
        std::abs(Dot(p32, p03)) > .001 || std::abs(Dot(p03, p01)) > .001)
        Error("Infinite light portal isn't a planar quadrilateral");
    portalFrame = Frame::FromXY(p03, p01);

    // Resample environment map into rectified image
    image = Image(PixelFormat::Float, equalAreaImage.Resolution(), {"R", "G", "B"},
                  equalAreaImage.Encoding(), alloc);
    ParallelFor(0, image.Resolution().y, [&](int y) {
        for (int x = 0; x < image.Resolution().x; ++x) {
            // Resample _equalAreaImage_ to compute rectified image pixel $(x,y)$
            // Find $(u,v)$ coordinates in equal-area image for pixel
            Point2f uv((x + 0.5f) / image.Resolution().x,
                       (y + 0.5f) / image.Resolution().y);
            Vector3f w = RenderFromImage(uv);
            w = Normalize(renderFromLight.ApplyInverse(w));
            Point2f uvEqui = EqualAreaSphereToSquare(w);

            for (int c = 0; c < 3; ++c) {
                Float v =
                    equalAreaImage.BilerpChannel(uvEqui, c, WrapMode::OctahedralSphere);
                image.SetChannel({x, y}, c, v);
            }
        }
    });

    // Initialize sampling distribution for portal image infinite light
    auto duv_dw = [&](Point2f p) {
        Float duv_dw;
        (void)RenderFromImage(p, &duv_dw);
        return duv_dw;
    };
    Array2D<Float> d = image.GetSamplingDistribution(duv_dw);
    distribution = WindowedPiecewiseConstant2D(d, alloc);
}

SampledSpectrum PortalImageInfiniteLight::Phi(SampledWavelengths lambda) const {
    // We're really computing fluence, then converting to power, for what
    // that's worth..
    SampledSpectrum sumL(0.);

    for (int y = 0; y < image.Resolution().y; ++y) {
        for (int x = 0; x < image.Resolution().x; ++x) {
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({x, y}, c);

            Point2f st((x + 0.5f) / image.Resolution().x,
                       (y + 0.5f) / image.Resolution().y);
            Float duv_dw;
            (void)RenderFromImage(st, &duv_dw);

            sumL +=
                RGBIlluminantSpectrum(*imageColorSpace, ClampZero(rgb)).Sample(lambda) /
                duv_dw;
        }
    }

    return scale * Area() * sumL / (image.Resolution().x * image.Resolution().y);
}

PBRT_CPU_GPU SampledSpectrum PortalImageInfiniteLight::Le(const Ray &ray,
                                             const SampledWavelengths &lambda) const {
    pstd::optional<Point2f> uv = ImageFromRender(Normalize(ray.d));
    pstd::optional<Bounds2f> b = ImageBounds(ray.o);
    if (!uv || !b || !Inside(*uv, *b))
        return SampledSpectrum(0.f);
    return ImageLookup(*uv, lambda);
}

PBRT_CPU_GPU SampledSpectrum PortalImageInfiniteLight::ImageLookup(
    Point2f uv, const SampledWavelengths &lambda) const {
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(uv, c);
    RGBIlluminantSpectrum spec(*imageColorSpace, ClampZero(rgb));
    return scale * spec.Sample(lambda);
}

PBRT_CPU_GPU pstd::optional<LightLiSample> PortalImageInfiniteLight::SampleLi(
    LightSampleContext ctx, Point2f u, SampledWavelengths lambda,
    bool allowIncompletePDF) const {
    // Sample $(u,v)$ in potentially visible region of light image
    pstd::optional<Bounds2f> b = ImageBounds(ctx.p());
    if (!b)
        return {};
    Float mapPDF;
    pstd::optional<Point2f> uv = distribution.Sample(u, *b, &mapPDF);
    if (!uv)
        return {};

    // Convert portal image sample point to direction and compute PDF
    Float duv_dw;
    Vector3f wi = RenderFromImage(*uv, &duv_dw);
    if (duv_dw == 0)
        return {};
    Float pdf = mapPDF / duv_dw;
    CHECK(!IsInf(pdf));

    // Compute radiance for portal light sample and return _LightLiSample_
    SampledSpectrum L = ImageLookup(*uv, lambda);
    Point3f pl = ctx.p() + 2 * sceneRadius * wi;
    return LightLiSample(L, wi, pdf, Interaction(pl, &mediumInterface));
}

PBRT_CPU_GPU Float PortalImageInfiniteLight::PDF_Li(LightSampleContext ctx, Vector3f w,
                                       bool allowIncompletePDF) const {
    // Find image $(u,v)$ coordinates corresponding to direction _w_
    Float duv_dw;
    pstd::optional<Point2f> uv = ImageFromRender(w, &duv_dw);
    if (!uv || duv_dw == 0)
        return 0;

    // Return PDF for sampling $(u,v)$ from reference point
    pstd::optional<Bounds2f> b = ImageBounds(ctx.p());
    if (!b)
        return 0;
    Float pdf = distribution.PDF(*uv, *b);
    return pdf / duv_dw;
}

PBRT_CPU_GPU pstd::optional<LightLeSample> PortalImageInfiniteLight::SampleLe(
    Point2f u1, Point2f u2, SampledWavelengths &lambda, Float time) const {
    Float mapPDF;
    Bounds2f b(Point2f(0, 0), Point2f(1, 1));
    pstd::optional<Point2f> uv = distribution.Sample(u1, b, &mapPDF);
    if (!uv)
        return {};

    // Convert infinite light sample point to direction
    // Note: ignore WorldToLight since we already folded it in when we
    // resampled...
    Float duv_dw;
    Vector3f w = -RenderFromImage(*uv, &duv_dw);
    if (duv_dw == 0)
        return {};

    // Compute PDF for sampled infinite light direction
    Float pdfDir = mapPDF / duv_dw;

#if 0
    // Just sample within the portal.
    // This works with the light path integrator, but not BDPT :-(
    Point3f p = portal[0] + u2[0] * (portal[1] - portal[0]) +
        u2[1] * (portal[3] - portal[0]);
    // Compute _PortalImageInfiniteLight_ ray PDFs
    Ray ray(p, w, time);

    // Cosine to account for projected area of portal w.r.t. ray direction.
    Normal3f n = Normal3f(portalFrame.z);
    Float pdfPos = 1 / (Area() * AbsDot(n, w));
#else
    // Compute infinite light sample ray
    Frame wFrame = Frame::FromZ(-w);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * wFrame.FromLocal(Vector3f(cd.x, cd.y, 0));
    Ray ray(pDisk + sceneRadius * -w, w, time);

    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));
#endif

    SampledSpectrum L = ImageLookup(*uv, lambda);

    return LightLeSample(L, ray, pdfPos, pdfDir);
}

PBRT_CPU_GPU void PortalImageInfiniteLight::PDF_Le(const Ray &ray, Float *pdfPos,
                                      Float *pdfDir) const {
    // TODO: negate here or???
    Vector3f w = -Normalize(ray.d);
    Float duv_dw;
    pstd::optional<Point2f> uv = ImageFromRender(w, &duv_dw);

    if (!uv || duv_dw == 0) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    Bounds2f b(Point2f(0, 0), Point2f(1, 1));
    Float pdf = distribution.PDF(*uv, b);

#if 0
    Normal3f n = Normal3f(portalFrame.z);
    *pdfPos = 1 / (Area() * AbsDot(n, w));
#else
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
#endif

    *pdfDir = pdf / duv_dw;
}

std::string PortalImageInfiniteLight::ToString() const {
    return StringPrintf("[ PortalImageInfiniteLight %s filename:%s scale: %f portal: %s "
                        " portalFrame: %s ]",
                        BaseToString(), filename, scale, portal, portalFrame);
}

// SpotLight Method Definitions
SpotLight::SpotLight(const Transform &renderFromLight,
                     const MediumInterface &mediumInterface, Spectrum Iemit, Float scale,
                     Float totalWidth, Float falloffStart)
    : LightBase(LightType::DeltaPosition, renderFromLight, mediumInterface),
      Iemit(LookupSpectrum(Iemit)),
      scale(scale),
      cosFalloffEnd(std::cos(Radians(totalWidth))),
      cosFalloffStart(std::cos(Radians(falloffStart))) {
    CHECK_LE(falloffStart, totalWidth);
}

PBRT_CPU_GPU Float SpotLight::PDF_Li(LightSampleContext, Vector3f, bool allowIncompletePDF) const {
    return 0.f;
}

PBRT_CPU_GPU SampledSpectrum SpotLight::I(Vector3f w, SampledWavelengths lambda) const {
    return SmoothStep(CosTheta(w), cosFalloffEnd, cosFalloffStart) * scale *
           Iemit->Sample(lambda);
}

SampledSpectrum SpotLight::Phi(SampledWavelengths lambda) const {
    return scale * Iemit->Sample(lambda) * 2 * Pi *
           ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2);
}

pstd::optional<LightBounds> SpotLight::Bounds() const {
    Point3f p = renderFromLight(Point3f(0, 0, 0));
    Vector3f w = Normalize(renderFromLight(Vector3f(0, 0, 1)));
    Float phi = scale * Iemit->MaxValue() * 4 * Pi;
    Float cosTheta_e = std::cos(std::acos(cosFalloffEnd) - std::acos(cosFalloffStart));
    // Allow a little slop here to deal with fp round-off error in the computation of
    // cosTheta_p in the importance function.
    if (cosTheta_e == 1 && cosFalloffEnd != cosFalloffStart)
        cosTheta_e = 0.999f;
    return LightBounds(Bounds3f(p, p), w, phi, cosFalloffStart, cosTheta_e, false);
}

PBRT_CPU_GPU pstd::optional<LightLeSample> SpotLight::SampleLe(Point2f u1, Point2f u2,
                                                  SampledWavelengths &lambda,
                                                  Float time) const {
    // Choose whether to sample spotlight center cone or falloff region
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};
    Float sectionPDF;
    int section = SampleDiscrete(p, u2[0], &sectionPDF);

    // Sample chosen region of spotlight cone
    Vector3f wLight;
    Float pdfDir;
    if (section == 0) {
        // Sample spotlight center cone
        wLight = SampleUniformCone(u1, cosFalloffStart);
        pdfDir = UniformConePDF(cosFalloffStart) * sectionPDF;

    } else {
        // Sample spotlight falloff region
        Float cosTheta = SampleSmoothStep(u1[0], cosFalloffEnd, cosFalloffStart);
        DCHECK(cosTheta >= cosFalloffEnd && cosTheta <= cosFalloffStart);
        Float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
        Float phi = u1[1] * 2 * Pi;
        wLight = SphericalDirection(sinTheta, cosTheta, phi);
        pdfDir = SmoothStepPDF(cosTheta, cosFalloffEnd, cosFalloffStart) * sectionPDF /
                 (2 * Pi);
    }

    // Return sampled spotlight ray
    Ray ray =
        renderFromLight(Ray(Point3f(0, 0, 0), wLight, time, mediumInterface.outside));
    return LightLeSample(I(wLight, lambda), ray, 1, pdfDir);
}

PBRT_CPU_GPU void SpotLight::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};
    *pdfPos = 0;
    // Find spotlight directional PDF based on $\cos \theta$
    Float cosTheta = CosTheta(renderFromLight.ApplyInverse(ray.d));
    if (cosTheta >= cosFalloffStart)
        *pdfDir = UniformConePDF(cosFalloffStart) * p[0] / (p[0] + p[1]);
    else
        *pdfDir = SmoothStepPDF(cosTheta, cosFalloffEnd, cosFalloffStart) * p[1] /
                  ((p[0] + p[1]) * (2 * Pi));
}

std::string SpotLight::ToString() const {
    return StringPrintf(
        "[ SpotLight %s Iemit: %s cosFalloffStart: %f cosFalloffEnd: %f ]",
        BaseToString(), Iemit, cosFalloffStart, cosFalloffEnd);
}

SpotLight *SpotLight::Create(const Transform &renderFromLight, Medium medium,
                             const ParameterDictionary &parameters,
                             const RGBColorSpace *colorSpace, const FileLoc *loc,
                             Allocator alloc) {
    Spectrum I = parameters.GetOneSpectrum("I", &colorSpace->illuminant,
                                           SpectrumType::Illuminant, alloc);
    Float sc = parameters.GetOneFloat("scale", 1);

    Float coneangle = parameters.GetOneFloat("coneangle", 30.);
    Float conedelta = parameters.GetOneFloat("conedeltaangle", 5.);
    // Compute spotlight rendering to light transformation
    Point3f from = parameters.GetOnePoint3f("from", Point3f(0, 0, 0));
    Point3f to = parameters.GetOnePoint3f("to", Point3f(0, 0, 1));

    Transform dirToZ = (Transform)Frame::FromZ(Normalize(to - from));
    Transform t = Translate(Vector3f(from.x, from.y, from.z)) * Inverse(dirToZ);
    Transform finalRenderFromLight = renderFromLight * t;

    sc /= SpectrumToPhotometric(I);

    Float phi_v = parameters.GetOneFloat("power", -1);
    if (phi_v > 0) {
        Float cosFalloffEnd = std::cos(Radians(coneangle));
        Float cosFalloffStart = std::cos(Radians(coneangle - conedelta));
        Float k_e =
            2 * Pi * ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2);
        sc *= phi_v / k_e;
    }

    return alloc.new_object<SpotLight>(finalRenderFromLight, medium, I, sc, coneangle,
                                       coneangle - conedelta);
}

SampledSpectrum Light::Phi(SampledWavelengths lambda) const {
    auto phi = [&](auto ptr) { return ptr->Phi(lambda); };
    return DispatchCPU(phi);
}

void Light::Preprocess(const Bounds3f &sceneBounds) {
    auto preprocess = [&](auto ptr) { return ptr->Preprocess(sceneBounds); };
    return DispatchCPU(preprocess);
}

PBRT_CPU_GPU pstd::optional<LightLeSample> Light::SampleLe(Point2f u1, Point2f u2,
                                              SampledWavelengths &lambda,
                                              Float time) const {
    auto sample = [&](auto ptr) { return ptr->SampleLe(u1, u2, lambda, time); };
    return Dispatch(sample);
}

PBRT_CPU_GPU void Light::PDF_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    auto pdf = [&](auto ptr) { return ptr->PDF_Le(ray, pdfPos, pdfDir); };
    return Dispatch(pdf);
}

pstd::optional<LightBounds> Light::Bounds() const {
    auto bounds = [](auto ptr) { return ptr->Bounds(); };
    return DispatchCPU(bounds);
}

std::string Light::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto str = [](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(str);
}

PBRT_CPU_GPU void Light::PDF_Le(const Interaction &intr, Vector3f w, Float *pdfPos,
                   Float *pdfDir) const {
    auto pdf = [&](auto ptr) { return ptr->PDF_Le(intr, w, pdfPos, pdfDir); };
    return Dispatch(pdf);
}

Light Light::Create(const std::string &name, const ParameterDictionary &parameters,
                    const Transform &renderFromLight,
                    const CameraTransform &cameraTransform, Medium outsideMedium,
                    const FileLoc *loc, Allocator alloc) {
    Light light = nullptr;
    if (name == "point")
        light = PointLight::Create(renderFromLight, outsideMedium, parameters,
                                   parameters.ColorSpace(), loc, alloc);
    else if (name == "virtualpoint")
        light = VirtualPointLight::Create(renderFromLight, outsideMedium, parameters,
                                          parameters.ColorSpace(),
                                          loc, alloc);
    else if (name == "spot")
        light = SpotLight::Create(renderFromLight, outsideMedium, parameters,
                                  parameters.ColorSpace(), loc, alloc);
    else if (name == "goniometric")
        light = GoniometricLight::Create(renderFromLight, outsideMedium, parameters,
                                         parameters.ColorSpace(), loc, alloc);
    else if (name == "projection")
        light = ProjectionLight::Create(renderFromLight, outsideMedium, parameters, loc,
                                        alloc);
    else if (name == "distant")
        light = DistantLight::Create(renderFromLight, parameters, parameters.ColorSpace(),
                                     loc, alloc);
    else if (name == "infinite") {
        const RGBColorSpace *colorSpace = parameters.ColorSpace();
        std::vector<Spectrum> L =
            parameters.GetSpectrumArray("L", SpectrumType::Illuminant, alloc);
        Float scale = parameters.GetOneFloat("scale", 1);
        std::vector<Point3f> portal = parameters.GetPoint3fArray("portal");
        std::string filename = ResolveFilename(parameters.GetOneString("filename", ""));
        Float E_v = parameters.GetOneFloat("illuminance", -1);

        if (L.empty() && filename.empty() && portal.empty()) {
            // Scale the light spectrum to be equivalent to 1 nit
            scale /= SpectrumToPhotometric(&colorSpace->illuminant);
            if (E_v > 0) {
                // If the scene specifies desired illuminance, first calculate
                // the illuminance from a uniform hemispherical emission
                // of L_v then use this to scale the emission spectrum.
                Float k_e = Pi;
                scale *= E_v / k_e;
            }

            // Default: color space's std illuminant
            light = alloc.new_object<UniformInfiniteLight>(
                renderFromLight, &colorSpace->illuminant, scale);
        } else if (!L.empty() && portal.empty()) {
            if (!filename.empty())
                ErrorExit(loc, "Can't specify both emission \"L\" and "
                               "\"filename\" with ImageInfiniteLight");

            // Scale the light spectrum to be equivalent to 1 nit
            scale /= SpectrumToPhotometric(L[0]);

            if (E_v > 0) {
                // If the scene specifies desired illuminance, first calculate
                // the illuminance from a uniform hemispherical emission
                // of L_v then use this to scale the emission spectrum.
                Float k_e = Pi;
                scale *= E_v / k_e;
            }

            light = alloc.new_object<UniformInfiniteLight>(renderFromLight, L[0], scale);
        } else {
            // Either an image was provided or it's "L" with a portal.
            ImageAndMetadata imageAndMetadata;
            if (filename.empty()) {
                // Create a uniform image with the L spectrum converted to
                // RGB so it can be stored in an Image. This usually should
                // be ok, but it is the best we can do under the circumstances.
                // (More generally, for uniform L, it's just as well to create
                // an emissive bilinear patch at the portal location, though
                // that doesn't allow things like putting an emissive sphere out
                // there for the sun, so here we go...
                if (!L[0].Is<RGBIlluminantSpectrum>())
                    Warning(loc, "Converting non-RGB \"L\" parameter to RGB so that a "
                                 "portal light can be used.");
                XYZ xyz = SpectrumToXYZ(L[0]);
                RGB rgb = RGBColorSpace::sRGB->ToRGB(xyz);

                int res = 1;  // happily, this all just works.
                imageAndMetadata.image =
                    Image(PixelFormat::Float, {res, res}, {"R", "G", "B"});
                imageAndMetadata.metadata.colorSpace = RGBColorSpace::sRGB;
                for (int y = 0; y < res; ++y)
                    for (int x = 0; x < res; ++x)
                        for (int c = 0; c < 3; ++c)
                            imageAndMetadata.image.SetChannel({x, y}, c, rgb[c]);
            } else {
                imageAndMetadata = Image::Read(filename, alloc);

                if (imageAndMetadata.image.HasAnyInfinitePixels())
                    ErrorExit(loc,
                              "%s: image has infinite pixel values and so is not "
                              "suitable as a light.",
                              filename);
                if (imageAndMetadata.image.HasAnyNaNPixels())
                    ErrorExit(loc,
                              "%s: image has not-a-number pixel values and so is not "
                              "suitable as a light.",
                              filename);
            }

            const RGBColorSpace *colorSpace = imageAndMetadata.metadata.GetColorSpace();

            ImageChannelDesc channelDesc =
                imageAndMetadata.image.GetChannelDesc({"R", "G", "B"});
            if (!channelDesc)
                ErrorExit(loc,
                          "%s: image provided to \"infinite\" light must "
                          "have R, G, and B channels.",
                          filename);

            // Scale the light spectrum to be equivalent to 1 nit
            scale /= SpectrumToPhotometric(&colorSpace->illuminant);

            if (E_v > 0) {
                // Upper hemisphere illuminance calculation for converting map to physical
                // units
                float illuminance = 0;
                const Image &image = imageAndMetadata.image;
                RGB lum = imageAndMetadata.metadata.GetColorSpace()->LuminanceVector();
                for (int y = 0; y < image.Resolution().y; ++y) {
                    float v = (float(y) + 0.5f) / float(image.Resolution().y);
                    for (int x = 0; x < image.Resolution().x; ++x) {
                        Float u = (x + 0.5f) / image.Resolution().x;
                        Vector3f w = EqualAreaSquareToSphere(Point2f(u, v));
                        // We could be more clever and see if we're in the inner rotated
                        // square, but not a big deal...
                        if (w.z <= 0)
                            continue;

                        ImageChannelValues values = image.GetChannels({x, y});
                        for (int c = 0; c < 3; ++c)
                            illuminance += values[c] * lum[c] * CosTheta(w);
                    }
                }
                illuminance *= 2 * Pi / (image.Resolution().x * image.Resolution().y);

                // scaling factor is just the ratio of the target
                // illuminance and the illuminance of the map multiplied by
                // the illuminant spectrum
                Float k_e = illuminance;
                scale *= E_v / k_e;
            }

            Image image = imageAndMetadata.image.SelectChannels(channelDesc, alloc);

            if (!portal.empty()) {
                for (Point3f &p : portal)
                    p = cameraTransform.RenderFromWorld(p);

                light = alloc.new_object<PortalImageInfiniteLight>(
                    renderFromLight, std::move(image), colorSpace, scale, filename,
                    portal, alloc);
            } else
                light = alloc.new_object<ImageInfiniteLight>(renderFromLight,
                                                             std::move(image), colorSpace,
                                                             scale, filename, alloc);
        }
    } else
        ErrorExit(loc, "%s: light type unknown.", name);

    if (!light)
        ErrorExit(loc, "%s: unable to create light.", name);

    parameters.ReportUnused();
    return light;
}

Light Light::CreateArea(const std::string &name, const ParameterDictionary &parameters,
                        const Transform &renderFromLight,
                        const MediumInterface &mediumInterface, const Shape shape,
                        FloatTexture alpha, const FileLoc *loc, Allocator alloc) {
    Light area = nullptr;
    if (name == "diffuse")
        area =
            DiffuseAreaLight::Create(renderFromLight, mediumInterface.outside, parameters,
                                     parameters.ColorSpace(), loc, alloc, shape, alpha);
    else
        ErrorExit(loc, "%s: area light type unknown.", name);

    if (!area)
        ErrorExit(loc, "%s: unable to create area light.", name);

    parameters.ReportUnused();
    return area;
}

}  // namespace pbrt
