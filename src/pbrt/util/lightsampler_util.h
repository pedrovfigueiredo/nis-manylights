#ifndef PBRT_LIGHTSAMPLERS_UTIL_H
#define PBRT_LIGHTSAMPLERS_UTIL_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>  // LightBounds. Should that live elsewhere?
#include <pbrt/util/containers.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace pbrt {

    PBRT_CPU_GPU
    inline Float AbsMaxDistAlong(Point3f p, Vector3f dir, const Point3f &pMin, const Point3f &pMax) {
      Vector3f dir_p = {dir.x * p.x, dir.y * p.y, dir.z * p.z};
      Vector3f mx0 = {std::abs(dir.x * pMin.x - dir_p.x), std::abs(dir.y * pMin.y - dir_p.y), std::abs(dir.z * pMin.z - dir_p.z)};
      Vector3f mx1 = {std::abs(dir.x * pMax.x - dir_p.x), std::abs(dir.y * pMax.y - dir_p.y), std::abs(dir.z * pMax.z - dir_p.z)};
      return std::max(mx0.x, mx1.x) + std::max(mx0.y, mx1.y) + std::max(mx0.z, mx1.z);
    }

    PBRT_CPU_GPU
    inline Float AbsMinDistAlong(Point3f p, Vector3f dir, const Point3f &pMin, const Point3f &pMax) {
      bool hasPositive = false, hasNegative = false;
      Float a = Dot(dir, pMin - p);
      Float b = Dot(dir, Point3f(pMin.x, pMin.y, pMax.z) - p);
      Float c = Dot(dir, Point3f(pMin.x, pMax.y, pMin.z) - p);
      Float d = Dot(dir, Point3f(pMin.x, pMax.y, pMax.z) - p);
      Float e = Dot(dir, Point3f(pMax.x, pMin.y, pMin.z) - p);
      Float f = Dot(dir, Point3f(pMax.x, pMin.y, pMax.z) - p);
      Float g = Dot(dir, Point3f(pMax.x, pMax.y, pMin.z) - p);
      Float h = Dot(dir, pMax - p);
      hasPositive = a > 0 || b > 0 || c > 0 || d > 0 || e > 0 || f > 0 || g > 0 || h > 0;
      hasNegative = a < 0 || b < 0 || c < 0 || d < 0 || e < 0 || f < 0 || g < 0 || h < 0;
      if (hasPositive && hasNegative)  return 0.f;
      return std::min(
        std::min(
          std::min(std::abs(a), std::abs(b)), 
          std::min(std::abs(c), std::abs(d))
        ), 
        std::min(
          std::min(std::abs(e), std::abs(f)), 
          std::min(std::abs(g), std::abs(h))
        )
      );
    }

    PBRT_CPU_GPU
    inline Float GeomTermBound(Point3f p, Vector3f n, const Point3f &pMin, const Point3f &pMax) {
      Float nrm_max = AbsMaxDistAlong(p, n, pMin, pMax);
      if (nrm_max <= 0.f)  return 0.f;
      Vector3f T, B;
      CoordinateSystem(n, &T, &B);
      Float y_amin = AbsMinDistAlong(p, T, pMin, pMax);
      Float z_amin = AbsMinDistAlong(p, B, pMin, pMax);
      Float hyp = SafeSqrt(y_amin * y_amin + z_amin * z_amin + nrm_max * nrm_max);
      return nrm_max / hyp;
    }

    PBRT_CPU_GPU
    static inline Float SquaredDistanceToClosestPoint(Point3f p, const Bounds3f &bounds) {
      Vector3f d = Vector3f(Min(Max(p, bounds.pMin), bounds.pMax));
      return Dot(d, d);
    }

    PBRT_CPU_GPU
    inline Float SquaredDistanceToFarthestPoint(Point3f p, const Bounds3f &bounds) {
      Vector3f d = Vector3f(Max(Abs(p - bounds.pMin), Abs(p - bounds.pMax)));
      return Dot(d, d);
    }
 
    PBRT_CPU_GPU
    inline Float normalizedWeights(Float l2_0, Float l2_1, Float intensGeom0, Float intensGeom1){
      Float ww0 = intensGeom0 / l2_0;
      Float ww1 = intensGeom1 / l2_1;
      return ww0 / (ww0 + ww1);
    }

    // Converts from [pMin, pMax] to [0,1] based on bounds
    PBRT_CPU_GPU
    inline Vector3f normalizePosition(Point3f p, const Bounds3f& bounds){
      Vector3f boundsDiag = bounds.Diagonal();
      Float boundsDiagNorm = Length(boundsDiag);
      Float boundsDiagNormInflationDelta = 0.005f * boundsDiagNorm;
      Bounds3f inflatedBounds = bounds;
      inflatedBounds.Inflate(boundsDiagNormInflationDelta);
      boundsDiag = inflatedBounds.Diagonal();
      Vector3f offsetP = inflatedBounds.Offset(p);
      return offsetP;
    }

    // Converts from [-1,1] to [0,1]
    PBRT_CPU_GPU
    inline Vector3f normalizeDirection(Vector3f d){
      return Vector3f((d.x + 1.f) / 2.f, (d.y + 1.f) / 2.f, (d.z + 1.f) / 2.f);
    }

    // Convert from cartesian to spherical coordinates
    PBRT_CPU_GPU
    inline Vector2f cartesianToSpherical(const Vector3f& v){
      const Vector3f vn = Normalize(v);
      Vector2f sph = {SafeACos(vn.z), atan2(vn.y, vn.x)};
      if (sph[1] < 0) sph[1] += Pi2;
      return sph;
    }

    PBRT_CPU_GPU
    inline Vector2f cartesianToSpherical(const Normal3f& n){
      return cartesianToSpherical(Vector3f(n));
    }

    PBRT_CPU_GPU
    inline Vector2f cartesianToSphericalNormalized(const Vector3f& v){
      Vector2f sph = cartesianToSpherical(v);
      return {sph[0] / Pi, sph[1] / Pi2};
    }

    PBRT_CPU_GPU
    inline Vector2f cartesianToSphericalNormalized(const Normal3f& n){
      return cartesianToSphericalNormalized(Vector3f(n));
    }

    // Snaps vector3f to nearest resolution
    PBRT_CPU_GPU
    inline Vector3f discretizeVec(const Vector3f& v, const int resolution){
      return Vector3f(
        (float) int(v.x * resolution) / resolution, 
        (float) int(v.y * resolution) / resolution, 
        (float) int(v.z * resolution) / resolution
      );
    }

    class LightBVHTree;
    struct LightBVHNode;
    class ImportanceFunction;
    template <typename LightCutType, uint32_t max_cut_size>
    PBRT_CPU_GPU
    inline void getLightCut(const Point3f& p, const Normal3f& n, const LightBVHTree& tree, const Float errorLimit, const ImportanceFunction& F, LightCutType* lightCut){
        const Bounds3f& allb = tree.allLightBounds;
        Point3f allLightCenter;
        Float allLightRadius;
        allb.BoundingSphere(&allLightCenter, &allLightRadius);
        auto errorFunction = [p, n, allLightRadius, errorLimit](unsigned int isLeaf, const Bounds3f& bounds, Float intensity) -> Float {
            if (isLeaf) return 0.f;
            Float dlen2 = SquaredDistanceToClosestPoint(p, bounds);
            Float SR2 = errorLimit * allLightRadius;
            SR2 *= SR2;
            if (dlen2 < SR2) dlen2 = SR2;
            const Float atten = GeomTermBound(p, Vector3f(n), bounds.pMin, bounds.pMax) / dlen2;
            return intensity * atten;
        };

        size_t maxCutSize = std::min((uint32_t) tree.bvhLights.size(), lightCut->Capacity());     
        size_t numLights = 1;
        
        // Add root to cut
        const LightBVHNode *root = &tree.nodes[0];
        Float rootError = errorFunction(root->isLeaf, root->lightBounds.Bounds(allb), root->lightBounds.Phi());
        lightCut->Append(0, 0, 0, 1.f);
        lightCut->NormalizeWeights();

        // If root is already below error limit, return
        if (rootError < errorLimit * root->lightBounds.Phi()){
            return;
        }

        struct ErrorElement{
            Float error;
            size_t cutIndex;
        };
        ErrorElement errorVec[max_cut_size];
        unsigned int maxErrorIdx = 0;
        errorVec[0] = {rootError, 0};

        while(numLights < maxCutSize){
          const unsigned int cutIndex = errorVec[maxErrorIdx].cutIndex;
          const unsigned int nodeIndex = (*lightCut)[cutIndex].nodeIndex;
          const LightBVHNode *node = &tree.nodes[nodeIndex];
          if (node->isLeaf){
              LOG_ERROR("Trying to split leaf node when building light cut");
              break;
          } 

          unsigned int depth = (*lightCut)[cutIndex].depth;
          uint32_t bitTrail = (*lightCut)[cutIndex].bitTrail;

          // If node is not a leaf, replace it with first child. Append second child
          const LightBVHNode *children[2] = {&tree.nodes[nodeIndex + 1],
                                              &tree.nodes[node->childOrLightIndex]};
          const Float probNode = lightCut->Weight(cutIndex);
          Float prob0Child;
          if (!F.compute(p, n, allb, children[0]->lightBounds, children[1]->lightBounds, prob0Child))
            break;
          
          lightCut->Replace(cutIndex, nodeIndex + 1, depth + 1, bitTrail,
              probNode * prob0Child);
          lightCut->Append(node->childOrLightIndex, depth + 1, bitTrail | (1u << depth),
              probNode * (1.f - prob0Child));
          lightCut->NormalizeWeights();
          
          // lightCut->Replace(cutIndex, nodeIndex + 1, depth + 1, bitTrail,
          //     F.compute(p, n, allb, children[0]->lightBounds));
          // lightCut->Append(node->childOrLightIndex, depth + 1, bitTrail | (1u << depth),
          //     F.compute(p, n, allb, children[1]->lightBounds));

          // Compute errors and update errorVec
          errorVec[maxErrorIdx] = {
              errorFunction(children[0]->isLeaf, children[0]->lightBounds.Bounds(allb), children[0]->lightBounds.Phi()),
              cutIndex
          };
          errorVec[numLights] = {
              errorFunction(children[1]->isLeaf, children[1]->lightBounds.Bounds(allb), children[1]->lightBounds.Phi()),
              numLights
          };
          numLights++;

          // Find next node to replace
          Float maxCurError = -1e10;
          for (int i = 0; i < numLights; i++){
              const Float curError = errorVec[i].error;
              if (curError > maxCurError){
                  maxCurError = curError;
                  maxErrorIdx = i;
              }
          }
          if (maxCurError < errorLimit * tree.nodes[nodeIndex].lightBounds.Phi()) break;
        }
    }
};

#endif  // PBRT_LIGHTSAMPLERS_UTIL_H