// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/lightsamplers.h>
#include <pbrt/lightsamplers_constants.h>

#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/log.h>
#include <pbrt/wavefront/workitems.h>
#include <pbrt/wavefront/train.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/pair.h>      // if using thrust::pair
#include <thrust/copy.h>      // for debug, etc.
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <curand_kernel.h>
#include <thrust/binary_search.h>

#include <cuda/atomic>
#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace pbrt {

    thrust::device_vector<HashTableEntry> VARLLightSampler_hashMapBuffer;

    VARLLightSampler::VARLLightSampler(pstd::span<const Light> lights, int maxQueueSize, const Bounds3f& sceneBounds, Allocator alloc, const std::string& importanceFunctionName)
    :   lightTree(lights, alloc),
        maxQueueSize(maxQueueSize),
        sceneBounds(sceneBounds),
        gamma(0),
        learningRate(1e-3),
        clusterSize(POSITIONAL_DISCRETIZATION_RESOLUTION*POSITIONAL_DISCRETIZATION_RESOLUTION*POSITIONAL_DISCRETIZATION_RESOLUTION),
        directionalResolution(DIRECTIONAL_DISCRETIZATION_RESOLUTION),
        directionalGridSize{Pi / directionalResolution, 2 * Pi / directionalResolution}, // Phi, Theta
        initLightCutSize(8),
        noChangeIterationLimit(128),
        globalLightCut(
            lightTree.GenerateLightCut<MAX_CUT_SIZE>(
                0.125f / initLightCutSize, initLightCutSize,
                [] (const LightBVHNode& node) -> Float {
                    return node.lightBounds.Phi();
                })
        ),
        trainPixelStride(1),
        vStar(2) // "zero"
    {
        importanceFunction = ImportanceFunction::Create(importanceFunctionName, alloc);
        VARLLightSampler_hashMapBuffer = thrust::device_vector<HashTableEntry>(clusterSize * directionalResolution * directionalResolution, globalLightCut);
        hashMap = thrust::raw_pointer_cast(VARLLightSampler_hashMapBuffer.data());
    }

    PBRT_CPU_GPU
    inline int GridShadingPointIndex(const Point3f& normalizedP, int clusterSizeDim) {
        int x = std::min<int>(normalizedP.x * clusterSizeDim, clusterSizeDim - 1);
        int y = std::min<int>(normalizedP.y * clusterSizeDim, clusterSizeDim - 1);
        int z = std::min<int>(normalizedP.z * clusterSizeDim, clusterSizeDim - 1);
        return x + y * clusterSizeDim + z * clusterSizeDim * clusterSizeDim;
    }

    PBRT_CPU_GPU
    inline uint32_t VARLLightSampler::UniformGridIndex(const Point3f& p, const Vector3f wo) const{
        const int clusterSizeDim = pow(clusterSize, 1.0f / 3.0f);
        Float lightFieldEncoding[2] = { SphericalTheta(wo), SphericalPhi(wo) };
        const Vector3f pPmin = (p - sceneBounds.pMin);
        const Vector3f pMaxPmin = (sceneBounds.pMax - sceneBounds.pMin);
        const Point3f normalizedP = {pPmin.x / pMaxPmin.x, pPmin.y / pMaxPmin.y, pPmin.z / pMaxPmin.z};
        const int shadingPointIndex = GridShadingPointIndex(normalizedP, clusterSizeDim);
        if (shadingPointIndex == -1){
            printf("Point not found in spTree\n");
            return 0;
        } 

        int thetaIndex = (int) std::floor(lightFieldEncoding[0] / directionalGridSize[0]);
        thetaIndex = std::min(thetaIndex, (int)directionalResolution - 1);
        thetaIndex =  std::max(thetaIndex, 0);

        int phiIndex = (int) std::floor(lightFieldEncoding[1] / directionalGridSize[1]);
        phiIndex = std::min(phiIndex, (int)directionalResolution - 1);
        phiIndex = std::max(phiIndex, 0);

        return shadingPointIndex * directionalResolution * directionalResolution
            + thetaIndex * directionalResolution
            + phiIndex;
    }

    void VARLLightSampler::EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size) {
        if (size == 0) return;
        constexpr uint16_t residualInfoDim = sizeof(LightSamplerResidualInfo) / sizeof(float);
        assert(residualInfoDim >= 9); // 3 for p, 3 for n, 3 for wo

        // Goal is to update lightcut weights at first sample (replacing power importance with importanceFunction)
        thrust::device_vector<uint32_t> lightCutIndices(size);
        uint32_t* lightCutIndicesPtr = lightCutIndices.data().get();
        thrust::device_vector<uint32_t> sizeIndices(size);
        uint32_t* sizeIndicesPtr = sizeIndices.data().get();
        
        GPUParallelFor("VARLLightSampler::EvalOrSample Index Computation", size, [=] PBRT_GPU(int sizeIdx) {
            const float* residualInfo = residualInfoBuffer + sizeIdx * residualInfoDim; // 9 floats per sample (3 for p, 3 for n, 3 for wo)
            const Point3f p = Point3f(residualInfo[0], residualInfo[1], residualInfo[2]);
            const Normal3f ns = Normal3f(residualInfo[3], residualInfo[4], residualInfo[5]);
            const Vector3f wo = Vector3f(residualInfo[6], residualInfo[7], residualInfo[8]);

            // Calculate uniform grid coordinates
            uint32_t index = UniformGridIndex(p, wo);
            // Store index for later use
            lightCutIndicesPtr[sizeIdx] = index;
            sizeIndicesPtr[sizeIdx] = sizeIdx;
        });
        GPUWait();

        // Sort lightCutIndices by index and reduce to unique indices
        thrust::sort_by_key(lightCutIndices.begin(), lightCutIndices.end(), sizeIndices.begin());
        auto reducedEnd = thrust::reduce_by_key(lightCutIndices.begin(), lightCutIndices.end(), sizeIndices.begin(), lightCutIndices.begin(), sizeIndices.begin(), thrust::equal_to<uint32_t>(), thrust::minimum<uint32_t>());
        int reducedSize = reducedEnd.first - lightCutIndices.begin();

        // Update lightcut weights at first iteration with importanceFunction evaluated at ctx
        GPUParallelFor("VARLLightSampler::EvalOrSample Importance Update", reducedSize, [=] PBRT_GPU(int reducedIdx) {
            const uint32_t sizeIdx = sizeIndicesPtr[reducedIdx];
            const float* residualInfo = residualInfoBuffer + sizeIdx * residualInfoDim; // 9 floats per sample (3 for p, 3 for n, 3 for wo)
            const Point3f p = Point3f(residualInfo[0], residualInfo[1], residualInfo[2]);
            const Normal3f ns = Normal3f(residualInfo[3], residualInfo[4], residualInfo[5]);
            const Vector3f wo = Vector3f(residualInfo[6], residualInfo[7], residualInfo[8]);

            const uint32_t index = lightCutIndices[reducedIdx];
            LightCut<MAX_CUT_SIZE>& lightCut = hashMap[index].lightCut;
            uint32_t lightCutSize = lightCut.Size();
            for (int i = 0; i < lightCutSize; i++) {
                unsigned int nodeIndex = lightCut[i].nodeIndex;
                const LightBVHNode& node = lightTree.nodes[nodeIndex];
                const Float importanceCtx = importanceFunction.compute(p, ns, lightTree.allLightBounds, node.lightBounds);
                lightCut.UpdateWeight(i, importanceCtx);
            }
            lightCut.NormalizeWeights();
        });
    }


    PBRT_CPU_GPU
    pstd::optional<SampledLight> VARLLightSampler::Sample(const LightSampleContext &ctx, Float u) const {
        // Compute infinite light sampling probability _pInfinite_
        Float pInfinite = lightTree.pInfinite();

        if (u < pInfinite) {
            // Sample infinite lights with uniform probability
            u /= pInfinite;
            int index =
                std::min<int>(u * lightTree.infiniteLights.size(), lightTree.infiniteLights.size() - 1);
            Float pmf = pInfinite / lightTree.infiniteLights.size();
            return SampledLight{lightTree.infiniteLights[index], pmf};
        }

        if (lightTree.nodes.empty())
            return {};
        
        // Adjust u for the odds of not sampling infinite lights
        u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);

        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        Vector3f wo = ctx.wo;
        // Calculate uniform grid coordinates
        uint32_t index = UniformGridIndex(p, wo);

        // LightCut sampling
        Float pmf =(1 - pInfinite);
        Float pmfLightCut;
        const LightCut<MAX_CUT_SIZE>& lightCut = hashMap[index].lightCut;
        LightCutNode lightCutNode = lightCut.Sample(u, &pmfLightCut, &u);
        unsigned int nodeIndex = lightCutNode.nodeIndex;
        
        pmf *= pmfLightCut;
        
        if (nodeIndex >= lightTree.nodes.size()) {
            printf("index: %d Invalid nodeIndex: %u\n", index, nodeIndex);
            return {};
        }

        // Traverse subtree to sample light
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

    PBRT_CPU_GPU
    Float VARLLightSampler::PMF(const LightSampleContext &ctx, Light light) const {
        // Handle infinite _light_ PMF computation
        if (!lightTree.lightToBitTrail.HasKey(light))
            return 1.f / (lightTree.infiniteLights.size() + (lightTree.nodes.empty() ? 0 : 1));

        const Point3f& p = ctx.p();
        const Normal3f& n = ctx.ns;
        const Vector3f& wo = ctx.wo;

        uint32_t uniformGridIndex = UniformGridIndex(p, wo);
        const LightCut<MAX_CUT_SIZE>& lightCut = hashMap[uniformGridIndex].lightCut;
        uint32_t bitTrail = lightTree.lightToBitTrail[light];
        LightCutNode lightCutNode;
        Float pmf = (1 - lightTree.pInfinite()) * lightCut.PMF(bitTrail, &lightCutNode);
        unsigned int nodeIndex = lightCutNode.nodeIndex;
        bitTrail >>= lightCutNode.depth;

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


    struct AppendData {
        Float rewardsSum;
        Float val;
        Float val2;
        uint32_t sizeIdx;
        uint32_t count;

        // We sum rewards, val, val2 and count
        // For sizeIdx, we just keep one value so we can refer later to compute the importance metric
        PBRT_CPU_GPU
        AppendData operator+(const AppendData& other) const {
            return {rewardsSum + other.rewardsSum, val + other.val, val2 + other.val2, sizeIdx, count + other.count};
        }
    };

    struct TentativeChildren {
        LightCutNode leftChild;
        LightCutNode rightChild;
        Float leftWeight;
        Float rightWeight;
        uint32_t lightCutIndex;
        Float pSplit;

        PBRT_CPU_GPU
        bool operator<(const TentativeChildren& other) const {
            return pSplit < other.pSplit;
        }
    };


    struct CurandFunctor {
        unsigned long long seed;

        __host__ __device__
        CurandFunctor(unsigned long long _seed) : seed(_seed) {}

        __device__
        float operator()(int tid) const {
            // Initialize CURAND state with a unique sequence for each thread
            curandState state;
            curand_init(seed, tid, 0, &state);
            // Generate a uniform random number in (0, 1)
            return curand_uniform(&state);
        }
    };

    
    Float VARLLightSampler::TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size) {
        if (size == 0) return 0.f;
        constexpr uint16_t residualInfoDim = sizeof(LightSamplerResidualInfo) / sizeof(float);
        assert(residualInfoDim >= 9); // 3 for p, 3 for n, 3 for wo

        thrust::device_vector<uint32_t> gridIndices(size);
        thrust::device_vector<AppendData> appendData(size);
        uint32_t* gridIndicesPtr  = thrust::raw_pointer_cast(gridIndices.data());
        AppendData* appendDataPtr = thrust::raw_pointer_cast(appendData.data());

        GPUParallelFor("VARLLightSampler::TrainStep Compute Rewards", size, [=] PBRT_GPU(int sizeIdx) {
            const float* residualInfo = residualInfoBuffer + sizeIdx * residualInfoDim;
            const Light light = lightTree.bvhLights[lightIndices[sizeIdx]];
            const RGB& radiance = radiances[sizeIdx];

            // Calculate rewards
            const float radiancePDF = radiancePDFs[sizeIdx];
            if (!lightTree.lightToNodeIndex.HasKey(light)) return;
            const uint32_t nodeIndex = lightTree.lightToNodeIndex[light];
            const LightBVHNode& node = lightTree.nodes[nodeIndex];
            const CompactLightBounds& lightBounds = node.lightBounds;
            const Bounds3f& lightBounds3f = lightBounds.Bounds(lightTree.allLightBounds);
            const Float lightSurfaceArea = lightBounds3f.SurfaceArea();

            RGB rewardRGB = radiance * radiance * lightSurfaceArea / radiancePDF;
            Float reward = std::sqrt(rewardRGB.Average() / radiancePDF);

            const Point3f p(residualInfo[0], residualInfo[1], residualInfo[2]);
            const Normal3f ns(residualInfo[3], residualInfo[4], residualInfo[5]);
            const Vector3f wo(residualInfo[6], residualInfo[7], residualInfo[8]);

            uint32_t index = UniformGridIndex(p, wo);

            // Map light to a node in the lightcut
            uint32_t bitTrail = lightTree.lightToBitTrail[light];
            LightCut<MAX_CUT_SIZE>& lightCut = hashMap[index].lightCut;
            int lightCutIndex = lightCut.Find(bitTrail);
            if (lightCutIndex == -1) {
                printf("LightCut not found for light %d\n", lightIndices[sizeIdx]);
                return;
            }

            // Our "bucket" is index*MAX_CUT_SIZE + lightCutIndex
            uint32_t gridIndex = index * MAX_CUT_SIZE + lightCutIndex;

            // Write to the raw pointers
            gridIndicesPtr[sizeIdx]  = gridIndex;
            Float val = radiance.Average() / radiancePDF;
            appendDataPtr[sizeIdx] = {reward, val, val * radiance.Average(), (uint32_t) sizeIdx, 1};
        });
        GPUWait();

        // One entry per lightcut node
        thrust::sort_by_key(gridIndices.begin(), gridIndices.end(), appendData.begin());
        thrust::device_vector<uint32_t> gridIndicesReduced(size);
        uint32_t* gridIndicesReducedPtr = thrust::raw_pointer_cast(gridIndicesReduced.data());
        thrust::device_vector<AppendData> appendDataReduced(size);
        AppendData* appendDataReducedPtr = thrust::raw_pointer_cast(appendDataReduced.data());
        int newSize = 0;
        {
            std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents("VARLLightSampler::TrainStep Reduce By Key");
            cudaEventRecord(events.first);
            auto newEnd = thrust::reduce_by_key(gridIndices.begin(), gridIndices.end(), 
                appendData.begin(), gridIndicesReduced.begin(), appendDataReduced.begin(), 
                thrust::equal_to<uint32_t>(), thrust::plus<AppendData>());
            cudaEventRecord(events.second);
            newSize = newEnd.first - gridIndicesReduced.begin();
        }
        if (newSize == 0){
            printf("VARLLightSampler::TrainStep Reduce to One Entry per LightCut Node Failed\n");
            return 0;
        }
        

        GPUParallelFor("VARLLightSampler::TrainStep Append Sampling Results and Compute Var", newSize, [=] PBRT_GPU(int reducedIdx) {
            const uint32_t gridIndex = gridIndicesReducedPtr[reducedIdx];
            const uint32_t index = gridIndex / MAX_CUT_SIZE;
            int lightCutIndex = gridIndex % MAX_CUT_SIZE;

            const AppendData& appendData = appendDataReducedPtr[reducedIdx];
            Float val = appendData.val / appendData.count;
            Float val2 = appendData.val2 / appendData.count;
            
            // Appending batched val, val2 to the sampling results
            LightTreeSamplingResults<MAX_CUT_SIZE>& samplingResults = hashMap[index].samplingResults;
            samplingResults.Append(lightCutIndex, val, val2);
        });
        GPUWait();

        thrust::device_vector<float> random_numbers(newSize);
        thrust::counting_iterator<int> index_seq(0);
        const unsigned long long seed = static_cast<unsigned long long>(std::time(nullptr));

        // Use thrust::transform with our CURAND-based functor
        thrust::transform(index_seq, index_seq + newSize,
                        random_numbers.begin(),
                        CurandFunctor(seed));
        
        
        thrust::device_vector<TentativeChildren> tentativeChildrenNodesReduced(newSize);
        TentativeChildren* tentativeChildrenNodesReducedPtr = thrust::raw_pointer_cast(tentativeChildrenNodesReduced.data());
        GPUParallelFor("VARLLightSampler::TrainStep Cluster Tentative Split", newSize, [=] PBRT_GPU(int reducedIdx) {
            const uint32_t gridIndex = gridIndicesReducedPtr[reducedIdx];
            const uint32_t index = gridIndex / MAX_CUT_SIZE;
            const uint32_t lightCutIndex = gridIndex % MAX_CUT_SIZE;
            const uint32_t sizeIdx = appendDataReducedPtr[reducedIdx].sizeIdx; // We pick the first sizeIdx of each lightcut node to compute the importance metric
            const HashTableEntry& entry = hashMap[index];
            const LightCut<MAX_CUT_SIZE>& lightCut = entry.lightCut;

            // Only try to subdivide if lightcut permits
            const bool canSubdivide = lightCut.Size() < MAX_CUT_SIZE && entry.noChangeIterations < noChangeIterationLimit;
            if (!canSubdivide)
                return;
        
            const LightTreeSamplingResults<MAX_CUT_SIZE>& samplingResults = entry.samplingResults;
            const LightCutNode& lightCutNode = lightCut[lightCutIndex];
            const uint32_t nodeIndex = lightCutNode.nodeIndex;
            if (nodeIndex >= lightTree.nodes.size()) {
                printf("Invalid nodeIndex: %u\n", nodeIndex);
                return;
            }
            const LightBVHNode& node = lightTree.nodes[nodeIndex];
            
            Float variance = samplingResults.Var(lightCutIndex);
            uint32_t nSamples = samplingResults.SampleCount(lightCutIndex);
            if (nSamples == 0) return;

            Float sumVariances = 0;
            for (int i = 0; i < lightCut.Size(); i++)
                sumVariances += samplingResults.Var(i);

            if (sumVariances == 0) return;
            
            Float pEPtr = 1.0f / (1.0f + ((float) lightCut.Size() / initLightCutSize) * std::exp(-1.0f * sumVariances));
            Float pT = 1.f - (1.f / nSamples);
            Float pV = variance / sumVariances;
            Float pSplit = pEPtr * pT * pV;
            Float uSplit = random_numbers[reducedIdx];

            // Returning if node is leaf or if split probability is less than random number
            if (node.isLeaf || uSplit > pSplit || nodeIndex + 1 >= lightTree.nodes.size() || node.childOrLightIndex >= lightTree.nodes.size())
                return;

            // Splitting the lightcut node
            const LightBVHNode& child0 = lightTree.nodes[nodeIndex + 1];
            const LightBVHNode& child1 = lightTree.nodes[node.childOrLightIndex];

            const float* residualInfo = residualInfoBuffer + sizeIdx * residualInfoDim;
            const Point3f p(residualInfo[0], residualInfo[1], residualInfo[2]);
            const Normal3f ns(residualInfo[3], residualInfo[4], residualInfo[5]);
            const Vector3f wo(residualInfo[6], residualInfo[7], residualInfo[8]);

            Float lr = 1.0f / (4.0f * std::pow(entry.iteration, 0.857f));
            if (learningRate > 0.f)
                lr = learningRate;

            Float lErrBound = importanceFunction.compute(p, ns, lightTree.allLightBounds, child0.lightBounds);
            Float rErrBound = importanceFunction.compute(p, ns, lightTree.allLightBounds, child1.lightBounds);
            Float lrErrBound = lErrBound + rErrBound;
            if (lrErrBound == 0) return;
            Float lw = lErrBound / lrErrBound;
            Float rw = rErrBound / lrErrBound;
            Float lSampleCount = lw * nSamples;
            Float rSampleCount = rw * nSamples;
            Float lE = lw * samplingResults.Exp2(lightCutIndex);
            Float rE = rw * samplingResults.Exp2(lightCutIndex);
            Float lC = std::pow(1.0f - lr, lSampleCount);
            Float rC = std::pow(1.0f - lr, rSampleCount);

            const float curMaxQValue = entry.futureValue(vStar);
            Float lWeight = (std::sqrt(lE * 1.f) + gamma * curMaxQValue) * 
              (1.0f - lC) + lErrBound * lC;
            Float rWeight = (std::sqrt(rE * 1.f) + gamma * curMaxQValue) * 
              (1.0f - rC) + rErrBound * rC;

            // We store the left and right child nodes for tentative update at the lightcut level
            LightCutNode leftChildNode = {nodeIndex + 1, lightCutNode.depth + 1, lightCutNode.bitTrail};
            LightCutNode rightChildNode = {node.childOrLightIndex, lightCutNode.depth + 1, lightCutNode.bitTrail | (1 << lightCutNode.depth)};
            tentativeChildrenNodesReducedPtr[reducedIdx] = {leftChildNode, rightChildNode, lWeight, rWeight, lightCutIndex, pSplit};
        });
        GPUWait();
        
        // One entry per lightcut
        thrust::device_vector<uint32_t> gridIndicesReducedUnique(newSize);
        uint32_t* gridIndicesReducedUniquePtr = thrust::raw_pointer_cast(gridIndicesReducedUnique.data());
        thrust::device_vector<TentativeChildren> tentativeChildrenNodesReducedUnique(newSize);
        TentativeChildren* tentativeChildrenNodesReducedUniquePtr = thrust::raw_pointer_cast(tentativeChildrenNodesReducedUnique.data());

        thrust::copy(gridIndicesReduced.begin(), gridIndicesReduced.begin() + newSize, gridIndicesReducedUnique.begin());
        thrust::transform(gridIndicesReducedUnique.begin(), gridIndicesReducedUnique.end(), gridIndicesReducedUnique.begin(), [] __device__ (uint32_t a) { return a / MAX_CUT_SIZE; });
        auto newEndUnique = thrust::reduce_by_key(gridIndicesReducedUnique.begin(), gridIndicesReducedUnique.end(), 
            tentativeChildrenNodesReduced.begin(), gridIndicesReducedUnique.begin(), tentativeChildrenNodesReducedUnique.begin(), thrust::equal_to<uint32_t>(), thrust::maximum<TentativeChildren>());
        int newSizeUnique = newEndUnique.first - gridIndicesReducedUnique.begin();

        if (newSizeUnique == 0){
            printf("VARLLightSampler::TrainStep Reduce to One Entry per LightCut Failed\n");
            return 0;
        }

        GPUParallelFor("VARLLightSampler::TrainStep Cluster Apply Updates", newSizeUnique, [=] PBRT_GPU(int reducedUniqueIdx) {
            const uint32_t index = gridIndicesReducedUniquePtr[reducedUniqueIdx];
            
            hashMap[index].iteration += 1;
            // Preemptively increment noChangeIterations. If split is applied, we reset it to 0
            hashMap[index].noChangeIterations += 1;

            LightCut<MAX_CUT_SIZE>& lightCut = hashMap[index].lightCut;
            LightTreeSamplingResults<MAX_CUT_SIZE>& samplingResults = hashMap[index].samplingResults;
            
            const bool canSubdivide = lightCut.Size() < MAX_CUT_SIZE && hashMap[index].noChangeIterations - 1 < noChangeIterationLimit;
            if (!canSubdivide)
                return;
        

            const TentativeChildren& tentativeChildren = tentativeChildrenNodesReducedUniquePtr[reducedUniqueIdx];
            const Float pSplit = tentativeChildren.pSplit;
            
            // If no split is proposed for the cut, we return
            if (pSplit == 0.f)
                return;
            
            const LightCutNode& leftChildNode = tentativeChildren.leftChild;
            const LightCutNode& rightChildNode = tentativeChildren.rightChild;
            const Float lWeight = tentativeChildren.leftWeight;
            const Float rWeight = tentativeChildren.rightWeight;
            const uint32_t lightCutIndex = tentativeChildren.lightCutIndex;

            // Replace the lightcut node with the left child inplace
            lightCut.Replace(lightCutIndex, leftChildNode.nodeIndex, leftChildNode.depth, leftChildNode.bitTrail, lWeight);
            samplingResults.Reset(lightCutIndex);

            // Append the right child to the lightcut
            lightCut.Append(rightChildNode.nodeIndex, rightChildNode.depth, rightChildNode.bitTrail, rWeight);

            // Set noChangeIterations to 0 if split is applied
            hashMap[index].noChangeIterations = 0;
        });
        GPUWait();


        GPUParallelFor("VARLLightSampler::TrainStep QLearning Update Weight", newSize, [=] PBRT_GPU(int reducedIdx) {
            const uint32_t gridIndex = gridIndicesReducedPtr[reducedIdx];
            const uint32_t index = gridIndex / MAX_CUT_SIZE;
            int lightCutIndex = gridIndex % MAX_CUT_SIZE;

            const AppendData& appendData = appendDataReducedPtr[reducedIdx];
            Float rewardsSum = appendData.rewardsSum;
            uint32_t rewardsCount = appendData.count;
            if (rewardsCount == 0){
                printf("No rewards found\n");
                return;
            }

            // Q-learning update over average rewards
            Float avgRewards = rewardsSum / rewardsCount;
            Float curMaxQValue = hashMap[index].futureValue(vStar);
            LightCut<MAX_CUT_SIZE>& lightCut = hashMap[index].lightCut;

            Float lr = 1.0f / (4.0f * std::pow(hashMap[index].iteration - 1, 0.857f)); // We use iteration - 1 to account for the increment in the previous kernel
            if (learningRate > 0.f)
                lr = learningRate;

            Float oldWeight = lightCut.Weight(lightCutIndex);
            Float newWeight = (1.f - lr) * oldWeight + lr * (avgRewards + gamma * curMaxQValue);
            lightCut.UpdateWeight(lightCutIndex, newWeight);
        });
        GPUWait();

        GPUParallelFor("VARLLightSampler::TrainStep QLearning Normalize Weights", newSizeUnique, [=] PBRT_GPU(int reducedUniqueIdx) {
            const uint32_t index = gridIndicesReducedUniquePtr[reducedUniqueIdx];
            LightCut<MAX_CUT_SIZE>& lightCut = hashMap[index].lightCut;
            lightCut.NormalizeWeights();
        });
        GPUWait();
        

        return 0;
    }

}  // namespace pbrt
