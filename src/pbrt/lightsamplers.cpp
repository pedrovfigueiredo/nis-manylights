// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/lightsamplers.h>

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

#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>
#include <iostream>
namespace pbrt {

std::string SampledLight::ToString() const {
    return StringPrintf("[ SampledLight light: %s p: %f ]",
                        light ? light.ToString().c_str() : "(nullptr)", p);
}

std::string CompactLightBounds::ToString() const {
    return StringPrintf(
        "[ CompactLightBounds qb: [ [ %u %u %u ] [ %u %u %u ] ] w: %s (%s) phi: %f "
        "qCosTheta_o: %u (%f) qCosTheta_e: %u (%f) twoSided: %u ]",
        qb[0][0], qb[0][1], qb[0][2], qb[1][0], qb[1][1], qb[1][2], w, Vector3f(w), phi,
        qCosTheta_o, CosTheta_o(), qCosTheta_e, CosTheta_e(), twoSided);
}

std::string CompactLightBounds::ToString(const Bounds3f &allBounds) const {
    return StringPrintf(
        "[ CompactLightBounds b: %s qb: [ [ %u %u %u ] [ %u %u %u ] ] w: %s (%s) phi: %f "
        "qCosTheta_o: %u (%f) qCosTheta_e: %u (%f) twoSided: %u ]",
        Bounds(allBounds), qb[0][0], qb[0][1], qb[0][2], qb[1][0], qb[1][1], qb[1][2], w,
        Vector3f(w), phi, qCosTheta_o, CosTheta_o(), qCosTheta_e, CosTheta_e(), twoSided);
}

LightSampler LightSampler::Create(const std::string &name, pstd::span<const Light> lights,
                                  Allocator alloc) {
    if (name == "uniform")
        return alloc.new_object<UniformLightSampler>(lights, alloc);
    else if (name == "power")
        return alloc.new_object<PowerLightSampler>(lights, alloc);
    else if (name == "bvh")
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    else if (name == "slc")
        return alloc.new_object<SLCLightTreeSampler>(lights, alloc, "slc");
    else if (name == "slcrt")
        return alloc.new_object<SLCLightTreeSampler>(lights, alloc, "slcrt");
    else if (name == "slcadt")
        return alloc.new_object<SLCLightTreeSampler>(lights, alloc, "slcats");
    else if (name == "exhaustive")
        return alloc.new_object<ExhaustiveLightSampler>(lights, alloc);
    else if (name.find("varl") == 0) {
        Error(R"(Light sample distribution type "%s" requires maxQueueSize. Using "bvh" instead.)",
              name.c_str());
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    }
    else if (name.find("neural") == 0) {
        Error(R"(Light sample distribution type "%s" requires maxQueueSize. Using "bvh" instead.)",
              name.c_str());
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    }   
    else {
        Error(R"(Light sample distribution type "%s" unknown. Using "bvh".)",
              name.c_str());
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    }
}

LightSampler LightSampler::Create(const std::string &name, pstd::span<const Light> lights,
                                  int maxQueueSize, const Bounds3f& sceneBounds, Vector2i resolution, Allocator alloc) {
    if (name == "uniform")
        return alloc.new_object<UniformLightSampler>(lights, alloc);
    else if (name == "power")
        return alloc.new_object<PowerLightSampler>(lights, alloc);
    else if (name == "bvh")
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    else if (name == "slc")
        return alloc.new_object<SLCLightTreeSampler>(lights, alloc, "slc");
    else if (name == "slcrt")
        return alloc.new_object<SLCLightTreeSampler>(lights, alloc, "slcrt");
    else if (name == "slcats")
        return alloc.new_object<SLCLightTreeSampler>(lights, alloc, "slcats");
    else if (name == "neural_slc")
        return alloc.new_object<NeuralSLCLightSampler>(lights, maxQueueSize, alloc, "slc");
    else if (name == "neural_slcrt")
        return alloc.new_object<NeuralSLCLightSampler>(lights, maxQueueSize, alloc, "slcrt");
    else if (name == "neural_slcats")
        return alloc.new_object<NeuralSLCLightSampler>(lights, maxQueueSize, alloc, "slcats");
    else if (name == "varl_slc")
        return alloc.new_object<VARLLightSampler>(lights, maxQueueSize, sceneBounds, alloc, "slc");
    else if (name == "varl_slcrt")
        return alloc.new_object<VARLLightSampler>(lights, maxQueueSize, sceneBounds, alloc, "slcrt");
    else if (name == "varl_slcats")
        return alloc.new_object<VARLLightSampler>(lights, maxQueueSize, sceneBounds, alloc, "slcats");
    else if (name == "exhaustive")
        return alloc.new_object<ExhaustiveLightSampler>(lights, alloc);
    
    else {
        Error(R"(Light sample distribution type "%s" unknown. Using "bvh".)",
              name.c_str());
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    }
}

std::string LightSampler::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

///////////////////////////////////////////////////////////////////////////
// PowerLightSampler

// PowerLightSampler Method Definitions
PowerLightSampler::PowerLightSampler(pstd::span<const Light> lights, Allocator alloc)
    : lights(lights.begin(), lights.end(), alloc),
      lightToIndex(alloc),
      aliasTable(alloc) {
    if (lights.empty())
        return;
    // Initialize _lightToIndex_ hash table
    for (size_t i = 0; i < lights.size(); ++i)
        lightToIndex.Insert(lights[i], i);

    // Compute lights' power and initialize alias table
    pstd::vector<Float> lightPower;
    SampledWavelengths lambda = SampledWavelengths::SampleVisible(0.5f);
    for (const auto &light : lights) {
        SampledSpectrum phi = SafeDiv(light.Phi(lambda), lambda.PDF());
        lightPower.push_back(phi.Average());
    }
    if (std::accumulate(lightPower.begin(), lightPower.end(), 0.f) == 0.f)
        std::fill(lightPower.begin(), lightPower.end(), 1.f);
    aliasTable = AliasTable(lightPower, alloc);
}

std::string PowerLightSampler::ToString() const {
    return StringPrintf("[ PowerLightSampler aliasTable: %s ]", aliasTable);
}

///////////////////////////////////////////////////////////////////////////
// BVHLightSampler

STAT_MEMORY_COUNTER("Memory/Light BVH", lightBVHBytes);
STAT_INT_DISTRIBUTION("Integrator/Lights sampled per lookup", nLightsSampled);

// LightTree Method Definitions
LightBVHTree::LightBVHTree(pstd::span<const Light> lights, Allocator alloc)
    : bvhLights(alloc),
      infiniteLights(alloc),
      nodes(alloc),
      lightToBitTrail(alloc),
      lightToIndex(alloc),
      lightToNodeIndex(alloc) {
    // Initialize _infiniteLights_ array and light BVH
    std::vector<std::pair<int, LightBounds>> bvhLightBounds;
    size_t j = 0;
    for (size_t i = 0; i < lights.size(); ++i) {
        // Store $i$th light in either _infiniteLights_ or _bvhLights_
        Light light = lights[i];
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        if (!lightBounds)
            infiniteLights.push_back(light);
        else if (lightBounds->phi > 0){
            bvhLights.push_back(light);
            bvhLightBounds.push_back(std::make_pair(j++, *lightBounds));
            allLightBounds = Union(allLightBounds, lightBounds->bounds);
        }
    }
    if (!bvhLightBounds.empty())
        buildBVH(bvhLightBounds, 0, bvhLightBounds.size(), 0, 0);
    
    pInf = Float(infiniteLights.size()) /
            Float(infiniteLights.size() + (nodes.empty() ? 0 : 1));
    
    LOG_VERBOSE("Built light BVH with %d lights (%d infinite, pInf: %f), %d nodes", bvhLights.size(),
                infiniteLights.size(), pInf, nodes.size());

    lightBVHBytes += nodes.size() * sizeof(LightBVHNode) +
                     lightToBitTrail.capacity() * sizeof(uint32_t) +
                     lightToIndex.capacity() * sizeof(uint32_t) +
                     lightToNodeIndex.capacity() * sizeof(uint32_t) +
                     bvhLights.size() * sizeof(Light) +
                     infiniteLights.size() * sizeof(Light);
}

std::pair<int, LightBounds> LightBVHTree::buildBVH(
    std::vector<std::pair<int, LightBounds>> &bvhLightBounds, int start, int end,
    uint32_t bitTrail, int depth) {
    DCHECK_LT(start, end);
    // Initialize leaf node if only a single light remains
    if (end - start == 1) {
        int nodeIndex = nodes.size();
        CompactLightBounds cb(bvhLightBounds[start].second, allLightBounds);
        int lightIndex = bvhLightBounds[start].first;
        nodes.push_back(LightBVHNode::MakeLeaf(lightIndex, cb));
        lightToBitTrail.Insert(bvhLights[lightIndex], bitTrail);
        lightToIndex.Insert(bvhLights[lightIndex], lightIndex);
        lightToNodeIndex.Insert(bvhLights[lightIndex], nodeIndex);
        return {nodeIndex, bvhLightBounds[start].second};
    }

    // Choose split dimension and position using modified SAH
    // Compute bounds and centroid bounds for lights
    Bounds3f bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        const LightBounds &lb = bvhLightBounds[i].second;
        bounds = Union(bounds, lb.bounds);
        centroidBounds = Union(centroidBounds, lb.Centroid());
    }

    Float minCost = Infinity;
    int minCostSplitBucket = -1, minCostSplitDim = -1;
    constexpr int nBuckets = 12;
    for (int dim = 0; dim < 3; ++dim) {
        // Compute minimum cost bucket for splitting along dimension _dim_
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim])
            continue;
        // Compute _LightBounds_ for each bucket
        LightBounds bucketLightBounds[nBuckets];
        for (int i = start; i < end; ++i) {
            Point3f pc = bvhLightBounds[i].second.Centroid();
            int b = nBuckets * centroidBounds.Offset(pc)[dim];
            if (b == nBuckets)
                b = nBuckets - 1;
            DCHECK_GE(b, 0);
            DCHECK_LT(b, nBuckets);
            bucketLightBounds[b] = Union(bucketLightBounds[b], bvhLightBounds[i].second);
        }

        // Compute costs for splitting lights after each bucket
        Float cost[nBuckets - 1];
        for (int i = 0; i < nBuckets - 1; ++i) {
            // Find _LightBounds_ for lights below and above bucket split
            LightBounds b0, b1;
            for (int j = 0; j <= i; ++j)
                b0 = Union(b0, bucketLightBounds[j]);
            for (int j = i + 1; j < nBuckets; ++j)
                b1 = Union(b1, bucketLightBounds[j]);

            // Compute final light split cost for bucket
            cost[i] = EvaluateCost(b0, bounds, dim) + EvaluateCost(b1, bounds, dim);
        }

        // Find light split that minimizes SAH metric
        for (int i = 1; i < nBuckets - 1; ++i) {
            if (cost[i] > 0 && cost[i] < minCost) {
                minCost = cost[i];
                minCostSplitBucket = i;
                minCostSplitDim = dim;
            }
        }
    }

    // Partition lights according to chosen split
    int mid;
    if (minCostSplitDim == -1)
        mid = (start + end) / 2;
    else {
        const auto *pmid = std::partition(
            &bvhLightBounds[start], &bvhLightBounds[end - 1] + 1,
            [=](const std::pair<int, LightBounds> &l) {
                int b = nBuckets *
                        centroidBounds.Offset(l.second.Centroid())[minCostSplitDim];
                if (b == nBuckets)
                    b = nBuckets - 1;
                DCHECK_GE(b, 0);
                DCHECK_LT(b, nBuckets);
                return b <= minCostSplitBucket;
            });
        mid = pmid - &bvhLightBounds[0];
        if (mid == start || mid == end)
            mid = (start + end) / 2;
        DCHECK(mid > start && mid < end);
    }

    // Allocate interior _LightBVHNode_ and recursively initialize children
    int nodeIndex = nodes.size();
    nodes.push_back(LightBVHNode());
    CHECK_LT(depth, 64);
    std::pair<int, LightBounds> child0 =
        buildBVH(bvhLightBounds, start, mid, bitTrail, depth + 1);
    DCHECK_EQ(nodeIndex + 1, child0.first);
    std::pair<int, LightBounds> child1 =
        buildBVH(bvhLightBounds, mid, end, bitTrail | (1u << depth), depth + 1);

    // Initialize interior node and return node index and bounds
    LightBounds lb = Union(child0.second, child1.second);
    CompactLightBounds cb(lb, allLightBounds);
    nodes[nodeIndex] = LightBVHNode::MakeInterior(child1.first, cb);
    return {nodeIndex, lb};
}

inline bool LinearBVHNodeIsLeaf(const LinearBVHNode* node) {
    CHECK(node != nullptr);
    return node->nPrimitives ^ 0;
}

// SceneBVHTree Method Definitions
ShadingPointBVHTree::ShadingPointBVHTree(const LinearBVHNode* sceneNodes, uint32_t cSize, Allocator alloc)
    : nodes(alloc), clusters(alloc), clusterSize(4)
{
    // 1. Extracting allBounds from the scene BVH
    using ShadingPointClusterBuildInfo = std::pair<int, Bounds3f>;
    const Vector3f threshold = Vector3f(1e-6f, 1e-6f, 1e-6f);
    auto compareBbox = [](const ShadingPointClusterBuildInfo& a, const ShadingPointClusterBuildInfo& b) {
        return Length(a.second.Diagonal()) < Length(b.second.Diagonal());
    };
    std::priority_queue<ShadingPointClusterBuildInfo, std::vector<ShadingPointClusterBuildInfo>, 
        decltype(compareBbox)> queueBbox(compareBbox);
    
    queueBbox.push(ShadingPointClusterBuildInfo(0, sceneNodes[0].bounds));
    while(queueBbox.size() < clusterSize) {
        const ShadingPointClusterBuildInfo curCluster = queueBbox.top();
        queueBbox.pop();
        if (LinearBVHNodeIsLeaf(&sceneNodes[curCluster.first])) {
            Bounds3f bounds = curCluster.second;
            int maxExtent = bounds.MaxDimension();
            Point3f max0 = bounds.pMax;
            Point3f min1 = bounds.pMin;
            max0[maxExtent] = (bounds.pMax[maxExtent] + bounds.pMin[maxExtent]) / 2.0f;
            min1[maxExtent] = (bounds.pMax[maxExtent] + bounds.pMin[maxExtent]) / 2.0f;
            Bounds3f b0 = Bounds3f(bounds.pMin, max0);
            Bounds3f b1 = Bounds3f(min1, bounds.pMax);
            queueBbox.push(ShadingPointClusterBuildInfo(curCluster.first, b0));
            queueBbox.push(ShadingPointClusterBuildInfo(curCluster.first, b1));
        } else {
            const Bounds3f& boundsLeftChild = sceneNodes[curCluster.first].bounds;
            queueBbox.push(ShadingPointClusterBuildInfo(curCluster.first+1, 
                    boundsLeftChild));
            int secondChildIndex = sceneNodes[curCluster.first].secondChildOffset;
            const Bounds3f& boundsRightChild = sceneNodes[secondChildIndex].bounds;
            queueBbox.push(ShadingPointClusterBuildInfo(secondChildIndex, 
                    boundsRightChild));
            
            const Bounds3f unionBounds = Union(boundsLeftChild, boundsRightChild);
            if (curCluster.second != unionBounds) {
                printf("Parent bounds %s, union of children bounds %s\n", curCluster.second.ToString().c_str(), unionBounds.ToString().c_str());
            }
        }
    }

    // clusters.reserve(queueBbox.size());
    // while(!queueBbox.empty()) {
    //     const auto& curCluster = queueBbox.top();
    //     clusters.push_back(curCluster.second);
    //     queueBbox.pop();
    // }
    std::vector<ShadingPointClusterBuildInfo> containerIndex;
    containerIndex.reserve(clusterSize);
    while(!queueBbox.empty()) {
        const auto& curCluster = queueBbox.top();
        containerIndex.push_back(curCluster);
        queueBbox.pop();
    }
    auto compareIndex = [](const ShadingPointClusterBuildInfo& a, const ShadingPointClusterBuildInfo& b) {
        return a.first < b.first;
    };
    std::sort(containerIndex.begin(), containerIndex.end(), compareIndex);


    // 2. Building shading point BVH from clusters
    std::vector<std::pair<int, Bounds3f>> bvhSceneBounds;
    for (size_t i = 0; i < containerIndex.size(); ++i){
        bvhSceneBounds.push_back(std::make_pair(i, containerIndex[i].second));
        totalBounds = Union(totalBounds, containerIndex[i].second);    
    }

    if (!bvhSceneBounds.empty())
        buildBVH(bvhSceneBounds, 0, bvhSceneBounds.size());
    
    LOG_VERBOSE("Built shading point BVH with %d clusters, %d nodes", clusters.size(),
                nodes.size());
    
    // Check if union of children bounds match parent bounds
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i].isLeaf) {
            Bounds3f childBounds = Union(nodes[nodes[i].childOrClusterIndex].bounds,
                                         nodes[i + 1].bounds);
            // Check if union of children bounds match parent bounds
            if (childBounds != nodes[i].bounds) {
                LOG_VERBOSE("Node %d: %s, children union: %s", i, nodes[i].bounds, childBounds);
            }
        }
    }
    LOG_VERBOSE("Total bounds: %s", totalBounds);
    Point3f examplePoint = Point3f(-4.694394, -1.102731, -1.637287);
    int clusterIndex = ClusterIndex(examplePoint);
    for (size_t i = 0; i < containerIndex.size(); ++i) {
        if (InsideExclusive(examplePoint, containerIndex[i].second)) {
            if (i != clusterIndex) {
                printf("%d Point %s is inside bounds %s but got cluster %d\n", i, examplePoint.ToString().c_str(), containerIndex[i].second.ToString().c_str(), clusterIndex);
            }else{
                printf("%d Point %s is inside bounds %s and got cluster %d\n", i, examplePoint.ToString().c_str(), containerIndex[i].second.ToString().c_str(), clusterIndex);
            }
        }
    }
    exit(1);
}


PBRT_CPU_GPU
int ShadingPointBVHTree::ClusterIndex(const Point3f &p) const {
    // for (size_t i = 0; i < clusters.size(); ++i) {
    //     if (InsideExclusive(p, clusters[i]))
    //         return i;
    // }

    int index = 0; // start at root
    while (!nodes[index].isLeaf) {
        // By convention:
        //   left child = index + 1
        //   right child = nodes[index].childOrClusterIndex
        // Check if inside "left child" bounds first
        int leftChild = index + 1;
        int rightChild = nodes[index].childOrClusterIndex;

        // If p is in the left child, go left
        if (InsideExclusive(p, nodes[leftChild].bounds)){
            printf("nodeIndex: %d Point %f %f %f is inside left bounds %f %f %f %f %f %f\n", index, p.x, p.y, p.z, nodes[leftChild].bounds.pMin.x, nodes[leftChild].bounds.pMin.y, nodes[leftChild].bounds.pMin.z, nodes[leftChild].bounds.pMax.x, nodes[leftChild].bounds.pMax.y, nodes[leftChild].bounds.pMax.z);
            index = leftChild;
        }   
        else if (InsideExclusive(p, nodes[rightChild].bounds)){
            // Otherwise, go right
            printf("nodeIndex: %d Point %f %f %f is inside right bounds %f %f %f %f %f %f\n", index, p.x, p.y, p.z, nodes[rightChild].bounds.pMin.x, nodes[rightChild].bounds.pMin.y, nodes[rightChild].bounds.pMin.z, nodes[rightChild].bounds.pMax.x, nodes[rightChild].bounds.pMax.y, nodes[rightChild].bounds.pMax.z);
            index = rightChild;
        }
        else{
            printf("nodeIndex: %d Point %f %f %f is outside left bounds %f %f %f %f %f %f and right bounds %f %f %f %f %f %f\n", index, p.x, p.y, p.z, nodes[leftChild].bounds.pMin.x, nodes[leftChild].bounds.pMin.y, nodes[leftChild].bounds.pMin.z, nodes[leftChild].bounds.pMax.x, nodes[leftChild].bounds.pMax.y, nodes[leftChild].bounds.pMax.z, nodes[rightChild].bounds.pMin.x, nodes[rightChild].bounds.pMin.y, nodes[rightChild].bounds.pMin.z, nodes[rightChild].bounds.pMax.x, nodes[rightChild].bounds.pMax.y, nodes[rightChild].bounds.pMax.z);
            return -1;
        }
    }
    // We've reached a leaf node, check if it truly contains p
    if (InsideExclusive(p, nodes[index].bounds))
        return nodes[index].childOrClusterIndex;
    
    // -1 if it's outside
    printf("nodeIndex: %d Point %f %f %f is outside bounds %f %f %f %f %f %f\n", index, p.x, p.y, p.z, nodes[index].bounds.pMin.x, nodes[index].bounds.pMin.y, nodes[index].bounds.pMin.z, nodes[index].bounds.pMax.x, nodes[index].bounds.pMax.y, nodes[index].bounds.pMax.z);
    return -1;
}



std::pair<int, Bounds3f> ShadingPointBVHTree::buildBVH(
    std::vector<std::pair<int, Bounds3f>> &bvhSceneBounds,
    int start, int end) 
{
    // If only one cluster, make a leaf node
    if (end - start == 1) {
        int nodeIndex = (int)nodes.size();
        int clusterIndex = bvhSceneBounds[start].first;
        const Bounds3f &cb = bvhSceneBounds[start].second;

        // Make a leaf node that references "clusterIndex"
        nodes.push_back(ShadingPointBVHNode::MakeLeaf(clusterIndex, cb));
        return {nodeIndex, cb};
    }

    // 1. Compute overall bounding box for [start, end)
    Bounds3f nodeBounds;
    for (int i = start; i < end; ++i)
        nodeBounds = Union(nodeBounds, bvhSceneBounds[i].second);

    // 2. Compute centroid bounds to decide split axis
    Bounds3f centroidBounds;
    for (int i = start; i < end; ++i) {
        Point3f c = bvhSceneBounds[i].second.Centroid();
        centroidBounds = Union(centroidBounds, c);
    }

    // 3. Choose an axis to split on (largest dimension in nodeBounds)
    Vector3f diag = nodeBounds.Diagonal();
    int axis = 0;
    if (diag.y > diag.x && diag.y > diag.z) axis = 1;
    else if (diag.z > diag.x) axis = 2;

    // 4. Sort [start..end) by the centroid in 'axis'
    std::sort(bvhSceneBounds.begin() + start,
              bvhSceneBounds.begin() + end,
              [axis](const std::pair<int, Bounds3f> &a,
                     const std::pair<int, Bounds3f> &b)
    {
        float ca = 0.5f * (a.second.pMin[axis] + a.second.pMax[axis]);
        float cb = 0.5f * (b.second.pMin[axis] + b.second.pMax[axis]);
        return ca < cb;
    });

    // 5. Split in the middle
    int mid = (start + end) / 2;

    // 6. Create a placeholder for this interior node
    int nodeIndex = (int)nodes.size();
    nodes.push_back(ShadingPointBVHNode()); // we’ll fill it in after children

    // 7. Recursively build left subtree (nodeIndex + 1)
    auto leftChild  = buildBVH(bvhSceneBounds, start, mid);
    // 8. Recursively build right subtree => returns (rightIndex, rightBounds)
    auto rightChild = buildBVH(bvhSceneBounds, mid, end);

    // 9. The interior node's bounding box is the union of child boxes
    Bounds3f combinedBounds = Union(leftChild.second, rightChild.second);

    // 10. Fill in the interior node:
    //   by PBRT convention,
    //   left child = nodeIndex + 1
    //   right child = rightChild.first
    nodes[nodeIndex] = ShadingPointBVHNode::MakeInterior(rightChild.first,
                                                         combinedBounds);

    // 11. Return node index and combined bounds
    return { nodeIndex, combinedBounds };
}


// BVHLightSampler Method Definitions
std::string BVHLightSampler::ToString() const {
    return StringPrintf("[ BVHLightSampler nodes: %s ]", lightTree.nodes);
}

ImportanceFunction ImportanceFunction::Create(const std::string &name, Allocator alloc) {
    if (name == "uniform")
        return alloc.new_object<UniformImportance>();
    else if (name == "slc")
        return alloc.new_object<SLCImportance>();
    else if (name == "slcrt")
        return alloc.new_object<SLCRTImportance>();
    else if (name == "slcats")
        return alloc.new_object<ATSImportance>();
    else {
        Error(R"(Importance Function type "%s" unknown. Using "slc".)",
              name.c_str());
        return alloc.new_object<SLCImportance>();
    }
}

std::string ImportanceFunction::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return DispatchCPU(ts);
}

std::string LightBVHNode::ToString() const {
    return StringPrintf(
        "[ LightBVHNode lightBounds: %s childOrLightIndex: %d isLeaf: %d ]", lightBounds,
        childOrLightIndex, isLeaf);
}

// ExhaustiveLightSampler Method Definitions
ExhaustiveLightSampler::ExhaustiveLightSampler(pstd::span<const Light> lights,
                                               Allocator alloc)
    : lights(lights.begin(), lights.end(), alloc),
      boundedLights(alloc),
      infiniteLights(alloc),
      lightBounds(alloc),
      lightToBoundedIndex(alloc) {
    for (const auto &light : lights) {
        if (pstd::optional<LightBounds> lb = light.Bounds(); lb) {
            lightToBoundedIndex.Insert(light, boundedLights.size());
            lightBounds.push_back(*lb);
            boundedLights.push_back(light);
        } else
            infiniteLights.push_back(light);
    }
}

PBRT_CPU_GPU pstd::optional<SampledLight> ExhaustiveLightSampler::Sample(const LightSampleContext &ctx,
                                                            Float u) const {
    Float pInfinite = Float(infiniteLights.size()) /
                      Float(infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));

    // Note: shared with BVH light sampler...
    if (u < pInfinite) {
        u /= pInfinite;
        int index = std::min<int>(u * infiniteLights.size(), infiniteLights.size() - 1);
        Float pdf = pInfinite * 1.f / infiniteLights.size();
        return SampledLight{infiniteLights[index], pdf};
    } else {
        u = std::min<Float>((u - pInfinite) / (1 - pInfinite), OneMinusEpsilon);

        uint64_t seed = MixBits(FloatToBits(u));
        WeightedReservoirSampler<Light> wrs(seed);

        for (size_t i = 0; i < boundedLights.size(); ++i)
            wrs.Add(boundedLights[i], lightBounds[i].Importance(ctx.p(), ctx.n));

        if (!wrs.HasSample())
            return {};

        Float pdf = (1.f - pInfinite) * wrs.SampleProbability();
        return SampledLight{wrs.GetSample(), pdf};
    }
}

PBRT_CPU_GPU Float ExhaustiveLightSampler::PMF(const LightSampleContext &ctx, Light light) const {
    if (!lightToBoundedIndex.HasKey(light))
        return 1.f / (infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));

    Float importanceSum = 0;
    Float lightImportance = 0;
    for (size_t i = 0; i < boundedLights.size(); ++i) {
        Float importance = lightBounds[i].Importance(ctx.p(), ctx.n);
        importanceSum += importance;
        if (light == boundedLights[i])
            lightImportance = importance;
    }
    Float pInfinite = Float(infiniteLights.size()) /
                      Float(infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));
    Float pdf = lightImportance / importanceSum * (1. - pInfinite);
    return pdf;
}

std::string ExhaustiveLightSampler::ToString() const {
    return StringPrintf("[ ExhaustiveLightSampler lightBounds: %s]", lightBounds);
}

}  // namespace pbrt
