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

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace pbrt {

    using nlohmann::json;
    using precision_t = tcnn::network_precision_t;
    using namespace tcnn;
    template <typename T>
    using GPUMatrix = tcnn::GPUMatrix<T, tcnn::MatrixLayout::ColumnMajor>;

    std::shared_ptr<NetworkWithInputEncoding<precision_t>> NeuralSLCLightSampler_network;
    std::shared_ptr<tcnn::Optimizer<precision_t>> NeuralSLCLightSampler_optimizer;
    std::shared_ptr<tcnn::Loss<precision_t>> NeuralSLCLightSampler_loss;
    std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> NeuralSLCLightSampler_trainer;

    GPUMemory<precision_t> NeuralSLCLightSampler_trainOutputBuffer;
    GPUMemory<precision_t> NeuralSLCLightSampler_gradientOutputBuffer;
    GPUMemory<float> NeuralSLCLightSampler_lossBuffer;
    GPUMemory<float> NeuralSLCLightSampler_inferenceOutputBuffer;

    GPUMemory<uint32_t> NeuralSLCLightSampler_pixelIndexToOutputIndexBuffer;

    NeuralSLCLightSampler::NeuralSLCLightSampler(pstd::span<const Light> lights, int maxQueueSize, 
        Allocator alloc, const std::string& importanceFunctionName)
    : lightTree(lights, alloc), maxQueueSize(maxQueueSize), 
    optBucketToNodeIndex(NEURAL_OUTPUT_DIM_MAX, alloc),
    lightToOptBucket(alloc),
    neuralOutputDim(NEURAL_OUTPUT_DIM_MAX - buildOptBucketNodeIndexMappings(0, 0, 0, 0).second),
    neuralOutputDimPadded((neuralOutputDim + 16 - 1) / 16 * 16),
    trainPixelStride(1)
    {
        importanceFunction = ImportanceFunction::Create(importanceFunctionName, alloc);
        LOG_VERBOSE("Output dim: %d (max = %d), Padded output dim: %d", neuralOutputDim, NEURAL_OUTPUT_DIM_MAX, neuralOutputDimPadded);

        NeuralSLCLightSampler_pixelIndexToOutputIndexBuffer = GPUMemory<uint32_t>(MAX_INFERENCE_NUM);
        pixelIndexToOutputIndex = NeuralSLCLightSampler_pixelIndexToOutputIndexBuffer.data();


        // Allocate buffers for inference and training
        NeuralSLCLightSampler_trainOutputBuffer = GPUMemory<precision_t>(neuralOutputDimPadded * TRAIN_BATCH_SIZE);
        NeuralSLCLightSampler_gradientOutputBuffer = GPUMemory<precision_t>(neuralOutputDimPadded * TRAIN_BATCH_SIZE);
        NeuralSLCLightSampler_lossBuffer = GPUMemory<float>(TRAIN_BATCH_SIZE);
        NeuralSLCLightSampler_inferenceOutputBuffer = GPUMemory<float>(neuralOutputDim * MAX_INFERENCE_NUM);
        
        // Assign pointer to inference output buffer so it can be referred to inside PMF and Sample functions
        inferenceOutputPtr = NeuralSLCLightSampler_inferenceOutputBuffer.data();

        // Initialize network, optimizer, loss and trainer
        auto loadJson = [](const std::string& filepath) -> json {
            if (!std::filesystem::exists(filepath)) {
                LOG_FATAL("Cannot locate file at %s", filepath.c_str());
                return {};
            }
            std::ifstream f(filepath);
            if (f.fail()) {
                LOG_FATAL("Failed to read JSON file at %s", filepath.c_str());
                return {};
            }
            json file = json::parse(f, nullptr, true, true);
            return file;
        };
    #if (LEARN_PRODUCT_SAMPLING == 1)
        #if (NEURAL_GRID_DISCRETIZATION == 0)
            std::string path = "common/configs/base_product.json";
        #else
            std::string path = "common/configs/base_product_discretized.json";
        #endif
    #else
        #if (NEURAL_GRID_DISCRETIZATION == 0)
            std::string path = "common/configs/base_radiance.json";
        #else
            std::string path = "common/configs/base_radiance_discretized.json";
        #endif
    #endif
        json config = loadJson(path)["nn"];
        LOG_VERBOSE("Successfully loaded JSON config from %s", path.c_str());

        // Create network with input encoding
        json& encoding_config = config["encoding"];
        json& optimizer_config = config["optimizer"];
        json& network_config = config["network"];

        json& loss_config = config["loss"];			// just a dummy loss used to make trainer happy, will be by-passed

        // Reset loss
        NeuralSLCLightSampler_loss.reset(create_loss<precision_t>(loss_config));
        
        // Reset optimizers
        NeuralSLCLightSampler_optimizer.reset(create_optimizer<precision_t>(optimizer_config));

        NeuralSLCLightSampler_network = std::make_shared<NetworkWithInputEncoding<precision_t>>(
			NEURAL_INPUT_DIM, neuralOutputDim, encoding_config, network_config);

        NeuralSLCLightSampler_trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
			NeuralSLCLightSampler_network, NeuralSLCLightSampler_optimizer, NeuralSLCLightSampler_loss, 7272);

        LOG_VERBOSE("Network has a padded output width of %d. Total params: %d", NeuralSLCLightSampler_network->padded_output_width(), NeuralSLCLightSampler_network->n_params());

        NeuralSLCLightSampler_trainer->initialize_params();
        cudaDeviceSynchronize();
    }

    int NeuralSLCLightSampler::buildOptBucketNodeIndexMappings(int nodeIndex, int optBucket){
        const LightBVHNode *node = &lightTree.nodes[nodeIndex];
        // Push the current optBucket from height H to every light ( leaf node in pre-order traversal)
        if (node->isLeaf){
            optBucketToNodeIndex[optBucket++] = nodeIndex;
            Light& light = lightTree.bvhLights[node->childOrLightIndex];
            lightToOptBucket.Insert(light, optBucket - 1);
            return optBucket;
        }

        const int returnLeft = buildOptBucketNodeIndexMappings(nodeIndex + 1, optBucket);
        return buildOptBucketNodeIndexMappings(node->childOrLightIndex, returnLeft);
    }

    std::pair<int, int> NeuralSLCLightSampler::buildOptBucketNodeIndexMappingsHeightCapped(int nodeIndex, int optBucket, uint32_t depth, int n_collapsed_outputdims){
        if (depth == NEURAL_HEIGHT){
            PBRT_DBG("buildOptBucketNodeIndexMappingsHeightCapped:: Saving nodeIndex %d to optBucket %d", nodeIndex, optBucket);
            optBucketToNodeIndex[optBucket++] = nodeIndex;
        }

        const LightBVHNode *node = &lightTree.nodes[nodeIndex];
        // Push the current optBucket from height H to every light ( leaf node in pre-order traversal)
        if (node->isLeaf){
            // If a leaf is above the NEURAL_HEIGHT, we must increment the optBucket AND add to the number of deprecated output dims
            if (depth < NEURAL_HEIGHT){
                PBRT_DBG("buildOptBucketNodeIndexMappingsHeightCapped:: Saving nodeIndex %d to optBucket %d", nodeIndex, optBucket);
                optBucketToNodeIndex[optBucket++] = nodeIndex;
                n_collapsed_outputdims += powi(2, NEURAL_HEIGHT - depth) - 1;
            }

            Light& light = lightTree.bvhLights[node->childOrLightIndex];
            lightToOptBucket.Insert(light, optBucket - 1);
            PBRT_DBG("buildOptBucketNodeIndexMappingsHeightCapped:: Light %d: OptBucket %d", node->childOrLightIndex, optBucket - 1);
            return {optBucket, n_collapsed_outputdims};
        }

        const std::pair<int, int> returnLeft = buildOptBucketNodeIndexMappingsHeightCapped(nodeIndex + 1, optBucket, depth + 1, n_collapsed_outputdims);
        return buildOptBucketNodeIndexMappingsHeightCapped(node->childOrLightIndex, returnLeft.first, depth + 1, returnLeft.second);
    }

    void NeuralSLCLightSampler::PopulateLightToOptBucket(int nodeIndex, int optBucket){
        const LightBVHNode *node = &lightTree.nodes[nodeIndex];
        if (node->isLeaf){
            Light& light = lightTree.bvhLights[node->childOrLightIndex];
            lightToOptBucket.Insert(light, optBucket);
            return;
        }
        PopulateLightToOptBucket(nodeIndex + 1, optBucket);
        PopulateLightToOptBucket(node->childOrLightIndex, optBucket);
    }
    
    // Learn all lights in the light tree if #lights <= NEURAL_OUTPUT_DIM_MAX. Otherwise learns up to NEURAL_HEIGHT in light tree
    std::pair<int, int> NeuralSLCLightSampler::buildOptBucketNodeIndexMappings(int nodeIndex, int optBucket, uint32_t depth, int n_collapsed_outputdims) {                
        const size_t nLights = lightTree.bvhLights.size();
        if (nLights <= NEURAL_OUTPUT_DIM_MAX){
            const int lastOptBucket = buildOptBucketNodeIndexMappings(nodeIndex, optBucket);
            return {lastOptBucket, NEURAL_OUTPUT_DIM_MAX-lastOptBucket};
        }

        return buildOptBucketNodeIndexMappingsHeightCapped(nodeIndex, optBucket, depth, n_collapsed_outputdims);
    }

    template <uint32_t N, typename T>
    PBRT_CPU_GPU inline void softmax(T vals[N]) {
        float max = 0.0f;
        TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                max = fmaxf(max, (float)vals[i]);
            }
        
        float sum_exp_diff = 0.0f;
        TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                sum_exp_diff += expf((float)vals[i] - max);
            }

        // Ensure that the sum of exponentials is not zero.
        // In case this fails, vals[i] is likely to be NaN or Inf.
        DCHECK(sum_exp_diff > 0);

        TCNN_PRAGMA_UNROLL
            for  (uint32_t i = 0; i < N; ++i) {
                vals[i] = (T) (expf((float)vals[i] - max - logf(sum_exp_diff)));
            }
    }

    template <typename T>
    PBRT_CPU_GPU inline void softmax(T* vals, uint32_t N) {
        float max = 0.0f;
        TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                max = fmaxf(max, (float)vals[i]);
            }
        
        float sum_exp_diff = 0.0f;
        TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                sum_exp_diff += expf((float)vals[i] - max);
            }

        // Ensure that the sum of exponentials is not zero.
        // In case this fails, vals[i] is likely to be NaN or Inf.
        DCHECK(sum_exp_diff > 0);

        TCNN_PRAGMA_UNROLL
            for  (uint32_t i = 0; i < N; ++i) {
                vals[i] = (T) (expf((float)vals[i] - max - logf(sum_exp_diff)));
            }
    }

    template <typename T>
    PBRT_CPU_GPU inline void softmaxNoZero(T* vals, uint32_t N) {
        float max = 0.0f;
        TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                max = fmaxf(max, (float)vals[i]);
            }
        
        float sum_exp_diff = 0.0f;
        TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                sum_exp_diff += expf((float)vals[i] - max);
            }

        // Ensure that the sum of exponentials is not zero.
        // In case this fails, vals[i] is likely to be NaN or Inf.
        DCHECK(sum_exp_diff > 0);

        // We add a small epsilon to the end of the softmax to avoid zero probabilities
        // Then, we normalize by the new sum to ensure that the sum of probabilities is 1 (valid PDF)
        constexpr Float EPSILON = 1e-3f;
        const Float sum_softmax_EPSILON_inv = 1.f / (1.f + N * EPSILON);

        TCNN_PRAGMA_UNROLL
            for  (uint32_t i = 0; i < N; ++i) {
                vals[i] = (T) ((expf((float)vals[i] - max - logf(sum_exp_diff)) + EPSILON) * sum_softmax_EPSILON_inv);
            }
    }

    template <typename T>
    PBRT_CPU_GPU void NeuralSLCLightSampler::computeResidualPMF(Point3f p, Normal3f n, T* networkOutputsi) const {
        // 1. Add log (baseline importances) to raw network output
    TCNN_PRAGMA_UNROLL
        for (uint32_t i = 0; i < neuralOutputDim; i++){
            const uint32_t nodeIndex = optBucketToNodeIndex[i];
            const LightBVHNode *node = &lightTree.nodes[nodeIndex];
            const Float baselineImportance = importanceFunction.compute(p, n, lightTree.allLightBounds, node->lightBounds);
            constexpr Float EPSILON = 1e-6f;
            networkOutputsi[i] += (T) logf(
                std::max<Float>(baselineImportance, EPSILON) // Avoid log(0)
            );
        }

        // 2. Compute softmax over combined output
        softmaxNoZero<T>(networkOutputsi, neuralOutputDim);
    }

    void NeuralSLCLightSampler::EvalOrSample(const float* inputBuffer, const int32_t* pixelIndexBuffer, const float* residualInfoBuffer, int size) {
        if (!NeuralSLCLightSampler_network) LOG_FATAL("Network not initialized!");
	    if (size == 0) return;
        constexpr uint16_t residualInfoDim = sizeof(LightSamplerResidualInfo) / sizeof(float);
        assert(residualInfoDim >= 9); // 3 for p, 3 for n, 3 for wo
        int paddedBatchSize = next_multiple(size, 128);
        PBRT_DBG("EvalOrSample:: Preprocessing %d samples (padded %d)", size, paddedBatchSize);

        GPUMatrix<float> networkInputs((float*) inputBuffer, NEURAL_INPUT_DIM, paddedBatchSize);
        GPUMatrix<float> networkOutputs(inferenceOutputPtr, neuralOutputDim, paddedBatchSize);

        {
            std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents("NeuralSLCLightSampler::EvalOrSample Inference");	
            cudaEventRecord(events.first);
            NeuralSLCLightSampler_network->inference(networkInputs, networkOutputs);
            cudaEventRecord(events.second);
        }
        
        GPUParallelFor("NeuralSLCLightSampler::EvalOrSample PostInference", size, [=] PBRT_GPU(int index) {
            if (index >= size) return;
            const int32_t pixelIndex = pixelIndexBuffer[index];
            float* networkOutputsi = inferenceOutputPtr + index * neuralOutputDim;
            const float* residualInfo = residualInfoBuffer + index * residualInfoDim; // 9 floats per sample (3 for p, 3 for n, 3 for wo)

            #if (NEEONLY == 0)
                 #if (RESIDUAL_LEARNING == 1)
                    // Incorporate network residual to importance baseline for the general case
                    computeResidualPMF({residualInfo[0], residualInfo[1], residualInfo[2]}, {residualInfo[3], residualInfo[4], residualInfo[5]}, networkOutputsi);
                #else
                    softmaxNoZero(networkOutputsi, neuralOutputDim);
                #endif
            #endif

            // Save the pixelIndex -> outputIndex mapping
            pixelIndexToOutputIndex[pixelIndex] = index;
        });
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> NeuralSLCLightSampler::Sample(const LightSampleContext &ctx, Float u) const {
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
        
        // Sample light from the cached network output
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;
        const int32_t pixelIndex = ctx.pixelIndex;
        const uint32_t outputIndex = pixelIndexToOutputIndex[pixelIndex];
        float* networkOutputsi = inferenceOutputPtr + outputIndex * neuralOutputDim;

        // When NEEONLY is enabled, we can get away with computing this here for runtime efficiency
        #if (NEEONLY == 1)
            #if (RESIDUAL_LEARNING == 1)
                computeResidualPMF(p, n, networkOutputsi);
            #else
                softmaxNoZero(networkOutputsi, neuralOutputDim);
            #endif
        #endif

        // CDF sampling
        const int optBucket = SampleDiscrete({networkOutputsi, neuralOutputDim}, u, nullptr, &u);

        DCHECK_LT(optBucket, neuralOutputDim);
        int nodeIndex = (int) optBucketToNodeIndex[optBucket];
        Float pmf = (1 - pInfinite) * networkOutputsi[optBucket];

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

    PBRT_CPU_GPU Float NeuralSLCLightSampler::HeuristicPMF(const LightSampleContext &ctx, Light light) const {
        if (!lightToOptBucket.HasKey(light) || !lightTree.lightToBitTrail.HasKey(light)){
            return 1.f / (lightTree.infiniteLights.size() + (lightTree.nodes.empty() ? 0 : 1));
        }

        uint32_t nodeIndex = optBucketToNodeIndex[lightToOptBucket[light]];
        uint32_t bitTrail = lightTree.lightToBitTrail[light] >> NEURAL_HEIGHT; // Skip the first NEURAL_HEIGHT bits
        Float pmf = 1.f;
        Point3f p = ctx.p();
        Normal3f n = ctx.ns;

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
              PBRT_DBG("Importance is 0");
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
    
    PBRT_CPU_GPU
    Float NeuralSLCLightSampler::PMF(const LightSampleContext &ctx, Light light) const {
        // Handle infinite _light_ PMF computation
        if (!lightTree.lightToBitTrail.HasKey(light))
            return 1.f / (lightTree.infiniteLights.size() + (lightTree.nodes.empty() ? 0 : 1));
        
        const int32_t pixelIndex = ctx.pixelIndex;
        const uint32_t outputIndex = pixelIndexToOutputIndex[pixelIndex];
        const float* networkOutputsi = inferenceOutputPtr + outputIndex * neuralOutputDim;

        DCHECK(lightToOptBucket.HasKey(light));
        const int optBucket = lightToOptBucket[light];
        DCHECK_LT(optBucket, neuralOutputDim);

        const Float pmf = (1 - lightTree.pInfinite()) * networkOutputsi[optBucket];
        const Float heuristicPMF = HeuristicPMF(ctx, light);
        // printf("%d Heuristic PMF: %f Network PMF: %f\n", pixelIndex, heuristicPMF, pmf);
        return pmf * heuristicPMF;
    }


    Float NeuralSLCLightSampler::TrainStep(const float* inputs, const RGB* radiances, const int32_t* lightIndices, const float* residualInfoBuffer, const float* radiancePDFs, int size) {
        if (!NeuralSLCLightSampler_network) LOG_FATAL("Network not initialized!");
        if (size == 0) return 0.f;
        
        constexpr uint16_t residualInfoDim = sizeof(LightSamplerResidualInfo) / sizeof(float);
        assert(residualInfoDim >= 9); // 3 for p, 3 for n, 3 for wo
        const int maxTrainBatches = size / TRAIN_BATCH_SIZE + 1;
        const int numTrainBatches = std::min(maxTrainBatches, BATCH_PER_FRAME);
        const int iterOffset = (maxTrainBatches - numTrainBatches) > 0 ? (float) rand() / RAND_MAX * (maxTrainBatches - numTrainBatches) : 0; // TODO: use PBRT's random number generator instead of rand()
        precision_t* networkOutputsPtr = NeuralSLCLightSampler_trainOutputBuffer.data();
        precision_t* dL_doutputPtr = NeuralSLCLightSampler_gradientOutputBuffer.data();
        float* lossPtr = NeuralSLCLightSampler_lossBuffer.data();

        float loss = 0.f;
        for (int iter = iterOffset; iter < numTrainBatches + iterOffset; iter++) {
            int localBatchSize = std::min(size - iter * TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE);
            localBatchSize -= localBatchSize % 128;
            if (localBatchSize < MIN_TRAIN_BATCH_SIZE) break;

            float* inputs_i = (float*)(inputs + iter * TRAIN_BATCH_SIZE * NEURAL_INPUT_DIM);
            RGB* radiances_i = (RGB*)(radiances + iter * TRAIN_BATCH_SIZE);
            float* radiancePDFs_i = (float*)(radiancePDFs + iter * TRAIN_BATCH_SIZE);
            int32_t* lightIndices_i = (int32_t*)(lightIndices + iter * TRAIN_BATCH_SIZE);
            const float* residualInfoBuffer_i = residualInfoBuffer + iter * TRAIN_BATCH_SIZE * residualInfoDim;

            GPUMatrix<float> networkInputs(inputs_i, NEURAL_INPUT_DIM, localBatchSize);
            GPUMatrix<precision_t> networkOutputs(networkOutputsPtr, neuralOutputDimPadded, localBatchSize);
            GPUMatrix<precision_t> dL_doutput(dL_doutputPtr, neuralOutputDimPadded, localBatchSize);
            
            std::unique_ptr<tcnn::Context> ctx;
            {
                std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents("NeuralSLCLightSampler::TrainStep Forward");	
                cudaEventRecord(events.first);
                ctx = NeuralSLCLightSampler_network->forward(networkInputs, &networkOutputs, false, false);
                cudaEventRecord(events.second);
            }
            
            GPUParallelFor("NeuralSLCLightSampler::TrainStep softmax and gradient/loss computation", localBatchSize, [=] PBRT_GPU(int index) {
                if (index >= localBatchSize) return;
                precision_t* networkOutputsi = networkOutputsPtr + index * neuralOutputDimPadded;
                precision_t* gradientDatai = dL_doutputPtr + index * neuralOutputDimPadded;
                float* lossi = lossPtr + index;
                const float* residualInfoi = residualInfoBuffer_i + index * residualInfoDim; // 9 floats per sample (3 for p, 3 for n, 3 for wo)

                #if (RESIDUAL_LEARNING == 1)
                    // Incorporate network residual to importance baseline
                    computeResidualPMF({residualInfoi[0], residualInfoi[1], residualInfoi[2]}, {residualInfoi[3], residualInfoi[4], residualInfoi[5]}, networkOutputsi);
                #else
                    softmaxNoZero(networkOutputsi, neuralOutputDim);
                #endif

                // Resolve PMF
                const Light light = lightTree.bvhLights[lightIndices_i[index]];
                DCHECK(lightToOptBucket.HasKey(light));
                int optBucket = lightToOptBucket[light];
                const Float EPSILON = 1e-3f;
                float guidedPMF = (float) networkOutputsi[optBucket];

                // Loss computation
                float loss_scale = (float)TRAIN_LOSS_SCALE / localBatchSize;
                float Li = radiancePDFs_i[index] > 0.f ? radiances_i[index].Average() / radiancePDFs_i[index] : 0.f;
                DCHECK_GT(Li, 0);
                *lossi = -Li * logf(guidedPMF);

                // Gradient computation
                const Float sum_softmax_EPSILON = (1.f + neuralOutputDim * EPSILON);
                const float prefix = -Li / guidedPMF * loss_scale / sum_softmax_EPSILON;
                const float pmfBucket = (float) networkOutputsi[optBucket] * sum_softmax_EPSILON - EPSILON;

                TCNN_PRAGMA_UNROLL
                for (int i = 0; i < neuralOutputDim; i++) {
                    const float pmfI = (float) networkOutputsi[i] * sum_softmax_EPSILON - EPSILON;
                    const float Jiidx = optBucket == i ? pmfBucket * (1.0f - pmfBucket) : -pmfBucket * pmfI;
                    gradientDatai[i] = (precision_t) (prefix * Jiidx);
                }
            });

            if (!ctx) LOG_FATAL("Context not initialized!");
            {
                std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents("NeuralSLCLightSampler::TrainStep Backward+OptimStep");
                cudaEventRecord(events.first);
                NeuralSLCLightSampler_network->backward(nullptr, *ctx, networkInputs, networkOutputs, dL_doutput, nullptr, false, EGradientMode::Overwrite);
                NeuralSLCLightSampler_trainer->optimizer_step(TRAIN_LOSS_SCALE);
                cudaEventRecord(events.second);
            }
            

            // loss compute, logging and plotting
            {
                std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents("NeuralSLCLightSampler::TrainStep Loss Reduction");
                cudaEventRecord(events.first);
                loss += thrust::reduce(thrust::device, lossPtr, lossPtr + localBatchSize, 0.f, thrust::plus<float>()) / localBatchSize;
                cudaEventRecord(events.second);
            }
            
        }

        return loss / numTrainBatches;
    }

    void NeuralSLCLightSampler::SaveModelWeights(const std::string &filename) const {
        if (!NeuralSLCLightSampler_network) LOG_FATAL("Network not initialized!");
        if (!NeuralSLCLightSampler_trainer) LOG_FATAL("Trainer not initialized!");

        // Based on the comments above, save for our network
        std::ofstream network_weight_file(filename, std::ios::binary);
        if (!network_weight_file.good()){
            LOG_ERROR("Failed to open file for saving network weights");
            return;
        }

        std::vector<float> host_weights_net(NeuralSLCLightSampler_network->n_params());
        CUDA_CHECK(cudaMemcpy(host_weights_net.data(), NeuralSLCLightSampler_trainer->params_full_precision(), NeuralSLCLightSampler_network->n_params() * sizeof(float), cudaMemcpyDeviceToHost));
        network_weight_file.write(reinterpret_cast<const char*>(host_weights_net.data()), NeuralSLCLightSampler_network->n_params() * sizeof(float));
        network_weight_file.close();

        LOG_VERBOSE("Saved network weights to file %s", filename.c_str());
    }


}  // namespace pbrt
