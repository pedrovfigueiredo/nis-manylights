// lightsamplers_constants.h
// Copyright (c) Pedro Figueiredo

#ifndef LIGHTSAMPLERS_CONSTANTS_H
#define LIGHTSAMPLERS_CONSTANTS_H

template <uint32_t base, uint32_t exponent> struct powi_compiletime {
    static const uint32_t value = base * powi_compiletime<base, exponent - 1>::value;
};

template <uint32_t base> struct powi_compiletime<base, 0> {
    static const uint32_t value = 1;
};

template <uint32_t dim, uint32_t multiple> struct padded_dim_compiletime {
    static const uint32_t value = (dim + multiple - 1) / multiple * multiple;
};


#define LEARN_PRODUCT_SAMPLING 1 // Learn product of Li*BSDF*cos term
#define NEEONLY 1 // Only direct illumination, no BSDF sampling
#define RESIDUAL_LEARNING 1 // Use residual learning on top of baseline method
#define VPL_USED false // Set to true for rendering scenes with VPLs (e.g. San Miguel), otherwise false
#define NEURAL_GRID_DISCRETIZATION 0 // Use grid discretization for neural network inputs

#if (NEEONLY == 1)
    #define MAX_TRAIN_DEPTH 2u // We set this to 2 because we ignore specular bounces (e.g. mirror)
#else
    #define MAX_TRAIN_DEPTH 5u
#endif

#define MAX_CUT_SIZE 64

constexpr uint32_t NEURAL_HEIGHT = 6u; // k height in the paper
constexpr uint32_t NEURAL_OUTPUT_DIM_MAX = powi_compiletime<2, NEURAL_HEIGHT>::value;
constexpr uint32_t NEURAL_OUTPUT_DIM_MAX_PADDED = padded_dim_compiletime<NEURAL_OUTPUT_DIM_MAX, 16>::value;
#if (LEARN_PRODUCT_SAMPLING == 1)
    #if (NEURAL_GRID_DISCRETIZATION == 1)
        constexpr uint32_t NEURAL_INPUT_DIM = 6u; // Adds wo to input. Does not include normal input
    #else
        constexpr uint32_t NEURAL_INPUT_DIM = 8u; // Adds wo to input
    #endif
#else
    #if (NEURAL_GRID_DISCRETIZATION == 1)
        constexpr uint32_t NEURAL_INPUT_DIM = 3u; // Does not include normal input
    #else
        constexpr uint32_t NEURAL_INPUT_DIM = 5u;
    #endif
#endif
constexpr unsigned int MAX_RESOLUTION = 1920 * 1080; // max size of the rendering frame 
constexpr int MAX_INFERENCE_NUM = MAX_RESOLUTION;
constexpr int TRAIN_BATCH_SIZE = 65'536 * 8;
constexpr int MIN_TRAIN_BATCH_SIZE = 10'000;
constexpr int BATCH_PER_FRAME = 4;
constexpr int TRAIN_LOSS_SCALE = 128;
constexpr float EPSILON = 1e-5f;

// NEURAL_GRID_DISCRETIZATION grid parameters
constexpr int DIRECTIONAL_DISCRETIZATION_RESOLUTION = 8;
constexpr int POSITIONAL_DISCRETIZATION_RESOLUTION = 32;

#endif // LIGHTSAMPLERS_CONSTANTS_H