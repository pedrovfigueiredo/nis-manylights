#ifndef PBRT_WAVEFRONT_TRAIN_H
#define PBRT_WAVEFRONT_TRAIN_H

#include <pbrt/pbrt.h>

#include <pbrt/options.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#endif  // PBRT_BUILD_GPU_RENDERER
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>

#include <atomic>
#include <utility>

#ifdef __CUDACC__

#ifdef PBRT_IS_WINDOWS
#if (__CUDA_ARCH__ < 700)
#define PBRT_USE_LEGACY_CUDA_ATOMICS
#endif
#else
#if (__CUDA_ARCH__ < 600)
#define PBRT_USE_LEGACY_CUDA_ATOMICS
#endif
#endif  // PBRT_IS_WINDOWS

#ifndef PBRT_USE_LEGACY_CUDA_ATOMICS
#include <cuda/atomic>
#endif

#endif  // __CUDACC__

namespace pbrt {

struct LightSamplerTrainInput {
    Vector3f pOffset; // Offset from the minimum corner of the bounding box

// Only includes normal if discretization is not used
#if (NEURAL_GRID_DISCRETIZATION == 0)
    Vector2f n; // Normal is encoded in spherical coordinates
#endif

// Only includes direction if product sampling is used
#if (LEARN_PRODUCT_SAMPLING == 1)
    Vector3f wo;
#endif
};

struct LightSamplerTrainOutput {
    RGB L;
    uint32_t lightBitTrail;
};

// Used for computing the baseline importances to which network residuals are applied
struct LightSamplerResidualInfo {
    Point3f p;
    Normal3f n;
    Vector3f wo;
};


template <typename Tin, typename Tout>
class InOutBuffer {
public:
    InOutBuffer() = default;

    InOutBuffer(int n, Allocator alloc) :
        mMaxSize(n) {
        mInputs = alloc.allocate_object<Tin>(n);
        mOutputs = alloc.allocate_object<Tout>(n);
    }

    PBRT_CPU_GPU int Size() const {
        int size;
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        size = mSize;
#else
        size = mSize.load(cuda::std::memory_order_relaxed);
#endif
#else
        size = mSize.load(std::memory_order_relaxed);
#endif
        return std::min<int>(size, mMaxSize);
    }

    PBRT_CPU_GPU int Capacity() const { return mMaxSize; }

    PBRT_CPU_GPU void Reset() { 
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        mSize = 0;
#else
        mSize.store(0, cuda::std::memory_order_relaxed);
#endif
#else
        mSize.store(0, std::memory_order_relaxed);
#endif	
    }

    PBRT_CPU_GPU int Push(const Tin& input, const Tout& output) {
        int index = AllocateEntry();
        DCHECK_LT(index, mMaxSize);
        mInputs[index] = input;
        mOutputs[index] = output;
        return index;
    }

    PBRT_CPU_GPU Tin* Inputs() const { return mInputs; }

    PBRT_CPU_GPU Tout* Outputs() const { return mOutputs; }

protected:
    PBRT_CPU_GPU
    int AllocateEntry() {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        return atomicAdd(&mSize, 1) % mMaxSize;
#else
        return mSize.fetch_add(1, cuda::std::memory_order_relaxed) % mMaxSize;
#endif
#else
        return mSize.fetch_add(1, std::memory_order_relaxed) % mMaxSize;
#endif
    }

private:
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
    int mSize = 0;
#else
    cuda::atomic<int, cuda::thread_scope_device> mSize{0};
#endif
#else
    std::atomic<int> mSize{0};
#endif  // PBRT_IS_GPU_CODE

    Tin* mInputs;
    Tout* mOutputs;
    int mMaxSize{ 0 };
};

template <typename Tin, typename Tout1, typename Tout2>
class In2xOutBuffer {
public:
    In2xOutBuffer() = default;

    In2xOutBuffer(int n, Allocator alloc) :
        mMaxSize(n) {
        mInputs = alloc.allocate_object<Tin>(n);
        mOutputs1 = alloc.allocate_object<Tout1>(n);
        mOutputs2 = alloc.allocate_object<Tout2>(n);
    }

    PBRT_CPU_GPU int Size() const {
        int size;
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        size = mSize;
#else
        size = mSize.load(cuda::std::memory_order_relaxed);
#endif
#else
        size = mSize.load(std::memory_order_relaxed);
#endif
        return std::min<int>(size, mMaxSize);
    }

    PBRT_CPU_GPU void Reset() { 
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        mSize = 0;
#else
        mSize.store(0, cuda::std::memory_order_relaxed);
#endif
#else
        mSize.store(0, std::memory_order_relaxed);
#endif	
    }

    PBRT_CPU_GPU int Push(const Tin& input, 
        const Tout1& output1, const Tout2& output2) {
        int index = AllocateEntry();
        mInputs[index] = input;
        mOutputs1[index] = output1;
        mOutputs2[index] = output2;
        return index;
    }

    PBRT_CPU_GPU Tin* Inputs() const { return mInputs; }

    PBRT_CPU_GPU Tout1* Outputs1() const { return mOutputs1; }

    PBRT_CPU_GPU Tout2* Outputs2() const { return mOutputs2; }

protected:
    PBRT_CPU_GPU
    int AllocateEntry() {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        return atomicAdd(&mSize, 1) % mMaxSize;
#else
        return mSize.fetch_add(1, cuda::std::memory_order_relaxed) % mMaxSize;
#endif
#else
        return mSize.fetch_add(1, std::memory_order_relaxed) % mMaxSize;
#endif
    }

private:
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
    int mSize = 0;
#else
    cuda::atomic<int, cuda::thread_scope_device> mSize{0};
#endif
#else
    std::atomic<int> mSize{0};
#endif  // PBRT_IS_GPU_CODE

    Tin* mInputs;
    Tout1* mOutputs1;
    Tout2* mOutputs2;
    int mMaxSize{ 0 };
};

template <typename Tin, typename Tout1, typename Tout2, typename Tout3>
class In3xOutBuffer {
public:
    In3xOutBuffer() = default;

    In3xOutBuffer(int n, Allocator alloc) :
        mMaxSize(n) {
        mInputs = alloc.allocate_object<Tin>(n);
        mOutputs1 = alloc.allocate_object<Tout1>(n);
        mOutputs2 = alloc.allocate_object<Tout2>(n);
        mOutputs3 = alloc.allocate_object<Tout3>(n);
    }

    PBRT_CPU_GPU int Size() const {
        int size;
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        size = mSize;
#else
        size = mSize.load(cuda::std::memory_order_relaxed);
#endif
#else
        size = mSize.load(std::memory_order_relaxed);
#endif
        return std::min<int>(size, mMaxSize);
    }

    PBRT_CPU_GPU void Reset() { 
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        mSize = 0;
#else
        mSize.store(0, cuda::std::memory_order_relaxed);
#endif
#else
        mSize.store(0, std::memory_order_relaxed);
#endif	
    }

    template <typename TinPush, typename Tout1Push, typename Tout2Push, typename Tout3Push>
    PBRT_CPU_GPU int Push(const TinPush& input, 
        const Tout1Push& output1, const Tout2Push& output2, const Tout3Push& output3) {
        int index = AllocateEntry();
        DCHECK_LT(index, mMaxSize);
        mInputs[index] = input;
        mOutputs1[index] = output1;
        mOutputs2[index] = output2;
        mOutputs3[index] = output3;
        return index;
    }

    PBRT_CPU_GPU Tin* Inputs() const { return mInputs; }

    PBRT_CPU_GPU Tout1* Outputs1() const { return mOutputs1; }

    PBRT_CPU_GPU Tout2* Outputs2() const { return mOutputs2; }

    PBRT_CPU_GPU Tout3* Outputs3() const { return mOutputs3; }

protected:
    PBRT_CPU_GPU
    int AllocateEntry() {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        return atomicAdd(&mSize, 1) % mMaxSize;
#else
        return mSize.fetch_add(1, cuda::std::memory_order_relaxed) % mMaxSize;
#endif
#else
        return mSize.fetch_add(1, std::memory_order_relaxed) % mMaxSize;
#endif
    }

private:
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
    int mSize = 0;
#else
    cuda::atomic<int, cuda::thread_scope_device> mSize{0};
#endif
#else
    std::atomic<int> mSize{0};
#endif  // PBRT_IS_GPU_CODE

    Tin* mInputs;
    Tout1* mOutputs1;
    Tout2* mOutputs2;
    Tout3* mOutputs3;
    int mMaxSize{ 0 };
};



template <typename Tin, typename... Touts>
class InNxOutBuffer {
public:
    InNxOutBuffer() = default;

    // Constructor that allocates memory for input and all outputs
    InNxOutBuffer(int n, Allocator alloc) : mMaxSize(n) {
        mInputs = alloc.allocate_object<Tin>(n);
        allocateOutputs(n, alloc);
    }

    PBRT_CPU_GPU int Size() const {
        int size;
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        size = mSize;
#else
        size = mSize.load(cuda::std::memory_order_relaxed);
#endif
#else
        size = mSize.load(std::memory_order_relaxed);
#endif
        return std::min<int>(size, mMaxSize);
    }

    PBRT_CPU_GPU void Reset() { 
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        mSize = 0;
#else
        mSize.store(0, cuda::std::memory_order_relaxed);
#endif
#else
        mSize.store(0, std::memory_order_relaxed);
#endif	
    }

    // Push a single input + N outputs. We enforce that the number of Touts
    // matches the number of push arguments for outputs.
    PBRT_CPU_GPU int Push(const Tin &input, const Touts&... outs) {
        static_assert(sizeof...(outs) == sizeof...(Touts),
                      "Number of output arguments to Push must match Touts...");
        int index = AllocateEntry();
        // Optionally add a debug check: assert(index < mMaxSize); 
        mInputs[index] = input;
        setOutputs(index, outs...);
        return index;
    }

    // Accessors for the underlying arrays:
    PBRT_CPU_GPU Tin* Inputs() const { return mInputs; }

    // Get the I-th output pointer (returns a pointer to the array for the I-th type in Touts...).
    // Usage example: `auto ptr = buf.Outputs<0>();` for the first output type, etc.
    template <std::size_t I>
    PBRT_CPU_GPU auto Outputs() const {
        return std::get<I>(mOutputs);
    }

private:
    // --------------------------------------------------------------------------
    // Allocate memory for each Touts* in the tuple.
    // --------------------------------------------------------------------------
    template <std::size_t I = 0>
    void allocateOutputs(int n, Allocator &alloc) {
        if constexpr (I < sizeof...(Touts)) {
            using CurrentType = std::tuple_element_t<I, std::tuple<Touts...>>;
            std::get<I>(mOutputs) = alloc.template allocate_object<CurrentType>(n);
            allocateOutputs<I + 1>(n, alloc);
        }
    }

    // --------------------------------------------------------------------------
    // Helper to set each Touts... pointer at [index] using parameter pack.
    // --------------------------------------------------------------------------
    template <typename... Ts>
    PBRT_CPU_GPU void setOutputs(int index, const Ts&... outs) {
        setOutputsImpl(index, std::index_sequence_for<Touts...>{}, outs...);
    }

    template <typename... Ts, std::size_t... Is>
    PBRT_CPU_GPU void setOutputsImpl(int index, std::index_sequence<Is...>, const Ts&... outs) {
        // Expand in a fold expression:
        ((std::get<Is>(mOutputs)[index] = std::get<Is>(std::tie(outs...))), ...);
    }

    PBRT_CPU_GPU
    int AllocateEntry() {
#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
        return atomicAdd(&mSize, 1) % mMaxSize;
#else
        return mSize.fetch_add(1, cuda::std::memory_order_relaxed) % mMaxSize;
#endif
#else
        return mSize.fetch_add(1, std::memory_order_relaxed) % mMaxSize;
#endif
    }

#ifdef PBRT_IS_GPU_CODE
#ifdef PBRT_USE_LEGACY_CUDA_ATOMICS
    int mSize = 0;
#else
    cuda::atomic<int, cuda::thread_scope_device> mSize{0};
#endif
#else
    std::atomic<int> mSize{0};
#endif  // PBRT_IS_GPU_CODE

    Tin* mInputs = nullptr;
    // Tuple of pointers for each output type:
    std::tuple<Touts*...> mOutputs{};
    int mMaxSize = 0;
};


}; // namespace pbrt

#endif  // PBRT_WAVEFRONT_TRAIN_H

