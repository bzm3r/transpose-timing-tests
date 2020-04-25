Executing a task on a GPU often requires breaking it up into "primitives". Some primitives can naturally be executed in parallel, but others require coordination between the GPU's threads. An example of such a primitive is the transposition of a square bitmap matrix, , an experimental renderer based on  the bitmap represents a boolean matrix storing whether object `i` interacts with tile `j`. This post will examine the performance of this transposition task in detail. 

To transpose the bitmap matrix:

1. (the threadgroup approach) threads must coordinate read/write access to the bitmap, which is stored in a central location (threadgroup shared memory).
2. (the subgroup approach) threads must coordinate data swaps amongst themselves, as the bitmap is stored in a distributed manner amongst their registers.

The threadgroup approach provides a programmer with a flexible interface through which stored data in threadgroup shared memory can be accessed and manipulated. This interface is widely supported by hardware, graphics APIs, and shader languages. 

On the other hand, the subgroup approach offers better performance, The subgroup approach is appealing because of significantly better performance due to low latency of read/write operations on the threads' registers, in comparison to read/write operations on threadgroup shared memory:

![memory-hierarchy](./diagrams/memory-hierarchy.png)

However, the subgroup approach struggles with portability: it is not supported or only partially supported on older hardware and APIs. Even modern shader languages do not uniformly support subgroup operations; for example HLSL with SM 6.0 does not provide the subgroup shuffle intrinsic. 

Here's some relevant resources for those who'd like to learn more about subgroups:
* [the Vulkan subgroup tutorial](https://www.khronos.org/blog/vulkan-subgroup-tutorial)
* [Vulkan Subgroup Explained, by Daniel Koch](./ref-docs/06-subgroups.pdf) ([direct link](https://www.khronos.org/assets/uploads/developers/library/2018-vulkan-devday/06-subgroups.pdf)), also provides a table comparing subgroup interface in GLSL vs. HLSL
* [AMD GCN Assembly: Cross-Lane Operations](https://gpuopen.com/amd-gcn-assembly-cross-lane-operations/)
* [Using CUDA Warp-Level Primitives](https://devblogs.nvidia.com/using-cuda-warp-level-primitives/)
* [GPU resources collection](https://raphlinus.github.io/gpu/2020/02/12/gpu-resources.html)

Without further ado, let's find out if the performance gain from the subgroup approach worth it, given its downsides.

## Performance of threadgroup approach vs. subgroup approach

We wrote kernels using threadgroup and subgroup approaches to solve the bitmap transposition problem. In the [threadgroup kernel](https://github.com/bzm3r/transpose-timing-tests/blob/master/kernels/templates/transpose-Threadgroup1d32-template.comp) (referred to as `Threadgroup1d32`), each threadgroup transposes some number, 1 in the least case, of 32x32 bitmaps, depending on the size of the threadgroup specified by the programmer. In the [subgroup kernel](https://github.com/bzm3r/transpose-timing-tests/blob/master/kernels/templates/transpose-Shuffle32-template.comp) (referred to as `Shuffle32`), each subgroup transposes a 32x32 bitmap. While the number of threadgroups can be specified by the programmer, the number of subgroups in a threadgroup depends upon the subgroup size, which is hardware specific. As a general rule of thumb, Nvidia devices have subgroups with 32 threads (1 bitmap matrix), while AMD devices have subgroups with 64 threads (2 bitmap matrices). 

To compare performance, we calculate from our timing results the number of bitmap transpositions performed per second. Let's plot this rate with respect to varying threadgroup size. This chart shows results from `Threadgroup1d32` and `Shuffle32` kernels, on 3 different GPUs.

![](./plots/dedicated_simd_tg_comparison.png)

We see something interesting right away: while the subgroup kernel outperforms the threadgroup-based kernel on both the AMD device and Nvidia devices, the effect is particularly pronounced on Nvidia devices. On the AMD device, the performance gain is marginal, suggesting that threadgroup shared memory is remarkably fast on AMD devices.

**Mystery:** why is AMD RX 570's threadgroup shared memory competitive in performance with subgroup registers?

We can also plot transposition rate versus varying number of bitmaps (payload) uploaded for transposition. Changing payload size varies the number of threads dispatched for the compute task. So, it provides the following information:
* (at low dispatch size) the relative performance of a single thread on a particular device with respect to that on another device;
* (at increasing dispatch size) the maximum number of threads the device can muster, after which it is must queue up tasks to wait for free threadgroups. 

![](./plots/amd_vs_nvd_loading_comparison.png)

In the above graph, note carefully the number of dispatched threads at which performance begins to flatten out, which is the maximum number of threads available on the device. So, while Nvidia GTX 1060's threads individually outperform those of the AMD RX 570's (better performance at low number of dispatched threads), it has less available than the AMD device. Thus, the AMD RX 570 achieves higher transposition rates at large payload sizes.

Comparing Nvidia devices alone, individual thread performance between the Nvidia GTX 1060 (mid-tier GPU) and the Nvidia RTX 2060 (high end GPU) is comparable. However, the RTX 2060 has many more threads available, since at large payload sizes, its performance dominates.

## Intel devices, hybrid shuffles, and 8x8 bitmap transpositions

Recall that a programmer can easily control the size of a threadgroup, but is at the mercy of the hardware for subgroup sizes. On Intel devices, this problem is especially trouble, since they have a physical subgroup size of 8, but a logical subgroup size of up to 32. The choice of logical subgroup size is left up to the shader compiler, unless the programmer uses the `VK_EXT_subgroup_size_control` extension for GLSL. Sadly, the [`VK_EXT_subgroup_size_control`](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_subgroup_size_control.html) extension is not widely supported by Intel drivers. 

(Side note: [vulkan.gpuinfo.org](https://vulkan.gpuinfo.org/) is a good source for checking out which Vulkan extensions a particular device driver supports, and how well supported, in general, an extension is. To check out the latter click on `Extensions` in the top bar, and type in the name of the extension you are interested in (e.g. `VK_EXT_subgroup_size_control`). Note that `VK_EXT_subgroup_size_control` isn't commonly supported, and the situation is particularly woolly on mobile.)

Therefore, we tried a "hybrid approach" specifically designed for Intel devices, which mixes the threadgroup and subgroup approaches. Our implementation of matrix transposition executes `lg(n)` (n being the number of bits in the bitmap) "block swap operations". First, 16x16 blocks of the 32x32 matrix are swapped with their opposing neighbours, then 8x8 blocks are swapped, and so on. To do a N/2 x N/2 block swap, you need at least N threads in a subgroup. 

Since the minimum subgroup size on Intel devices is 8, we can always do 4x4 and lower block swaps via subgroup operations. This version of the hybrid threadgroup-subgroup kernel is called `HybridShuffle32`. It is possible to be smarter, if we can determine which N was chosen by the shader compiler. N should be available to kernels via the `gl_SubgroupSize` constant. This adaptive version of the hybrid threadgroup-subgroup kernel is called `HybridShuffleAdaptive32`. 

Doing this, we discovered a bug on Intel devices! The `gl_SubgroupSize` variable reports only the maximum logical size of a subgroup (32), instead of the size selected by the shader compiler. To get around this issue, we calculated subgroup size as [`gl_WorkgroupSize/gl_NumSubgroups`](https://github.com/bzm3r/transpose-timing-tests/blob/a78b46523cecd5483ea154ccc34080f581dda413/kernels/templates/transpose-HybridShuffleAdaptive32-template.comp#L51). 

Before we present the graph, remember that since threadgroup shared memory access is expensive compared to subgroup register access, we expect the hybrid approach to be better than a pure threadgroup approach.

![](./plots/integrated_hybrid_tg_comparison.png)

Surprise! The hybrid kernel underperformed compared to the pure threadgroup kernel on Intel HD 630. The adaptive hybrid kernel has worse performance than the simpler kernel. The Intel HD 520 results are frankly weird, since the hybrid kernel does perform better at lower threadgroup sizes, but at larger threadgroup sizes, the situation becomes similar to that of the Intel HD 630. 

To check out the situation on dedicated devices, we tested `HybridShuffle32`:

![](./plots/dedicated_hybrid_tg_comparison.png)

As expected, hybrid shuffle had performance middling between threadgroup and shuffle kernels. However, given that we are working with a conceptual model which assumes threadgroup shared memory operations to be very expensive in comparison to subgroup operations, we would have expected the hybrid kernel to do much better than the threadgroup kernel, since it does 3 out of 5 shuffle operations using the subgroup approach. 

**Mystery:** why does the hybrid shuffle perform worse than pure threadgroup kernel on Intel devices? How can we explain the poor performance of the threadgroup approach at low threadgroup sizes on Intel HD 520?

There is a way to shuffle bitmap matrices using only subgroup operations on Intel: transpose smaller matrices! Since the physical subgroup size on Intel devices is 8, let's transpose 8x8 bit matrices using only subgroup operations. Note that 16 8x8 bit matrices fit inside one
 32x32 bit matrix, so we need not fiddle with our data representation or reported performance metric (transpose/sec) too much. Just consider an unqualified transpose in the 8x8 setting to be be the operation of transposing 16 8x8 bit matrices at once, by one threadgroup, instead of 32x32 bit matrix. The `Shuffle8` kernel has impressive performance on Intel devices:

![](./plots/intel_8vs32_comparison.png)


Note that this is not because the `Shuffle8` kernel is simply doing less work, since the `Threadgroup1d8` kernel (which does the same work as `Shuffle8`, except using threadgroups) is not significantly more performant than the `Threadgroup1d32` kernel. Furthermore, `Shuffle8` kernels are also not significantly more performant than `Shuffle32` kernels on AMD and Nvidia devices:

![](./plots/shuffle_8vs32_comparison.png)

Wow! Performance of pure-subgroup 16x8x8 bit matrix transposition on Intel is at the same order of magnitude as pure-subgroup 16x8x8 bit matrix transposition on Nvidia or AMD devices!

Now, let's look at the performance of the Intel devices with respect to changing payload size. As we might expect, Intel devices are able to muster fewer lanes than the dedicated GPUs, as their performance begins saturate out earlier than the dedicated devices.  

![](./plots/intel_loading_comparison.png)

This plot also reveals that Intel devices have less available parallelism for threadgroup shared memory than for subgroup operations. This effect is expected, as the hardware shares a relatively limited number of shared memory subsystems per "slice" of execution units ([see Wikichip page on Gen9 microarchitecture](https://en.wikichip.org/wiki/intel/microarchitectures/gen9#Gen9)). A plot showing only 8x8 transposition results cleanly shows the parallelism difference between the two approaches on Intel devices.

![](./plots/tg8_shuffle8_loading_comparison.png)

## A ballot based subgroup approach

Recall that subgroup operations are not supported uniformly by modern shader languages. In particular, our matrix transposition code relies heavily on the subgroup shuffle intrinsic, and this is not available in HLSL. See the [test on Tim Jones' shader-playground](http://shader-playground.timjones.io/63db661366c97e9d1e0b5e05fa5d89c2), and [relevant DXC issue](https://github.com/microsoft/DirectXShaderCompiler/issues/2692).

Since HLSL does support the subgroup ballot intrinsic, we explored the performance of [a kernel](https://github.com/bzm3r/transpose-timing-tests/blob/master/kernels/templates/transpose-Ballot32-template.comp) based on the ballot intrinsic compared to those using the shuffle intrinsic. The ballot approach is only relevant to 32x32 bitmap transpositions, due to how we have structured our data, so we are going to look at its performance on the discrete (AMD/Nvidia) GPUs:

![](./plots/dedicated_simd_tg_ballot_comparison.png)

`Ballot32` kernel performance is poor. The loss in performance is particularly pronounced on the Nvidia devices. Part of the poor performance can be explained by the fact that  the ballot-based kernel requires on the order of n (n being the number of bits in the matrix) instructions to execute a transpose, while the shuffle-based kernel requires only on the order of lg(n) instructions. Another issue is divergence: the ballot kernel makes heavy use of branching. Since GPUs are fundamentally SIMD machines, threads which diverge from others (i.e. want to execute different instructions) due to branching are temporarily inactivated, until the other threads complete execution of their instruction. Thus, divergence should be avoided as much as possible, as it disrupts parallelism. 

## TL;DR

GPUs are, at their core, [SIMD](https://en.wikipedia.org/wiki/SIMD) devices. Traditionally, graphics APIs and shader languages do not make this apparent to the programmer, and instead encourage what we call the "threadgroup approach" but in the past few years, modern graphics APIs and shader languages have been exposing this underlying truth via subgroup (Vulkan)/wavefront (DX12) operations.
 
Making a decision between using the threadgroup approach and the subgroup approach is not straightforward. While the subgroup approach promises faster computation times, the threadgroup approach is easy to use, and reliably portable. This post is meant to help guide a decision between using subgroups vs. threadgroups.

Our key findings are:

1. subgroup approach’s performance gains are device dependent: marginal on our AMD device, significant on our Nvidia devices, and dramatic on our Intel devices;  
2. it can be difficult to write subgroup kernels for Intel, since it is not yet easy to force a particular subgroup size, and a hybrid subgroup+threadgroup approach has surprisingly poor performance;
3. if you're writing kernels using HLSL, then you may be missing the subgroup intrinsics necessary for a performant implementation of your kernel. 

So, if performance matters, you’re okay with supporting a narrow set of hardware, and are willing to use GLSL + Vulkan, then the subgroup approach is the winner. Finally, some of our observations are mysterious to us. These are marked with `Mystery:` (CTRL+F for it). If you think you can shed light on these mysteries, we'd love to hear!

