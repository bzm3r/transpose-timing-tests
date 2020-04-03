Solving problems on a GPU requires breaking it up into "primitives". Some primitives can naturally be executed in parallel, but others require 
coordination between the GPU's threads. An example of a primitive which requires coordination between threads for execution is the 
transposition of a square bitmap matrix. In piet-gpu, the bitmap represents a boolean matrix storing whether object `i` interacts with 
tile `j`. This post will examine the performance of this transposition task in detail. 

As we've hinted, the transposition of a square matrix bitmap requires coordination between threads. Depending on how the problem is
approached, either: 

1. (the threadgroup approach) the read/write access to a centrally stored (in threadgroup shared memory) bitmap must be coordinated between 
threads, or 
2. (the subgroup approach) if the bitmap is stored in a distributed manner amongst the registers of the threads, then data must be shuffled 
around between these registers to perform transposition. 

The threadgroup approach provides a programmer with a flexible interface through which stored data in threadgroup shared memory can be 
accessed and manipulated, and this interface is widely supported by hardware, graphics APIs, and shader languages. On the other hand, in the 
subgroup approach, a programmer has a limited interface available for interacting with stored data in the registers of threads within a 
subgroup, and this is interface is not available in older hardware, and not uniform amongst modern graphics APIs and shader languages. For 
example, HLSL and DX12 do not make the shuffle intrinsic available. 

The subgroup approach is appealing because of amazing performance when reading/writing between data stored in subgroup's thread registers. 
Threadgroup shared memory, in comparison, has poorer performance:

![memory-hierarchy](./diagrams/memory-hierarchy.png)

Is the performance gain from the subgroup approach worth it, given its downsides? Let's find out!

## Performance of threadgroup approach vs. subgroup approach

To compare performance, we calculate from our timing results the number of bitmap transpositions performed per second. Let's plot this rate with respect to varying threadgroup size:

![](./plots/dedicated_simd_tg_comparison.png)

What jumps out is that while the the subgroup kernel (`Shuffle32`) outperforms the threadgroup-based kernel (`Threadgroup1d32`) on both the AMD device and Nvidia devices, the effect is particularly pronounced on Nvidia devices. On the AMD device, the performance gain is marginal, suggesting that threadgroup shared memory is remarkably fast on AMD devices. Furthermore, effective utilization of the Nvidia RTX 2060 (a high end Nvidia GPU) for the bit matrix transposition task with respect to the Nvidia GTX 1060 relies on using SIMD techniques.



