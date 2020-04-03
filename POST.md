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



