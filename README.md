# Analyzing results from timing bit matrix  transpositions on the GPU

Raph Levien's planned performance improvements to [`piet-metal`]() will rely on efficiently transposing 32x32 bit matrices. Therefore, it will be of interest to examine the performance of programs designed to execute this task. In particular, in this post, we will investigate **time performance**. 

The bit matrix transposition problem can be addressed, broadly speaking, with two types of programs:
1. Classical **threadgroup** ("warp", "workgroup") based programs, relying on shared memory available to threads within the group.
2. SIMD ("subgroup", "wavefront") based programs relying on rapid sharing of registers associated with the lanes making up the SIMD group.

In general, we will see that the SIMD class of programs will outperform threadgroup-based programs. However, we will also see that the timing results reveal much about the underlying GPU hardware.

### Introduction to SIMD

In 1966, Michael J. Flynn proposed a classification of computer architectures which is eponymously referred to as "Flynn's taxonomy". There are four categories in this classification:
* Single Instruction stream, Single Data stream (SISD)
* Single Instruction stream, Multiple Data streams (SIMD)
* Multiple Instruction streams, Single Data stream (MISD)
* Multiple Instruction streams, Multiple Data streams(MIMD)

A computing device is SISD if at each [clock cycle](https://en.wikipedia.org/wiki/Clock_signal), it fetches an instruction from a single instruction stream, and data (for that instruction) from a single data stream. Such a device cannot work in parallel (at least in a useful way). 

A computing device is SIMD, if at each clock cycle, it fetches an instruction from a single stream, but fetches data from multiple streams, so that the fetched instruction can be executed simultaneously upon the data acquired from each stream. One way to create a SIMD device might be to group some processing units together so that they are all connected to the same instruction stream, but have their own data streams. Such a device can be meaningfully used to perform parallel computations. It is worth re-iterating that in a SIMD device, every processing unit must be executing the same instruction every cycle; only the data provided to the instruction may differ amongst the processing units.

MISD, and MIMD should be relatively straightforward to understand from the definition of SISD and SIMD. [Wikipedia provides a more complete explanation](https://en.wikipedia.org/wiki/Flynn%27s_taxonomy) of the categories in Flynn's taxonomy.

Flynn's taxonomy will not be considered useful in this article beyond providing an understanding of what is meant by SIMD. 

### Introduction to Threadgroups and Subgroups

Modern GPU hardware is not only complex, but many of the relevant details are also closed-source. Thus, to understand understand program performance, programmers must rely on abstractions of this hardware. These abstractions in turn are strongly influenced by APIs (e.g. OpenGL or Vulkan) specifying the interaction between hardware and application.

The most hardware abstraction, is unsurprisingly then, the one which APIs present front and center: that of threadgroups. In this model, a GPU's processors ("threads") can be thought of being divided into groups ("threadgroups"). The size of these threadgroups can be set by the application. Special programs called "shaders" (or, in the context of general-purpose GPU computing, "kernels") are executed by each thread within the threadgroup. The instructions specified by the kernel are the same across the threads, but a thread can access different memory for these instructions to work upon. Furthermore, threadgroups can include branching control flow statements (e.g. if statements), which can also cause particular threads within a threadgroup to be executing different instructions than another thread, at any given time.

![threadgroup-model](https://i.imgur.com/Z8TohZ9.png)

While this model captures the gist of "what is a GPU?", it is incomplete. To build a more refined mental model, one must start by considering how data flows within a GPU, i.e. understanding hierarchy within which GPU memory is organized.

![memory-hierarchy](https://i.imgur.com/kxpt4Fq.png)


Each thread has associated with it registers which store data using a [configuration of logic gates](https://en.wikipedia.org/wiki/Flip-flop_(electronics)) called a flip-flop. This storage is expensive in the energy required to maintain it, but powerful in terms of performance. However, its capacity is also quite small (**Question 0:** Why? Probably because we want to keep each thread close by on the chip, and the more of these we have, the hotter everything would get?). Direct access to these registers was unavailable to programmers until quite recently, when APIs began to expose them via subgroup operations.  

A subgroup is a group of threads which are tightly associated due to hardware architecture, in that they can access each others' registers efficiently. A change of terminology is in order: we will call threads within a subgroup "lanes", taking this terminology from the SIMD-world. The term "lane" is not overloaded by this decision, as a subgroup *is* a SIMD unit. Crucially, all the lanes within a subgroup must be executing the same instruction. If due to branching statements two lanes within a subgroup "diverge" (so that one is performing a different instruction than the other), then one of the lanes is temporarily made inactive! In other words, if only two lanes were available, the device would temporarily become SISD.


One level up in the hierarchy, we have threadgroups, which can be profitably understood as a group of subgroups. Perhaps a change in terminology is in order for threadgroups, but let's restrict ourselves to one change in term per post; world domination one step at a time :). Subgroups within a threadgroup have access to "threadgroup shared memory", allowing for data to be shared between subgroups. While threadgroup shared memory has greater storage capacity, it also has lower bandwidth, and greater latency, making it less performant. Note that threadgroup shared memory storage is also register (flip-flop) based.

On the top, we have global memory, which unlike the lower levels in the hierarchy, uses capacitors (DRAM) as storage elements. Global memory is accessible to all threadgroups on a device, but its performance characters mean that two threadgroups cannot communicate with each other on useful timescales. If the threads within a threadgroup can pass send messages to each other with a day's delay, it may take up to 10 days for messages to pass between threadgroups. You wouldn't want to have your threads sitting around, twiddling their thumbs, in that time, and thus design your programs to minimize dependency on inter-threadgroup communication. 

### Transposing bit matrices

Before we jump into using our new-found understanding of threadgroups and subgroups to start kernels which transpose bit matrices, let us discuss how bit matrices will be reprsented in memory, and the general concept of the transposition algorithm for bit matrices.

**Memory representation.** Recall that we are interested in transposing 32x32 bit matrices. Note that an unqualified unsigned integer is 32 bits wide. Therefore, a 32x32 bit matrix can be represented as an array of 32 unsigned integers. 

![](https://i.imgur.com/ZPoTfJz.png)


**General algorithm.** The algorithm we use to transpose the bit matrix is recursive. Label the 32x32 matrix `M_32`, and consider its division into 4 16x16 sub-matrices ("blocks"):

![](https://i.imgur.com/cZ3AEBN.png)

16x16 blocks of the same colour are swapped with each other, giving us a new matrix `M_16`. Divide `M_16` up into 8x8 blocks:

![](https://i.imgur.com/iNFDP6z.png)

Once again, blocks of the same colour are swapped with each other, giving us a new matrix, `M_8`. We repeat this procedure 3 more times: 4x4 blocks (produces `M_4` from `M_8`), 2x2 blocks (produces `M_2` from `M_4`), and 1x1 blocks (produces `M_1` from `M_2`).

For some hand-built intuition, start with a 4x4 matrix, labelled `M_4`. By hand, write down its transposition, and label that `T`. Produce `M_2` from `M_4`, and then `M_1` from `M_2`. Confirm that `M_1` and `T` are the same.  

### Parellelizing the recursive transposition algorithm. 

Note that when you are generating `M_16` from `M_32`, the `i`th row of `M_16` will depend upon the `(i + 16) % 32` row of `M_32`. When `0 <= i < 16`, `M_16[i] = M_32[i + 16] << 16 | M_32[i + 16] >> 16` (i.e. the first 16 bits of `M_32[i + 16]`) become the last 16 bits of `M_16[i]`, and the last 16 bits of `M_32[i + 16]` become the first 16 bits of `M_16[i]`. 

So, if you have 32 processors, the `i`th processor can read `M_32[(i + 16)%32]`, and "shuffle round" its bits, storing the result in a new array called `M_16`. The `i`th processor does its work independently of the other processors, so it is embarrassingly easy to parallelize this problem. Note that each processor will execute the exact same sequence of operations, but will have different input data (SIMD).

For the other cases, the concept is the same, but one must be more careful with the indexing, and the bit shifting. All of this is implemented most plainly in the threadgroup based kernel's "shuffle round" function [`shuffle_round`](https://github.com/bzm3r/transpose-timing-tests/blob/74559f2ecc76d1ee65880eb5b77585059e0e1090/kernels/templates/transpose-threadgroup-template.comp#L20). Given some thread `i`, figuring out which row's data it should read is done on [this line](https://github.com/bzm3r/transpose-timing-tests/blob/74559f2ecc76d1ee65880eb5b77585059e0e1090/kernels/templates/transpose-threadgroup-template.comp#L60).


**Aside:** for those unfamiliar with bitwise operations, please don't be afraid to understand the code. It will take some time, but the concepts are straightforward. Keep [Wolfram Alpha](https://www.wolframalpha.com/input/?i=BitAnd%5Bffff00_16%2C+ffff_16%5D) or [Rust Playground](https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=3db32d1ea9ff7b7e8a49a0b0b7292cf1) handy (I can't recommend Python, because it does not have unsigned integers, which causes all sorts of problems). Begin by carefully going through these points:
* (important) the [most significant bit](https://en.wikipedia.org/wiki/Bit_numbering#Most_significant_bit), the [least significant bit](https://en.wikipedia.org/wiki/Bit_numbering#Least_significant_bit), and [endianness](https://en.wikipedia.org/wiki/Endianness)
* (very important) bitwise operators [and, or, not and xor](https://en.wikipedia.org/wiki/Bitwise_operation#Bitwise_operators)
* (somewhat important) [hexadecimal numbers](https://en.wikipedia.org/wiki/Hexadecimal). Note that `0xf` is the  form of the integer with binary form `1111` (4 `1`s). In 32-bit form, `0xf` will be `00000000000000000000000000001111` (16 `0`s followed by 4 `1s`) The integer with 32-bit binary form of 16 `0s` followed by  get 16 `1`s, will be `0xffff`. What is `0x9`? What is `0xa`? 
* The modulo of `x` by number `y`, where `y` is a power of 2, is `x & (y - 1)` (`&` is bitwise and). Why is that? 
* Suppose that `y` is the `m`th power of 2 (`m >= 0`). Why is `x/y = x >> m` (`>>` is logical right shift)?
* What is the bitwise xor of various numbers between 0 and 31 (inclusive), by various other numbers between 0 and 31 inclusive? What patterns can you see? (The Rust playground link may be helpful.)

The SIMD kernel using the shuffle_xor operation also follows the same concept, but the `^` (bitwise xor) operation happens within the `subgroupShuffleXor` instruction. 

The Shuffle+Threadgroup hybrid kernel generates `M_16` and `M_8` using threadgroups, and `M_4` to `M_1` using subgroups. It is particularly well suited to Intel devices, where the subgroup size is variable between 32 and 8, but is guaranteed to be at least as large as 8.

The SIMD ballot kernel uses a much simpler algorithm: for the `i`th row, it asks for the `i`th bit of all the processors (from `0` to `31`) inclusive (i.e. it effectively computes the `i`th column using the `subgroupBallot` operation). It stores the result as the row of the transposed matrix.

### Why bother with SIMD (subgroups)?

Apart from the fact that the parallelization of the bit matrix transposition algorithm presented had a distinctly "Single Instruction Multiple Data" flavour to it, why else would we bother using subgroups?

The answer is: better performance. In the threadgroup kernel, a bit matrix is loaded into threadgroup shared memory, and this is then accessed by all threads in the group as they do their calculation. Because of the `shuffle_round` function, we are able to avoid any if statements in transpose kernel, so we don't suffer from divergence due to branching (a major point necessary for performance whether or not you are using subgroups), but the GPU does have to organize reads of the threads from shared memory.

On the other hand, in the shuffle kernel, a bit matrix (or two, if you are using an AMD machine with a subgroup size of 64) is loaded into the *fast* subroup-level registers. Lanes within the subgroup need not look up the data they need from threadgroup shared memory, but instead can take advantage of the close connection they have to the other registers in their subgroup. 

