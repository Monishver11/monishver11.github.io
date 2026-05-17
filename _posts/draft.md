GPU Mem Hierarchy and Tiling

These two topics are intertwined with each other, and we need to understand and expand on each to get a good mental model. first lets start with gpu mem hierarchy. why an hierarchy, because, its layered as such based on its functionality, bandwidth, usage and access. its very similar to CPU's hierarchy of HDD, RAM, Cache and registers. Here, it is GMEM(global memory)/HBM(high bandwidth mem)/Off chip memory, L2 cache, SMEM(shared memory)/on-chip memory, L1 cache, registers. this is the ordering in terms of decresing size, increasing bandwidth, increasing latency. a rough estimate of numbers to have in mind for each is GMEM in GBs, L2 cache in MBs, shared memory and L1 cache are shared and can we set as portions by the users in code and its of combioned size of KBs, and registers are per SM basis and of 64K or 65536 to be exact. 

Highly recommend Aleksa gordic nvidia gpus anatomy for more details if you wish to read before proceeding.

there is a clear tradeoff in terms of designing kernels to manage this hardware characteristics. this is one of the reasons i love this domain, you can clearly see to what one has to offer or degrade. an example of this is occupancy. we need to maximize the number of warps per SM, but to do so, we need to make sure that there are enough resources in the SM, and that in term depends on how you framed the algorithm and how you're using the memory and registers. The beautiful balance of this is what it takes to achieve high utilization. sorry for going off-topic, but can't help myself rather than appreciate.

we've summarized the mem hierarchy, lets do a similar one for compute. the kernels we wrote are launched as grids, which has multiple blocks, and which in turn is composed on mulitple warps, and each warp is of 32 threads. The number of threads per warp is fixed by hardware, the notion of warps are not directly supported as a software unit for tuning. the blocks and grids are given as software accessible and tunable things. there can be mutiple blocks within a SM, and it depends on the block resource availability of the SM. the threads within the block, all can share the SMEM of that SM to communicate, share data etc. this is all about the strucutre of parallel compute and not the compute itself. the computes of the SM are Arithmetic logic unit(ALU), Tensor cores, Special function units(XU), and CUDA cores. each are made for specific usecases to make it highly and properly utilized, like cuda cores for normal math, tensor cores for matmul/fused mutiply add(FMA), exponentials in XU. also, since these compute units are separate hardware, we can use them in parallel. the reason i mentioned this is because this is also one of the optimiation we'll do to make the attention faster, that is to use the tensor cores for matmul, while simultaneouly doing softmax in cuda cores for a differnt block. just mentioning to give you an idea. 

one more important thing to understand and very foundation of GPU programming is Single instruction multiple data/threads(SIMD/SIMT). so, here we can issue a single instruction and do the same instruction on multiple data. this is the absolute core of GPU's and why deep learning operations got very fast in such acclerators. see, the matrix mulitplications are a suitable computation for GPUs because, you can do the compuation of a each result element in parallel. that is for the first element of the result matrix, its the dot product of first row of A with first column of B. while this is computing, you can compute the second element of the result matrix, which takes the same row of A, and takes the second column of B. so here, the instruction is to do the dot product, but the data to take and do the computation is differnt. if we assume a basic model of one thread to compute one element of the result matrix, then the instructions are same and identical, but each thread use differnt data.  

<excalidraw img of matrix A@B=C>

now, with both the memory and compute hierarchies in place, we need to combine them in a way to make the utilization better. and that is done via tiling.

Tiling

before, actually seeing what is tiling, we need to see why do we need tiling, or why does tiling helps us combine the mem and compute topologies. there are two main reasons, one is we can't have the whole data in memory to do the compute, so we break it and do it in chuncks or blocks or parts. second is that, we need to avoid the memory transfers from gmem to smem. as this is major time spent and we need to reduce this as much as possible. this is because the bandwidth is not that great to move the data faster, so the compute has to wait until the data is present to actually take and do the compute. so, we need to avoid the memory stalls as much as possible. and since the smem is smaller is size, we can only store a small portion at a time. this again in turn leads to breaking the original data into small pieces. and that is why, we break the original data into small parts. but, why its called tiles? the reason is that, we chunck it in 2D, along both row and columns, so the word tiles. this is efficient, as we can do the computation in parallel and the ampere and hopper generations of GPUs, have tensor cores, which can directly take two matrix of MK and KN size to compute MN shape, so tiles have become the norm. now a days, we've many libraries like triton, cutile, whose primitives are tiles rahter than simple elements.

<excalidraw img of matrix A@B=C tiled>

now, with tiling, we need to store partial results somewhere to accumulate it later to get the exact final results. and that somewhere is shared memory and registers. a simple example of understanding this is the matmul. you'll have three matrices, A, B and result C. in tiling, you can load a small tile(or first tile of the first row) of A and store it in share memory from gmem. now, for each tile in the of the B's row, you need to multiple it with this first tile of the first row of A. now, since its stored in smem, you don't need to get it some smem each tile, you can use it directly and compute the partial results of the first row of the C matrix. then once this first tile of A is all computed, we can move to the next tile of A. <need to clarify this example more clearly or a better one to explain tiling.>

code of three loop, or one element per thread.

tiled version code.

see simon's gemm worklog for better understanding the tiling concept.

---

Note: in this section, i'm going that deep into each of the concepts, but will do in the upcoming sections when and where we need it. like shared memory and bank conflicts will be talked when we dive into tiling, avoiding bank conflicts with swizzling, tma loads etc. and when we're designing kernels, we'll talk and see some ablations of how changing one affects the other, plus see whats happening with NCU profle results, or atleast that's the plan.

Note: you'll come across a lot of acronyms. i'll expand the term in the first usage. so, if you missed or not sure, just do a ctrl+f.


