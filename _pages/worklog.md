<!-- ---
layout: page
permalink: /worklog/
title: Worklog
description: 
nav: True
nav_order: 6
--- -->


T - X Days

Coffee + Article Read + Standing

Coding - ML Systems & GPU

Coding - Problem Solving (Try for 30 mins minimum) + Learning the approach

Routine & Discipline

Document the work each day, what you learnt, express in writing.

Specialized in-depth knowledge about ML Systems and GPU Kernel writing

Ruthless work-ethic

12/09/25 (T - 154)

[Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)

Always First Principles thinking and approach;

GPU kernels for MOE, refined idea using first principles, and then proceeded to claude code. Now, will read and understand the code better and will see what to do next.

Slides for BDML presentation - Done, will prepare for the presentation tmr morning and will do 2 dry run of the content i want to present.

[Intro to Triton: A Parallel Programming Compiler and Language, esp for AI acceleration (updated)](https://www.youtube.com/watch?v=s1ILGG0TyYM)

12/10/2025 (T - 153)

[GPU MODE Lecture 14: Practitioners Guide to Triton](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/)

Have some doubts in the above blog's Triton coding parts, will check them using LLM and understand after BDML presentation.

[MegaBlocks: Eﬃcient SparseTraining with Mixture-of-Experts](https://www.youtube.com/watch?v=JndztlScZLA)

Solve Interesting Technical Work, again and always from first principles;

I need to code here, from scratch and on my own. I feel this to a be big lacking and decrease in confidence level. Need to start doing even small things, but do it myself. 

[Why PyTorch is an amazing place to work... and Why I'm Joining Thinking Machines](https://www.thonking.ai/p/why-pytorch-is-an-amazing-place-to)

Done two dry runs, got a descent flow of what and how i want to present. All the best Monish!

Presentation done and did well, kudos. Next focus on the custom kernel for MOE(with triton) and then report for BDML project.

Ping Sohan to ask, If he can be my mentor and accountability partner, with whom i can share my bi-weekly progress update, short and long term goals and discuss, someone with whom i can connect and talk and ask for advice - Done

[GPU MODE Lecture 14: Practitioners Guide to Triton](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/) + Gemini

so, as per my understanding of how triton is similar and different from CUDA is:
- in triton the basic unit is blocks, which consists of processing a vector of 128, rather than as in cuda, where the basic unit is threads, which processes one scalar value.
- next, when a kernel is launched, its split into program ids, and each program id needs to handle a block of data, in this case of 128 values and takes care of its processing. we use pid and block_size, to get the offsets, which we ultimately use to get the memory addresses of the values the block needs to process. We also have a mask, which is very similar to cuda if condition, where we ensure that the idx < n. next, we know that for that block the starting memory address is x_ptr and we use the offsets, to get the x_ptrs for the whole block of data. since the x_ptrs are adjacent to each other, its already memory coalesced and efficient enough, rather as in cuda, we need to take care of this manually. next, for both load and store, we use to mask to avoid any invalid memory access and does a no-op for those memory address respectively.
- @triton.jit decorator tells python to don't run this on the CPU, instead compile this function into a GPU kernel.
- - in triton, since we define in terms of a block, which handles a certain portion of data, we don't define # threads singularly, instead we define as warps to process each of the block of data.
- now, this python function will be handled by the JIT(Triton JIT Compiler), which does the loop unrolling, allocates registers, and generates ptx that is matched to the optimized cuda kernel.

When you run your Python script, here is the exact sequence:
- Python Source: Your @triton.jit function.
- Triton IR: The Triton compiler converts your Python AST into an intermediate representation (IR) that understands "blocks" and "layouts."
- Optimizer: It performs block-level optimizations (e.g., "I can load these two blocks together").
- LLVM IR: It converts Triton IR into LLVM (the same backend C++ uses). This is where Loop Unrolling and Register Allocation happen.
- PTX/Assembly: LLVM generates the final Nvidia PTX code (or AMD GCN).

- PyTorch JIT (torch.compile): This is a high-level graph optimizer. It looks at your PyTorch code and says, "I see a ReLU followed by a Softmax. I should fuse them." It might call Triton to generate code, but it doesn't compile the kernel itself.

- riton JIT (@triton.jit): This is the low-level compiler. It takes your Python function, parses the Abstract Syntax Tree (AST), and converts it directly into machine code.

Next, MatMul in Triton and (Swizzling) Concept.

Coding the Deepseek-MOE custom kernel with claude, trying to make it as a package and work within modded-nanogpt.

If it works, and i'll make it work. Then, i'll understand this very clearly, to a precision, where i can say at this line this happens and we do this for this very specific reason, and only then i'll share it someone. I don't care about anything, but to understand what i'm telling i did, must be of know doubt and ifs. 

Discussed on RBDA, was little hesitant to present as i don't have all the facts. But, i read an article that leaders lead in the realm of uncertainty. So, i'll gather as much information i need by tmr afternoon, prep myself well enough to have a clear narrative and practice and present it well. 

12/11/2025 (T - 152)

[How To Build and Use a Multi GPU System for Deep Learning](https://timdettmers.com/2014/09/21/how-to-build-and-use-a-multi-gpu-system-for-deep-learning/)

- The main bottleneck is network bandwidth, i.e, how much data is transferred from computer to computer per second.
- The network bandwidth of network cards (affordable cards are at about 4GB/s) does not come even close to the speed of PCIe 3.0 bandwidth (15.75 GB/s). So GPU-to-GPU communication within a computer will be fast, but it will be slow between computers.
- On top of that most network card only work with memory that is registered with the CPU and so the GPU to GPU transfer between two nodes would be like this: GPU 1 to CPU 1 to Network Card 1 to Network Card 2 to CPU 2 to GPU 2. What this means is, if one chooses a slow network card then there might be no speedups over a single computer. 
- PCIe: Inside one machine, connects - CPU ↔ GPU, GPU ↔ GPU (via PCIe switches), CPU ↔ SSD, etc.
- Network cards(Ethernet / InfiniBand NICs): Between different machines (nodes), connects - Server 1 ↔ Server 2 ↔ Server 3 ...
- GPUDrirect RDMA, a network card driver that can make sense of GPU memory addresses and thus can transfer data directly from GPU to GPU between computers.
- As deep learning programs use a single thread(from CPU) for a GPU most of the time, a CPU with as many cores as GPUs you have is often sufficient. 
- There are basically two options how to do multi-gpu programming. You do it in CUDA and have a single thread and manage the GPUs directly by setting the current device and by declaring and assigning a dedicated memory-stream to each GPU, or the other options is to use CUDA-aware MPI where a single thread is spawned for each GPU and all communication and synchronization is handled by the MPI. The second option is much more efficient and clean. Also, MPI is the standard in HPC and its standardized library means that you can be sure that a MPI method really does what it is supposed to do.
- MPI(Message passing interface): A standard API for communication between multiple processes running on multiple machines.

Made the RBDA slides to cook a story, next is to serve to the audience, and i'll practice that next, after a short break, feeling bit tired and head weighted.

Practiced a few times, feels ok. Will do a few more time to feel confident. 

Added GPU Lecture notes to my personal webpage, and tweaked and refined the RBDA 1 & 2 Lecture notes.

Practiced 3 more times and got the flow. The 5 minutes constraint is slightly hard, and i feel like rushing, but that's fine, give your best. All the best Monish.

Completed the presentation, had a bit of slack at the start, but eventually picked up the pace, and did well overall expect for the first bit. Good command over voice. Definitely can improve, and need to take up more presentation opportunity to master this skill, as this communication matters a lot than it appears to be. 

Put everything you learn here in your own words.

12/12/2025 (T - 151)

Good morning Monish!

[Einsum is All you Need - Einstein Summation in Deep Learning](https://rockt.ai/2018/04/30/einsum)

einsum notes from above;
- einstein summation, its a notation implemented in numpy, and also now in pytorch and TF. 
- It is an elegant way to present and express dot products, outer products, transposes and matrix-vector or matrix-matrix multiplications. 
- It is a DSL, that is domain specific language, but why?
- DSL like einsum can be sometimes be compiled to high-performing code.


unsqueeze operation;
- It's an pytorch operation.
- It adds a new dimension of size 1 at a specified axis.
- If x.shape == (3, 4), and you unsqueeze at axis = 1, given by x = x.unsqueeze(1), results in shape: (3, 1, 4).
- Numpy equivalent is np.expand_dims(x, axis)
- Used for: adding a batch dimension, preparing for broadcasting or matching the dimensions of a model.
- The opposite of this is squeeze, which exists as both numpy and pytorch operation, it removes dimension of size 1 across all dimensions by default, but optinally takes axis too.


A more detailed and clear view of einsum with numpy;
[Einstein Summation in Numpy](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/)

Need clarity in derivatives with respect to certain forms in backprop calculations, noted down a example of MLP with einsum;

[MLP Standard Derivatives Derivation](/blog/2025/mlp-derivatives/)

[Simple MLP – Forward and Backward Pass (with Einsum)](/blog/2025/mlp-fw-bwd/)

Above are drafts, may refine it further, but not now;


[A basic introduction to NumPy's einsum](https://ajcr.net/Basic-guide-to-einsum/)
- einsum does not promote data types when summing, what it means is that if you're using a more limited datatype, you might get unexpected results.
- einsum is not always the fastest option in NumPy. Functions such as dot and inner often link to lightening-quick BLAS routines which can outperform einsum and certainly shouldn't be forgotten about. 


Finish Einsum today; - Done

Draft a mail for Prof Zaharan for GPU project, by EOD, scheduled mail for 12/15/25, 8AM - Done

Need to finish the backprop of MLP with einsum after dinner; 
Check on the page's content for MLP derivation, or write a new one for this compact MLP forward and backward pass; - First learn, and then finish this. - Next; - Done;
Along with this, start transformer;

12/13/2025 (T - 150)

[NanoGPT-inference LLM inference from scratch, by Pieter Delobelle](https://pieter.ai/blog/2025/nanogpt-inference/)

[NanoGPT-inference - Baseline, by Pieter Delobelle](https://pieter.ai/blog/2025/nanogpt-inference---baseline/)
- Time to first token(ms): the time it takes for the model to start responding. This is usually the first forward pass, once we get to KV caching we also call this first pass the 'prefill'. Lower is better.
- Inter-token latency(ms): the time it takes for the model to respond to the next token. Lower is better again.
- Throughput per request(tokens/s): The inverse of the inter-token latency. Higher is better.
- Total throughput(tokens/s): The total number of tokens we can process per second. This is the product of the throughput per request and the batch size.
- Memory usage(GB): The GPU memory usage of our inference engine. A model needs to fit into the GPU memory, as well as activations, some intermediate tensors, and later on some cached values. The higher the memory usage, the bigger our VRAM(/HBM) needs to be. 

[What is the roofline model? - GPU Glossary](https://modal.com/gpu-glossary/perf/roofline-model)
- The roofline model is a simplified, visual model of performance used to quickly determine whether a program is bound by memory bandwidth or arithmetic bandwidth.
- In this, we have two hardware-derived "roofs" put a "ceiling" on the possible performance:
  - the "compute roof" - the peak rate of the target hardware (CUDA cores or Tensor cores), aka the arithmetic bandwidth.
  - the "memory roof" - the peak memory throughput of the target hardware, aka the memory bandwidth.
- x-axis: arithmetic intensity (in operations per byte) and  y-axis: performance (in operations per second).
- The compute roof is a horizontal line with the height equal to the arithmetic bandwidth. 
- The memory roof is a slanted line with slope equal to the memory bandwidth. Slope is "rise over run", and so the line has units of bytes per second (operations per second divided by operations per byte).
- A specific kernel's x-coordinate tells you instantly whether its is fundamentally compute-bound (points beneath the flat roof) or memory-bound (points beneath the slanted roof). 
- Kernels are rarely up against either roof due to the effects of overhead.
- The point on boundary, i.e. where the diagonal and horizontal roof meet, is called the "ridge point". 
- Its x-coordinate is the minimum arithmetic intensity required to be able to escape the memory bottleneck. 
- The compute and memory roofs need only be derived once per subsystems (though importantly they vary depending on the subsystem, not just the system; Tensor cores have more FLOPS than CUDA cores).

Next:
Annotated Transformer; - Glossed over, got very high-level gist. 
Simple MLP with Einsum - Colab; - Done;

Need to learn Transformers from first principles

[Learning Triton One Kernel at a Time: Matrix Multiplication](https://medium.com/data-science-collective/learning-triton-one-kernel-at-a-time-matrix-multiplication-44851b4146dd)
- The L2 cache is coherent across SMs, meaning that updates from one SM are visible to others, enabling synchronization between thread blocks. 
- Parallel Tiled GEMM: Here we extend our tiled GEMM to parallelize the computation of each pairs of tiles over several thread blocks. (See the animation, it clearly shows the difference from Tiled GEMM)
- Naive GEMM -> Tiled GEMM -> Parallel Tiled GEMM
- Memory coalescing is achieved when subsequent threads in a warp access subsequent memory addresses. (Nice analogy: Imagine a librarian needing to fetch books for a client, if all books are side-by-side on a shelf, they can grab them all at once. In contrast, if all books are lying on different shelves, they’ll have to grab them one by one, which takes significantly longer.)
- Frameworks like pytorch adapt a row-major layout, meaning that elements of a matrix are per-row contiguous in memory. For instance, elements of our (2,2) matrix would be stored as follows: [(0,0), (0,1), (1,0), (1,1)], notice that elements of the same row are contiguous (touching) while elements of the same column have a stride of 1(separated by one element).
- This implies that we can load rows using coalesced loads, but columns do not satisfy this condition. However, we need to access columns of Y to compute dot products. In order to maximize performance, a good practice is to transpose Y so that we iterate on its rows rather than its columns.
- However, transposing Y isn't enough to modify its layout in memory. As mentioned previously, PyTorch stores machine in a flat array. Each matrix dimension is associated with a stride attribute, denoting the jump necessary to go from one element to the next one along this dimension. for instance, a (10,10) matrix would have strides=(10, 1). Indeed, starting from element [0,0], element [1,0] is 10 memory slots(i.e. one row) away, whereas element [0,1] is adjacent.
- When transposing a tensor, PyTorch doesn't modify the layout in memory, but simply recomputes the strides. In order to make the transpose effective from a memory standpoint we need to call Y.T.contiguous().
- These are the required steps the load columns of Y efficiently, however we'll need to transpose the loaded blocks within the kernel to perform the dot product properly: `z_block = tl.dot(X_block, Y_block.T)`.
- See the image for a more clear visual notion.
- Naive GEMM -> Tiled GEMM -> Parallel Tiled GEMM -> Memory Coalescing.
- Triton Implementation(with memory coalescing): The Nsight profile shows that the transpose inside the kernel causes shared-memory bank conflicts, which serialize accesses and stall warps. As a result, the warp scheduler has no eligible warp to run 87.6% of the time, meaning the SMs are mostly idle waiting for these conflicts to resolve. The low DRAM throughput (8.2%) indicates the kernel is not limited by global memory bandwidth, and the modest compute throughput (21.1%) shows the ALUs are also underutilized. Since neither memory nor compute resources are saturated and the dominant issue is waiting on stalls, the kernel is latency bound. In contrast, the earlier kernel is compute bound because its compute throughput is high relative to DRAM usage, so performance there is limited by computation rather than stalls.
- **Q: If threads in a warp should access contiguous memory for performance, but shared memory accesses must avoid bank conflicts to be parallel, isn’t this contradictory?**
- A: No, because these rules apply to two different memory systems. Contiguous access refers to global memory, where consecutive addresses allow requests from a warp to be coalesced into fewer DRAM transactions, maximizing bandwidth. Bank conflicts apply to shared memory, which is divided into banks; here, threads must access different banks to avoid serialization. In practice, an optimal kernel loads data from global memory using coalesced (contiguous) accesses, then rearranges it in shared memory into a layout that avoids bank conflicts for computation. Both conditions are required: coalescing ensures efficient data movement into the SM, while conflict-free shared memory ensures fast, parallel use of that data once it is on-chip. Reference(check image): [4/7: The GPU – a performance-centric overview](https://performanceguidelines.blogspot.com/2013/08/the-gpu-performance-centric-overview.html?utm_source=chatgpt.com)
  

Do, Triton implementation of this next. 
Also, go through this for a Triton implementation revision - [Learning Triton One Kernel At a Time: Vector Addition](https://towardsdatascience.com/learning-triton-one-kernel-at-a-time-vector-addition/?source=post_page-----44851b4146dd---------------------------------------)

12/14/2025 (T - 149)

[GPU Compiler Architecture: CUDA, Triton & DeepSeek](https://qsysarch.com/posts/cuda-compiler-architecture/)
- Got a brief mental idea of what's supposedly happening, but not having much clarity on each of the specific parts. But, this is exactly the kind of things you need to know for first principles, and to gain expertise. I'll revisit this blog, and read Aleksagordic's blog to understand this better.

Will code the triton matmul tmr;

12/15/2025 (T - 148)

Coded Triton MatMul, both tiled GEMM and Memory Coalesced. Found a small bug in the author's code, and posted it as a comment in Medium's blogpost page. Didn't profile with NSight, but that's ok, i've seen the numbers of flops and throughput.

[https://www.aleksagordic.com/blog/matmul](https://www.aleksagordic.com/blog/matmul)
Reading this, even though i might not be able to finish in one sitting, But, i'll do in parts. 
Also add to feeder.co - [Aleksa Gordić](https://www.aleksagordic.com/); - Having a limit on the feeds, so wasn't able to add. Need to look for another alternative.

Went to college, proctored for the Calculus 1 Final exam.

Adding RBDA notes for concepts prior to mid-term; Done.

Make an entry for TA experience, after dinner; -> Tmr;

Finished Hive slides and notes. 

12/16/2025 (T - 147)

"Explain from first principles so that I can do this without you next time" - Priyansh Agarwal 
 
Reading RBDA - HBase(Done, added notes), Zookeeper(Done, added notes), starting with Kakfa;

Came across this and want to later see it(saving for reference): [Timilearning - Tagged “learning-diary”](https://timilearning.com/tags/learning-diary/)

12/17/2025 (T - 146)

Read to need a 1 page from Aleska Gordic Blog;
Review BDML Report;

Starting with Kafka. (Done, edits pending for publishing)

Starting with Flink. Read, added some notes, but only for places i had question. Need to input all the content with this and make notes, and that's after exam;

Start 12-data format. Done;

12/18/2025 (T - 145) 

[Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)
- Read from CUDA programming model to L1 model. 
- Have to make notes
- Add nuanced questions(wrt specific details) as tags, to revisit and clarify after reading the whole blog

Went through the pre-midterm cheat sheet and now going to take mock exam; Done.

Finished pre-midterm 2 slides, took some rest, now starting again the rest of the pre-midterm slides;

Done with pre-midterm, made notes till kafka, and now will be starting exam in 5 mins.

Finished exam, was slightly harder than the mock, but overall fine. Properly understanding stuff, the precise first principles interals helped a lot. Overall, good, learnt well.

Work on RBDA project report. Done.

12/19/2025 (T - 144)

Read BDML report and marked for changes needed yesterday itself. Will be connecting in zoom and finishing that up. Done, my part and my teammate will submit the doc. - Done;

Email BDML prof for ML Systems work - Done;

12/20/2025 (T - 143)

[Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul); Read till PTX code;