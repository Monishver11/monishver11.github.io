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

12/21/2025 (T - 142)

[Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul); 

Pulse NYC Hackathon Build;
[VisaPulse: Practice your visa interview before it decides your future.](https://devpost.com/software/visapulse)

12/22/2025 (T - 141)

[Shape Suffixes — Good Coding Style](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd)

[Interview Prep: Classic System Design](https://twopug.com/interview-prep-classic-system-design/)

[Interview Prep: Keep At It](https://twopug.com/interview-prep-keep-at-it/#fnref-1)

Working on adding the RBDA notes, GPU project and BDML project to webpage.

12/23/2025 (T - 140)

[Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul); 

12/24/2025 (T - 139)

[Performance Hints](https://abseil.io/fast/hints.html)

[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)

12/25/2025 (T - 138)

[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)

12/26/2025 (T - 137)

Write updates mail to Sohan; Done.

12/27/2025 (T - 136)

Talked to get a GTC pass for work in ML systems, will get it anyhow;

12/28/2025 (T - 135)

12/29/2025 (T - 134)

Leisure in holidays;

12/30/2025 (T - 133)

12/31/2025 (T - 132)

01/01/2026 (T - 131)

GPU Kernel Opt., ML Systems or ML Infra - Courant 4th sem, co-op with great learning in a great startup.

01/02/2026 (T - 130)

Building a ML Job tracker for strategic applying, preparing and breaking into my dream job;

01/03/2026 (T - 129)

Was working a bit on ML-Job Aggregator, and have a base MVP working;

01/04/2026 (T - 128)

Made notes and added changes to update resume, linkedin. Made ML Sys resume, looks good. Noted other changes to make as well.

ML-Agg-Job is setup with Github Actions, and the jobs are added in G-Sheets. Will see whether its running for every 6 hrs, and make some small fixes(if i find any). 

01/05/2026 (T - 127)

Was working a bit more on Job aggregator tool.

01/06/2026 (T - 126)

Want to finish the ML Job Agg App by today and start applying.
Also want to see and experiment with CP-31 sheet.

01/07/2026 (T - 125)

[DATA MOVEMENT IS ALL YOU NEED: A CASE STUDY ON OPTIMIZING TRANSFORMERS](https://arxiv.org/pdf/2007.00072)

01/08/2026 (T - 124)

01/09/2026 (T - 123)

01/10/2026 (T - 122)

01/11/2026 (T - 121)

Trying to get an opensource issue on GPU kernel optmization/ML systems/ML infra. Was going through many issues, got some unassigned and unworked after good amount of time, commented for those and got one to work on. Will start after it's been assigned to me.

[deduplicate is_attention_module between compressed-tensors and llm-compressor #2079](https://github.com/vllm-project/llm-compressor/issues/2079#issuecomment-3734272853) - waiting for assignment; starting the work anyway

[[Performance] Remove instances of torch.nonzero() where appropriate #9889](https://github.com/sgl-project/sglang/issues/9889) - waiting for confirmation and then ask for assignment; Its already done it seems and PR is pending.

**NVIDIA**

Started with github setup for opensource project.

01/12/2026 (T - 120)

Completed the change to the llm-compressor issue; Raised PR and waiting for feedback. Wasn't able to add the 'ready' label, so tagged the author of the issue to take a look. 

[deduplicate is_attention_module between compressed-tensors and llm-compressor #2079](https://github.com/vllm-project/llm-compressor/issues/2079#issuecomment-3734272853)

Got some feedback and made fixes accordingly, waiting for review and hope its merges.

Got it merged, Great Monish.

[[torch.compile][Performance]: Unwrap custom ops and improve fusion (Inductor and custom) #24629](https://github.com/vllm-project/vllm/issues/24629) - Asked to work on this issue next, will start once the moderator comments or assigns it to me;

[[Performance]: Custom fused kernel tracking #25179](https://github.com/vllm-project/vllm/issues/25179) - Sub-issue of the above to work after what's mentioned in the comments i made on the above.

[Model Performance Bash! #23963](https://github.com/vllm-project/vllm/issues/23963) - Main parent issue

Other issues to work on:
[[Feature]: Add SM120 (RTX 6000/5000 Blackwell) support for native NVFP4 MoE kernels #31085](https://github.com/vllm-project/vllm/issues/31085)
[[cuBLAS] setting cublas/cublaslt backend seems to be a no-op for torch.matmul in eager. #172231](https://github.com/pytorch/pytorch/issues/172231) - Some asked to work on, will take up if they've abondoned.

Got assigned and now working on this issue now(on main vLLM repo, and on kernels) - [[Feature]: Support norm+quant & silu+quant fusion for block (group) quantization #27847](https://github.com/vllm-project/vllm/issues/27847); Such a good work i got to work on. Must finish this and get this PR merged.

(Ask HN: How to understand the large codebase of an open-source project?)[https://news.ycombinator.com/item?id=16299125]

Understanding the codebase for this dev, the flow, structure and nuances. Also, taking reference of the rms_norm_block_quant PR that was merged last month.

Made the initial changes, wasn't able to install, build and test due to unavailability of CUDA. Also, the mac is slow, will restart now. Will commit the changes tmr, and proceed with the setup and testing for these changes; Made the initial commit, with reference of this to the actual git issue to link.

01/13/2026 (T - 119)

Gave short AI interview in Calyptus platform.

Learnt and understood what the silu+block_quant kernel works. The present kernel is the base version, and if it works and gives a performance boost, i can tune it more with other optimizations and thinking.

Checked that i still have access to cuda5 cluster, so'll test the changes there by setting up; The first build is taking a very long time and still running;

[gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources)

01/14/2026 (T - 118)

Got this issue to work on from NVIDIA/cuda-python. Working on this along with the vllm's silu+block_quant feature.

[[FEA]: Add public class methods for retrieving the legacy default stream as well as the per-thread default stream #1445](https://github.com/NVIDIA/cuda-python/issues/1445#issuecomment-3747428451) - Made the changes, tested it and raised the PR. Waiting for review and feedback. Merged this PR, but the author mentioned some missed idea, after its merged and told he'll raise a PR to fix it. Learn what's the mistake and raise a PR for that again(Do, own and finish something end to end).

[[BUG]: Legacy/per-thread default streams should be singletons #1494](https://github.com/NVIDIA/cuda-python/issues/1494) - The maintainer raised this critical bug(on the changes i've made) and added as a bug, to fix it before the next core deployment. I've understood the bug and asked for fixing it myself, by mentioning that in the comment he had made. Will see; 

Fixed the above bug with this [Fix/default stream singletons #1496](https://github.com/NVIDIA/cuda-python/pull/1496); Asked for review from the maintainer. I've tested it again, and the logic looks fine to me now. Let's see. 

The silu_mul+block_quant kernel is working now. Making a small bug fix, and then will thoroughly understand this first. Once done, will check on the vectorized changes to make it more performant. And then start with the more standard unit test, performance and benchmarks. So, fix the bug, and learn it in and out tmr.

01/15/2026 (T - 117)

Fixed the kernel bug for transposed. Got a bit of reasoning for the bug. Made up the vectorized kernel, and build is happening. In the meantime, will see both the non-vectorized and vectorized kernel(the full line by line) code.

[Deep Dive: Optimizing LLM inference](https://www.youtube.com/watch?v=hMs8VNRy5Ys)

Got the scalar kernel working, understood the entire flow of what i've done till now. Benchmarking and testing now. Will do more tests, clean up and ask for feedback in the github issue. Let's see.

Check more on the above here - [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/)

01/16/2026 (T - 116)

Got feedback on the stream bug, made the changes, and posted the comment. Now, testing it in local, to be sure.

Other opensource works to contribute for(noting it down here):
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/issues)
- [triton-inference-server](https://github.com/triton-inference-server/server/issues)
- [jax-ml/jax](https://github.com/jax-ml/jax/issues)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/issues)
  
One thing to do, while at airport tmr:
- Work on the Silu_mul+Block_Quant kernel. See how vLLM works in the first place, read their webpage, and workflow.

01/17/2026 (T - 115)

Checked vLLM, and understood the basics from their documentation page, next is to work on the code.

01/18/2026 (T - 114)

01/19/2026 (T - 113)

Work on silu_mul_block_quant work;

["I shipped code I don't understand and I bet you have too" – Jake Nations, Netflix](https://www.youtube.com/watch?v=eIoohUmYpGI)

01/20/2026 (T - 112)

All the best Monish.

Done 1 problem in CP-31 sheet. Tmr check on basic C++ code setup in VS Code, and get the level-1 course. - Done.

Work on the kerenl from basics, without claude tmr.

01/21/2026 (T - 111)

Solved 1 in CP-31.

[Cracking FAANG Interviews: Guide to Prepare for Tech Interviews](https://medium.com/@victor.arango93/cracking-faang-interviews-guide-to-prepare-for-tech-interviews-e113db07f8a1)

VS Code, setup done.

Worked on vllm kernel, made some fixes and changes, started the build now

01/22/2026 (T - 110)

Working on vllm kernel, the kernel is working correctly and passes all the test cases. Now working on the pattern matching compilation phase, once this is working, raise a PR, explain your design choices, share results and ask for feedback. Do this by EOD;

[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

Tmr: 
- raise PR for kernel, with how much is done for now, mention the approach, design and ask for feedback. That will give you more direction, than doing all by yourself.
- GPU kernel optimization/ML Systems/ML Infra Partime cold-mailing on the selected startups.

01/23/2026 (T - 109)

Working on the vllm kernel, refactoring unwanted bits.

Raise PR, comment, tag and ask for feedback on the work done; Done.

Do CP-31; Done. Made a small mistake in the workout and that made me stuck. After 30 mins, saw the hint, got my mistake, coded the solution and got it accepted. This is a learning, if you're not able to follow/find a pattern for these, then just recheck your workings once and you might find your answer.

Working on the kernel; Will share the updates with Sohan tmr.

Raised vllm kernel PR(made it to a draft) and asked for review. 

01/24/2026 (T - 108)

Read the slides, and made rough notes. Will format with claude and publish. Done.

Applied to jobs, gave a short 25 mins interview as part of one of the application. Learnt that, i need to learn a lot and gain more knowledge of production ML, pipeline and inference.

Next, cp-31; Done.

vllm kernel; Checked what i wanted too. Waiting for the code review. And i missed to check and resolve the reviews mentioned by gemini and cursor bots, will do it tmr. - Tmr

Also, commented to work on another performance kernel. The CR is still very early, but i just wanted to get my place to work on that, so commented right away.

[Anthropic's Original Performance Take-Home](https://github.com/anthropics/original_performance_takehome) - X

01/25/2026 (T - 107)

Working on vllm kernel fusion pattern matching. Got some new info, the silu_mul is always running as native inline ops, and its not able to match with matcher_silu_mul(existing one). Tried with nsys, and its calling this kernel: act_and_mul_kernel, which is the unfused silu_mul ops. Ok, now question, how its calling this kernel, it should be native pytorch ops right, how it able to match that with this? Got it, actually in my test, i'm trying to use the custom ops directly to see what is being called, and as result its calling this. So, for now, the conclusion is that silu_mul is executed as the native inline ops, and in the compilations, its not being detected to match my needed version of sil_mul. And, because of this, the whole silu_mul_block_pattern isn't working.

Participated in my first CF contest. Spent 2hr 15 mins, solved 4 questions, with 3 accepted and 1 WA in 1 test case.

cp-31, 1q done.

Read Dist. Sys Lec 1, and published notes.

01/26/2026 (T - 106)

Check and resolve vllm kernel bot's review changes and merge;

cp-31 1q done.



