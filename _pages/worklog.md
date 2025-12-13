<!-- ---
layout: page
permalink: /worklog/
title: Worklog
description: 
nav: False
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

https://horace.io/brrr_intro.html

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

Need clarity in derivatives with respect to certain forms in backprop calculations, noted down a example og MLP with einsum;

**Softmax Definition**

$$
y_i = \mathrm{softmax}(a)_i = \frac{e^{a_i}}{\sum_k e^{a_k}}
\qquad\text{let } S = \sum_k e^{a_k}.
$$

---

**Softmax Jacobian:** \( \frac{\partial y_i}{\partial a_j} \)

Start from:

$$
y_i = \frac{e^{a_i}}{S}.
$$

Differentiate w.r.t. \(a_j\) using the quotient rule:

$$
\frac{\partial y_i}{\partial a_j}
= \frac{S\cdot\frac{\partial}{\partial a_j}e^{a_i}
      \;-\;
      e^{a_i}\cdot\frac{\partial S}{\partial a_j}}{S^2}.
$$

Compute derivatives:

- \( \frac{\partial}{\partial a_j} e^{a_i} = e^{a_i}\delta_{ij} \)
- \( \frac{\partial S}{\partial a_j} = e^{a_j} \)

Substitute:

$$
\frac{\partial y_i}{\partial a_j}
= \frac{S(e^{a_i}\delta_{ij}) - e^{a_i}e^{a_j}}{S^2}
= \frac{e^{a_i}}{S^2}(S\delta_{ij} - e^{a_j}).
$$

Use softmax definitions:

$$
y_i = \frac{e^{a_i}}{S}, \qquad
y_j = \frac{e^{a_j}}{S}.
$$

Final form:

$$
\frac{\partial y_i}{\partial a_j}
= y_i\delta_{ij} - y_i y_j
= y_i (\delta_{ij} - y_j).
$$


**Cross-Entropy Loss**

For one-hot target \(t\):

$$
L = -\sum_i t_i \log y_i.
$$

Differentiate w.r.t. \(a_j\):

$$
\frac{\partial L}{\partial a_j}
= \sum_i \frac{\partial L}{\partial y_i}
       \frac{\partial y_i}{\partial a_j}
= \sum_i \left( -\frac{t_i}{y_i} \right)
           y_i(\delta_{ij} - y_j).
$$

Simplify:

$$
= \sum_i -t_i(\delta_{ij} - y_j)
= -\sum_i t_i\delta_{ij}
  + \sum_i t_i y_j.
$$

Use:

- \( \sum_i t_i\delta_{ij} = t_j \)
- \( \sum_i t_i = 1 \) for one-hot \(t\)

Thus:

$$
\frac{\partial L}{\partial a_j}
= -t_j + y_j
= y_j - t_j.
$$



$$
\boxed{
\frac{\partial L}{\partial a_j} = y_j - t_j
}
$$


[A basic introduction to NumPy's einsum](https://ajcr.net/Basic-guide-to-einsum/)
- einsum does not promote data types when summing, what it means is that if you're using a more limited datatype, you might get unexpected results.
- einsum is not always the fastest option in NumPy. Functions such as dot and inner often link to lightening-quick BLAS routines which can outperform einsum and certainly shouldn't be forgotten about. 


Finish Einsum today; - Done

Draft a mail for Prof Zaharan for GPU project, by EOD, scheduled mail for 12/15/25, 8AM - Done

Need to finish the backprop of MLP with einsum after dinner;
Check on the page's content for MLP derivation, or write a new one for this compact MLP forward and backward pass; - First learn, and then finish this. - Next;
Along with this, start transformer;





