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

[https://www.youtube.com/watch?v=s1ILGG0TyYM](Intro to Triton: A Parallel Programming Compiler and Language, esp for AI acceleration (updated))

12/10/2025 (T - 153)

[https://christianjmills.com/posts/cuda-mode-notes/lecture-014/](GPU MODE Lecture 14: Practitioners Guide to Triton)

Have some doubts in the above blog's Triton coding parts, will check them using LLM and understand after BDML presentation.

[https://www.youtube.com/watch?v=JndztlScZLA](MegaBlocks: Eﬃcient SparseTraining with Mixture-of-Experts)

Solve Interesting Technical Work, again and always from first principles;

I need to code here, from scratch and on my own. I feel this to a be big lacking and decrease in confidence level. Need to start doing even small things, but do it myself. 

[https://www.thonking.ai/p/why-pytorch-is-an-amazing-place-to](Why PyTorch is an amazing place to work... and Why I'm Joining Thinking Machines)

Done two dry runs, got a descent flow of what and how i want to present. All the best Monish!

Presentation done and did well, kudos. Next focus on the custom kernel for MOE(with triton) and then report for BDML project.

Ping Sohan to ask, If he can be my mentor and accountability partner, with whom i can share my bi-weekly progress update, short and long term goals and discuss, someone with whom i can connect and talk and ask for advice - Done

[https://christianjmills.com/posts/cuda-mode-notes/lecture-014/](GPU MODE Lecture 14: Practitioners Guide to Triton) + Gemini

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