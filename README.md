# INR-Arch: A Dataflow Architecture and Compiler for Arbitrary-Order Gradient Computations in Implicit Neural Representation Processing

Stefan Abi-Karam, Rishov Sarkar, Dejia Xu, Zhiwen Fan, Zhangyang Wang, Cong Hao

> An increasing number of researchers are finding use for nth-order gradient computations for a wide variety of applications, including graphics, meta-learning (MAML), scientific computing, and most recently, implicit neural representations (INRs). Recent work shows that the gradient of an INR can be used to edit the data it represents directly without needing to convert it back to a discrete representation. However, given a function represented as a computation graph, traditional architectures face challenges in efficiently computing its nth-order gradient due to the higher demand for computing power and higher complexity in data movement. This makes it a promising target for FPGA acceleration. In this work, we introduce INR-Arch, a framework that transforms the computation graph of an nth-order gradient into a hardware-optimized dataflow architecture. We address this problem in two phases. First, we design a dataflow architecture that uses FIFO streams and an optimized computation kernel library, ensuring high memory efficiency and parallel computation. Second, we propose a compiler that extracts and optimizes computation graphs, automatically configures hardware parameters such as latency and stream depths to optimize throughput, while ensuring deadlock-free operation, and outputs High-Level Synthesis (HLS) code for FPGA implementation. We utilize INR editing as our benchmark, presenting results that demonstrate 1.8-4.8x and 1.5-3.6x speedup compared to CPU and GPU baselines respectively. Furthermore, we obtain 3.1-8.9x and 1.7-4.3x lower memory usage, and 1.7-11.3x and 5.5-32.8x lower energy-delay product. Our framework will be made open-source and available on GitHub.

Paper URL: https://arxiv.org/abs/2308.05930

This project is developed under Sharc Lab @ Georgia Tech.

More updates in the readme to come...
