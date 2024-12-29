# Parallel Ford-Fulkerson Flow

This is an implementation of the Ford-Fulkerson algorithm designed for solving the Max-Flow/Min-Cut problem using CUDA for GPU-based acceleration. The implementation leverages parallel processing to improve computational efficiency.

## Overview

The Max-Flow problem identifies the maximum capacity path from a source to a sink in a flow network. Traditional implementations of the Ford-Fulkerson algorithm are constrained by sequential execution and high time complexity. This implementation addresses these constraints by utilizing CUDA-enabled GPUs to execute computations in parallel.

## Motivation

Solving network flow problems such as Max-Flow/Min-Cut is critical in various domains, including transportation, communication networks, and computer vision. Traditional sequential approaches often struggle with large-scale graphs due to high time complexity. With the advent of GPU computing, parallel processing offers an opportunity to significantly accelerate such algorithms. This project aims to explore and implement these advancements, making network flow analysis faster and more efficient.

## Features

- Parallelized depth-first search (DFS) for efficient augmenting path identification.
- GPU-based processing for graph cuts and bottleneck capacity computations.
- Optimized memory usage and synchronization mechanisms.
- Support for both dense and sparse graph structures.

## Technologies Used

- **Programming Language:** C++ with CUDA extensions
- **GPU Framework:** NVIDIA CUDA
- **Parallel Computing:** Thread and block-based execution
- **Libraries:** CUDA Runtime API

## Setup

### Prerequisites

1. An NVIDIA GPU with CUDA capability.
2. CUDA Toolkit installed.
3. A C++ compiler (e.g., GCC) compatible with the CUDA toolkit.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TakedoNick/Parallel-FF-Flow.git
   cd Parallel-FF-Flow
   ```

2. Compile the code
   ```bash
   nvcc -o parallel_ff_flow parallel_ff_flow.cu
   ```

3. Prepare the input graph in the required format
   	•	<graph_size>: Number of nodes in the graph.
   
	  •	<input_file>: File containing the adjacency matrix representation of the graph.

    Input Format
   ```bash
   1,capacity 0,0 1,capacity ...
   ```

4. Execution
   ```bash
   ./parallel_ff_flow <graph_size> <input_file>
   ```

## Project Architecture

 1.	Kernel Functions:
	•	Push and pull kernels for updating flow values.

	•	Relabeling kernels for dynamic graph updates.
	
 2.	Memory Management:
	•	Shared memory for intra-block communication.

	•	Global memory for inter-thread synchronization.

 3.	Execution Flow:
	•	Initialization of graph structures.

	•	Iterative kernel execution until convergence.

## Applications
	
 •	Traffic Optimization: Identifies bottlenecks and optimizes vehicular flow on urban roads.

 •	Image Segmentation: Applies graph cuts for image processing tasks such as object detection and medical imaging.

 •	Network Analysis: Evaluates network reliability and identifies optimal paths for data transmission.

## Future Work
	
 •	Scaling for larger graphs through distributed computing.

 •	Real-time graph updates for dynamic scenarios.

 •	Exploration of alternative parallelization techniques for diverse hardware architectures.

## References
	
 1.	Vineet, Vibhav, and P. J. Narayanan. “CUDA cuts: Fast graph cuts on the GPU.”
	
 2.	Surakhi, Ola M., et al. “A Parallel Genetic Algorithm for Maximum Flow Problem.”
	
 3.	Singh, Dhirendra P., et al. “Efficient Parallel Implementation of Single Source Shortest Path Algorithm on GPU Using CUDA.”
