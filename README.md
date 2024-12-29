# Parallel Ford-Fulkerson Flow

This is an implementation of the Ford-Fulkerson algorithm designed for solving the Max-Flow/Min-Cut problem using CUDA for GPU-based acceleration. The implementation leverages parallel processing to improve computational efficiency.

## Overview

The Max-Flow problem identifies the maximum capacity path from a source to a sink in a flow network. Traditional implementations of the Ford-Fulkerson algorithm are constrained by sequential execution and high time complexity. This implementation addresses these constraints by utilizing CUDA-enabled GPUs to execute computations in parallel.

## Motivation

Solving network flow problems such as Max-Flow/Min-Cut is critical in various domains, including transportation, communication networks, and computer vision. Traditional sequential approaches often struggle with large-scale graphs due to high time complexity. With the advent of GPU computing, parallel processing offers an opportunity to significantly accelerate such algorithms. This project aims to explore and implement these advancements, making network flow analysis faster and more efficient.

## Project Architecture

The implementation leverages a CUDA-enabled GPU as a co-processor to parallelize the Ford-Fulkerson algorithm for solving the Max-Flow problem. The architecture combines CPU-based control and GPU-based computation for optimal performance.

 1.	Kernel Functions:
	•	Push and pull kernels for updating flow values.

	•	Relabeling kernels for dynamic graph updates.
	
 2.	Memory Management:
	•	Shared memory for intra-block communication.

	•	Global memory for inter-thread synchronization.

 3.	Execution Flow:
	•	Initialization of graph structures.

	•	Iterative kernel execution until convergence.

![Architechture](imgs/Architecture%20Diagram.jpg)

### Components

1. **CPU (Host)**:
   - Performs input preparation and initial Breadth-First Search (BFS).
   - Allocates tasks and kernel instructions to the GPU.
   - Transfers data to the GPU via CUDA `memcpy`.

2. **CUDA-Enabled GPU (Device)**:
   - Processes augmenting paths in parallel using 2D compute grids.
   - Assigns threads to explore paths and calculate bottleneck capacities.
   - Uses shared memory for thread communication and global memory for synchronization.

3. **Memory Management**:
   - **Host Memory**: Stores input data and BFS results.
   - **Device Memory**: Manages path exploration and flow updates.

### Workflow

1. **Input Preparation**:
   - Graph adjacency matrix and capacities are loaded into CPU memory.
2. **BFS Execution**:
   - CPU computes initial augmenting paths and transfers results to the GPU.
3. **Kernel Execution**:
   - GPU assigns augmenting paths to 2D blocks, with threads processing paths independently.
   - Shared memory handles intra-block communication; global memory handles inter-block updates.
4. **Path Exploration**:
   - Threads calculate bottleneck capacities and update flows concurrently.
   - Loops continue until no more augmenting paths are found.
5. **Output Generation**:
   - Results, including maximum flow and computation times, are sent back to the CPU.

### Key Optimizations

- **Parallel Processing**: GPU processes augmenting paths concurrently, reducing execution time.
- **Hierarchical Memory**: Shared memory optimizes intra-block operations; global memory supports synchronization.
- **Scalability**: Architecture supports large graphs, limited by GPU hardware constraints.


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

    Input Format: Example format for a 10 Node Graph
```bash
0 14 0 0 0 0 0 0 0 0
0 0 10 0 0 0 0 0 0 0
0 0 0 5 0 0 0 0 0 0
0 0 0 0 10 0 0 0 0 0
0 0 0 0 0 8 0 0 0 0
0 0 0 0 0 0 6 0 0 0
0 0 0 0 0 0 0 11 0 0
0 0 0 0 0 0 0 0 16 0
0 0 0 0 0 0 0 0 0 20
0 0 0 0 0 0 0 0 0 0
```
   

4. Execution
```bash
./parallel_ff_flow <graph_size> <input_file>
```

5. Output
- Augmented Paths
```bash
Augmented Path: 0 -> 1 -> 2 -> 3
Bottleneck of augmented path: 14
```

6. Performance Metrics
```bash
Time taken to run complete parallel Ford-Fulkerson algorithm: 45559 µs
Time taken for kernel: 11583 µs
Time taken for memory from host to device: 34610 µs
Maximum Flow: 324
```



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
