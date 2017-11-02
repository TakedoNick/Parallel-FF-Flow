#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// data on the CPU = h_
// data on the GPU = d_
// GPU memory pointers
float * d_in;
float * d_out;


#include <limits.h>
#include <stdbool.h>
#include <string.h>

#define N 6 // Number of nodes in the graph

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))

int head, tail;
int q[N + 2];

const int ARRAY_SIZE = 64;
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

void enqueue(int x) {
	q[tail] = x;
	tail++;
}

int dequeue() {
	int x = q[head];
	head++;
	return x;
}

bool bfs(int rGraph[N][N], int s, int t, int parent[]) // bfs=1 if there is a path from 's' to 't'
{
	// Array - all nodes initially non-visited
	bool visited[N];
	memset(visited, 0, sizeof(visited));

	// Add source node to queue and mark it
	enqueue(s);
	visited[s] = true;
	parent[s] = -1;

	// Standard BFS Loop
	while (head != tail)
	{
		int u = dequeue();

		for (int v = 0; v<N; v++)
		{
			if (visited[v] == false && rGraph[u][v] > 0)
			{
				enqueue(v);
				parent[v] = u;
				visited[v] = true;
			}
		}
	}
	return (visited[t] == true);
}

// Kernel program
// __global__ - declaration specifier (declspec) - this is the way
// CUDA knows that this is a kernel
__global__ void fordFulkerson(int graph[N][N], int s, int t)
{

	int u, v;
	int rGraph[N][N]; // Graph with Residual Capacities
	for (u = 0; u < N; u++)
		for (v = 0; v < N; v++)
			rGraph[u][v] = graph[u][v];

	int parent[N];  // Filled by BFS to store path

	int max_flow = 0;

	for (int i = 0; i<N; i++) {
		d_in[i] = graph[i];
	}

	// allocate GPU memory
	cudaMalloc((void **)&d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// Augmenting Path
	while (bfs(rGraph, s, t, parent))
	{
		// Find minimum residual capacity of the edges
		// which is nothing but the min-cuts
		// which is equal to the Maximum Flow acc. to Min-Cut-Max-Flow
		int path_flow = INT_MAX;
		for (v = t; v != s; v = parent[v])
		{
			u = parent[v];
			path_flow = min(path_flow, rGraph[u][v]);
		}
		// update residual capacities of the edges and reverse edges
		// along the path
		for (v = t; v != s; v = parent[v])
		{
			u = parent[v];
			rGraph[u][v] -= path_flow;
			rGraph[v][u] += path_flow;
		}
		// Add to overall flow
		max_flow += path_flow;
	}
	return max_flow;
}


int main()
{
	// {A, B, C, D, E, F}
	int graph[N][N] = { { 0, 5, 5, 0, 0, 0 },
	{ 0, 0, 0, 6, 3, 0 },
	{ 0, 0, 0, 3, 1, 0 },
	{ 0, 0, 0, 0, 0, 6 },
	{ 0, 0, 0, 0, 0, 6 },
	{ 0, 0, 0, 0, 0, 0 }
	};

	fordFulkerson<<<2, ARRAY_SIZE >>> (graph, 0, 5);
	printf("Maximum Flow: %d \n", fordFulkerson(graph, 0, 5));

	// free the GPU memory
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
