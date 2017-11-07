#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <time.h>

using namespace std;

//Check for edges valid to be part of augmented path
__global__ void kernel(bool* adj_mat, const int N, bool* visited, int* frontier, bool* new_frontier, bool* par_mat, int* cap_mat, int* cap_max_mat) {
	int row_idx = frontier[blockIdx.x+1];
	long offset = N * row_idx;

	int col_idx = threadIdx.x;
	long offset2 = N * col_idx;
	if(adj_mat[offset + col_idx] && (cap_mat[offset + col_idx] < cap_max_mat[offset + col_idx]) && !visited[col_idx]) {
		new_frontier[col_idx] = true;
		par_mat[offset2 + row_idx] = true;
	}

	if(adj_mat[offset2 + row_idx] && (cap_mat[offset2 + row_idx] > 0) && !visited[col_idx]) {
		new_frontier[col_idx] = true;
		par_mat[offset2 + row_idx] = true;
	}
}

//Update frontier
__global__ void k2(const int N, bool* visited, int* frontier, bool* new_frontier, bool* augFound) {
	int count = 0;
	for(int i=0;i<N;i++) {
		if(new_frontier[i]) {
			new_frontier[i] = false;
			frontier[++count] = i;
			visited[i] = true;
		}
	}
	frontier[0] = count;

  //Complete search if sink has been reached
  for(int i = 0; i < frontier[0]; i++)
    if(frontier[i + 1] == (N - 1))
      augFound[0] = true;
}

__global__ void k3(const int N, int* augPath, bool* visited, int* frontier, bool* new_frontier, bool* par_mat, int* cap_mat, bool* adj_mat, int* cap_max_mat, int* maxflow, bool* augFound) {
	augFound[0] = false;

	//Find the augmented path
  augPath[0] = N - 1;
  int i = 1, vertex = N - 1;
  while(vertex != 0) {
    for(int j = 0; j < N; j++) {
      if(par_mat[vertex * N + j]) {
        vertex = j;
        augPath[i] = vertex;
        i++;
        break;
      }
    }
  }

  //Compute the bottleneck for the augmented path
  int bottleneck = -1;
  for(int i = 0; i < N; i++) {
    if(augPath[i] == 0)
      break;
    else {
      int k = augPath[i];
      int j = augPath[i + 1];
      int freeCap;
      if(adj_mat[j * N + k]) {
        freeCap = cap_max_mat[j * N + k] - cap_mat[j * N + k];
      } else {
        freeCap = cap_mat[k * N + j];
      }

      if(bottleneck == -1)
        bottleneck = freeCap;
      else if(freeCap < bottleneck)
        bottleneck = freeCap;
    }
  }
  maxflow[0] += bottleneck;

  //Update capacities in d_cap_mat
  for(int i = 0; i < N; i++) {
    if(augPath[i] == 0)
      break;
    else {
      int k = augPath[i];
      int j = augPath[i + 1];
      if(adj_mat[j * N + k]) {
        cap_mat[j * N + k] += bottleneck;
      } else {
        cap_mat[k * N + j] -= bottleneck;
      }
    }
  }

  //Initialize par_mat
  for(int i=0;i<N*N;i++)
    par_mat[i] = false;

  //Initialize visited and frontier
  for(int i=0;i<N;i++) visited[i] = false;
  for(int i=0;i<N;i++) new_frontier[i] = false;

  visited[0] = true;
  frontier[0] = 1;
  frontier[1] = 0;
}

std::vector<std::string> split(std::string str,std::string sep) {
    char* cstr=const_cast<char*>(str.c_str());
    char* current;
    std::vector<std::string> arr;
    current=strtok(cstr,sep.c_str());
    while(current!=NULL) {
        arr.push_back(current);
        current=strtok(NULL,sep.c_str());
    }
    return arr;
}

int main(int arg, char** argv) {
	if(arg!=2) {
		cout<<"Please run in the following manner: ./<out> <graph_size> <<input>.txt"<<endl;
		return -1;
	}
	const int N = atoi(argv[1]);

	//Read graph from <input>.txt
	bool* h_adj_mat = (bool*)malloc(N*N*sizeof(bool));
	int* h_cap_mat = (int*)malloc(N*N*sizeof(int));
	int* h_cap_max_mat = (int*)malloc(N*N*sizeof(int));

  bool* h_augFound = (bool*)malloc(sizeof(bool));
  int* h_maxflow = (int*)malloc(sizeof(int));

	for(int i=0;i<N*N;i++) {
		string a;
		cin>>a;

		std::vector<std::string> arr;
    arr=split(a, ",");

		if(arr[0]=="1") h_adj_mat[i] = true;
		else h_adj_mat[i] = false;

		h_cap_mat[i] = 0;
		h_cap_max_mat[i] = atoi(arr[1].c_str());
	}

	clock_t start, end, s, e;
	start = clock();

	//Allocate device memory for adj_mat, cap_mat and cap_max_mat
	bool *d_adj_mat, *d_par_mat, *d_visited, *d_new_frontier, *d_augFound;
	int *d_cap_mat, *d_cap_max_mat, *d_frontier, *d_augPath, *d_maxflow;
	cudaMalloc((void**) &d_adj_mat, sizeof(bool) * N * N);
	cudaMemcpy((void*) d_adj_mat, (void*) h_adj_mat, sizeof(bool)*N*N, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_cap_mat, sizeof(int) * N * N);
	cudaMemcpy((void*) d_cap_mat, (void*) h_cap_mat, sizeof(int)*N*N, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_cap_max_mat, sizeof(int) * N * N);
	cudaMemcpy((void*) d_cap_max_mat, (void*) h_cap_max_mat, sizeof(int)*N*N, cudaMemcpyHostToDevice);

	//Allocate host memory visited and frontier
	bool* h_par_mat = (bool*)malloc(N*N*sizeof(bool));
	bool* h_visited = (bool*)malloc(N*sizeof(bool));
	int* h_frontier = (int*)malloc((N + 1)*sizeof(int));
	bool* h_new_frontier = (bool*)malloc(N*sizeof(bool));
  int* h_augPath = (int*)malloc(N*sizeof(int));

	//Allocate device memory for par_mat, visited and frontier
	cudaMalloc((void**) &d_par_mat, sizeof(bool) * N * N);
	cudaMalloc((void**) &d_visited, sizeof(bool) * N);
	cudaMalloc((void**) &d_frontier, sizeof(int) * (N + 1));
	cudaMalloc((void**) &d_new_frontier, sizeof(bool) * N);
  cudaMalloc((void**) &d_augPath, sizeof(int) * N);
  cudaMalloc((void**) &d_augFound, sizeof(bool));
  cudaMalloc((void**) &d_maxflow, sizeof(int));

  //Initialize par_mat
  for(int i=0;i<N*N;i++)
    h_par_mat[i] = false;

  //Initialize visited and frontier
  for(int i=0;i<N;i++) h_visited[i] = false;
  for(int i=0;i<N;i++) h_new_frontier[i] = false;

  h_visited[0] = true;
  h_frontier[0] = 1;
  h_frontier[1] = 0;
  h_maxflow[0] = 0;
  h_augFound[0] = false;

  //Copy to device memory for par_mat, visited and frontier
  cudaMemcpy((void*) d_par_mat, (void*) h_par_mat, sizeof(bool)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy((void*) d_visited, (void*) h_visited, sizeof(bool)*N, cudaMemcpyHostToDevice);
  cudaMemcpy((void*) d_frontier, (void*) h_frontier, sizeof(int)*N, cudaMemcpyHostToDevice);
  cudaMemcpy((void*) d_new_frontier, (void*) h_new_frontier, sizeof(bool)*N, cudaMemcpyHostToDevice);
	cudaMemcpy((void*) d_augPath, (void*) h_augPath, sizeof(int)*N, cudaMemcpyHostToDevice);
  cudaMemcpy((void*) d_augFound, (void*) h_augFound, sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy((void*) d_maxflow, (void*) h_maxflow, sizeof(int), cudaMemcpyHostToDevice);

	double t = 0;
	while(1) {
		while((h_frontier[0]) != 0 && !h_augFound[0]) {
			//Call to kernels
			s = clock();

			cudaMemcpy((void*) h_frontier, (void*) d_frontier, sizeof(int), cudaMemcpyDeviceToHost);

			kernel<<<h_frontier[0], N>>>(d_adj_mat,N,d_visited,d_frontier, d_new_frontier, d_par_mat, d_cap_mat, d_cap_max_mat);
			k2<<<1, 1>>>(N, d_visited, d_frontier, d_new_frontier, d_augFound);
			e = clock();
			t += double(e - s);

			cudaMemcpy((void*) h_frontier, (void*) d_frontier, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy((void*) h_augFound, (void*) d_augFound, sizeof(bool), cudaMemcpyDeviceToHost);
		}

		if(h_augFound[0]) {
      k3<<<1, 1>>>(N, d_augPath, d_visited, d_frontier, d_new_frontier, d_par_mat, d_cap_mat, d_adj_mat, d_cap_max_mat, d_maxflow, d_augFound);
      h_augFound[0] = false;
		} else {
			cout<<"\nFord Fulkerson complete!\n";
      cudaMemcpy((void*) h_maxflow, (void*) d_maxflow, sizeof(int), cudaMemcpyDeviceToHost);
			cout<<"\nMaximum Flow: "<<h_maxflow[0]<<"\n";
			break;
		}
	}

	//Display execution times
	end = clock();
	cout<<"\nTime taken to run complete parallel Ford Fulkerson algorithm: "<<double(end - start)<<"us"<<endl;
	cout<<"Time taken to run kernel: "<<t<<"us"<<endl;
	cout<<"Time taken for memcpy from host to device: "<<double(end - start) - t<<"us"<<endl;

	//Free all memory
	free(h_augPath);
	free(h_par_mat);
	free(h_visited);
	free(h_frontier);
	free(h_new_frontier);
	free(h_adj_mat);
	free(h_cap_mat);
	free(h_cap_max_mat);
	cudaFree(d_par_mat);
	cudaFree(d_visited);
	cudaFree(d_frontier);
	cudaFree(d_new_frontier);
	cudaFree(d_adj_mat);
	cudaFree(d_cap_mat);
	cudaFree(d_cap_max_mat);

	return 0;
}
