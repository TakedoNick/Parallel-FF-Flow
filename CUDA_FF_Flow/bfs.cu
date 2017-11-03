// This is a parallel Breath First Search (BFS) Implementation using CUDA
// Input: randomly generated graph in form of adjacent matrix stored in 'mygraph.txt'
// BFS starts from node 0 and stops when the logical BFS tree is formed
// usage: ./<out> <graph_size> <mygraph.txt

#include <cstdlib>
#include <iostream>
#include <time.h>

using namespace std;

__global__ void kernel(bool* adj_mat, const int N, bool* visited, int* frontier, bool* new_frontier, bool* par_mat, int* cap_mat, int* cap_max_mat){
	int row_idx = frontier[blockIdx.x+1];
	long offset = N * row_idx;

	// update new_frontier in threads
	int col_idx = threadIdx.x;
	long offset2 = N * col_idx;
	if(adj_mat[offset + col_idx] && (cap_mat[offset + col_idx] < cap_max_mat[offset + col_idx]) && !visited[col_idx]){
		new_frontier[col_idx] = true;
		par_mat[offset2 + row_idx] = true;
	}

	if(adj_mat[offset2 + row_idx] && (cap_mat[offset2 + row_idx] > 0) && !visited[col_idx]){
		new_frontier[col_idx] = true;
		par_mat[offset2 + row_idx] = true;
	}
}
__global__ void k2(const int N, bool* visited, int* frontier, bool* new_frontier){
	int cn = 0;
	for(int i=0;i<N;i++){
		if(new_frontier[i]){
			new_frontier[i] = false;
			frontier[++cn] = i;
			visited[i] = true;
		}
	}
	frontier[0] = cn;
}

std::vector<std::string> split(std::string str,std::string sep){
    char* cstr=const_cast<char*>(str.c_str());
    char* current;
    std::vector<std::string> arr;
    current=strtok(cstr,sep.c_str());
    while(current!=NULL){
        arr.push_back(current);
        current=strtok(NULL,sep.c_str());
    }
    return arr;
}

int main(int arg, char** argv){
	if(arg!=2){
		cout<<"usage: ./<out> <graph_size> <mygraph.txt"<<endl;
		return -1;
	}
	const int N = atoi(argv[1]);

	//read graph from <input>.txt
	bool* h_adj_mat = (bool*)malloc(N*N*sizeof(bool));
	int* h_cap_mat = (int*)malloc(N*N*sizeof(int));
	int* h_cap_max_mat = (int*)malloc(N*N*sizeof(int));

	for(int i=0;i<N*N;i++){
		string a;
		cin>>a;

		std::vector<std::string> arr;
    arr=split(a, ",");

		if(arr[0]=="1") h_adj_mat[i] = true;
		else h_adj_mat[i] = false;

		h_cap_mat[i] = atoi(arr[1].c_str());
		h_cap_max_mat[i] = atoi(arr[1].c_str());

		//cout<<a<<":"<<h_adj_mat[i]<<" "<<h_cap_mat[i]<<"<>";
	}

	bool* h_par_mat = (bool*)malloc(N*N*sizeof(bool));
	for(int i=0;i<N*N;i++)
		h_par_mat[i] = false;

	//generate visited and frontier vector; init them with node 0;
	bool* h_visited = (bool*)malloc(N*sizeof(bool));
	for(int i=0;i<N;i++) h_visited[i] = false;
	int* h_frontier = (int*)malloc(N*sizeof(int));
	bool* h_new_frontier = (bool*)malloc(N*sizeof(bool));
	for(int i=0;i<N;i++) h_new_frontier[i] = false;

	h_visited[0] = true;
	h_frontier[0] = 1;
	h_frontier[1] = 0;

	//malloc mem in gpu
	clock_t start,end, s, e;
	start = clock();
	bool *d_adj_mat, *d_par_mat, *d_visited, *d_new_frontier;
	int *d_cap_mat, *d_cap_max_mat, *d_frontier;
	cudaMalloc((void**) &d_adj_mat, sizeof(bool) * N * N);
	cudaMemcpy((void*) d_adj_mat, (void*) h_adj_mat, sizeof(bool)*N*N, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_cap_mat, sizeof(int) * N * N);
	cudaMemcpy((void*) d_cap_mat, (void*) h_cap_mat, sizeof(int)*N*N, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_cap_max_mat, sizeof(int) * N * N);
	cudaMemcpy((void*) d_cap_max_mat, (void*) h_cap_max_mat, sizeof(int)*N*N, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_par_mat, sizeof(bool) * N * N);
	cudaMemcpy((void*) d_par_mat, (void*) h_par_mat, sizeof(bool)*N*N, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_visited, sizeof(bool) * N);
	cudaMemcpy((void*) d_visited, (void*) h_visited, sizeof(bool)*N, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_frontier, sizeof(int) * (N+1));
	cudaMemcpy((void*) d_frontier, (void*) h_frontier, sizeof(int)*N, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_new_frontier, sizeof(bool) * N);
	cudaMemcpy((void*) d_new_frontier, (void*) h_new_frontier, sizeof(bool)*N, cudaMemcpyHostToDevice);

	//loop until frontier vector is empty
	int cn=1;
	double t=0;
	bool augFound = false;
	while(1) {
		while(h_frontier[0]!=0){
			cn+=h_frontier[0];

			for(int i = 0; i < h_frontier[0]; i++)
				if(h_frontier[i + 1] == (N - 1)) {
					augFound = true;
					break;
				}

			//lauch kernel : launch threads to update frontier_len, visited and frontier in gpu local mem
			s=clock();
			kernel<<<h_frontier[0], N>>>(d_adj_mat,N,d_visited,d_frontier, d_new_frontier, d_par_mat, d_cap_mat, d_cap_max_mat);

			k2<<<1,1>>>(N, d_visited,d_frontier, d_new_frontier);
			e=clock();
			t+=double(e-s);

			cudaMemcpy((void*) h_frontier, (void*) d_frontier, sizeof(int)*1, cudaMemcpyDeviceToHost);
		}

		if(augFound) {
			cout<<"Augmented path found!"<<endl;
			cudaMemcpy((void*) h_par_mat, (void*) d_par_mat, sizeof(bool) * N * N, cudaMemcpyDeviceToHost);

			//Find the augmented path
			int* augPath = (int*)malloc(N*sizeof(int));
			augPath[0] = N - 1;
			int i = 1, vertex = N - 1;
			while(vertex != 0) {
				for(int j = 0; j < N; j++) {
					if(h_par_mat[i * N + j]) {
						vertex = j;
						augPath[i] = vertex;
						i++;
						break;
					}
				}
			}

			//Display augmented path
			for(int i = 0; i < N; i++) {
				if(augPath[i] == 0) {
					cout<<augPath[i]<<endl;
					break;
				} else {
					cout<<augPath[i]<<" <- ";
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
					if(h_adj_mat[j * N + k]) {
						freeCap = h_cap_max_mat[j * N + k] - h_cap_mat[j * N + k];
					} else {
						freeCap = h_cap_mat[k * N + j];
					}

					if(bottleneck == -1)
						bottleneck = freeCap;
					else if(freeCap < bottleneck)
						bottleneck = freeCap;
				}
			}

			//Update capacities in h_cap_mat
			for(int i = 0; i < N; i++) {
				if(augPath[i] == 0)
					break;
				else {
					int k = augPath[i];
					int j = augPath[i + 1];
					if(h_adj_mat[j * N + k]) {
						h_cap_mat[j * N + k] += bottleneck;
					} else {
						h_cap_mat[k * N + j] -= bottleneck;
					}
				}
			}

			cudaMemcpy((void*) d_cap_mat, (void*) h_cap_mat, sizeof(int)*N*N, cudaMemcpyHostToDevice);
		} else {
			cout<<"Completed Ford Fulkerson!\n";
			break;
		}
	}

	end = clock();
	cout<<"queue through put: "<< cn<<endl;
	cout << "parallel BFS uses " << double(end - start) << " us in total"<< endl;
	cout << "kernel launching and computing uses " <<t<<" us"<<endl;
	cout << "mem copy uses " <<double(end - start) - t<<" us"<<endl;

	return 0;
}
