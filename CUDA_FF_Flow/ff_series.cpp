#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string.h>
#include <time.h>

using namespace std;

//Check for edges valid to be part of augmented path
void updateFrontier(bool* adj_mat, const int N, bool* visited, int* frontier, bool* new_frontier, bool* par_mat, int* cap_mat, int* cap_max_mat) {
	for(int i = 0; i < frontier[0]; i++) {
		int row_idx = frontier[i + 1];
		long offset = N * row_idx;

		for(int j = 0; j < N; j++){
			int col_idx = j;
			long offset2 = N * col_idx;
			if(adj_mat[offset + col_idx] && (cap_mat[offset + col_idx] < cap_max_mat[offset + col_idx]) && !visited[col_idx]) {
				new_frontier[col_idx] = true;
				par_mat[offset2 + row_idx] = true;
				visited[col_idx] = true;
			}

			if(adj_mat[offset2 + row_idx] && (cap_mat[offset2 + row_idx] > 0) && !visited[col_idx]) {
				new_frontier[col_idx] = true;
				par_mat[offset2 + row_idx] = true;
				visited[col_idx] = true;
			}
		}
	}

	int count = 0;
	for(int i=0;i<N;i++) {
		if(new_frontier[i]) {
			new_frontier[i] = false;
			frontier[++count] = i;
		}
	}
	frontier[0] = count;
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

	int maxflow = 0;

	//Read graph from <input>.txt
	bool* adj_mat = (bool*)malloc(N*N*sizeof(bool));
	int* cap_mat = (int*)malloc(N*N*sizeof(int));
	int* cap_max_mat = (int*)malloc(N*N*sizeof(int));

	for(int i=0;i<N*N;i++) {
		string a;
		cin>>a;

		std::vector<std::string> arr;
    arr=split(a, ",");

		if(arr[0]=="1") adj_mat[i] = true;
		else adj_mat[i] = false;

		cap_mat[i] = 0;
		cap_max_mat[i] = atoi(arr[1].c_str());
	}

	clock_t start, end;
	start = clock();

	bool* par_mat = (bool*)malloc(N*N*sizeof(bool));
	bool* visited = (bool*)malloc(N*sizeof(bool));
	int* frontier = (int*)malloc(N*sizeof(int));
	bool* new_frontier = (bool*)malloc(N*sizeof(bool));
	int* augPath = (int*)malloc(N*sizeof(int));

	double t = 0;
	while(1) {
		//Generate and initialize par_mat
		for(int i=0;i<N*N;i++)
			par_mat[i] = false;

		//Generate and initialize visited and frontier
		for(int i=0;i<N;i++) visited[i] = false;
		for(int i=0;i<N;i++) new_frontier[i] = false;

		visited[0] = true;
		frontier[0] = 1;
		frontier[1] = 0;

		bool augFound = false;
		cout<<"\nFrontier:"<<endl<<endl;
		while(frontier[0] != 0) {
			for(int i = 0; i < frontier[0]; i++)
				cout<<frontier[i + 1]<<" ";
			cout<<endl;

			//Complete search if sink has been reached
			for(int i = 0; i < frontier[0]; i++)
				if(frontier[i + 1] == (N - 1)) {
					augFound = true;
					break;
				}

			//Call to update frontier
			updateFrontier(adj_mat, N, visited, frontier, new_frontier, par_mat, cap_mat, cap_max_mat);
		}

		if(augFound) {
			cout<<"\nAugmented path found!"<<endl;

			cout<<"\nParent matrix:"<<endl<<endl;
			for(int i = 0; i < N; i++) {
				for(int j = 0; j < N; j++) {
					cout<<par_mat[i * N + j]<<" ";
				}
				cout<<endl;
			}

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

			//Display augmented path
			cout<<"\nAugmented Path:\n\n";
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
			cout<<"\nBottleneck of augmented path: "<<bottleneck<<endl;
			maxflow += bottleneck;

			//Update capacities in cap_mat
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
		} else {
			cout<<"\nFord Fulkerson complete!\n";
			cout<<"\nMaximum Flow: "<<maxflow<<"\n";
			break;
		}
	}

	//Display execution times
	end = clock();
	cout<<"\nTime taken to run complete series Ford Fulkerson algorithm: "<<double(end - start)<<"us"<<endl;

	free(augPath);
	free(par_mat);
	free(visited);
	free(frontier);
	free(new_frontier);
	free(adj_mat);
	free(cap_mat);
	free(cap_max_mat);

	return 0;
}
