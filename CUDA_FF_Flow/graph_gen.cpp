#include <cstdlib>
#include <iostream>

using namespace std;

int main(int arg, char** argv) {
	if(arg != 3){
		cout<<"Please run in the following manner: ./<out> <graph_size> <edge_number> >mygraph.txt"<<endl;
		return -1;
	}
	const int N = atoi(argv[1]);
	const int EDGE_NUM = atoi(argv[2]);

	int** adj_mat = (int**) malloc(N * sizeof(int*));
	for(int row = 0; row < N; row++)
		adj_mat[row] = (int*) malloc(N * sizeof(int));
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			adj_mat[i][j] = 0;

	int n = EDGE_NUM;
	while(n > 0) {
		int u = rand() % N;
		int v = rand() % N;
		if(adj_mat[u][v] == 0 && u != v) {
			adj_mat[u][v] = 1;
			n--;
		}
	}

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			if(adj_mat[i][j] == 1)
				cout<<adj_mat[i][j]<<','<<rand() % 100 + 10<<' ';
			else
				cout<<adj_mat[i][j]<<",0 ";
		}
		cout<<endl;
	}
	return 0;
}
