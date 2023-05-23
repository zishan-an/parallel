#include <iostream>
#include<sys/time.h>

#define N 512
#define C 1
using namespace std;
float A[N][N];

void init() {
    //cout<<"Initializing...";
    //cout<<"Initializing ...\n";
    for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
    }
    for (int r = 0; r < 5 * N; r++) {
        int i1 = rand() % N;
        int i2 = rand() % N;
        float rate = rand() % 10 / 10.0;;
        if (i1 != i2) {
            for (int j = 0; j < N; j++) {
                A[i1][j] += rate * A[i2][j];
            }
        }
    }
    //cout<<"Finished. \n";
}
//平凡算法
void normal() {
    struct timeval h, t;
    float time = 0.0;
    //Normal
    cout<<"Normal:(N = "<<N<<")\n";
    for(int r = 0; r < C; r++) {
        init();
        gettimeofday(&h, NULL);
        for(int k = 0; k < N; k++) {
            for(int j = k + 1; j < N; j++) {
                A[k][j] = A[k][j]/A[k][k];
            }
            A[k][k] = 1.0;
            for(int i = k + 1; i < N; i++) {
                for(int j = k + 1; j < N; j++) {
                    A[i][j] = A[i][j] - A[i][k]*A[k][j];
                }
                A[i][k] = 0;
            }
        }
        gettimeofday(&t, NULL);
        time += 1000*(t.tv_sec - h.tv_sec) + 0.001*(t.tv_usec - h.tv_usec);
    }
    cout<<"TIME: "<<time/C<<" ms"<<endl;
}
int main() 
{
    normal();
}
