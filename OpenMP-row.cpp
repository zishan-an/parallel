#include <iostream>
#include<sys/time.h>
#include<omp.h>
#define N 2048
#define C 1
#define P 4
using namespace std;
float A[N][N];
void init() {
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
}
//openMP线程数为7行主元消元
void omRow() {
    struct timeval h, t;
    float time = 0.0;
    //Normal
    cout << "ROW:(N = " << N << ", P = 4)\n";
    for (int r = 0; r < C; r++) {
        init();
        gettimeofday(&h, NULL);
        int i, j, k;
        float tmp;
        for (k = 0; k < N; k++) {
            {
                tmp = A[k][k];
                for (j = k + 1; j < N; j++) {
                    A[k][j] = A[k][j] / tmp;
                }
                A[k][k] = 1.0;
            }
            for (i = k + 1; i < N; i++) {
                tmp = A[i][k];
                for (j = k + 1; j < N; j++) {
                    A[i][j] = A[i][j] - tmp * A[k][j];
                }
                A[i][k] = 0;
            }
        }
        gettimeofday(&t, NULL);
        time += 1000 * (t.tv_sec - h.tv_sec) + 0.001 * (t.tv_usec - h.tv_usec);
    }
    cout << "TIME: " << time / C << " ms" << endl;
}

int main()
{
    omRow();
    return 0;
}
