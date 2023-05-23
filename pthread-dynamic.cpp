#include <iostream>
#include<sys/time.h>
#include<pthread.h>
#include<semaphore.h>
#define N 2048
#define C 1
#define P 4
using namespace std;
float A[N][N];
typedef struct {
    int k;//消去的轮次
    int t_id;//线程id
}threadParam_t;
sem_t sem_main;
sem_t sem_workerstart[P];
sem_t sem_workerend[P];
void* threadFunc(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;//消去的轮次
    int t_id = p->t_id;//线程编号
    int i = k + t_id + 1;//获取自己的计算任务
    for (int j = k + 1; j < N; j++) {
        A[i][j] = A[i][j] - A[i][k] * A[k][j];
    }
    A[i][k] = 0;
    pthread_exit(nullptr);
}

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

//动态线程
void pth() {
    struct timeval h, t;
    float time = 0.0;
    cout << "Pthread:(N = " << N << ")\n";
    for (int r = 0; r < C; r++) {
        init();
        gettimeofday(&h, NULL);
        for (int k = 0; k < N; k++) {
            //主线程做除法操作
            for (int j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            //创建工作线程，进行消去操作
            int worker_count = N - 1 - k;
            pthread_t* handles = (pthread_t*)malloc(N * sizeof(pthread_t));
            threadParam_t* param = (threadParam_t*)malloc(N * sizeof(threadParam_t));
            //分配任务
            for (int t_id = 0; t_id < worker_count; t_id++) {
                param[t_id].k = k;
                param[t_id].t_id = t_id;
            }
            //创建线程
            for (int t_id = 0; t_id < worker_count; t_id++) {
                pthread_create(handles + t_id, nullptr, threadFunc, param + t_id);
            }
            //主线程挂起等待所有工作线程完成此轮消去工作
            for (int t_id = 0; t_id < worker_count; t_id++) {
                pthread_join(handles[t_id], nullptr);
            }
        }
        gettimeofday(&t, NULL);
        time += 1000 * (t.tv_sec - h.tv_sec) + 0.001 * (t.tv_usec - h.tv_usec);
    }
    cout << "TIME: " << time / C << " ms" << endl;
}

int main() 
{
    pth();
}