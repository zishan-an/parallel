#include <iostream>
#include<sys/time.h>
#include<pthread.h>
#include<semaphore.h>
#include<xmmintrin.h>
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
void* threadFunc3(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workerstart[t_id]);//阻塞，等待主线程完成除法操作
        //循环划分任务
        for (int i = k + t_id + 1; i < N; i += P) {
            float* aik = new float[4];
            *aik = *(aik + 1) = *(aik + 2) = *(aik + 3) = A[i][k];
            __m128 vaik = _mm_load_ps(aik);//将四个单精度浮点数从内存加载到向量寄存器
            delete[] aik;
            for (int j = k + 1; j + 3 < N; j += 4) {
                __m128 vaij = _mm_loadu_ps(&A[i][j]);
                __m128 vakj = _mm_loadu_ps(&A[k][j]);
                __m128 vx = _mm_mul_ps(vaik, vakj);
                vaij = _mm_sub_ps(vaij, vx);//A[i][j] = A[i][j] - A[i][k]*A[k][j];
                _mm_storeu_ps(&A[i][j], vaij);//存储到内存
            }
            for (int j = N - N % 4; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }//结尾几个元素串行计算
            A[i][k] = 0;
        }
        sem_post(&sem_main);//唤醒主线程
        sem_wait(&sem_workerend[t_id]);//阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(nullptr);
}
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
void pthRowSSE() {
    struct timeval h, t;
    float time = 0.0;
    cout << "PthreadRowSSE:(N = " << N << " and P = " << P << ")\n";
    for (int r = 0; r < C; r++) {
        init();
        gettimeofday(&h, NULL);
        //初始化信号量
        sem_init(&sem_main, 0, 0);
        for (int i = 0; i < P; i++) {
            sem_init(&sem_workerstart[i], 0, 0);
            sem_init(&sem_workerend[i], 0, 0);
        }
        //创建线程
        pthread_t handles[P];
        threadParam_t param[P];
        for (int t_id = 0; t_id < P; t_id++) {
            param[t_id].t_id = t_id;
            pthread_create(handles + t_id, nullptr, threadFunc3, param + t_id);
        }
        for (int k = 0; k < N; k++) {
            //主线程做除法操作
            float* akk = new float[4];
            *akk = *(akk + 1) = *(akk + 2) = *(akk + 3) = A[k][k];
            __m128 vt = _mm_load_ps(akk);//将四个单精度浮点数从内存加载到向量寄存器
            delete[] akk;
            for (int j = k + 1; j + 3 < N; j += 4) {
                __m128 va = _mm_loadu_ps(&A[k][j]);
                va = _mm_div_ps(va, vt);//A[k][j] = A[k][j]/A[k][k];
                _mm_storeu_ps(&A[k][j], va);//储存到内存
            }
            for (int j = N - N % 4; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }//结尾几个元素串行计算
            A[k][k] = 1.0;
            //开始唤醒工作线程
            for (int t_id = 0; t_id < P; t_id++) {
                sem_post(&sem_workerstart[t_id]);
            }
            //主线程睡眠
            for (int t_id = 0; t_id < P; t_id++) {
                sem_wait(&sem_main);
            }
            //主线程再次唤醒工作线程
            for (int t_id = 0; t_id < P; t_id++) {
                sem_post(&sem_workerend[t_id]);
            }
        }
        for (int t_id = 0; t_id < P; t_id++) {
            pthread_join(handles[t_id], nullptr);
        }
        sem_destroy(&sem_main);
        for (int i = 0; i < P; i++) {
            sem_destroy(&sem_workerstart[i]);
            sem_destroy(&sem_workerend[i]);
        }
        gettimeofday(&t, NULL);
        time += 1000 * (t.tv_sec - h.tv_sec) + 0.001 * (t.tv_usec - h.tv_usec);
    }
    cout << "TIME: " << time / C << " ms" << endl;
}

int main()
{
    pthRowSSE();
}
