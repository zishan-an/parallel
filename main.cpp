#include <iostream>
#include<sys/time.h>
#include<immintrin.h>
#include<x86intrin.h>
#define N 4096
using namespace std;
float** A;
void display(float A[N][N]) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            cout<<A[i][j]<<" ";
        }
        cout<<endl;
    }
}//check outcome
void init0(float** &A) {
    cout<<"Initializing ...\n";
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            A[i][j] = rand();
        }
    }
}
void init(float** &A) {
    cout<<"Initializing ...\n";
    A = new float*[N];
    for(int i = 0; i < N; i++) {
        *(A + i) = (float*)_aligned_malloc(N*sizeof(float), 32);
    }
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            A[i][j] = rand();
        }
    }//initialize
}
void checkAlgn(float** A) {
    cout<<&A[0][0]<<endl;
}//check alignment
void normal(float** A) {
    struct timeval h, t;
    checkAlgn(A);
    //Normal
    cout<<"Normal:(N = "<<N<<")\n";
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
    cout<<"TIME: "<<(1000*(t.tv_sec - h.tv_sec) + 0.001*(t.tv_usec - h.tv_usec))<<" ms"<<endl;
}

//SSE第一部分并行（不对齐）
void sseY1(float** &A) {
    struct timeval h, t;
    checkAlgn(A);
    //SSE
    cout<<"SSEY1 Para:(N = "<<N<<")\n";
    gettimeofday(&h, NULL);
    for(int k = 0; k < N; k++) {
        float* akk = new float[4];
        *akk = *(akk+1) = *(akk+2) = *(akk+3) = A[k][k];
        __m128 vt = _mm_load_ps(akk);
        delete[] akk;
        for(int j = k + 1; j + 3 < N; j += 4) {
            __m128 va = _mm_loadu_ps(&A[k][j]);
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&A[k][j], va);
        }
        for(int j = N - N%4; j < N; j++) {
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
    cout<<"TIME: "<<(1000*(t.tv_sec - h.tv_sec) + 0.001*(t.tv_usec - h.tv_usec))<<" ms"<<endl;
}

//SSE第二部分并行（不对齐）
void sseY2(float** &A) {
    struct timeval h, t;
    checkAlgn(A);
    //SSE
    cout<<"SSEY2 Para:(N = "<<N<<")\n";
    gettimeofday(&h, NULL);
    for(int k = 0; k < N; k++) {
        for(int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1; i < N; i++) {
            float* aik = new float[4];
            *aik = *(aik+1) = *(aik+2) = *(aik+3) = A[i][k];
            __m128 vaik = _mm_load_ps(aik);
            delete[] aik;
            for(int j = k + 1; j + 3 < N; j += 4) {
                __m128 vaij = _mm_loadu_ps(&A[i][j]);
                __m128 vakj = _mm_loadu_ps(&A[k][j]);
                __m128 vx = _mm_mul_ps(vaik, vakj);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&A[i][j], vaij);
            }
            for(int j = N - N%4; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            A[i][k] = 0;
        }
    }
    gettimeofday(&t, NULL);
    cout<<"TIME: "<<(1000*(t.tv_sec - h.tv_sec) + 0.001*(t.tv_usec - h.tv_usec))<<" ms"<<endl;
}

//SSE全部并行（不对齐）
void sse(float** &A) {
    struct timeval h, t;
    checkAlgn(A);
    //SSE
    cout<<"SSE Para:(N = "<<N<<")\n";
    gettimeofday(&h, NULL);
    for(int k = 0; k < N; k++) {
        float* akk = new float[4];
        *akk = *(akk+1) = *(akk+2) = *(akk+3) = A[k][k];
        __m128 vt = _mm_load_ps(akk);
        delete[] akk;
        for(int j = k + 1; j + 3 < N; j += 4) {
            __m128 va = _mm_loadu_ps(&A[k][j]);
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&A[k][j], va);
        }
        for(int j = N - N%4; j < N; j++) {
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1; i < N; i++) {
            float* aik = new float[4];
            *aik = *(aik+1) = *(aik+2) = *(aik+3) = A[i][k];
            __m128 vaik = _mm_load_ps(aik);
            delete[] aik;
            for(int j = k + 1; j + 3 < N; j += 4) {
                __m128 vaij = _mm_loadu_ps(&A[i][j]);
                __m128 vakj = _mm_loadu_ps(&A[k][j]);
                __m128 vx = _mm_mul_ps(vaik, vakj);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&A[i][j], vaij);
            }
            for(int j = N - N%4; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            A[i][k] = 0;
        }
    }
    gettimeofday(&t, NULL);
    cout<<"TIME: "<<(1000*(t.tv_sec - h.tv_sec) + 0.001*(t.tv_usec - h.tv_usec))<<" ms"<<endl;
}
//SSE全部并行（对齐）
void sseAlgn(float** &A) {
    struct timeval h, t;
    checkAlgn(A);
    //SSE
    cout<<"SSE Para Aligned:(N = "<<N<<")\n";
    gettimeofday(&h, NULL);
    for(int k = 0; k < N; k++) {
        float* akk = new float[4];
        *akk = *(akk+1) = *(akk+2) = *(akk+3) = A[k][k];
        __m128 vt = _mm_load_ps(akk);
        delete[] akk;
        for(int j = k + 1; j < k + 4 - k%4; j++) {
            A[k][j] = A[k][j]/A[k][k];
        }
        for(int j = k + 4 - k%4; j + 3 < N; j += 4) {
            __m128 va = _mm_load_ps(&A[k][j]);
            va = _mm_div_ps(va, vt);
            _mm_store_ps(&A[k][j], va);
        }
        for(int j = N - N%4; j < N; j++) {
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1; i < N; i++) {
            float* aik = new float[4];
            *aik = *(aik+1) = *(aik+2) = *(aik+3) = A[i][k];
            __m128 vaik = _mm_load_ps(aik);
            delete[] aik;
            for(int j = k + 1; j < k + 4 - k%4; j++) {
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            for(int j = k + 4 - k%4; j + 3 < N; j += 4) {
                __m128 vaij = _mm_load_ps(&A[i][j]);
                __m128 vakj = _mm_load_ps(&A[k][j]);
                __m128 vx = _mm_mul_ps(vaik, vakj);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_store_ps(&A[i][j], vaij);
            }
            for(int j = N - N%4; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            A[i][k] = 0;
        }
    }
    gettimeofday(&t, NULL);
    cout<<"TIME: "<<(1000*(t.tv_sec - h.tv_sec) + 0.001*(t.tv_usec - h.tv_usec))<<" ms"<<endl;
}

//AVX第一部分并行（不对齐）
void avxY1(float** &A) {
    struct timeval h, t;
    checkAlgn(A);
    //AVX
    cout<<"AVXY1 Para:(N = "<<N<<")\n";
    gettimeofday(&h, NULL);
    for(int k = 0; k < N; k++) {
        float* akk = new float[8];
        *akk = *(akk+1) = *(akk+2) = *(akk+3) = *(akk+4) = *(akk+5) = *(akk+6) = *(akk+7) = A[k][k];
        __m256 vt = __builtin_ia32_loadups256(akk);
        delete[] akk;
        for(int j = k + 1; j + 7 < N; j += 8) {
            __m256 va = __builtin_ia32_loadups256(&A[k][j]);
            va = _mm256_div_ps(va, vt);
            __builtin_ia32_storeups256(&A[k][j], va);
        }
        for(int j = N - N%8; j < N; j++) {
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
    cout<<"TIME: "<<(1000*(t.tv_sec - h.tv_sec) + 0.001*(t.tv_usec - h.tv_usec))<<" ms"<<endl;
}

//AVX第二部分并行（不对齐）
void avxY2(float** &A) {
    struct timeval h, t;
    checkAlgn(A);
    //AVX
    cout<<"AVXY2 Para:(N = "<<N<<")\n";
    gettimeofday(&h, NULL);
    for(int k = 0; k < N; k++) {
        for(int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1; i < N; i++) {
            float* aik = new float[8];
            *aik = *(aik+1) = *(aik+2) = *(aik+3) = *(aik+4) = *(aik+5) = *(aik+6) = *(aik+7) = A[i][k];
            __m256 vaik = __builtin_ia32_loadups256(aik);
            delete[] aik;
            for(int j = k + 1; j + 7 < N; j += 8) {
                __m256 vaij = __builtin_ia32_loadups256(&A[i][j]);
                __m256 vakj = __builtin_ia32_loadups256(&A[k][j]);
                __m256 vx = __builtin_ia32_mulps256(vaik, vakj);
                vaij = __builtin_ia32_subps256(vaij, vx);
                __builtin_ia32_storeups256(&A[i][j], vaij);
            }
            for(int j = N - N%8; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            A[i][k] = 0;
        }
    }
    gettimeofday(&t, NULL);
    cout<<"TIME: "<<(1000*(t.tv_sec - h.tv_sec) + 0.001*(t.tv_usec - h.tv_usec))<<" ms"<<endl;
}

//AVX全部并行（不对齐）
void avx(float** &A) {
    struct timeval h, t;
    checkAlgn(A);
    //AVX
    cout<<"AVX Para:(N = "<<N<<")\n";
    gettimeofday(&h, NULL);
    for(int k = 0; k < N; k++) {
        float* akk = new float[8];
        *akk = *(akk+1) = *(akk+2) = *(akk+3) = *(akk+4) = *(akk+5) = *(akk+6) = *(akk+7) = A[k][k];
        __m256 vt = __builtin_ia32_loadups256(akk);
        delete[] akk;
        for(int j = k + 1; j + 7 < N; j += 8) {
            __m256 va = __builtin_ia32_loadups256(&A[k][j]);
            va = _mm256_div_ps(va, vt);
            __builtin_ia32_storeups256(&A[k][j], va);
        }
        for(int j = N - N%8; j < N; j++) {
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1; i < N; i++) {
            float* aik = new float[8];
            *aik = *(aik+1) = *(aik+2) = *(aik+3) = *(aik+4) = *(aik+5) = *(aik+6) = *(aik+7) = A[i][k];
            __m256 vaik = __builtin_ia32_loadups256(aik);
            delete[] aik;
            for(int j = k + 1; j + 7 < N; j += 8) {
                __m256 vaij = __builtin_ia32_loadups256(&A[i][j]);
                __m256 vakj = __builtin_ia32_loadups256(&A[k][j]);
                __m256 vx = __builtin_ia32_mulps256(vaik, vakj);
                vaij = __builtin_ia32_subps256(vaij, vx);
                __builtin_ia32_storeups256(&A[i][j], vaij);
            }
            for(int j = N - N%8; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            A[i][k] = 0;
        }
    }
    gettimeofday(&t, NULL);
    cout<<"TIME: "<<(1000*(t.tv_sec - h.tv_sec) + 0.001*(t.tv_usec - h.tv_usec))<<" ms"<<endl;
}
//AVX全部并行（对齐）
void avxAlgn(float** &A) {
    struct timeval h, t;
    checkAlgn(A);
    //AVX
    cout<<"AVX Para Aligned:(N = "<<N<<")\n";
    gettimeofday(&h, NULL);
    for(int k = 0; k < N; k++) {
        float* akk = new float[8];
        *akk = *(akk+1) = *(akk+2) = *(akk+3) = *(akk+4) = *(akk+5) = *(akk+6) = *(akk+7) = A[k][k];
        __m256 vt = __builtin_ia32_loadups256(akk);
        delete[] akk;
        for(int j = k + 1; j < k + 8 - k%8; j++) {
            A[k][j] = A[k][j]/A[k][k];
        }
        for(int j = k + 8 - k%8; j + 7 < N; j += 8) {
            __m256 va = __builtin_ia32_loadups256(&A[k][j]);
            va = __builtin_ia32_divps256(va, vt);
            __builtin_ia32_storeups256(&A[k][j], va);
        }
        for(int j = N - N%8; j < N; j++) {
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k + 1; i < N; i++) {
            float* aik = new float[8];
            *aik = *(aik+1) = *(aik+2) = *(aik+3) = *(aik+4) = *(aik+5) = *(aik+6) = *(aik+7) = A[i][k];
            __m256 vaik = __builtin_ia32_loadups256(aik);
            delete[] aik;
            for(int j = k + 1; j < k + 8 - k%8; j++) {
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            for(int j = k + 8 - k%8; j + 7 < N; j += 8) {
                __m256 vaij = __builtin_ia32_loadups256(&A[i][j]);
                __m256 vakj = __builtin_ia32_loadups256(&A[k][j]);
                __m256 vx = __builtin_ia32_mulps256(vaik, vakj);
                vaij = __builtin_ia32_subps256(vaij, vx);
                __builtin_ia32_storeups256(&A[i][j], vaij);
            }
            for(int j = N - N%8; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            A[i][k] = 0;
        }
    }
    gettimeofday(&t, NULL);
    cout<<"TIME: "<<(1000*(t.tv_sec - h.tv_sec) + 0.001*(t.tv_usec - h.tv_usec))<<" ms"<<endl;
}

int main()
{
    init(A);
    normal(A);
    init0(A);
    sseY1(A);
    init0(A);
    sseY2(A);
    init0(A);
    sse(A);
    init0(A);
    sseAlgn(A);
    init0(A);
    avxY1(A);
    init0(A);
    avxY2(A);
    init0(A);
    avx(A);
    init0(A);
    avxAlgn(A);
    return 0;
}
