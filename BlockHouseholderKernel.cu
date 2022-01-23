// CUDA rhypot function!!

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#include "BlockHouseholderKernel.h"

#define BLOCKDIM_X_HOUSE 32
#define BLOCKDIM_X_SUMSQUARE 512
#define BLOCKDIM_X_VPRIME 128
#define BLOCKDIM_X_ADD 32
#define BLOCKDIM_Y_ADD 8
#define BLOCKDIM_X_ADD_SEQ 32
#define BLOCKDIM_X_CALC_YPRIME 256
#define BLOCKDIM_X_WPRIME 32
#define BLOCKDIM_X_RADD 16
#define BLOCKDIM_Y_RADD 16
#define BLOCKDIM_X_CALCW 32
#define BLOCKDIM_X_COPYW 32
#define TILE_SIZE 16
#define TILE_SIZE_W 16
#define TILE_SIZE_A 16
#define TILE_SIZE_VERT 16
#define TILE_SIZE_ADD 16

#define WPRIME_VERT 16
#define WPRIME_N 32

#define POS(r, c, ld) ((c)*(ld) + (r))


__global__ void sumSquares(double *A, int start, int end, double *result) {
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ double s[BLOCKDIM_X_SUMSQUARE];

    s[x] = 0;
    for (unsigned int i = start + x; i < end; i += dimX) {
        s[x] += A[i]*A[i];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 0; bound /= 2) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x == 0) {
        *result = s[0];
    }
}

__global__ void house_internal(double *A, int m, int ld, int col, double *V, double *magX2) {
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    if (x < col) V[x] = 0;
    else if (x == col) V[x] = A[POS(x, col, ld)] - sqrt(*magX2);
    else if (x > col and x < m)
        V[x] = A[POS(x, col, ld)];
}

__global__ void calc_beta(double *magX2, double *oldVal, double *beta) {
    double newVal = *oldVal - sqrt(*magX2);
    double magV2 = *magX2 - (*oldVal)*(*oldVal) + newVal*newVal;
    *beta = 2/magV2;
}

void house(double *A, int m, int ld, int col, double *V, double *magX2, double *beta) {
    sumSquares<<<1, BLOCKDIM_X_SUMSQUARE>>>(A, POS(col, col, ld), POS(m, col, ld), magX2);
    calc_beta<<<1, 1>>>(magX2, A + POS(col, col, ld), beta);
    int blockxdim =  min(m, BLOCKDIM_X_HOUSE);
    house_internal<<<(m+blockxdim-1)/blockxdim, blockxdim>>>(A, m, ld, col, V, magX2);
}

__global__ void calc_Vprime(double *A, int m, int ld, int n, double *V, double startc, double *Vprime) {
    int col = blockIdx.x + startc;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ double s[BLOCKDIM_X_VPRIME];

    s[x] = 0;
    for (unsigned int i = startc + x; i < m; i += dimX) {
        s[x] += V[i]*A[POS(i, col, ld)];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 0; bound /= 2) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x == 0) {
        Vprime[col] = s[0];
    }
}

__global__ void add_VVprime(double *A, int m, int ld, int n, double *V, int startc, double *Vprime, double *beta) {
    int r = threadIdx.x + blockDim.x*blockIdx.x + startc;
    int c = threadIdx.y + blockDim.y*blockIdx.y + startc;
    if (r < m && c < n)
        A[POS(r, c, ld)] -= (*beta)*V[r]*Vprime[c];
}

__global__ void add_VVprime_seq(double *A, int m, int ld, int n, double *V, int startc, double *Vprime, double *beta) {
    int r = threadIdx.x + blockDim.x*blockIdx.x + startc;
    int rStep = blockDim.x*gridDim.x;
    int c = threadIdx.y + blockDim.y*blockIdx.y + startc;
    if (c < n) {
        for (int ri = r; ri < m; ri += rStep) {
            A[POS(ri, c, ld)] -= (*beta)*V[ri]*Vprime[c];
        }
    }
}

void rankOneUpdate(double *A, int m, int ld, int n, double *V, double *beta, int startc, double *Vprime) {
    calc_Vprime<<<n-startc, BLOCKDIM_X_VPRIME>>>(A, m, ld, n, V, startc, Vprime);
    int blockdimx = min((m - startc), BLOCKDIM_X_ADD);
    int blockdimy = min((n - startc), BLOCKDIM_Y_ADD);
    add_VVprime<<<dim3((m-startc+blockdimx-1)/blockdimx, (n - startc+blockdimy-1)/blockdimy, 1), dim3(blockdimx, blockdimy, 1)>>>(A, m, ld, n, V, startc, Vprime, beta);
    // int blockdimx = min((m - startc), BLOCKDIM_X_ADD_SEQ);
    // add_VVprime1_seq<<<dim3(1, n - startc, 1), dim3(blockdimx, 1, 1)>>>(A, m, ld, n, V, startc, Vprime, beta);
}

__global__ void calc_Wprime(double *A, int m, int ld, int n, int startc, int startn, int R, double *W, double *Wprime) {
    int col = blockIdx.x + startn;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    int j = threadIdx.y + blockIdx.y*blockDim.y;

    __shared__ double s[];

    s[x] = 0;
    for (unsigned int i = x + startc; i < m; i += dimX) {
        s[x] += W[POS(i, j, m)]*A[POS(i, col, ld)];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 0; bound /= 2) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x == 0) {
        Wprime[POS(col, j, n)] = s[0];
    }
}

__global__ void calc_Wprime2(double *A, int m, int ld, int n, int startc, int startn, int R, double *W, double *Wprime) {
    int bx = blockIdx.x; // n
    int by = blockIdx.y; // R
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ double Ws[TILE_SIZE][TILE_SIZE];
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    double Wprimesub = 0;

    for (int b = 0; b < (m+TILE_SIZE-1-startc)/TILE_SIZE; ++b) {
        if (b*TILE_SIZE+tx+startc < m and ty + by*TILE_SIZE < R) {
            Ws[ty][tx] = W[POS(b*TILE_SIZE+tx+startc, ty + by*TILE_SIZE, m)];
        } else {
            Ws[ty][tx] = 0;
        }
        if (b*TILE_SIZE+ty+startc < m and bx*TILE_SIZE+startn+tx < n) {
            As[ty][tx] = A[POS(b*TILE_SIZE+ty+startc, bx*TILE_SIZE+startn+tx, m)];
        } else {
            As[ty][tx] = 0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            Wprimesub += Ws[tx][k]*As[ty][k];
        }
        __syncthreads();
    }
    if (ty + by*TILE_SIZE < R and bx*TILE_SIZE+startn+tx < n) {
        Wprime[POS(bx*TILE_SIZE+startn+tx, ty + by*TILE_SIZE, n)] = Wprimesub;
    }
}

// __global__ void calc_Wprime3(double *A, int m, int ld, int n, int startc, int startn, int R, double *W, double *Wprime) {
//     int y = threadIdx.x + startc;
//     int stride = WPRIME_VERT;
//     int nx = threadIdx.y + blockIdx.y*blockDim.y + startn;
//     int Rx = threadIdx.y;

//     __shared__ double Ws[WPRIME_N][WPRIME_VERT];
//     __shared__ double As[WPRIME_N][WPRIME_VERT];
//     __shared__ double Wprimes[WPRIME_N][WPRIME_N];


//     for (; y < m; y += stride) {
//         Ws[Rx][threadIdx.x] = W[POS(y, Rx, m)];
//         As[threadIdx.y][threadIdx.x] = A[POS(y, threadIdx.y, m)];
//         __syncthreads();
//         for (int i = 0; i < stride; ++i) {
//             Wprimes[threadIdx.x][threadIdx.y] += Ws[threadIdx.x][i]*As[threadIdx.y][i];
//             Wprimes[threadIdx.x+16][threadIdx.y] += Ws[threadIdx.x+16][i]*As[threadIdx.y][i];
//         }
//     }
//     Wprime[POS(nx, Rx, n)] = Wprimes[threadIdx.x][threadIdx.y];
//     Wprime[POS(nx, Rx, n)] = Wprimes[threadIdx.x][threadIdx.y];
// }


__global__ void calc_Wprime_dumb(double *A, int m, int ld, int n, int startc, int startn, int R, double *W, double *Wprime) {
    Wprime[POS(blockIdx.x, blockIdx.y, n)] = 0;
    for (unsigned int i = 0; i < m; i += 1) {
        Wprime[POS(blockIdx.x, blockIdx.y, n)] += W[POS(i, blockIdx.y, m)]*A[POS(i, blockIdx.x, ld)];;

    }
}


__global__ void add_YWprime(double *A, int m, int ld, int n, int startc, int startn, int R, double *Y, double *Wprime) {
    int r = threadIdx.x + blockDim.x*blockIdx.x + startc;
    int c = threadIdx.y + blockDim.y*blockIdx.y + startn;

    if (r < m && c < n) {
        double addVal = 0;
        for (int i = 0; i < R; ++i) {
            addVal += Y[POS(r, i, m)]*Wprime[POS(c, i, n)];
        }
        A[POS(r, c, ld)] += addVal;
    }
}

// __global__ void add_YWprime2(double *A, int m, int ld, int n, int startc, int startn, int R, double *Y, double *Wprime) {
//     // int r = threadIdx.x + blockDim.x*blockIdx.x + startc;
//     // int c = threadIdx.y + blockDim.y*blockIdx.y + startn;
//     // if (r < m && c < n) {
//     //     double addVal = 0;
//     //     for (int i = 0; i < R; ++i) {
//     //         addVal += Y[POS(r, i, m)]*Wprime[POS(c, i, n)];
//     //     }
//     //     A[POS(r, c, ld)] += addVal;
//     // }

//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     int r = tx+bx*TILE_SIZE_ADD;
//     int c = ty+by*TILE_SIZE_ADD;

//     __shared__ double Ys[TILE_SIZE_ADD][TILE_SIZE_ADD];
//     __shared__ double Wprimes[TILE_SIZE_ADD][TILE_SIZE_ADD];
//     double Asub = 0;

//     for (int b = 0; b < R/TILE_SIZE_ADD; ++b) {
//         Ys[ty][tx] = Y[POS(r,b*TILE_SIZE_ADD+tx,m)];
//         Wprimes[ty][tx] = Wprime[POS(c,b*TILE_SIZE_ADD+tx,n)];
//         __syncthreads();

//         for (int k = 0; k < TILE_SIZE_ADD; ++k) {
//             Asub += Ys[ty][k]*Wprimes[k][tx];
//         }
//         __syncthreads();
//     }
//     A[POS(r, c, m)] += Asub;
// }

void rankRUpdate(double *A, int m, int ld, int n, int startc, int startn, int R, double *Y, double *W, double *Wprime) {
    // if (n - startn < 1200) {
    calc_Wprime<<<dim3(n-startn, R, 1), dim3(BLOCKDIM_X_WPRIME, 1, 1)>>>(A, m, ld, n, startc, startn, R, W, Wprime);
    // calc_Wprime_dumb<<<dim3(n, R, 1), dim3(1, 1, 1)>>>(A, m, ld, n, startc, startn, R, W, Wprime);
    // } else {
    // calc_Wprime2<<<dim3((n-startn+TILE_SIZE-1)/TILE_SIZE, (R+TILE_SIZE-1)/TILE_SIZE, 1), dim3(TILE_SIZE, TILE_SIZE, 1)>>>(A, m, ld, n, startc, startn, R, W, Wprime);
    // }
    // calc_Wprime3<<<dim3(1,(n-startn)/WPRIME_N, 1), dim3(WPRIME_VERT, WPRIME_N, 1)>>>(A, m, ld, n, startc, startn, R, W, Wprime);

    int blockdimx = min((m - startc), BLOCKDIM_X_RADD);
    int blockdimy = min((n - startn), BLOCKDIM_Y_RADD);
    add_YWprime<<<dim3((m-startc+blockdimx-1)/blockdimx, (n - startn + blockdimy-1)/blockdimy, 1), dim3(blockdimx, blockdimy, 1)>>>(A, m, ld, n, startc, startn, R, Y, Wprime);
    // add_YWprime2<<<dim3((m-startc+TILE_SIZE_ADD-1)/TILE_SIZE_ADD, (n - startn + TILE_SIZE_ADD - 1)/TILE_SIZE_ADD, 1), dim3(TILE_SIZE_ADD, TILE_SIZE_ADD, 1)>>>(A, m, ld, n, startc, startn, R, Y, Wprime);
}


__global__ void calc_Yprime(int m, int startc, double *Y, double *V, double *Yprime) {
    int j = blockIdx.x;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ double s[BLOCKDIM_X_CALC_YPRIME];

    s[x] = 0;
    for (unsigned int i = startc + x; i < m; i += dimX) {
        s[x] += V[i]*Y[POS(i, j, m)];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 0; bound /= 2) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x == 0) {
        Yprime[j] = s[0];
    }
}

__global__ void calc_W(int m, int col, int startc, double *W, double *Y, double *Yprime, double *beta) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    // if (x < startc + col) {
        // W[POS(x, col, m)] = 0;
    // }
    if (x < m) {
        double z = 0;
        for (int i = 0; i < col; ++i) {
            z += W[POS(x, i, m)]*Yprime[i];
        }
        z += Y[POS(x, col, m)];
        z *= -(*beta);
        W[POS(x, col, m)] = z;
    }
}

__global__ void copy_W(int m, int startc, double *Y, double *W, double *beta) {
    int x = threadIdx.x + blockIdx.x*blockDim.x + startc;
    if (x < m) {
        W[x] = -(*beta)*Y[x];
    }
}

void append_W(int m, int col, int startc, double *Y, double *W, double *Wprime, double *beta) {
    if (col == 0) {
        int blockdim = min(BLOCKDIM_X_COPYW, m - startc);
        copy_W<<<(m-startc+blockdim-1)/blockdim, blockdim>>>(m ,startc, Y, W, beta);
    } else {
        double *V = Y + POS(0, col, m);
        calc_Yprime<<<col, BLOCKDIM_X_CALC_YPRIME>>>(m, startc, Y, V, Wprime);
        int blockdim = min(BLOCKDIM_X_CALCW, m);
        calc_W<<<(m + blockdim-1)/blockdim, blockdim>>>(m, col, startc, W, Y, Wprime, beta);
    }
}

__global__ void printY(double *A, int m, int ld, int n, int R, double *Y, double *W) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    A[POS(x, y, ld)] = W[POS(x, y, m)];
}

__global__ void printbeta(double *A, double *beta) {
    *A = *beta;
}

void doOneBlock(double *A, int m, int ld, int n, int R, int col, double *Y, double *W, double *Wprime, double *beta, double *magX2) {
    for (int i = 0; i < R; ++i) {
        int curcol = col + i;
        double *curV = Y + POS(0, i, m);
        house(A, m, ld, curcol, curV, magX2, beta);
        rankOneUpdate(A, m, ld, col + R, curV, beta, curcol, Wprime);
        append_W(m, i, curcol, Y, W, Wprime, beta);
    }
    rankRUpdate(A, m, ld, n, col, col+R, R, Y, W, Wprime);
    // printY<<<dim3(m, R), dim3(1,1,1)>>>(A, m, ld, n, R, Y, W);
    // printbeta<<<1, 1>>>(A, beta);
}

// __global__ void zeroLowerTriangular(double *A, int rows, int cols, int cols_extended) {
//     int x = threadIdx.x;
//     int dimX = blockDim.x;
//     int blockX = blockIdx.x;
//     int numBlocks = gridDim.x;
//     for (int r = 0; r < rows; ++r) {
//         for (int c = x; c < cols && c < r; c += dimX) {
//             A[POS(rows, cols_extended, r, c)] = 0;
//         }
//     }
// }

void QRBlockSolve(double *A, int m, int na, int nb, int ld, int R) {
    double *magX2;
    double *beta;
    double *Y;
    double *W;
    double *Wprime;

    if (cudaMalloc((void**)&magX2, sizeof(magX2)) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&beta, sizeof(beta)) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&Y, sizeof(Y)*R*m) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&W, sizeof(W)*R*m) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&Wprime, sizeof(Wprime)*R*(na+nb)) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }

    for (int col = 0; col < na; col += R) {
        int curR = min(R, na - col);
        doOneBlock(A, m, ld, (na+nb), curR, col, Y, W, Wprime, beta, magX2);
    }

    // MagX2<<<dim3(1), dim3(BLOCKDIM_X)>>>(A, rows, 0, cols_extended, magX2);
    // for (int i = 0; i < iters; ++i) {
    //     house<<<dim3(1), dim3(cols_extended-i)>>>(A, rows, cols_extended, i, magX2);
    // }
    // zeroLowerTriangular<<<dim3(1), dim3(BLOCKDIM_X)>>>(A, rows, cols, cols_extended);
    // cudaFree(magX2);
}



__global__ void copy(double *target, double *dest, int num) {
    int x = threadIdx.x;
    int dimX = blockDim.x;
    int blockX = blockIdx.x;
    int numBlocks = gridDim.x;

    for (int i = x + dimX*blockX; i < num; i += dimX*numBlocks) {
        if (i < num) {
            target[i] = dest[i];
        }
    }
}
