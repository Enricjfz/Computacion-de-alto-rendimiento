
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16

/*
*********************************************************************
function name: inicializarMatrizRandom
descripcion: inicializa aleatoriamente los elementos de una matriz
parametros:
		- M: puntero a la matriz a inicializar
		- m: numero de filas de A
		- n: numero de columnas de A
*********************************************************************
*/

void inicializarMatrizRandom (float *M, int m, int n){
    int i;
    for (i = 0; i < m*n; i++) {
	    M[i] = rand() % 10;
}
}
/*
*********************************************************************
funcion: gpu_matrix_mult
descripcion: producto de dos matrices sin caches en GPU (no necesariamente cuadradas)
parametros:
		- a,b,c: punteros a las matrices con las que operar en GPU
		- m: numero de filas de A
		- n: numero de columnas de A
		- k: numero de columnas de B
*********************************************************************
*/
__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if( col < k && row < m)
    {
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

/*
*********************************************************************
funcion: gpu_square_matrix_mult
descripcion: producto de dos matrices utilizando caches en GPU (matriz cuadrada)
parametros:
		- d_a,d_b,d_result: punteros a las matrices device con las que operar en GPU
		- n: numero de columnas de A (se presupone que la matriz sera cuadrada)
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(float *d_a, float *d_b, float *d_result, int n)
{
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float tmp = 0.0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub)
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {

            tile_a[threadIdx.y][threadIdx.x] = 0.0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0.0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}


/*
*********************************************************************
funcion: cpu_matrix_mult
descripcion: producto de dos matrices (no necesariamente cuadradas) en CPU.
parametros:
		- h_a,h_b,h_result: punteros a las matrices host con las que operar en CPU
		- m: numero de filas de A
		- n: numero de columnas de A
		- k: numero de columnas de B
*********************************************************************
*/
void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m, int n, int k) {

/*
*********************************************************************
Version IKJ (secuencial optimizada)
*********************************************************************
*/
 float r=0.0;
 int i;
 int j;
 int h;
    for (i=0; i< m; i++)
      for (j=0; j< k; j++)
        h_result[i*k+j] = 0;

    for (i=0; i<m; i++)
      for (h=0; h<n; h++) {
        r = h_a[i*n+h];
        for (j=0; j<k; j++)
	  h_result[i*k+j]+= r * h_b[h*k+j];
	}

/*
*********************************************************************
Version IJK (secuencial original)
*********************************************************************
*/
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            float tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
  int m, n, k;
  if(argc < 4 || argc > 4){
    fprintf(stderr,"Uso: filas matriz A, columnas matriz A, columnas matriz B\n");
    exit(-1);
  }
  // Recogemos los parámetros de ejecución relativos al tamaño de las matrices
  m = atoi(argv[1]);
  n = atoi(argv[2]);
  k = atoi(argv[3]);

  // Reservamos memoria para almacenar las matrices en host y su posterior multiplicación en CPU
  float *h_a, *h_b, *h_c, *h_cc;
  h_a=(float*)malloc(sizeof(float)*m*n);
  h_b=(float*)malloc(sizeof(float)*n*k);
  h_c=(float*)malloc(sizeof(float)*m*k);
  h_cc=(float*)malloc(sizeof(float)*m*k);

  // Inicializamos las matrices con valores decimales aleatorios
  inicializarMatrizRandom (h_a, m, n);
  inicializarMatrizRandom (h_b, n, k);


  // Creamos cudaEvents para medir los tiempos de ejeución en CPU y GPU.
  float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  // Reservamos memoria para almacenar las matrices en device y su posterior multiplicación en GPU
  float *d_a, *d_b, *d_c;
  cudaMalloc((void **) &d_a, sizeof(float)*m*n);
  cudaMalloc((void **) &d_b, sizeof(float)*n*k);
  cudaMalloc((void **) &d_c, sizeof(float)*m*k);


  cudaMemcpy(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);

  // Establecemos las dimensiones del Grid (Rejilla) y el Bloque
  unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  // If y else para elegir entre ejecución sin caches o con caches (memoria sin compartir o compartida)
  if(m == n && n == k){
      gpu_square_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
  }
  else{
      gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
  }

  cudaMemcpy(h_c, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Medimos el tiempo empleado en ejecutar la multiplicación en GPU
  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("Tiempo empleado en la Mult de Matrices %dx%d . %dx%d en GPU: %fs.\n\n", m, n, n, k, gpu_elapsed_time_ms/1000);


  cudaEventRecord(start, 0);
  cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Medimos el tiempo empleado en ejecutar la multiplicación en CPU
  cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
  printf("Tiempo empleado en la Mult de Matrices %dx%d . %dx%d en CPU: %fs.\n\n", m, n, n, k, cpu_elapsed_time_ms/1000);

  // Comprobamos que los resultados de la multiplicación son correctos
  int all_ok = 1;
  for (int i = 0; i < m; ++i){
      for (int j = 0; j < k; ++j){
         if(h_cc[i*k + j] != h_c[i*k + j])
            {
              all_ok = 0;
            }
        }
    }

  // Calculamos el speedup empleado respecto a la CPU
  if(all_ok){
      printf("CORRECTO!, Speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
  }
  else{
      printf("INCORRECTO\n");
  }


  // Liberamos la memoria reservada anteriormente
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  cudaFreeHost(h_cc);

  return 0;
}
