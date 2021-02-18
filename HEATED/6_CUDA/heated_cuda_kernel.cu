/*
  Authors: Juan Bernal Mencía, Enrique Jiménez Fernández
*/

#define M 500
#define N 500
#define NUM_ELEMENTS M * N
#define DIM_GRID 256
#define DIM_BLOCK 1024


// Función que permite copiar el contenido de la matriz d_w en d_u para iniciar el cálculo de la ecuación de calor
__global__ void copy_grid(double *d_w, double *d_u)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < M && y < N)
        d_u[x + y * N] = d_w[x + y * N];

    __syncthreads();

    return;
}

__device__ double d_epsilon;

__device__ double d_epsilon_reduction[NUM_ELEMENTS];

__device__ double d_epsilon_reduction_results[DIM_BLOCK];


// Función que permite llevar a cabo el cómputo de la tolerancia de error (diff) mediante el uso de fabs() y un stride
__global__ void epsilon_reduction(double *d_w, double *d_u)
{
    __shared__ double local_reduction[DIM_BLOCK];

    int stride = blockDim.x * gridDim.x;

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int local_index = threadIdx.x;

    local_reduction[local_index] = fabs(d_w[index] - d_u[index]); 

    if ((index + stride) < NUM_ELEMENTS && local_reduction[local_index] < fabs(d_w[index + stride] - d_u[index + stride]))
        local_reduction[local_index] = fabs(d_w[index + stride] - d_u[index + stride]);

    __syncthreads();

    for (int i = blockDim.x>>1; i>0; i>>=1)
    {
        if (local_index < i && local_reduction[local_index] < local_reduction[local_index + i])
            local_reduction[local_index] = local_reduction[local_index + i];

        __syncthreads();
    }

    if(local_index == 0) 
        d_epsilon_reduction_results[blockIdx.x] = local_reduction[local_index];

    return;
}

// Función que permite devolver los resultados del cálculo de la tolerancia de error en la variable device d_epsilon
__global__ void epsilon_reduction_results()
{
    __shared__ double local_reduction[DIM_BLOCK];

    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < blockDim.x)
    {
        local_reduction[index] = 0;
        __syncthreads();

        local_reduction[index] = d_epsilon_reduction_results[index];
        __syncthreads();

        for (int i = blockDim.x>>1; i>0; i>>=1)
        {
            if (index < i && local_reduction[index] < local_reduction[index + i])
                local_reduction[index] = local_reduction[index + i];

            __syncthreads();
        }

        if (index == 0)
            d_epsilon = local_reduction[index];
        __threadfence();
    }

    return;
}

// Función que ejecuta el cálculo de la solución de la ecuación de calor mediante el método de Jacobi en GPU
__global__ void calculate_solution(double *d_w, double *d_u)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x > 0 && y > 0 && x < M - 1 && y < N - 1)
    {
        int index = x + y * N;

        int west = (x - 1) + y * N;
        int east = (x + 1) + y * N;
        int north = x + (y - 1) * N;
        int south = x + (y + 1) * N;

        d_w[index] = (d_u[north] + d_u[south] + d_u[east] + d_u[west]) / 4.0;
    }

    __syncthreads();

    return;
}


void calculate_solution_kernel(double w[M][N],  double epsilon)
{
    double diff;
    int iterations;
    int iterations_print;
    float ElapsedTime;
    cudaEvent_t cudaStart, cudaStop;

    // Creación de evento para el cálculo del tiempo empleado
    cudaEventCreate(&cudaStart);
    cudaEventCreate(&cudaStop);
    
    const unsigned int matrix_mem_size = sizeof(double) * M * N;

    // Reservas de memoria y establecimiento de la variable d_w a través de la matriz host W
    double *d_w = (double *)malloc(matrix_mem_size);
    double *d_u = (double *)malloc(matrix_mem_size);
    
    cudaMalloc((void **)&d_w, matrix_mem_size);
    cudaMalloc((void **)&d_u, matrix_mem_size);
    
    cudaMemcpy(d_w, w, matrix_mem_size, cudaMemcpyHostToDevice);
    
    
    diff = epsilon;   

	// Inicialización del número de bloques y el número de hilos
    dim3 dimGrid(16, 16);  // 256 bloques
    dim3 dimBlock(32, 32); // 1024 hilos

    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");

    cudaEventRecord(cudaStart, 0);

     for (;;)
    {
      
      // Guardamos W en U 
      copy_grid<<<dimGrid, dimBlock>>>(d_w, d_u);
	
      // Calculamos la solución en d_W utilizando el método de Jacobi
      calculate_solution<<<dimGrid, dimBlock>>>(d_w, d_u);

      // Calculamos la tolerancia de Error (Diff) mediante fabs() y el uso de un stride 
      epsilon_reduction<<<DIM_GRID, DIM_BLOCK>>>(d_w, d_u);
	
      // Devolvemos los resultados en d_epsilon
      epsilon_reduction_results<<<DIM_GRID, DIM_BLOCK>>>();
	
      // Sincronizamos los Threads de ejecución
      cudaDeviceSynchronize();
 
      // Obtenemos los valores de la variable device "d_epsilon" en diff
      cudaMemcpyFromSymbol(&diff, d_epsilon, sizeof(double), 0, cudaMemcpyDeviceToHost);
	
      if(diff<=epsilon)break;
      iterations++;
      if (iterations == iterations_print)
      {	   
          printf("  %8d  %lg\n", iterations, diff);
          iterations_print = 2 * iterations_print;
      }
   }

    
    // Toma de tiempos
    cudaEventRecord(cudaStop, 0);
    cudaEventSynchronize(cudaStop);
    cudaEventElapsedTime(&ElapsedTime, cudaStart, cudaStop);

    printf("\n");
    printf("  %8d  %lg\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  GPU time = %f\n", ElapsedTime / 1000);

    // Devolución de los valores de la variable device y liberación de Memoria
    cudaMemcpy(w, d_w, matrix_mem_size, cudaMemcpyDeviceToHost);

    cudaFree(d_w);
    cudaFree(d_u);

    // Destrucción de los eventos creados
    cudaEventDestroy(cudaStart);
    cudaEventDestroy(cudaStop);
}
