#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include "mpi.h"
#include <omp.h>

#define BACK_SIZE N2+1
#define go_SIZE 1

#define WORKING 100
#define END 10
#define MAGIC 255

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
void inicializarMatrizRandom (float **M, int m, int n){
    int i;
    int j;
    for (i = 0; i < m; i++) {
	for (j=0;j<n;j++)
	    M[i][j] = rand() % 10;
    }
}

/*
**********************************************************************************************
Funciones utilizadas para la reserva de memoria de las matrices a utilizar a través de calloc()
**********************************************************************************************
*/
void *xcalloc (size_t nmemb, size_t size){
	void * mem = calloc(nmemb,size);
	assert(mem);
	return mem;
}

void * MATRIX1D(int T, int I){
	return xcalloc(I,T);
}

void Free_M1D(void * M){
	free(M);
}

void * MATRIX2D(int T, int I, int J){
	void ** M2D;
	int i;
	M2D=xcalloc(I,sizeof(void*));
	M2D[0]=MATRIX1D(T,I*J);
	for (i=1;i<I;i++)
		M2D[i]=((void*)(M2D[i-1])) + J*T;
	return M2D;
}

void Free_M2D(void * M){
	Free_M1D(((void**)M)[0]);
	free(M);
}

int main(int argc, char **argv){
	float ** matA;
	float ** matB;
	float ** matC;
	float ** matSubA;
	float ** matSubC;
	int n_processes;
	int rank;
	int root=0;
	double start,finish;
	int begin,end;
	int n_rows;
	float result;
	int M1,N1,M2,N2;
	int i, j, k;
	
	// Inicialización de MPI y determinación del número de procesos e identificador de estos
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	
	// En caso de introducir correctamente los parámetros, recogemos los valores y los guardamos en variables
	if (argc <4) {
		if (rank==0)
			printf("Uso: Ejecutable n_filas(A) n_columnas(A) n_columnas(B)\n");
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(-1);
    }
	else{
		M1=atoi(argv[1]);
		N1=atoi(argv[2]);
		N2=atoi(argv[3]); // Numero de columnas de B
		M2=N1;		  // Numero de filas de B == numero de columnas de A
	}
	
	if (rank==0)
		printf("Mat producto %dx%d X %dx%d\n", M1,N1,M2,N2);
	
	// Redondeo del numero de filas a repartir en base al numero de procesos disponibles
	if(M1%n_processes==0)
		n_rows=M1/n_processes;
	else n_rows=M1/n_processes+1;
	
	begin=n_rows*rank;
	end=begin+n_rows;
	if (end>M1)
		end=M1;
	´
	// Reserva de memoria para las matrices e inicializacion a valores decimales aleatorios
	if (rank==0){
		matA=MATRIX2D(sizeof(float),n_rows*n_processes,N1); // M1*N1
		matB=MATRIX2D(sizeof(float),M2,N2);
		matC=MATRIX2D(sizeof(float),n_rows*n_processes,N2); // M1*N2
		
		inicializarMatrizRandom (matA, M1, N1);
		inicializarMatrizRandom (matB, M2, N2);
	}
	else{
		matA=MATRIX2D(sizeof(float),1,1); // M1*N1
		matB=MATRIX2D(sizeof(float),M2,N2);
		matC=MATRIX2D(sizeof(float),1,1); // M1*N2
	}
	
	// Establecemos un "cronómetro" para medir el tiempo de ejecución
	start=MPI_Wtime();
	
	// Realizamos el Broadcast y el Scatter a los diferentes procesos
	MPI_Bcast(&matB[0][0],M2*N2,MPI_FLOAT,root,MPI_COMM_WORLD);
	
	matSubA=MATRIX2D(sizeof(float),n_rows,N1);
	matSubC=MATRIX2D(sizeof(float),n_rows,N2);
	
	MPI_Scatter(&matA[0][0], n_rows*N1, MPI_FLOAT, &matSubA[0][0], n_rows*N1, MPI_FLOAT,root,MPI_COMM_WORLD);
	
	// Realizamos la multiplicación de matrices paralelizada con openmp utilizando la estrategia IKJ
	float r;

     for (i=0; i< n_rows; i++)
       	for (j=0; j< N2; j++)
          matSubC[i][j] = 0;
 #pragma omp parallel shared (n_rows,N1,N2)
 {
    #pragma omp for simd private(i,k,j,r)	
     for (i=0; i<n_rows; i++){
        for (k=0; k<N1; k++) {
          r = matSubA[i][k];
          for (j=0; j<N2; j++) 
               matSubC[i][j]+= r * matB[k][j];
        }
       }
}

	// Esperamos la finalización de tosos los procesos y realizamos el Gather de estos
	printf("Worker %d ends: de %d a %d\n",rank,begin,end);
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Gather(&matSubC[0][0], n_rows*N2, MPI_FLOAT, &matC[0][0], n_rows*N2, MPI_FLOAT, root, MPI_COMM_WORLD);
	finish=MPI_Wtime();
	MPI_Finalize();
	
	// Comprobamos los resultados obtenidos en la multiplicación 
	if (rank==0){
	   int wrong=0;
	printf("Total time was %f seconds\n", finish-start);
	for (i=0;i<M1;i++){
	  for (j=0; j< N2; j++){
	  result=0.0;
	    for (k=0;k<N1;k++){
		result+=matA[i][k]*matB[k][j];
		}
	if (matC[i][j]!=result)
		wrong=1;
	}
      }
	if (wrong)
		printf("ERROR EN LA MULTIPLICACION\n");
	else printf("CORRECTO!\n");
  }
	return 0;
}

	
