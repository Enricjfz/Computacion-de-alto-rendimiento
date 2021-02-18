# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <sys/time.h>
# include <omp.h>
# include "mpi.h"





//****************************************************************************
int main ( int argc, char *argv[] )
//****************************************************************************
//
//  Purpose:
//
//    MAIN is the main program for HEATED_PLATE.
//
//  Discussion:
//
//    This code solves the steady state heat equation on a rectangular region.
//
//    The sequential version of this program needs approximately
//    18/epsilon iterations to complete.
//
//
//    The physical region, and the boundary conditions, are suggested
//    by this diagram;
//
//                   W = 0
//             +------------------+
//             |                  |
//    W = 100  |                  | W = 100
//             |                  |
//             +------------------+
//                   W = 100
//
//    The region is covered with a grid of M by N nodes, and an N by N
//    array W is used to record the temperature.  The correspondence between
//    array indices and locations in the region is suggested by giving the
//    indices of the four corners:
//
//                  I = 0
//          [0][0]-------------[0][N-1]
//             |                  |
//      J = 0  |                  |  J = N-1
//             |                  |
//        [M-1][0]-----------[M-1][N-1]
//                  I = M-1
//
//    The steady state solution to the discrete heat equation satisfies the
//    following condition at an interior grid point:
//
//      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )
//
//    where "Central" is the index of the grid point, "North" is the index
//    of its immediate neighbor to the "north", and so on.
//
//    Given an approximate solution of the steady state heat equation, a
//    "better" solution is given by replacing each interior point by the
//    average of its 4 neighbors - in other words, by using the condition
//    as an ASSIGNMENT statement:
//
//      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )
//
//    If this process is repeated often enough, the difference between successive
//    estimates of the solution will go to zero.
//
//    This program carries out such an iteration, using a tolerance specified by
//    the user, and writes the final estimate of the solution to a file that can
//    be used for graphic processing.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    22 July 2008
//
//  Author:
//
//    Original C version by Michael Quinn.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Michael Quinn,
//    Parallel Programming in C with MPI and OpenMP,
//    McGraw-Hill, 2004,
//    ISBN13: 978-0071232654,
//    LC: QA76.73.C15.Q55.
//
//  Parameters:
//
//    Commandline argument 1, double EPSILON, the error tolerance.
//
//    Commandline argument 2, char *OUTPUT_FILENAME, the name of the file into which
//    the steady state solution is written when the program has completed.
//
//  Local parameters:
//
//    Local, double DIFF, the norm of the change in the solution from one iteration
//    to the next.
//
//    Local, double MEAN, the average of the boundary values, used to initialize
//    the values of the solution in the interior.
//
//    Local, double U[M][N], the solution at the previous iteration.
//
//    Local, double W[M][N], the solution computed at the latest iteration.
//
{
  # define M 500
  # define N 500

  #ifdef _OPENMP
  double start_time, run_time;
  #else
  struct timeval tv_start, tv_end;
  double run_time;
  #endif


  double diff;
  double epsilon;
  double global_diff;
  int i;
  int iterations;
  int iterations_print;
  int j;
  double mean;
  double resta;
  FILE *output;
  char output_filename[80];
  double absoluto;
  int success;
  double u[M][N];
  double w[M][N];
  int proc_rows;

  MPI_Status status;
  int myid, procs,myrows;
  int last_node;


  MPI_Init(&argc, &argv);
  MPI_Comm_size( MPI_COMM_WORLD, &procs );
  MPI_Comm_rank( MPI_COMM_WORLD, &myid );

  int display[procs];
  int sendCounts[procs];

  if(N%procs==0)
  {
    myrows=N/procs;
    last_node = myrows;
  }
  else
  {
    myrows=N/procs;
    last_node = myrows + N%procs;
  }



  if(myid == 0)
  {
    proc_rows = myrows + 1;

  }
  else if(myid < (procs-1))
  {
    proc_rows = myrows + 2;

  }
  else
  {
    proc_rows = last_node + 1;

  }

  double subMatriz[proc_rows][N];
  double subU[proc_rows][N];
  epsilon = atof(argv[1]);

  diff = epsilon;
  global_diff = epsilon;
  if(myid == 0)
  {
    printf ( "\n" );
    printf ( "HEATED_PLATE <epsilon> <fichero-salida>\n" );
    printf ( "  C/serie version\n" );
    printf ( "  A program to solve for the steady state temperature distribution\n" );
    printf ( "  over a rectangular plate.\n" );
    printf ( "\n" );
    printf ( "  Spatial grid of %d by %d points.\n", M, N );
  }

  for(i = 0; i < procs; i++)
  {
    if(i < (procs - 1))
    {

      sendCounts[i] = myrows*N;
    }
    else
    {

      sendCounts[i] = last_node*N;
    }
  }


  for(i = 0; i < procs;i++)
  {
    if(i == 0)
    {
      display[i] = 0;

    }
    else
    {
      display[i] = i*myrows*N;
    }
  }

  if(myid == 0)
  {

    for(i = 0; i < procs; i++)
    {
      printf("Posición: %d sendCount: %d\n",i,sendCounts[i]);
    }


    for(i = 0; i < procs; i++)
    {
      printf("Posición: %d Display: %d\n",i,display[i]);
    }
  }





  if(myid==0){
    printf("The iteration will be repeated until the change is <= %lf\n", epsilon);
    success = sscanf ( argv[2], "%s", output_filename );
    if ( success != 1 )
    {
      printf ( "\n" );
      printf ( "HEATED_PLATE\n" );
      printf ( " Error en la lectura del nombre del fichero de salida\n");
      return 1;
    }

    printf("  The steady state solution will be written to %s\n", output_filename);


    //
    //  Set the boundary values, which don't change.
    //
    //
    //  Set the boundary values, which don't change.
    //
    for(i = 1; i < M - 1; i++)
    w[i][0] = 100.0;

    for(i = 1; i < M - 1; i++)
    w[i][N-1] = 100.0;

    for(j = 0; j < N; j++)
    w[M-1][j] = 100.0;

    for(j = 0; j < N; j++)
    w[0][j] = 0.0;

    //  Average the boundary values, to come up with a reasonable
    //  initial value for the interior.
    mean = 0.0;
    for(i = 1; i < M - 1; i++)
    mean = mean + w[i][0];

    for(i = 1; i < M - 1; i++)
    mean = mean + w[i][N-1];

    for(j = 0; j < N; j++)
    mean = mean + w[M-1][j];

    for(j = 0; j < N; j++)
    mean = mean + w[0][j];

    mean = mean/(double)(2 * M + 2 * N - 4);

    printf("\n");
    printf("  MEAN = %lf\n", mean);

    //printf ( "  MEAN = %lf\n", mean );

    //  Initialize the interior solution to the mean value.

    for ( i = 1; i < N - 1; i++ )
    for ( j = 1; j < N - 1; j++ )
    w[i][j] = mean;

  }


  if(myid == 0)
  {
    MPI_Scatterv(&w[0][0], sendCounts, display,MPI_DOUBLE, &subMatriz[0][0],sendCounts[myid], MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatterv(&w[0][0], sendCounts,display,MPI_DOUBLE, &subU[0][0],sendCounts[myid], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Scatterv(&w[0][0], sendCounts, display,MPI_DOUBLE, &subMatriz[1][0],sendCounts[myid], MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatterv(&w[0][0], sendCounts,display,MPI_DOUBLE, &subU[1][0],sendCounts[myid], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  iterations = 0;
  iterations_print = 1;


  if(myid==0){
    printf ( "\n" );
    printf ( " Iteration  Change\n" );
    printf ( "\n" );
    #ifdef _OPENMP
    start_time = omp_get_wtime();
    #else
    gettimeofday(&tv_start, NULL);
    #endif
  }
  for(;;){
    global_diff = 0.0;


    if(iterations % 2 != 0)
    {
      if(myid>0)
      {
        MPI_Send (subU[1],N, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD);
      }
      if (myid < procs-1){
        MPI_Send (subU[proc_rows-2], N, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD);
        MPI_Recv (subU[proc_rows-1], N, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD,&status) ;
      }
      if(myid>0)
      {
        MPI_Recv (subU[0], N, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD,&status) ;
      }
      diff = 0.0;
      #pragma omp parallel for private(i,j) reduction(max:diff)
      for(i = 1; i < proc_rows - 1; i++){
        for(j = 1; j < N - 1; j++){
          subMatriz[i][j] = (subU[i-1][j] + subU[i+1][j] + subU[i][j-1] + subU[i][j+1])/4.0;
          if(fabs(subMatriz[i][j] - subU[i][j]) > diff)
          diff = fabs(subMatriz[i][j] - subU[i][j]);
        }
      }
    }
    else
    {
      if(myid>0)
      {
        MPI_Send (subMatriz[1],N, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD);
      }
      if (myid < procs-1){
        MPI_Send (subMatriz[proc_rows-2], N, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD);
        MPI_Recv (subMatriz[proc_rows-1], N, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD,&status) ;
      }
      if(myid>0)
      {
        MPI_Recv (subMatriz[0], N, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD,&status) ;
      }
      diff = 0.0;
      #pragma omp parallel for private(i,j) reduction(max:diff)
      for(i = 1; i < proc_rows - 1; i++){
        for(j = 1; j < N - 1; j++){
          subU[i][j] = (subMatriz[i-1][j] + subMatriz[i+1][j] + subMatriz[i][j-1] + subMatriz[i][j+1])/4.0;

          if(fabs(subU[i][j] - subMatriz[i][j]) > diff)
          diff = fabs(subU[i][j] - subMatriz[i][j]);
        }
      }
    }
    MPI_Allreduce (&diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(global_diff<=epsilon)
    {
      break;
    }
    iterations++;
    if(myid == 0)
    {

      if ( iterations == iterations_print )
      {
        printf ( "  %8d %lg\n", iterations, diff);
        iterations_print = 2 * iterations_print;
      }
    }


  }  //fin while epsilon
  //Solo lo hace el nodo maestro

  if(iterations % 2 != 0 && myid == 0)
  {
    MPI_Gatherv(&subMatriz[0][0],sendCounts[myid], MPI_DOUBLE, &w[0][0],sendCounts,display,MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  else if(iterations % 2 != 0 && myid > 0)
  {
    MPI_Gatherv(&subMatriz[1][0],sendCounts[myid], MPI_DOUBLE, &w[0][0],sendCounts,display,MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  else if(iterations % 2 == 0 && myid == 0)
  {
    MPI_Gatherv(&subU[0][0],sendCounts[myid], MPI_DOUBLE, &w[0][0],sendCounts,display,MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Gatherv(&subU[1][0],sendCounts[myid], MPI_DOUBLE, &w[0][0],sendCounts,display,MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  if(myid == 0)
  {

    #ifdef _OPENMP
    run_time = omp_get_wtime() - start_time;
    #else
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 +
    (tv_end.tv_usec - tv_start.tv_usec); //en us
    run_time = run_time/1000000; // en s
    #endif


    printf ( "\n" );
    printf ( "  %8d  %lg\n", iterations, diff );
    printf ( "\n" );
    printf ( "  Error tolerance achieved.\n" );
    printf("\n Tiempo version Distribuida = %lg s\n", run_time);

    //  Write the solution to the output file.
    //
    output = fopen(output_filename, "wt");

    fprintf(output, "%d\n", M);
    fprintf(output, "%d\n", N);

    for ( i = 0; i < M; i++ )
    {
      for ( j = 0; j < N; j++)
      {
        fprintf(output, "%lg ", w[i][j]);
      }
      fprintf(output, "\n");
    }
    fclose(output);

    printf ( "\n" );
    printf ( " Solucion escrita en el fichero %s\n", output_filename);
    //
    //  Terminate.
    //
    printf ( "\n" );
    printf ( "HEATED_PLATE_Serie:\n" );
    printf ( "  Normal end of execution.\n" );
  }

  //Finalizan los nodos
  MPI_Finalize();
  return 0;
}
