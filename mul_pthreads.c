/*
 ============================================================================
 Name        : mul_pthreads.c
 Author      : Manisha Agrawal
 Create a pThreads or C++11 program to multiply two square matrices of arbitrary size.
 Fill matrices with random double data. Include a non-threaded routine to multiply
 the matrices in order to check. Write a test routine to make sure that both methods
 produce the same answer to within a reasonable tolerance.
 Description : Hello World in C, Ansi-style
 ============================================================================
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NTHREADS      4
#define ARRAYSIZE   8

typedef struct
{
    int startIdx;
    double **arrC, **arrA, **arrB;
} mulType;

void *mulArrayPthread(void *x)
{
	mulType *p = (mulType *) x;
    int blockSize = ARRAYSIZE/NTHREADS;
    int end = (*p).startIdx + blockSize;
    printf("Start=%d\tEnd=%d\n",(*p).startIdx, end);

    int i, j ,k;

    for (i=(*p).startIdx;i<end;i++)
    {
       for(j=0; j<ARRAYSIZE; ++j)
      	 for(k=0; k<ARRAYSIZE; ++k)
      		(*p).arrC[i][j] += (*p).arrA[i][k]*(*p).arrB[k][j];
    }
    return NULL;
}

int main(int argc, char *argv[])
{
  int i, j, k, tids[NTHREADS];
  pthread_t threads[NTHREADS];
  pthread_attr_t attr;

  double **A_array=(double**)malloc(ARRAYSIZE*sizeof(double*));
  double **B_array=(double**)malloc(ARRAYSIZE*sizeof(double*));
  double **Non_array=(double**)malloc(ARRAYSIZE*sizeof(double*));
  double **Thread_array=(double**)malloc(ARRAYSIZE*sizeof(double*));

  for(i=0;i<ARRAYSIZE;i++)
  {
	  A_array[i]=(double*)malloc(ARRAYSIZE*sizeof(double));
	  B_array[i]=(double*)malloc(ARRAYSIZE*sizeof(double));
	  Non_array[i]=(double*)malloc(ARRAYSIZE*sizeof(double));
	  Thread_array[i]=(double*)malloc(ARRAYSIZE*sizeof(double));
  }

  for (i = 0; i < ARRAYSIZE; i++)
  {
	  for(j=0; j < ARRAYSIZE; j++)
	  {
		  A_array[i][j] = rand()/(float)RAND_MAX;
	  	  B_array[i][j] = rand()/(float)RAND_MAX;
	  }
  }

  //Non-threaded routine
  for(i=0; i<ARRAYSIZE; ++i)
  {
     for(j=0; j<ARRAYSIZE; ++j)
     {
    	 Non_array[i][j]=0.0;
    	 for(k=0; k<ARRAYSIZE; ++k)
    		 Non_array[i][j] = Non_array[i][j] + (A_array[i][k] * B_array[k][j]) ;
     }
  }

  int index=ARRAYSIZE/NTHREADS ;
  mulType **datas = malloc(NTHREADS*sizeof(mulType));
  for (i = 0; i != NTHREADS ; i++)
  {
	  datas[i] = malloc(sizeof(mulType));
	  datas[i]->startIdx = index*i;
	  datas[i]->arrA = A_array;
	  datas[i]->arrB = B_array;
	  datas[i]->arrC = Thread_array;
  }

  for (i=0; i< NTHREADS; i++)
  {
    tids[i] = i;
    pthread_create(&threads[i], NULL, mulArrayPthread, (void *)(datas[i]));
  }

  /* Wait for all threads to complete then print global sum */
  for (i=0; i<NTHREADS; i++)
  {
    pthread_join(threads[i], NULL);
  }

  for (i=0;i<ARRAYSIZE;i++)
  {
	  for (j=0;j<ARRAYSIZE;j++)
	  {
	    	 printf("[%d][%d]:Serial=%f\tThread=%f\n",i,j, Non_array[i][j], Thread_array[i][j]);
	  }
  }

  /* Clean up and exit */
  pthread_attr_destroy(&attr);
  pthread_exit (NULL);
  return 0;
}
