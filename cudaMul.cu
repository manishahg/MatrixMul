/*CUDA 2-D Matrix Multiplication*/

#include <stdio.h>
#include <math.h>

#define TILE_WIDTH 2
#define WIDTH  100

__global__ void MatrixMul( float *A_d , float *B_d , float *C_d)
{
    // calculate thread id
    unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
    unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

    C_d[row*WIDTH+col] = 0;
    for (int k = 0 ; k<WIDTH ; k++ )
    {
        C_d[row*WIDTH + col]+= A_d[row * WIDTH + k ] * B_d[ k * WIDTH + col] ;
    }
}

// main routine
int main ()
{
  float array1_h[WIDTH][WIDTH] ,array2_h[WIDTH][WIDTH];
  float non_array_h[WIDTH][WIDTH] ,M_result_array_h[WIDTH][WIDTH]  ;
  float *array1_d , *array2_d ,*M_result_array_d ; // device array
  int i , j ;
  
  //input in host array
  for ( i = 0 ; i<WIDTH ; i++ )
  {
     for (j = 0 ; j<WIDTH ; j++ )
     {
        array1_h[i][j] =(float)rand()/(float)RAND_MAX;
        array2_h[i][j] = (float)rand()/(float)RAND_MAX;
     }
  }

  //Non-threaded routine
  for(i=0; i<WIDTH; ++i)
  {
      for(j=0; j<WIDTH; ++j)
      {
           non_array_h[i][j]=0;
           for(int k=0; k<WIDTH; ++k)
           {
               non_array_h[i][j] += array1_h[i][k] * array2_h[k][j] ;
           }
      }
 }

  //create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;
  cudaMalloc((void **) &array1_d , WIDTH*WIDTH*sizeof (float) ) ;
  cudaMalloc((void **) &array2_d , WIDTH*WIDTH*sizeof (float) ) ;

  //copy host array to device array; cudaMemcpy ( dest , source , WIDTH , direction )
  cudaMemcpy ( array1_d , array1_h , WIDTH*WIDTH*sizeof (float) , cudaMemcpyHostToDevice ) ;
  cudaMemcpy ( array2_d , array2_h , WIDTH*WIDTH*sizeof (float) , cudaMemcpyHostToDevice ) ;

  //allocating memory for resultant device array
  cudaMalloc((void **) &M_result_array_d , WIDTH*WIDTH*sizeof (float) ) ;

  //calling kernel
  dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ;
  dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;

   MatrixMul <<<dimGrid,dimBlock>>> ( array1_d , array2_d ,M_result_array_d) ;

  //copy back result_array_d to result_array_h
  cudaMemcpy(M_result_array_h , M_result_array_d , WIDTH*WIDTH*sizeof(float) ,
                                    cudaMemcpyDeviceToHost) ;
  float error = 0;
  
  //print the result array
  printf("*****************************************\n");
  printf("Vector Multiplication of %d*%d matrix\n", WIDTH, WIDTH);
  printf("*****************************************\n");
  for ( i = 0 ; i<WIDTH ; i++ )
  {
      for ( j = 0 ; j < WIDTH ; j++ )
     {
       // printf ("%f %f\t",non_array_h[i][j], M_result_array_h[i][j] ) ;
       //Root mean square of the differences
        error += pow(M_result_array_h[i][j]-non_array_h[i][j], 2);
     }
     //printf ("\n") ;
 }
  error = error/(WIDTH*WIDTH);
  error = sqrt(error);

  printf("*****************************************\n");
  printf("Error:%f\n", error);
  printf("*****************************************\n");

  cudaFree(array1_d);
  cudaFree(array2_d);
  cudaFree(M_result_array_d);

  return 0;
}