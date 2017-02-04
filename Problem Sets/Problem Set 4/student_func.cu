//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
#include <stdio.h>


__global__ void print(const unsigned int* const d_arr, const size_t size)
{
  for(int i = 0; i < size; i++)
    printf("%d ", d_arr[i]);
  printf("\n");
}

__global__ void histogram(const unsigned int* const d_in,
                          unsigned int* const d_out,
                          const unsigned int mask, 
                          const unsigned int i)
{
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned int bin = (d_in[idx] & mask) >> i;
  atomicAdd(&(d_out[bin]), 1);
}

void radix_sort(unsigned int* const d_inputVals,
                           unsigned int* const d_inputPos,
                           unsigned int* const d_outputVals,
                           unsigned int* const d_outputPos,
                           const size_t numElems)
{
  const int numBits = 1;
  const int numBins = 1 << numBits;
  
  unsigned int* d_histogram = 0;
  cudaMalloc((void **)&d_histogram, numBins * sizeof(unsigned int));

  unsigned int* d_bin_scan = 0;
  cudaMalloc((void **)&d_histogram, numBins * sizeof(unsigned int));

  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) 
  {
      unsigned int mask = (numBins - 1) << i;

      cudaMemset(d_histogram, 0, numBins * sizeof(unsigned int));
      cudaMemset(d_bin_scan, 0, numBins * sizeof(unsigned int));

      int threads = 1024;
      int blocks = numElems / threads;
      // histogram build
      histogram<<<blocks, threads>>>(d_inputVals, d_histogram, mask, i);
  }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  print<<<1, 1>>>(d_inputPos, numElems);
}
