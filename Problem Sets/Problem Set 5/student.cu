/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include <stdio.h>

__global__
void histogram(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numVals)
{
  extern __shared__ unsigned int subhist[];

  const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned int tid = threadIdx.x;

  subhist[4 * tid] = 0;
  subhist[4 * tid + 1] = 0;
  subhist[4 * tid + 2] = 0;
  subhist[4 * tid + 3] = 0;
  __syncthreads();

  for(int i = 0; i < 8; ++i)
      atomicAdd(&(subhist[vals[8 * idx + i]]), 1);
  __syncthreads();

  atomicAdd(&(histo[4 * tid]), subhist[4 * tid]);
  atomicAdd(&(histo[4 * tid + 1]), subhist[4 * tid + 1]);
  atomicAdd(&(histo[4 * tid + 2]), subhist[4 * tid + 2]);
  atomicAdd(&(histo[4 * tid + 3]), subhist[4 * tid + 3]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  const unsigned int threads = 256;
  const unsigned int blocks = numElems / threads / 8;

  histogram<<<blocks, threads, numBins * sizeof(unsigned int)>>>(d_vals, d_histo, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
