/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "stdio.h"
#include <algorithm>

__global__ void arrprint(unsigned int* arr, int size)
{
  for(int i = 0; i < size; i++)
    printf("%d ", arr[i]);
  printf("\n\n");
}

__global__ void shmem_maximum(const float* const d_in, float* const d_out)
{
  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  sdata[tid] = d_in[myId];
  __syncthreads();
 
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if(tid < s)
    {
      sdata[tid] = fmaxf(sdata[tid + s], sdata[tid]);
    }
    __syncthreads();
  }

  if(tid == 0)
  {
    d_out[blockIdx.x] = sdata[0];
  }
}


__global__ void shmem_minimum(const float* const d_in, float* const d_out)
{
  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  sdata[tid] = d_in[myId];
  __syncthreads();

  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if(tid < s)
    {
      sdata[tid] = fminf(sdata[tid + s], sdata[tid]);
    }
    __syncthreads();
  }

  if(tid == 0)
  {
    d_out[blockIdx.x] = sdata[0];
  }
}


float minimum(const float* const d_in, const size_t size)
{
  const int maxThreadPerBlock = 1024;
  int threads = maxThreadPerBlock;
  int blocks = size / maxThreadPerBlock;

  // we should not modify the data, so copy an array
  float *d_tmp = 0;
  cudaMalloc((void **)&d_tmp, size * sizeof(float));
  cudaMemcpy(d_tmp, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);


  // create output array
  float *d_out = 0;
  cudaMalloc((void **)&d_out, sizeof(float) * threads);
  cudaMemset(d_out, 0, sizeof(float) * threads);

  shmem_minimum<<<blocks, threads, threads * sizeof(float)>>>(d_tmp, d_out);
  cudaDeviceSynchronize();

  threads = blocks;
  blocks  = 1;
  
  shmem_minimum<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_tmp);
  cudaDeviceSynchronize();

  //gets reduction value from d_tmp[0]
  float reduction_value[] = { 0 };
  cudaMemcpy(reduction_value, d_tmp, 1 * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_tmp);
  cudaFree(d_out);

  return reduction_value[0];
}

float maximum(const float* const d_in, const size_t size)
{
  const int maxThreadPerBlock = 1024;
  int threads = maxThreadPerBlock;
  int blocks = size / maxThreadPerBlock;

  // we should not modify the data, so copy an array
  float *d_tmp = 0;
  cudaMalloc((void **)&d_tmp, size * sizeof(float));
  cudaMemcpy(d_tmp, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);


  // create output array
  float *d_out = 0;
  cudaMalloc((void **)&d_out, sizeof(float) * threads);
  cudaMemset(d_out, 0, sizeof(float) * threads);

  shmem_maximum<<<blocks, threads, threads * sizeof(float)>>>(d_tmp, d_out);

  threads = blocks;
  blocks  = 1;
  
  shmem_maximum<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_tmp);
  cudaDeviceSynchronize();

  //gets reduction value from d_tmp[0]
  float reduction_value[] = { 0 };
  cudaMemcpy(reduction_value, d_tmp, 1 * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_tmp);
  cudaFree(d_out);

  return reduction_value[0];
}

__global__ void histo(const float* const d_in, 
                      unsigned int* const d_out, 
                      const size_t size, 
                      const int numBins, 
                      const float lumMin, 
                      const float lumRange)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int bin = umin(static_cast<unsigned int>((d_in[myId] - lumMin) / lumRange * numBins), static_cast<unsigned int>(numBins - 1));
  atomicAdd(&(d_out[bin]), 1);
}

unsigned int* histogram(const float* const d_in, const size_t size, const int numBins, const float lumMin, const float lumRange)
{
  unsigned int* d_histogram = 0;
  cudaMalloc((void **)&d_histogram, numBins * sizeof(unsigned int));
  cudaMemset(d_histogram, static_cast<unsigned int>(0), sizeof(unsigned int) * numBins);

  const int maxThreadPerBlock = 1024;
  int threads = maxThreadPerBlock;
  int blocks = size / maxThreadPerBlock;

  histo<<<blocks, threads>>>(d_in, d_histogram, size, numBins, lumMin, lumRange);
  cudaDeviceSynchronize();

  return d_histogram; 
}

// Blelloch scan
__global__ void exclusive_scan(const unsigned int* const d_in, unsigned int* const d_out, const int n)
{
  extern __shared__ unsigned int temp[];

  int tid = threadIdx.x;
  int offset = 1;

  temp[2 * tid] = d_in[2 * tid];
  temp[2 * tid + 1] = d_in[2 * tid + 1];

  for(unsigned int d = n / 2; d > 0; d >>= 1)
  {
    __syncthreads();
    if(tid < d)
    {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if(tid == 0)
  {
    temp[n - 1] = 0;
  }

  for(unsigned int d = 1; d < n; d *= 2)
  {
    offset >>= 1;
    __syncthreads();
    
    if(tid < d)
    {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  __syncthreads();

  d_out[2 * tid] = temp[2 * tid];
  d_out[2 * tid + 1] = temp[2 * tid + 1];
}


void prefix_sum(const unsigned int* const d_in, unsigned int* const d_out, const int size)
{
  int threads = size / 2;
  int blocks = 1;

  exclusive_scan<<<blocks, threads, 2 * threads * sizeof(unsigned int)>>>(d_in, d_out, size);
  cudaDeviceSynchronize();
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  //
  //  Here are the steps you need to implement
  //  1) find the minimum and maximum value in the input logLuminance channel
  //     store in min_logLum and max_logLum

  min_logLum = minimum(d_logLuminance, numRows * numCols);
  max_logLum = maximum(d_logLuminance, numRows * numCols);

  //  2) subtract them to find the range

  float range = max_logLum - min_logLum;
  
  //  3) generate a histogram of all the values in the logLuminance channel using
  //     the formula: bin = (lum[i] - lumMin) / lumRange * numBins

  unsigned int* d_histogram = histogram(d_logLuminance, numCols * numRows, numBins, min_logLum, range);
  
  //  4) Perform an exclusive scan (prefix sum) on the histogram to get
  //     the cumulative distribution of luminance values (this should go in the
  //     incoming d_cdf pointer which already has been allocated for you)
  prefix_sum(d_histogram, d_cdf, numBins);     

  cudaFree(d_histogram);
}
