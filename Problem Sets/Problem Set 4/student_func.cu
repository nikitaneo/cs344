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
#define BLOCK_SIZE 1024

__global__
void partial_exclusive_blelloch_scan(unsigned int* const d_list, 
                                     unsigned int* const d_block_sums, 
                                     const size_t numElems)
{
  /*
    has 'partial' in the name b/c it treats segments within a block as their
    own scan subproblem. To solve the whole scan subproblem, use this function
    on your whole array (solving many independent scan problems), then use this
    function again to scan the d_block_sums, which will contain the total of
    each independent scan subproblem. Then use increment_blelloch_scan_with_block_sums
    to increment each independent scan subproblem's solution with the scan
    results of d_block_sums.

    Note: This function will only work if you initialize 2^n threads to run it.
    The number of elements in d_list does not matter though, just the number of
    threads initialized.
  */
  extern volatile __shared__ unsigned int s_block_scan[];

  const unsigned int tid = threadIdx.x;
  const unsigned int id = blockDim.x * blockIdx.x + tid;

  // copy to shared memory, pad the block that is too small
  if (id >= numElems)
    s_block_scan[tid] = 0;
  else
    s_block_scan[tid] = d_list[id];
  __syncthreads();

  // reduce
  unsigned int i;
  for (i = 2; i <= blockDim.x; i <<= 1) 
  {
    if ((tid + 1) % i == 0) 
    {
      unsigned int neighbor_offset = i>>1;
      s_block_scan[tid] += s_block_scan[tid - neighbor_offset];
    }
    __syncthreads();
  }
  i >>= 1; // return i to last value before for loop exited
  // reset last (sum of whole block) to identity element
  if (tid == (blockDim.x-1)) 
  {
    d_block_sums[blockIdx.x] = s_block_scan[tid];
    s_block_scan[tid] = 0;
  }
  __syncthreads();

  // downsweep
  for (i = i; i >= 2; i >>= 1) 
  {
    if((tid + 1) % i == 0) 
    {
      unsigned int neighbor_offset = i>>1;
      unsigned int old_neighbor = s_block_scan[tid - neighbor_offset];
      s_block_scan[tid - neighbor_offset] = s_block_scan[tid]; // copy
      s_block_scan[tid] += old_neighbor;
    }
    __syncthreads();
  }

  // copy result to global memory
  if (id < numElems) 
  {
    d_list[id] = s_block_scan[tid];
  }
}

__global__
void increment_blelloch_scan_with_block_sums(unsigned int* const d_predicateScan,
                                             unsigned int* const d_blockSumScan, 
                                             const size_t numElems)
{
  /*
    companion to: partial_exclusive_blelloch_scan
  */
  const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  d_predicateScan[id] += d_blockSumScan[blockIdx.x];
}

__global__
void scatter(const unsigned int* const d_input, 
             unsigned int* const d_output,
             const unsigned int* const d_predicateTrueScan, 
             const unsigned int* const d_predicateFalseScan,
             const unsigned int* const d_predicateFalse,
             const unsigned int* const d_numPredicateTrueElements,
             const size_t numElems)
{
  const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  unsigned int newLoc;
  if (d_predicateFalse[id] == 1) 
    newLoc = d_predicateFalseScan[id] + *d_numPredicateTrueElements;
  else 
    newLoc = d_predicateTrueScan[id];

  d_output[newLoc] = d_input[id];
}

__global__
void generate_predicates(const unsigned int* const d_in, 
                         unsigned int* const d_out, 
                         const unsigned int bit,
                         const size_t size)
{
  const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(idx >= size)
    return;

  unsigned int value = ((d_in[idx] & bit) == 0);
  d_out[idx] = value; 
}


// flip 0 to 1 and 1 to 0
__global__
void flip_bit(unsigned int* const d_list, 
              const size_t numElems)
{
  const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  d_list[id] = 1 - d_list[id];
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  size_t size = numElems * sizeof(unsigned int);

  unsigned int blockSize = BLOCK_SIZE; 
  unsigned int gridSize = ceil((float)size / (float)blockSize);

  unsigned int* d_predicate;
  unsigned int* d_predicateTrueScan;
  unsigned int* d_predicateFalseScan;
  unsigned int* d_numPredicateTrueElements;
  unsigned int* d_numPredicateFalseElements;
  unsigned int* d_block_sums;

  checkCudaErrors(cudaMalloc((void**)&d_predicate, size));
  checkCudaErrors(cudaMalloc((void**)&d_predicateTrueScan, size));
  checkCudaErrors(cudaMalloc((void**)&d_predicateFalseScan, size));
  checkCudaErrors(cudaMalloc((void**)&d_numPredicateTrueElements, sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void**)&d_numPredicateFalseElements, sizeof(unsigned int))); // throwaway
  checkCudaErrors(cudaMalloc((void**)&d_block_sums, gridSize*sizeof(unsigned int)));

  checkCudaErrors(cudaMemset(d_block_sums, 0, gridSize*sizeof(unsigned int)));

  unsigned int nsb;
  unsigned int max_bits = 31;

  unsigned int* d_in = d_inputVals;
  unsigned int* d_out = d_outputVals;

  unsigned int* d_inPos = d_inputPos;
  unsigned int* d_outPos = d_outputPos;

  for (unsigned int bit = 0; bit < max_bits; bit++) {
    nsb = 1 << bit;

    generate_predicates<<<gridSize, blockSize>>>(d_in, d_predicate, nsb, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // scan predicateTrue
    checkCudaErrors(cudaMemcpy(d_predicateTrueScan, d_predicate, size, cudaMemcpyDeviceToDevice));

    partial_exclusive_blelloch_scan<<<gridSize, blockSize, sizeof(unsigned int)*blockSize>>>(d_predicateTrueScan, d_block_sums, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    partial_exclusive_blelloch_scan<<<1, blockSize, sizeof(unsigned int)*blockSize>>>(d_block_sums, d_numPredicateTrueElements, gridSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    increment_blelloch_scan_with_block_sums<<<gridSize, blockSize>>>(d_predicateTrueScan, d_block_sums, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // transform predicateTrue -> predicateFalse
    flip_bit<<<gridSize, blockSize>>>(d_predicate, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // scan predicateFalse
    checkCudaErrors(cudaMemcpy(d_predicateFalseScan, d_predicate, size, cudaMemcpyDeviceToDevice));

    partial_exclusive_blelloch_scan<<<gridSize, blockSize, sizeof(unsigned int)*blockSize>>>(d_predicateFalseScan, d_block_sums, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    partial_exclusive_blelloch_scan<<<1, blockSize, sizeof(unsigned int)*blockSize>>>(d_block_sums, d_numPredicateFalseElements, gridSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    increment_blelloch_scan_with_block_sums<<<gridSize, blockSize>>>(d_predicateFalseScan, d_block_sums, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // scatter values (flip input/output depending on iteration)
    scatter<<<gridSize, blockSize>>>(d_in, d_out, d_predicateTrueScan, d_predicateFalseScan,
                                       d_predicate, d_numPredicateTrueElements, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    scatter<<<gridSize, blockSize>>>(d_inPos, d_outPos, d_predicateTrueScan, d_predicateFalseScan,
                                       d_predicate, d_numPredicateTrueElements, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // ping-pong pointers
    std::swap(d_in, d_out);
    std::swap(d_inPos, d_outPos);

  }


  checkCudaErrors(cudaFree(d_predicate));
  checkCudaErrors(cudaFree(d_predicateTrueScan));
  checkCudaErrors(cudaFree(d_predicateFalseScan));
  checkCudaErrors(cudaFree(d_numPredicateTrueElements));
  checkCudaErrors(cudaFree(d_numPredicateFalseElements));
  checkCudaErrors(cudaFree(d_block_sums));
}
