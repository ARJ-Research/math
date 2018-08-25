#ifndef STAN_MATH_GPU_KERNELS_MULTIPLY_TRANSPOSE_HPP
#define STAN_MATH_GPU_KERNELS_MULTIPLY_TRANSPOSE_HPP
#ifdef STAN_OPENCL

#include <stan/math/gpu/kernel_cl.hpp>

namespace stan {
namespace math {
namespace opencl_kernels {
// \cond
const char* multiply_transpose_kernel_code = STRINGIFY(
    // \endcond
    /**
     * Matrix multiplication of the form A*A^T on the GPU
     *
     * @param[in] A matrix A
     * @param[out] B the output matrix
     * @param[in] M Number of rows for matrix A
     * @param[in] N Number of cols for matrix A and the number of rows for
     * matrix A^T
     */
    __kernel void multiply_transpose(const __global double* A,
                                     __global double* B, const int M,
                                     const int N) {
      // workitem index inside the workgroup
      const int workgroup_row = get_local_id(0);
      const int workgroup_col = get_local_id(1);

      // global workitem index
      const int i
          = WG_SIZE_MULT_SELF_TRANS * get_group_id(0) + workgroup_row;
      const int j
          = WG_SIZE_MULT_SELF_TRANS * get_group_id(1) + workgroup_col;

      // indexes that determine the last indexes that need to compute
      // in order to remove the unnecesary multiplications in the special
      // multiplication of A*A^T
      const int jMin = WG_SIZE_MULT_SELF_TRANS * get_group_id(1);
      const int iMax = WG_SIZE_MULT_SELF_TRANS * get_group_id(0)
                       + get_local_size(0);

      // local memory
      __local double A_local[WG_SIZE_MULT_SELF_TRANS]
                            [WG_SIZE_MULT_SELF_TRANS];
      __local double B_local[WG_SIZE_MULT_SELF_TRANS]
                            [WG_SIZE_MULT_SELF_TRANS];

      double acc[WORK_PER_WI_MULT_SELF_TRANS];
      for (int w = 0; w < WORK_PER_WI_MULT_SELF_TRANS; w++) {
        acc[w] = 0.0;
      }

      const int numTiles = (N + WG_SIZE_MULT_SELF_TRANS - 1)
                           / WG_SIZE_MULT_SELF_TRANS;
      // iterate over all tiles
      for (int t = 0; t < numTiles; t++) {
        // in each tile
        const int tiled_i = WG_SIZE_MULT_SELF_TRANS * t + workgroup_row;
        const int tiled_j = WG_SIZE_MULT_SELF_TRANS * t + workgroup_col;
        // if the data needs to be loaded to local memory
        if (jMin <= iMax) {
          // each workitem copies WORK_PER_WI_MULT_SELF_TRANS values to the
          // local memory
          for (int w = 0; w < WORK_PER_WI_MULT_SELF_TRANS; w++) {
            A_local[workgroup_col + w * WG_SIZE_MULT_SELF_TRANS_COL]
                   [workgroup_row]
                = A[i + (tiled_j + w * WG_SIZE_MULT_SELF_TRANS_COL) * M];
            B_local[workgroup_col + w * WG_SIZE_MULT_SELF_TRANS_COL]
                   [workgroup_row]
                = A[(j + w * WG_SIZE_MULT_SELF_TRANS_COL) + tiled_i * M];
          }
        }
        // wait till all tile values are loaded to the local memory
        barrier(CLK_LOCAL_MEM_FENCE);
        // multiply the tile products
        for (int k = 0; k < WG_SIZE_MULT_SELF_TRANS; k++) {
          // each workitem multiplies WORK_PER_WI_MULT_SELF_TRANS values
          for (int w = 0; w < WORK_PER_WI_MULT_SELF_TRANS; w++) {
            if (jMin <= iMax) {
              if ((j + w * WG_SIZE_MULT_SELF_TRANS_COL) <= i) {
                acc[w]
                    += A_local[k][workgroup_row]
                       * B_local[workgroup_col
                                 + w * WG_SIZE_MULT_SELF_TRANS_COL][k];
              }
            }
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      // save the values
      for (int w = 0; w < WORK_PER_WI_MULT_SELF_TRANS; w++) {
        // each workitem saves WORK_PER_WI_MULT_SELF_TRANS values
        if ((j + w * WG_SIZE_MULT_SELF_TRANS_COL) <= i) {
          B[i + (j + w * WG_SIZE_MULT_SELF_TRANS_COL) * M] = acc[w];
          B[(j + w * WG_SIZE_MULT_SELF_TRANS_COL) + i * M] = acc[w];
        }
      }
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/multiply_transpose.hpp add() \endlink
 */
const local_range_kernel<cl::Buffer, cl::Buffer, int, int> multiply_transpose(
    "multiply_transpose", multiply_transpose_kernel_code);

}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan
#endif
#endif
