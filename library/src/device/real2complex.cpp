// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "./kernels/common.h"
#include "kernel_launch.h"
#include "rocfft.h"
#include "rocfft_hip.h"
#include <iostream>
#include <numeric>

// Device function for embedding real data in a complex buffer
template <typename Tcomplex>
__global__ void real2complex_kernel(const size_t          input_size,
                                    const size_t          input_stride,
                                    const size_t          output_stride,
                                    const void* vinput,
                                    const size_t          input_distance,
                                    void*              voutput,
                                    const size_t          output_distance)
{
    // Cast to correct type.
    // Add  batch offset + multi-dimensional offset
    const real_type_t<Tcomplex>* input = (real_type_t<Tcomplex>*)vinput
        + hipBlockIdx_z * input_distance + hipBlockIdx_y * input_stride;
    Tcomplex* output = (Tcomplex*) voutput
        + hipBlockIdx_z * output_distance + hipBlockIdx_y * output_stride; 

    const size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(tid < input_size)
    {
        output[tid].y = 0.0;
        output[tid].x = input[tid];
    }
}

/// \brief auxiliary function
///
///    convert a real vector into a complex one by padding the imaginary part with 0.
///    @param[in] input_size
///           size of input buffer
///    @param[in] input_buffer
///          data type : float or double
///    @param[in] input_distance
///          distance between consecutive batch members for input buffer
///    @param[in,output] output_buffer
///          data type : complex type (float2 or double2)
///    @param[in] output_distance
///           distance between consecutive batch members for output buffer
///    @param[in] batch
///           number of transforms
///    @param[in] precision
///          data type of input buffer. rocfft_precision_single or rocfft_precsion_double
void real2complex(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    const size_t input_size = data->node->length[0]; // input_size is the innermost dimension

    const size_t input_distance  = data->node->iDist;
    const size_t output_distance = data->node->oDist;

    const size_t input_stride
        = (data->node->length.size() > 1) ? data->node->inStride[1] : input_distance;
    const size_t output_stride
        = (data->node->length.size() > 1) ? data->node->outStride[1] : output_distance;

    const void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    const size_t batch          = data->node->batch;
    const size_t high_dimension = std::accumulate(data->node->length.begin() + 1,
                                            data->node->length.end(),
                                            1, std::multiplies<size_t>());
    const rocfft_precision precision = data->node->precision;

    const size_t blocks = (input_size - 1) / 512 + 1;

    if(high_dimension > 65535 || batch > 65535)
    {
        std::cout << "2D and 3D or batch is too big; not implemented\n";
    }
    
    // The z dimension is used for batching,
    // If 2D or 3D, the number of blocks along y will multiple high dimensions.
    // Notice that the maximum # of thread blocks in y & z is 65535
    // according to HIP && CUDA
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(512, 1, 1); // use 512 threads (work items)

    hipStream_t rocfft_stream = data->rocfft_stream;

    hipLaunchKernelGGL((precision == rocfft_precision_single) ?
                       real2complex_kernel<float2> :
                       real2complex_kernel<double2>,
                       grid,
                       threads,
                       0,
                       rocfft_stream,
                       input_size,
                       input_stride,
                       output_stride,
                       input_buffer,
                       input_distance,
                       output_buffer,
                       output_distance);

}

// Device function for extracting non-redundant data from a Hermitian
// symmetric complex buffer
template <typename Tcomplex>
__global__ void complex2hermitian_kernel(const size_t input_size,
                                         const size_t input_stride,
                                         const size_t output_stride,
                                         const void*     vinput,
                                         const size_t input_distance,
                                         void*     voutput,
                                         const size_t output_distance)
{
    // Cast to correct type.
    // Add  batch offset + multi-dimensional offset.
    const Tcomplex* input = (Tcomplex*)vinput
        + hipBlockIdx_z * input_distance + hipBlockIdx_y * input_stride;
    Tcomplex* output = (Tcomplex*) voutput
        + hipBlockIdx_z * output_distance + hipBlockIdx_y * output_stride; 

    const size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // Only read and write the first [input_size/2+1] elements due to
    // conjugate redundancy.
    if(tid < (1 + input_size / 2))
    {
        output[tid] = input[tid];
    }
}

/// \brief auxiliary function
///    read from input_buffer and store the first  [1 + input_size/2] elements to
///   the output_buffer
///    @param[in] input_size
///           size of input buffer
///    @param[in] input_buffer
///          data type dictated by precision parameter but complex type (float2 or
///   double2)
///    @param[in] input_distance
///           distance between consecutive batch members for input buffer
///    @param[in,output] output_buffer
///          data type dictated by precision parameter but complex type (float2 or
///   double2) but only store first [1 + input_size/2] elements according to
///   conjugate symmetry
///    @param[in] output_distance
///           distance between consecutive batch members for output buffer
///    @param[in] batch
///           number of transforms
///    @param[in]  precision
///           data type of input and output buffer. rocfft_precision_single or
///   rocfft_precsion_double
void complex2hermitian(const void* data_p, void* back_p)
{
    const DeviceCallIn* data = (DeviceCallIn*)data_p;

    // input_size is the innermost dimension
    const size_t input_size = data->node->length[0]; 

    const size_t input_distance  = data->node->iDist;
    const size_t output_distance = data->node->oDist;

    const size_t input_stride
        = (data->node->length.size() > 1) ? data->node->inStride[1] : input_distance;
    const size_t output_stride
        = (data->node->length.size() > 1) ? data->node->outStride[1] : output_distance;

    const void* input_buffer  = data->bufIn[0];
 void* output_buffer = data->bufOut[0];

    const size_t batch          = data->node->batch;
    const size_t high_dimension = std::accumulate(data->node->length.begin() + 1,
                                            data->node->length.end(),
                                            1, std::multiplies<size_t>());
    const rocfft_precision precision = data->node->precision;

    const size_t blocks = (input_size - 1) / 512 + 1;

    if(high_dimension > 65535 || batch > 65535)
    {
        std::cout << "2D and 3D or batch is too big; not implemented\n";
    }
    
    // The z dimension is used for batching.
    // If 2D or 3D, the number of blocks along y will multiple high
    // dimensions notice the maximum # of thread blocks in y & z is
    // 65535 according to HIP and CUDA.
    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(512, 1, 1); // use 512 threads (work items)

    hipStream_t rocfft_stream = data->rocfft_stream;

    hipLaunchKernelGGL((precision == rocfft_precision_single) ?
                       complex2hermitian_kernel<float2> :
                       complex2hermitian_kernel<double2>,
                       grid,
                       threads,
                       0,
                       rocfft_stream,
                       input_size,
                       input_stride,
                       output_stride,
                       input_buffer,
                       input_distance,
                       output_buffer,
                       output_distance);

}


// GPU kernel for 1d r2c post-process and c2r pre-process.
// Tcomplex is memory allocation type, could be float2 or double2.
// Each thread handles 2 points.
template <typename Tcomplex, bool R2C>
__global__ void real_1d_pre_post_process_kernel(const size_t   half_N,
                                                const size_t   input_stride,
                                                const size_t   output_stride,
                                                const void*    vinput,
                                                const size_t   input_distance,
                                                void*    voutput,
                                                const size_t   output_distance,
                                                void* const vtwiddles)
{
    // Cast the input pointers to the correct type and add batch
    // offset + stride offset.
    // Notice for 1D, hipBlockIdx_y == 0 and thus has no effect.
    const Tcomplex* input = (Tcomplex*)vinput
        + hipBlockIdx_z * input_distance + hipBlockIdx_y * input_stride; 
    Tcomplex* output = (Tcomplex*)voutput
        + hipBlockIdx_z * output_distance + hipBlockIdx_y * output_stride;
    const Tcomplex* twiddles = (Tcomplex*)vtwiddles;
    
    const size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t idx_q = half_N - idx_p;

    if(idx_p == 0)
    {
        if(R2C)
        {
            output[half_N].x = input[0].x - input[0].y;
            output[half_N].y = 0.0;
            output[0].x      = input[0].x + input[0].y;
            output[0].y      = 0.0;
        }
        else
        {
            const Tcomplex p             = input[0];
            const Tcomplex q             = input[half_N];
            output[idx_p].x = p.x + q.x;
            output[idx_p].y = p.x - q.x;
        }
    }
    else if(idx_p <= half_N >> 1)
    {
        const Tcomplex p = R2C ? 0.5 * input[idx_p] : input[idx_p];
        const Tcomplex q = R2C ? 0.5 * input[idx_q] : input[idx_q];

        const Tcomplex u(p.x + q.x, p.y - q.y); // p + conj(q)
        const Tcomplex v(p.x - q.x, p.y + q.y); // p - conj(q)

        const Tcomplex twd_p(R2C ? twiddles[idx_p].x : -twiddles[idx_p].x, twiddles[idx_p].y);
        const Tcomplex twd_q(R2C ? twiddles[idx_q].x : -twiddles[idx_q].x, twiddles[idx_q].y);

        output[idx_p].x = u.x + v.x * twd_p.y + v.y * twd_p.x;
        output[idx_p].y = u.y + v.y * twd_p.y - v.x * twd_p.x;

        output[idx_q].x = u.x - v.x * twd_q.y + v.y * twd_q.x;
        output[idx_q].y = -u.y + v.y * twd_q.y + v.x * twd_q.x;
    }
}

// Function for launching a pre- or post-processing kernel for
// even-length real/complex transforms.
template <bool R2C>
void real_1d_pre_post(const void* data_p, void* back_p)
{
    DeviceCallIn* data = (DeviceCallIn*)data_p;

    // Input_size is the innermost dimension
    // The upper level provides always N/2, that is regular complex fft size
    const size_t half_N = data->node->length[0];

    const size_t input_distance  = data->node->iDist;
    const size_t output_distance = data->node->oDist;

    const size_t input_stride
        = (data->node->length.size() > 1) ? data->node->inStride[1] : input_distance;
    const size_t output_stride
        = (data->node->length.size() > 1) ? data->node->outStride[1] : output_distance;

    const void* input_buffer  = data->bufIn[0];
    void* output_buffer = data->bufOut[0];

    const size_t batch          = data->node->batch;
    const size_t high_dimension = std::accumulate(data->node->length.begin() + 1,
                                            data->node->length.end(),
                                            1, std::multiplies<size_t>());

      const size_t block_size = 512;
     const size_t blocks     = (half_N / 2 + 1 - 1) / block_size + 1;

    if(high_dimension > 65535 || batch > 65535)
    {
        std::cout << "2D and 3D or batch is too big; not implemented\n";
    }

    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(block_size, 1, 1);

    hipLaunchKernelGGL((data->node->precision == rocfft_precision_single) ?
                       real_1d_pre_post_process_kernel<float2, R2C> :
                       real_1d_pre_post_process_kernel<double2, R2C>,
                       grid,
                       threads,
                       0,
                       data->rocfft_stream,
                       half_N,
                       input_stride,
                       output_stride,
                       input_buffer,
                       input_distance,
                       output_buffer,
                       output_distance,
                       data->node->twiddles);
}

// Wrappers for templated real/complex even-length pre/post kernel launches
void r2c_1d_post(const void* data_p, void* back_p)
{
    real_1d_pre_post<true>(data_p, back_p);
}
void c2r_1d_pre(const void* data_p, void* back_p)
{
    real_1d_pre_post<false>(data_p, back_p);
}
