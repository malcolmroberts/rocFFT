// Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

/// Kernel for unpacking two complex arrays with Hermitian symmetry from one complex array
/// which is the output of a c2c transform where the input is two real arrays x and y.
///
/// That is, given Z = \mathcal{F}(x + iy) = X + i Y, we compute
///
/// X_0 = \re{Z_0},                Y_0 = \im{Z_0},
///
/// X_r = (Z_r + Z_{N - r}^*)/2,   Y_r = (Z_r - Z_{N - r}^*)/(2i)
///
/// for r = 1, ... , \floor{N/2} + 1.
template <typename Treal>
__global__ static void complex2pair_unpack_kernel(const size_t      half_N,
                                                  const void*       input0,
                                                  const void*       input1,
                                                  void*        output0_,
                                                  void*        output1_)
{
    const size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const size_t idx_q = half_N - idx_p;

    const auto quarter_N = (half_N + 1) / 2;
    
    if(idx_p < quarter_N)
    {
        const auto inputRe = (Treal*)input0;
        const auto inputIm = (Treal*)input1;
        auto output0 = (complex_type_t<Treal>*)output0_;
        auto output1 = (complex_type_t<Treal>*)output1_;
        
        if(idx_p == 0)
        {
            output0[idx_p].x = inputRe[idx_p];
            output0[idx_p].y = 0.0;

            output1[idx_p].x = inputIm[idx_p];
            output1[idx_p].y = 0.0;

        }
        else
        {
            const Treal Rep = inputRe[idx_p];
            const Treal Imp = inputIm[idx_p];
            const Treal Req = inputRe[idx_q];
            const Treal Imq = inputIm[idx_q];
            
            output0[idx_p].x = 0.5 * (Rep + Req);
            output0[idx_p].y = 0.5 * (Imp - Imq);

            output1[idx_p].x = 0.5 * (Imp + Imq);
            output1[idx_p].y = -0.5 * (Rep - Req);
        } 
    }
}

/// Unpack two (Hermitian-symmetric) complex arrays from a full-length complex array for a
/// real-to-complex transform
void complex2pair_unpack(const void* data_p, void*)
{
    std::cout << "complex2pair" << std::endl;
    const DeviceCallIn* data = (DeviceCallIn*)data_p;

    const size_t idist = data->node->iDist;
    const size_t odist = data->node->oDist;

    void* bufIn0  = data->bufIn[0];
    void* bufIn1  = data->bufIn[1];
    void* bufOut0 = data->bufOut[0];
    void* bufOut1 = data->bufOut[1];


    const size_t half_N = data->node->length[0];
    const size_t high_dimension = std::accumulate(
        data->node->length.begin() + 1, data->node->length.end(), 1, std::multiplies<size_t>());
    const size_t batch = data->node->batch;
    
    const size_t block_size = 512;
    size_t       blocks     = (half_N + block_size - 1) / block_size;

    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(block_size, 1, 1);

    std::cout << "threads: " << threads.x << std::endl; // FIXME: temp
    std::cout << "half_N: " << half_N << std::endl; // FIXME: temp
    
    switch(data->node->precision)
    {
    case rocfft_precision_single:
        complex2pair_unpack_kernel<float><<<grid, threads, 0>>>(half_N,
                                                                bufIn0, bufIn1, bufOut0, bufOut1);
        break;
    case rocfft_precision_double:
        complex2pair_unpack_kernel<double><<<grid, threads, 0>>>(half_N,
                                                                 bufIn0, bufIn1, bufOut0, bufOut1);
        break;
    default:
        std::cerr << "invalid precision for complex2pair\n";
        assert(false);
    }

}


// FIXME: document
template <typename Treal>
__global__ static void pair2complex_pack_kernel()
{
    // FIXME: implement
}

/// Pack two (Hermitian-symmetric) complex arrays into full-length complex array for a
/// complex-to-real transform.
void pair2complex_pack(const void* data_p, void*)
{
    // FIXME: implement
    const DeviceCallIn* data = (DeviceCallIn*)data_p;
    assert(false);
}

