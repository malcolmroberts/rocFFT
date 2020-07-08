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

/// Kernels for unpacking two complex arrays with Hermitian symmetry from one complex array
/// which is the output of a c2c transform where the input is two real arrays x and y.
///
/// That is, given Z = \mathcal{F}(x + iy) = X + i Y, we compute
///
/// X_0 = \re{Z_0},                Y_0 = \im{Z_0},
///
/// X_r = (Z_r + Z_{N - r}^*)/2,   Y_r = (Z_r - Z_{N - r}^*)/(2i)
///
/// for r = 1, ... , \floor{N/2} + 1.

/// Interleaved data version.
template <typename Treal>
__global__ static void complex2pair_unpack_kernel(const size_t      N,
                                                  const void*       input,
                                                  const size_t     ioffset,
                                                  void*        output,
                                                  const size_t ooffset)
{
    const size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto half_N = (N + 1) / 2;
    const auto quarter_N = (half_N + 1) / 2;
        
    if(idx_p <= quarter_N)
    {
        // NB: the nyquist is double-counted.
        
        const auto inputRe = (Treal*)input;
        const auto inputIm = inputRe + ioffset;
        
        auto outputX = (complex_type_t<Treal>*)output;
        auto outputY = outputX + ooffset;

        const Treal Rep = inputRe[idx_p];
        const Treal Imp = inputIm[idx_p];
        
        const size_t idx_q = N - idx_p;
        const Treal Req = inputRe[idx_q];
        const Treal Imq = inputIm[idx_q];

        // For in-place transforms, we need to run two sets of indices
        // in order to avoid race conditions.
        
        const size_t idx_r = half_N - idx_p;
        const Treal Rer = inputRe[idx_r];
        const Treal Imr = inputIm[idx_r];

        const size_t idx_s = N - idx_r;
        const Treal Res = inputRe[idx_s];
        const Treal Ims = inputIm[idx_s];

        
        complex_type_t<Treal> Xp;
        complex_type_t<Treal> Yp;

        complex_type_t<Treal> Xr;
        complex_type_t<Treal> Yr;
        
        if(idx_p == 0)
        {
            Xp.x = Rep;
            Xp.y = 0.0;

            Yp.x = Imp;
            Yp.y = 0.0;
        }
        else
        {
            Xp.x = 0.5 * (Rep + Req);
            Xp.y = 0.5 * (Imp - Imq);

            Yp.x = 0.5 * (Imp + Imq);
            Yp.y = -0.5 * (Rep - Req);

            Xr.x = 0.5 * (Rer + Res);
            Xr.y = 0.5 * (Imr - Ims);

            Yr.x = 0.5 * (Imr + Ims);
            Yr.y = -0.5 * (Rer - Res);

        }
        // For the Nyquist, we only set idx_p, so write it last: idx_r
        // equals idx_p in this case.
        outputX[idx_r] = Xr;
        outputY[idx_r] = Yr;
        
        outputX[idx_p] = Xp;
        outputY[idx_p] = Yp;
    }
}

/// Planar data version.
template <typename Treal>
__global__ static void complex2pair_unpack_kernel(const size_t      N,
                                                  const void*       input,
                                                  const size_t     ioffset,
                                                  void*        outputRe,
                                                  void*        outputIm,
                                                  const size_t ooffset)
{
    const size_t idx_p = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const auto half_N = (N + 1) / 2;

    // Planar data is assumed to be out-of-place, so we need only run
    // one set of indices.
    
    if(idx_p < half_N)
    {
        const auto inputRe = (Treal*)input;
        const auto inputIm = inputRe + ioffset;

        const auto Rep = inputRe[idx_p];
        const auto Imp = inputIm[idx_p];
        
        const size_t idx_q = N - idx_p;
        const auto Req = inputRe[idx_q];
        const auto Imq = inputIm[idx_q];

        auto outputXRe = (Treal*)outputRe;
        auto outputYRe = outputXRe + ooffset;
        
        auto outputXIm = (Treal*)outputIm;
        auto outputYIm = outputXIm + ooffset;
        
        complex_type_t<Treal> X;
        complex_type_t<Treal> Y;
                
        if(idx_p == 0)
        {
            X.x = Rep;
            X.y = 0.0;

            Y.x = Imp;
            Y.y = 0.0;
        }
        else
        {
            X.x = 0.5 * (Rep + Req);
            X.y = 0.5 * (Imp - Imq);

            Y.x = 0.5 * (Imp + Imq);
            Y.y = -0.5 * (Rep - Req);
        }
        
        outputXRe[idx_p] = X.x;
        outputXIm[idx_p] = X.y;

        outputYRe[idx_p] = Y.x;
        outputYIm[idx_p] = Y.y;
    }
}


/// Unpack two (Hermitian-symmetric) complex arrays from a full-length complex array for a
/// real-to-complex transform
void complex2pair_unpack(const void* data_p, void*)
{
//    std::cout << "complex2pair" << std::endl;
    const DeviceCallIn* data = (DeviceCallIn*)data_p;

    const size_t idist = data->node->iDist;
    const size_t odist = data->node->oDist;

    void* bufIn0  = data->bufIn[0];
    void* bufOut0 = data->bufOut[0];
    
    void* bufOut1 = data->bufOut[1];
        
    const ptrdiff_t ioffset
        = (data->node->parent->batch % 2 == 0)
        ?  data->node->iDist
        : data->node->inStride[data->node->pairdim];

    const ptrdiff_t ooffset
        = (data->node->parent->batch % 2 == 0)
        ? data->node->oDist
        : data->node->outStride[data->node->pairdim];
    
    const size_t N = data->node->length[0];
    const size_t high_dimension = std::accumulate(
        data->node->length.begin() + 1, data->node->length.end(), 1, std::multiplies<size_t>());
    const size_t batch = data->node->batch;
    
    const size_t block_size = 512;
    size_t       blocks     = (N + block_size - 1) / block_size;

    dim3 grid(blocks, high_dimension, batch);
    dim3 threads(block_size, 1, 1);

    switch(data->node->outArrayType)
    {
    case rocfft_array_type_hermitian_interleaved:
        switch(data->node->precision)
        {
        case rocfft_precision_single:
            complex2pair_unpack_kernel<float><<<grid, threads, 0>>>(N,
                                                                    bufIn0, ioffset,
                                                                    bufOut0, ooffset);
            break;
        case rocfft_precision_double:
            complex2pair_unpack_kernel<double><<<grid, threads, 0>>>(N,
                                                                     bufIn0, ioffset,
                                                                     bufOut0, ooffset);
            break;
        default:
            std::cerr << "invalid precision for complex2pair\n";
            assert(false);
        }
        break;
    case rocfft_array_type_hermitian_planar:
        switch(data->node->precision)
        {
        case rocfft_precision_single:
            complex2pair_unpack_kernel<float><<<grid, threads, 0>>>(N,
                                                                    bufIn0, ioffset,
                                                                    bufOut0, bufOut1, ooffset);
            break;
        case rocfft_precision_double:
            complex2pair_unpack_kernel<double><<<grid, threads, 0>>>(N,
                                                                     bufIn0, ioffset,
                                                                     bufOut0, bufOut1, ooffset);
            break;
        default:
            std::cerr << "invalid precision for complex2pair\n";
            assert(false);
        }
        break;
    default:
        std::cerr << "invalid output type for complex2pair" << std::endl;
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

