/******************************************************************************
* Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include <complex>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hipfft.h>

// Kernel for initializing the real-valued input data on the GPU.
__global__ void initdata(double* x, const size_t Nx, const size_t Ny, const size_t rstride)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx < Nx && idy < Ny)
    {
        const size_t pos = idx * rstride + idy;
	//x[pos]           = idx + idy;
    }
}

// Helper function for determining grid dimensions
template <typename Tint1, typename Tint2>
Tint1 ceildiv(const Tint1 nominator, const Tint2 denominator)
{
    return (nominator + denominator - 1) / denominator;
}

int main(int argc, char* argv[])
{
    std::cout << "hipfft 2D double-precision real-to-complex transform\n";

    const size_t Nx = (argc < 2) ? 4 : atoi(argv[1]);
    const size_t Ny = (argc < 3) ? 3 : atoi(argv[2]);
    const bool inplace = (argc < 4) ? true : atoi(argv[3]);
    
    std::cout << "Nx: " << Nx << "\tNy: " << Ny
              << "\tin-place: " << inplace << std::endl;

    const size_t Nycomplex = Ny / 2 + 1;
    const size_t rstride   = inplace ? (Nycomplex * 2) : Ny;

    std::cout << "rstride: " << rstride << std::endl;
    
    const size_t real_bytes    = sizeof(double) * Nx * rstride;
    const size_t complex_bytes = 2 * sizeof(double) * Nx * Nycomplex;

    hipError_t rt;
    
    double*    x = NULL;
    rt = hipMalloc(&x, real_bytes);
    assert(rt == HIP_SUCCESS);
        
    // Inititalize the data on the device
    const dim3 blockdim(32, 32);
    const dim3 griddim(ceildiv(Nx, blockdim.x), ceildiv(Ny, blockdim.y));
    hipLaunchKernelGGL(initdata, griddim, blockdim, 0, 0, x, Nx, Ny, rstride);
    hipDeviceSynchronize();
    rt = hipGetLastError();
    assert(rt == hipSuccess);

    // Copy the input data to the host and output
    std::vector<double> rdata(Nx * rstride);
    rt = hipMemcpy(rdata.data(), x, real_bytes, hipMemcpyDefault);
    assert(rt == HIP_SUCCESS);

    std::cout << "input:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < rstride; j++)
        {
            auto pos = i * rstride + j;
            std::cout << rdata[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    
    // Create plan:
    hipfftResult rc   = HIPFFT_SUCCESS;
    hipfftHandle plan = NULL;
    rc                = hipfftCreate(&plan);
    assert(rc == HIPFFT_SUCCESS);
    rc = hipfftPlan2d(&plan, // plan handle
                      Nx, // transform length
                      Ny, // transform length
                      HIPFFT_D2Z); // transform type (HIPFFT_R2C for single-precisoin)
    assert(rc == HIPFFT_SUCCESS);

    // Set up the destination buffer:
    hipfftDoubleComplex* y = inplace ? (hipfftDoubleComplex*)x : NULL;
    if(!inplace) {
        rt = hipMalloc(&x, complex_bytes);
        assert(rt == HIP_SUCCESS);
    }
    
    // Execute the forward transform:
    // hipfftExecD2Z: double precision, hipfftExecR2C: single-precision
    rc = hipfftExecD2Z(plan, x, y);
    assert(rc == HIPFFT_SUCCESS);

    // copy the output data to the host and output:
    std::vector<std::complex<double>> cdata(Nx * Ny);
    hipMemcpy(cdata.data(), y, complex_bytes, hipMemcpyDeviceToHost);
    std::cout << "output:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Nycomplex; j++)
        {
            auto pos = i * Nycomplex + j;
            std::cout << cdata[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    // Execute the backward transform:
    rc = hipfftExecZ2D(plan, y, x);
    assert(rc == HIPFFT_SUCCESS);

    std::cout << "back to (scaled) input:\n";
    hipMemcpy(rdata.data(), x, real_bytes, hipMemcpyDeviceToHost);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            auto pos = i * rstride + j;
            std::cout << rdata[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
    
    hipfftDestroy(plan);
    
    hipFree(x);
    x = NULL;
    if(y != NULL) {
        hipFree(y);
        y = NULL;
    }
    
    return 0;
}
