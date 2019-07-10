
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
__global__
void initreal(double* x, const size_t Nx, const size_t Ny, const size_t rstride)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx < Nx && idy < Ny) {
        const size_t pos = idx * rstride + idy;
        x[pos] = idx + idy;
    }
}

// Helper function for determining grid dimensions
template<typename Tint1, typename Tint2>
Tint1 ceildiv(const Tint1 nominator, const Tint2 denominator)
{
    return (nominator + denominator - 1) / denominator;
}
    
int main()
{
    std::cout << "hipfft 2D double-precision creal-to-complex transform\n";

    const size_t Nx         = 4;
    const size_t Ny         = 5;

    std::cout << "Nx: " << Nx << "\tNy " << Ny << std::endl;
    
    const size_t Nycomplex = Ny / 2 + 1;
    const size_t rstride = Nycomplex * 2; // Ny for out-of-place

    std::cout << "Nycomplex: " << Nycomplex << "\trstride " << rstride << std::endl;
    
    const size_t real_bytes = sizeof(double) * Nx * rstride;
    const size_t complex_bytes = 2 * sizeof(double) * Nx * Ny;

    std::cout << "input:\n";
    double* x;
    hipError_t rt;
    rt = hipMalloc(&x, real_bytes);
    assert(rt == HIP_SUCCESS);
    const dim3 blockdim(32, 32);
    const dim3 griddim(ceildiv(Nx, blockdim.x), ceildiv(Ny, blockdim.y));

    hipLaunchKernelGGL(initreal,
                       blockdim,
                       griddim,
                       0,0,
                       x, Nx, Ny, rstride);
    hipDeviceSynchronize();
    rt = hipGetLastError();
    assert(rt == hipSuccess);
    
    std::vector<double> rdata(Nx * rstride);
    rt = hipMemcpy(rdata.data(), x, real_bytes, hipMemcpyDefault);
    assert(rt == HIP_SUCCESS);
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

    hipfftResult rc = HIPFFT_SUCCESS;

    hipfftHandle plan = NULL;
    rc                = hipfftCreate(&plan);
    assert(rc == HIPFFT_SUCCESS);
    rc = hipfftPlan2d(&plan,       // plan handle
                      Nx,          // transform length
                      Ny,          // transform length
                      HIPFFT_D2Z); // transform type (HIPFFT_R2C for single-precisoin)
    assert(rc == HIPFFT_SUCCESS);

    // Execute plan:
    // D2Z: double precision, R2C: single-precision
    rc = hipfftExecD2Z(plan, x, (hipfftDoubleComplex*)x); 
    assert(rc == HIPFFT_SUCCESS);

    std::cout << "output:\n";
    std::vector<std::complex<double>> cdata(Nx * Ny);
    hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDeviceToHost);
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

    hipfftDestroy(plan);
    hipFree(x);

    return 0;
}
