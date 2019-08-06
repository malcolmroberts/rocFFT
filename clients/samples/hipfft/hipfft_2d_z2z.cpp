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
__global__ void initdata(hipfftDoubleComplex* x, const int Nx, const int Ny)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx < Nx && idy < Ny)
    {
        const int pos = idx * Ny + idy;
        x[pos].x      = idx + idy;
        x[pos].y      = 0.0;
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
    std::cout << "hipfft 2D double-precision complex-to-complex transform\n";

    const size_t Nx = (argc < 2) ? 4 : atoi(argv[1]);
    const size_t Ny = (argc < 3) ? 3 : atoi(argv[2]);
    const bool inplace = (argc < 4) ? true : atoi(argv[3]);

    std::vector<std::complex<double>> cdata(Nx * Ny);
    size_t complex_bytes = sizeof(decltype(cdata)::value_type) * cdata.size();

    hipError_t rt;
    
    // Create HIP device object and copy data to device:
    // hipfftComplex for single-precision
    hipfftDoubleComplex* x = NULL;
    rt = hipMalloc(&x, complex_bytes);
    assert(rt == HIP_SUCCESS);

    // Inititalize the data on the device
    const dim3 blockdim(32, 32);
    const dim3 griddim(ceildiv(Nx, blockdim.x), ceildiv(Ny, blockdim.y));
    hipLaunchKernelGGL(initdata, griddim, blockdim, 0, 0, x, Nx, Ny);
    hipDeviceSynchronize();
    rt = hipGetLastError();
    assert(rt == hipSuccess);

    std::cout << "input:\n";
    hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDefault);
    for(int i = 0; i < Nx; i++)
    {
        for(int j = 0; j < Ny; j++)
        {
            int pos = i * Ny + j;
            std::cout << cdata[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    // Create plan
    hipfftResult rc   = HIPFFT_SUCCESS;
    hipfftHandle plan = NULL;
    rc                = hipfftCreate(&plan);
    assert(rc == HIPFFT_SUCCESS);
    rc = hipfftPlan2d(&plan,       // plan handle
                      Nx,          // transform length
                      Ny,          // transform length
                      HIPFFT_Z2Z); // transform type (HIPFFT_C2C for single-precisoin)
    assert(rc == HIPFFT_SUCCESS);

    // Set up the destination buffer:
    hipfftDoubleComplex* y = inplace ? x : NULL;
    if(!inplace) {
        rt = hipMalloc(&x, complex_bytes);
        assert(rt == HIP_SUCCESS);
    }
    
    // Execute forward transform:
    // hipfftExecZ2Z: double precision, hipfftExecC2C: for single-precision
    rc = hipfftExecZ2Z(plan, x, y, HIPFFT_FORWARD);
    assert(rc == HIPFFT_SUCCESS);

    std::cout << "output:\n";
    hipMemcpy(cdata.data(), y, complex_bytes, hipMemcpyDeviceToHost);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            auto pos = i * Ny + j;
            std::cout << cdata[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    // Backward transform
    rc = hipfftExecZ2Z(plan, x, x, HIPFFT_BACKWARD);
    assert(rc == HIPFFT_SUCCESS);

    std::cout << "back to (scaled) input:\n";
    hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDefault);
    for(int i = 0; i < Nx; i++)
    {
        for(int j = 0; j < Ny; j++)
        {
            int pos = i * Ny + j;
            std::cout << cdata[pos] << " ";
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
