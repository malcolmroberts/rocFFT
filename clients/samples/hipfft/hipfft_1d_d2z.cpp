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

int main()
{
    std::cout << "hipfft 1D double-precision real-to-complex transform\n";

    const size_t N         = 8;
    const size_t Ncomplex  = N / 2 + 1;
    
    std::vector<double> rdata(N);
    size_t real_bytes = sizeof(decltype(rdata)::value_type) * rdata.size();
    std::vector<std::complex<decltype(rdata)::value_type>> cdata(Ncomplex);
    size_t complex_bytes = sizeof(std::complex<decltype(rdata)::value_type>) * cdata.size();

    std::cout << "input:\n";
    for(size_t i = 0; i < rdata.size(); i++)
    {
        rdata[i] = i;
    }
    for(size_t i = 0; i < rdata.size(); i++)
    {
        std::cout << rdata[i] << " ";
    }
    std::cout << std::endl;

    hipfftResult rc = HIPFFT_SUCCESS;

    // Create HIP device object and copy data to device:
    decltype(rdata)::value_type* x; 
    hipMalloc(&x, complex_bytes);
    hipMemcpy(x, rdata.data(), real_bytes, hipMemcpyHostToDevice);

    hipfftHandle plan = NULL;
    rc                = hipfftCreate(&plan);
    assert(rc == HIPFFT_SUCCESS);
    rc = hipfftPlan1d(&plan,      // plan handle
                      N,          // transform length
                      HIPFFT_D2Z, // transform type (HIPFFT_R2C for single-precisoin)
                      1);         // number of transforms
    assert(rc == HIPFFT_SUCCESS);

    // Execute plan:
    // D2Z: double precision, R2C: for single-precision
    // Direction is implied by real-to-complex direction
    rc = hipfftExecD2Z(plan, x, (hipfftDoubleComplex*) x); 
    assert(rc == HIPFFT_SUCCESS);

    std::cout << "output:\n";

    hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDeviceToHost);
    for(size_t i = 0; i < cdata.size(); i++)
    {
        std::cout << cdata[i] << " ";
    }
    std::cout << std::endl;

    hipfftDestroy(plan);
    hipFree(x);

    return 0;
}
