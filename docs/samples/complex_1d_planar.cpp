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

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>

#include <hip/hip_runtime_api.h>

#include "rocfft.h"

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define ENUMSTR(x) x, TO_STR(x)


std::string PrintStatus(const rocfft_status status)
{
  std::map<rocfft_status, const char*> StatustoString
    = {{ENUMSTR(rocfft_status_success)},
       {ENUMSTR(rocfft_status_failure)},
       {ENUMSTR(rocfft_status_invalid_arg_value)},
       {ENUMSTR(rocfft_status_invalid_dimensions)},
       {ENUMSTR(rocfft_status_invalid_array_type)},
       {ENUMSTR(rocfft_status_invalid_strides)},
       {ENUMSTR(rocfft_status_invalid_distance)},
       {ENUMSTR(rocfft_status_invalid_offset)}};
  return StatustoString.at(status);
}



int main()
{
    // The problem size
    const size_t N = 8;

    std::cout << "Complex 1d in-place FFT example\n";

    // Initialize data on the host
    std::vector<float> cxre(N);
    std::vector<float> cxim(N);
    for(size_t i = 0; i < N; i++)
    {
        cxre[i] = i;
        cxim[i] = 0;
    }

    std::cout << "Input:\n";
    for(size_t i = 0; i < N; i++)
    {
        std::cout << "( " << cxre[i] << "," << cxim[i] << ") ";
    }
    std::cout << "\n";

    rocfft_setup();

    // Create HIP device object.
    float* xre;
    hipMalloc(&xre, cxre.size() * sizeof(decltype(cxre)::value_type));
    float* xim;
    hipMalloc(&xim, cxim.size() * sizeof(decltype(cxim)::value_type));

    //  Copy data to device
    hipMemcpy(xre, cxre.data(), cxre.size() * sizeof(decltype(cxre)::value_type),
	      hipMemcpyHostToDevice);
    hipMemcpy(xim, cxim.data(), cxim.size() * sizeof(decltype(cxim)::value_type),
	      hipMemcpyHostToDevice);

    std::array<float* ,2> bufs = {xre, xim};

    rocfft_status status;
    
    rocfft_plan_description fdescription;
    status = rocfft_plan_description_create(&fdescription);
    assert(status == rocfft_status_success);

    const std::array<size_t, 1> in_offsets = {0};
    const std::array<size_t, 1> out_offsets = {0};

    const std::array<size_t, 1> in_strides = {1};
    const std::array<size_t, 1> out_strides = {1};

    
    status = rocfft_plan_description_set_data_layout(fdescription,
						     rocfft_array_type_complex_planar,
						     rocfft_array_type_complex_planar,
						     in_offsets.data(),  // size_t* in_offsets
						     out_offsets.data(), // size_t* out_offsets
						     in_strides.size(),  // size_t  in_strides_size
						     in_strides.data(),  // size_t* in_strides,
						     0,    // size_t  in_distance
						     out_strides.size(), // size_t  out_strides_size
						     out_strides.data(), // size_t* out_strides
						     0);   // size_t  out_distance
    std::cout << "status: " << PrintStatus(status) << std::endl;
    assert(status == rocfft_status_success);

    
    // Create plans
    rocfft_plan forward = NULL;
    status = rocfft_plan_create(&forward,
				rocfft_placement_inplace,
				rocfft_transform_type_complex_forward,
				rocfft_precision_single,
				1, // Dimensions
				&N, // lengths
				1, // Number of transforms
				fdescription); // Description
    std::cout << "status: " << PrintStatus(status) << std::endl;
    assert(status == rocfft_status_success);
    
    rocfft_plan_description bdescription = NULL;
    status = rocfft_plan_description_create(&bdescription);
    assert(status == rocfft_status_success);

    status = rocfft_plan_description_set_data_layout(bdescription,
						     rocfft_array_type_complex_planar,
						     rocfft_array_type_complex_planar,
						     in_offsets.data(),  // size_t* in_offsets
						     out_offsets.data(), // size_t* out_offsets
						     in_strides.size(),  // size_t  in_strides_size
						     in_strides.data(),  // size_t* in_strides,
						     0,    // size_t  in_distance
						     out_strides.size(), // size_t  out_strides_size
						     out_strides.data(), // size_t* out_strides
						     0);   // size_t  out_distance
    assert(status == rocfft_status_success);
    
    // Create plans
    rocfft_plan backward = NULL;
    status = rocfft_plan_create(&backward,
                       rocfft_placement_inplace,
                       rocfft_transform_type_complex_inverse,
                       rocfft_precision_single,
                       1, // Dimensions
                       &N, // lengths
                       1, // Number of transforms
                       bdescription); // Description
    std::cout << "status: " << PrintStatus(status) << std::endl;
    assert(status == rocfft_status_success);

    // Execute the forward transform
    status = rocfft_execute(forward,
			    (void**)bufs.data(), // in_buffer
                   NULL, // out_buffer
                   NULL); // execution info
    assert(status == rocfft_status_success);

    // Copy result back to host
    std::vector<float> cyre(N);
    std::vector<float> cyim(N);
    hipMemcpy(cyre.data(), xre, cyre.size() * sizeof(decltype(cyre)::value_type),
	      hipMemcpyDeviceToHost);
    hipMemcpy(cyim.data(), xim, cyre.size() * sizeof(decltype(cyim)::value_type),
	      hipMemcpyDeviceToHost);


    std::cout << "Transformed:\n";
    for(size_t i = 0; i < cyre.size(); i++)
    {
        std::cout << "( " << cyre[i] << "," << cyim[i] << ") ";
    }
    std::cout << "\n";

    // Execute the backward transform
    rocfft_execute(backward,
                   (void**)bufs.data(), // in_buffer
                   NULL, // out_buffer
                   NULL); // execution info
    hipMemcpy(cyre.data(), xre, cyre.size() * sizeof(decltype(cyre)::value_type),
	      hipMemcpyDeviceToHost);
    hipMemcpy(cyim.data(), xim, cyre.size() * sizeof(decltype(cyim)::value_type),
	      hipMemcpyDeviceToHost);
    std::cout << "Transformed back:\n";
    for(size_t i = 0; i < cyre.size(); i++)
    {
        std::cout << "( " << cyre[i] << "," << cyim[i] << ") ";
    }
    std::cout << "\n";

    const float overN = 1.0f / N;
    float       error = 0.0f;
    for(size_t i = 0; i < cxre.size(); i++)
    {
        float diff
            = std::max(std::abs(cxre[i] - cyre[i] * overN), std::abs(cxim[i] - cyim[i] * overN));
        if(diff > error)
        {
            error = diff;
        }
    }
    std::cout << "Maximum error: " << error << "\n";

    hipFree(xre);
    hipFree(xim);

    // Destroy plans
    rocfft_plan_destroy(forward);
    rocfft_plan_destroy(backward);

    rocfft_cleanup();

    return 0;
}
