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

#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../client_utils.h"
#include "accuracy_test.h"
#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"

// Print the test parameters
void print_params(const std::vector<size_t>&    length,
                  const size_t                  istride0,
                  const size_t                  ostride0,
                  const size_t                  nbatch,
                  const rocfft_result_placement place,
                  const rocfft_precision        precision,
                  const rocfft_transform_type   transformType,
                  const rocfft_array_type       itype,
                  const rocfft_array_type       otype)
{
    std::cout << "length:";
    for(const auto& i : length)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "istride0: " << istride0 << "\n";
    std::cout << "ostride0: " << ostride0 << "\n";
    std::cout << "nbatch: " << nbatch << "\n";
    if(place == rocfft_placement_inplace)
        std::cout << "in-place\n";
    else
        std::cout << "out-of-place\n";
    if(precision == rocfft_precision_single)
        std::cout << "single-precision\n";
    else
        std::cout << "double-precision\n";
    switch(transformType)
    {
    case rocfft_transform_type_complex_forward:
        std::cout << "complex forward:\t";
        break;
    case rocfft_transform_type_complex_inverse:
        std::cout << "complex inverse:\t";
        break;
    case rocfft_transform_type_real_forward:
        std::cout << "real forward:\t";
        break;
    case rocfft_transform_type_real_inverse:
        std::cout << "real inverse:\t";
        break;
    }
    switch(itype)
    {
    case rocfft_array_type_complex_interleaved:
        std::cout << "rocfft_array_type_complex_interleaved";
        break;
    case rocfft_array_type_complex_planar:
        std::cout << "rocfft_array_type_complex_planar";
        break;
    case rocfft_array_type_real:
        std::cout << "rocfft_array_type_real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        std::cout << "rocfft_array_type_hermitian_interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        std::cout << "rocfft_array_type_hermitian_planar";
        break;
    case rocfft_array_type_unset:
        std::cout << "rocfft_array_type_unset";
        break;
    }
    std::cout << " -> ";
    switch(otype)
    {
    case rocfft_array_type_complex_interleaved:
        std::cout << "rocfft_array_type_complex_interleaved";
        break;
    case rocfft_array_type_complex_planar:
        std::cout << "rocfft_array_type_complex_planar";
        break;
    case rocfft_array_type_real:
        std::cout << "rocfft_array_type_real";
        break;
    case rocfft_array_type_hermitian_interleaved:
        std::cout << "rocfft_array_type_hermitian_interleaved";
        break;
    case rocfft_array_type_hermitian_planar:
        std::cout << "rocfft_array_type_hermitian_planar";
        break;
    case rocfft_array_type_unset:
        std::cout << "rocfft_array_type_unset";
        break;
    }
    std::cout << std::endl;
}

// Test for comparison between FFTW and rocFFT.
TEST_P(accuracy_test, vs_fftw)
{
    const std::vector<size_t> length    = std::get<0>(GetParam());
    const size_t              istride0  = std::get<1>(GetParam());
    const size_t              ostride0  = std::get<2>(GetParam());
    const size_t              nbatch    = std::get<3>(GetParam());
    const rocfft_precision    precision = std::get<4>(GetParam());

    const std::
        tuple<rocfft_transform_type, rocfft_array_type, rocfft_array_type, rocfft_result_placement>
                                  tranio        = get<5>(GetParam());
    const rocfft_transform_type   transformType = std::get<0>(tranio);
    const rocfft_array_type       itype         = std::get<1>(tranio);
    const rocfft_array_type       otype         = std::get<2>(tranio);
    const rocfft_result_placement place         = std::get<3>(tranio);

    // NB: Input data is row-major.

    const size_t dim = length.size();

    if(verbose)
    {
        print_params(
            length, istride0, ostride0, nbatch, place, precision, transformType, itype, otype);
    }

    // Input data:
    auto ilength = length;
    if(transformType == rocfft_transform_type_real_inverse)
        ilength[dim - 1] = ilength[dim - 1] / 2 + 1;
    const auto cpu_istride = compute_stride(ilength, 1);
    auto       cpu_itype   = make_type_contiguous(itype);
    const auto cpu_idist
        = set_idist(rocfft_placement_notinplace, transformType, length, cpu_istride);

    auto olength = length;
    if(transformType == rocfft_transform_type_real_forward)
        olength[dim - 1] = olength[dim - 1] / 2 + 1;
    const auto cpu_ostride = compute_stride(olength, 1);
    const auto cpu_odist
        = set_odist(rocfft_placement_notinplace, transformType, length, cpu_ostride);
    auto cpu_otype = make_type_contiguous(otype);
    if(verbose > 3)
    {
        std::cout << "CPU  params:\n";
        std::cout << "\tilength:";
        for(auto i : ilength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_istride:";
        for(auto i : cpu_istride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_idist: " << cpu_idist << std::endl;

        std::cout << "\tolength:";
        for(auto i : olength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_ostride:";
        for(auto i : cpu_ostride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tcpu_odist: " << cpu_odist << std::endl;
    }

    // Generate the data:
    auto cpu_input = compute_input<fftwAllocator<char>>(
        precision, cpu_itype, length, cpu_istride, cpu_idist, nbatch);
    auto cpu_input_copy = cpu_input; // copy of input (might get overwritten by FFTW).

    // Compute the Linfinity and L2 norm of the CPU output:
    auto CPU_input_L2Linfnorm
        = LinfL2norm(cpu_input, ilength, nbatch, precision, cpu_itype, cpu_istride, cpu_idist);
    if(verbose > 2)
    {
        std::cout << "CPU Input Linf norm:  " << CPU_input_L2Linfnorm.first << "\n";
        std::cout << "CPU Input L2 norm:    " << CPU_input_L2Linfnorm.second << "\n";
    }
    ASSERT_TRUE(std::isfinite(CPU_input_L2Linfnorm.first));
    ASSERT_TRUE(std::isfinite(CPU_input_L2Linfnorm.second));

    if(verbose > 3)
    {
        std::cout << "CPU input:\n";
        printbuffer(precision, cpu_itype, cpu_input, ilength, cpu_istride, nbatch, cpu_idist);
    }

    // FFTW computation
    // NB: FFTW may overwrite input, even for out-of-place transforms.
    auto cpu_output = fftw_via_rocfft(length,
                                      cpu_istride,
                                      cpu_ostride,
                                      nbatch,
                                      cpu_idist,
                                      cpu_odist,
                                      precision,
                                      transformType,
                                      cpu_input);

    // Compute the Linfinity and L2 norm of the CPU output:
    auto CPU_output_L2Linfnorm
        = LinfL2norm(cpu_output, olength, nbatch, precision, cpu_otype, cpu_ostride, cpu_odist);
    if(verbose > 2)
    {
        std::cout << "CPU Output Linf norm: " << CPU_output_L2Linfnorm.first << "\n";
        std::cout << "CPU Output L2 norm:   " << CPU_output_L2Linfnorm.second << "\n";
    }
    if(verbose > 3)
    {
        std::cout << "CPU output:\n";
        printbuffer(precision, cpu_otype, cpu_output, olength, cpu_ostride, nbatch, cpu_odist);
    }
    ASSERT_TRUE(std::isfinite(CPU_output_L2Linfnorm.first));
    ASSERT_TRUE(std::isfinite(CPU_output_L2Linfnorm.second));

    if(place == rocfft_placement_inplace)
    {
        if(istride0 != ostride0)
        {
            // In-place transforms require identical input and output strides.
            if(verbose)
            {
                std::cout << "istride0: " << istride0 << " ostride0: " << ostride0
                          << " differ; skipped for in-place transforms: skipping test" << std::endl;
            }
            // TODO: mark skipped
            return;
        }
        if((transformType == rocfft_transform_type_real_forward
            || transformType == rocfft_transform_type_real_inverse)
           && (istride0 != 1 || ostride0 != 1))
        {
            // In-place real/complex transforms require unit strides.
            if(verbose)
            {
                std::cout << "istride0: " << istride0 << " ostride0: " << ostride0
                          << " must be unitary for in-place real/complex transforms: skipping test"
                          << std::endl;
            }
            // TODO: mark skipped
            return;
        }
    }

    auto gpu_istride = compute_stride(ilength,
                                      istride0,
                                      place == rocfft_placement_inplace
                                          && transformType == rocfft_transform_type_real_forward);

    auto gpu_ostride = compute_stride(olength,
                                      ostride0,
                                      place == rocfft_placement_inplace
                                          && transformType == rocfft_transform_type_real_inverse);

    const auto gpu_idist = set_idist(place, transformType, length, gpu_istride);
    const auto gpu_odist = set_odist(place, transformType, length, gpu_ostride);

    rocfft_status fft_status = rocfft_status_success;

    // Transform parameters from row-major to column-major for rocFFT:
    auto gpu_length_cm  = length;
    auto gpu_ilength_cm = ilength;
    auto gpu_olength_cm = olength;
    auto gpu_istride_cm = gpu_istride;
    auto gpu_ostride_cm = gpu_ostride;
    for(int idx = 0; idx < dim / 2; ++idx)
    {
        const auto toidx = dim - idx - 1;
        std::swap(gpu_istride_cm[idx], gpu_istride_cm[toidx]);
        std::swap(gpu_ostride_cm[idx], gpu_ostride_cm[toidx]);
        std::swap(gpu_length_cm[idx], gpu_length_cm[toidx]);
        std::swap(gpu_ilength_cm[idx], gpu_ilength_cm[toidx]);
        std::swap(gpu_olength_cm[idx], gpu_olength_cm[toidx]);
    }
    if(verbose > 1)
    {
        std::cout << "GPU params:\n";
        std::cout << "\tgpu_ilength_cm:";
        for(auto i : gpu_ilength_cm)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tgpu_istride_cm:";
        for(auto i : gpu_istride_cm)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tgpu_idist: " << gpu_idist << std::endl;

        std::cout << "\tgpu_olength_cm:";
        for(auto i : gpu_olength_cm)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tgpu_ostride_cm:";
        for(auto i : gpu_ostride_cm)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tgpu_odist: " << gpu_odist << std::endl;
    }

    // Create FFT description
    rocfft_plan_description desc = NULL;
    fft_status                   = rocfft_plan_description_create(&desc);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT description creation failure";
    const std::vector<size_t> ioffset = {0, 0};
    const std::vector<size_t> ooffset = {0, 0};
    fft_status                        = rocfft_plan_description_set_data_layout(desc,
                                                         itype,
                                                         otype,
                                                         ioffset.data(),
                                                         ooffset.data(),
                                                         gpu_istride_cm.size(),
                                                         gpu_istride_cm.data(),
                                                         gpu_idist,
                                                         gpu_ostride_cm.size(),
                                                         gpu_ostride_cm.data(),
                                                         gpu_odist);
    ASSERT_TRUE(fft_status == rocfft_status_success)
        << "rocFFT data layout failure: " << fft_status;

    // Create the plan
    rocfft_plan gpu_plan = NULL;
    fft_status           = rocfft_plan_create(&gpu_plan,
                                    place,
                                    transformType,
                                    precision,
                                    gpu_length_cm.size(),
                                    gpu_length_cm.data(),
                                    nbatch,
                                    desc);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan creation failure";

    // Create execution info
    rocfft_execution_info info = NULL;
    fft_status                 = rocfft_execution_info_create(&info);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT execution info creation failure";
    size_t workbuffersize = 0;
    fft_status            = rocfft_plan_get_work_buffer_size(gpu_plan, &workbuffersize);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT get buffer size get failure";

    // Number of value in input and output variables.
    const size_t isize = nbatch * gpu_idist;
    const size_t osize = nbatch * gpu_odist;

    // Sizes of individual input and output variables
    const size_t isize_t = var_size(precision, itype);
    const size_t osize_t = var_size(precision, otype);

    // Check if the problem fits on the device; if it doesn't skip it.
    if(!vram_fits_problem(isize * isize_t,
                          (place == rocfft_placement_inplace) ? 0 : osize * osize_t,
                          workbuffersize))
    {
        rocfft_plan_destroy(gpu_plan);
        rocfft_plan_description_destroy(desc);
        rocfft_execution_info_destroy(info);

        if(verbose)
        {
            std::cout << "Problem won't fit on device; skipped\n";
        }
        // TODO: mark as skipped via gtest.
        return;
    }

    hipError_t hip_status = hipSuccess;

    // Allocate work memory and associate with the execution info
    void* wbuffer = NULL;
    if(workbuffersize > 0)
    {
        hip_status = hipMalloc(&wbuffer, workbuffersize);
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
        fft_status = rocfft_execution_info_set_work_buffer(info, wbuffer, workbuffersize);
        ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT set work buffer failure";
    }

    // Formatted input data:
    auto gpu_input = allocate_host_buffer<fftwAllocator<char>>(
        precision, itype, length, gpu_istride, gpu_idist, nbatch);

    // Copy from contiguous_input to input.
    copy_buffers(cpu_input_copy,
                 gpu_input,
                 ilength,
                 nbatch,
                 precision,
                 cpu_itype,
                 cpu_istride,
                 cpu_idist,
                 itype,
                 gpu_istride,
                 gpu_idist);

    if(verbose > 4)
    {
        std::cout << "GPU input:\n";
        printbuffer(precision, itype, gpu_input, ilength, gpu_istride, nbatch, gpu_idist);
    }
    if(verbose > 5)
    {
        std::cout << "flat GPU input:\n";
        printbuffer_flat(precision, itype, gpu_input, gpu_idist);
    }

    // GPU input and output buffers:
    std::vector<void*> ibuffer = alloc_buffer(precision, itype, gpu_idist, nbatch);
    std::vector<void*> obuffer = (place == rocfft_placement_inplace)
                                     ? ibuffer
                                     : alloc_buffer(precision, otype, gpu_odist, nbatch);

    // Copy the input data to the GPU:
    for(int idx = 0; idx < gpu_input.size(); ++idx)
    {
        hip_status = hipMemcpy(
            ibuffer[idx], gpu_input[idx].data(), gpu_input[idx].size(), hipMemcpyHostToDevice);
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";
    }

    // Execute the transform:
    fft_status = rocfft_execute(gpu_plan, // plan
                                (void**)ibuffer.data(), // in buffers
                                (void**)obuffer.data(), // out buffers
                                info); // execution info
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan execution failure";

    // Copy the data back to the host:
    auto gpu_output = allocate_host_buffer<fftwAllocator<char>>(
        precision, otype, olength, gpu_ostride, gpu_odist, nbatch);
    for(int idx = 0; idx < gpu_output.size(); ++idx)
    {
        hip_status = hipMemcpy(
            gpu_output[idx].data(), obuffer[idx], gpu_output[idx].size(), hipMemcpyDeviceToHost);
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";
    }

    if(verbose > 2)
    {
        std::cout << "GPU output:\n";
        printbuffer(precision, otype, gpu_output, olength, gpu_ostride, nbatch, gpu_odist);
    }
    if(verbose > 5)
    {
        std::cout << "flat GPU output:\n";
        printbuffer_flat(precision, otype, gpu_output, gpu_odist);
    }

    // Compute the Linfinity and L2 norm of the GPU output:
    auto L2LinfnormGPU
        = LinfL2norm(gpu_output, olength, nbatch, precision, otype, gpu_ostride, gpu_odist);
    if(verbose > 2)
    {
        std::cout << "GPU output Linf norm: " << L2LinfnormGPU.first << "\n";
        std::cout << "GPU output L2 norm:   " << L2LinfnormGPU.second << "\n";
    }

    ASSERT_TRUE(std::isfinite(L2LinfnormGPU.first));
    ASSERT_TRUE(std::isfinite(L2LinfnormGPU.second));

    // Compute the l-infinity and l-2 distance between the CPU and GPU output:
    auto linfl2diff = LinfL2diff(cpu_output,
                                 gpu_output,
                                 olength,
                                 nbatch,
                                 precision,
                                 cpu_otype,
                                 cpu_ostride,
                                 cpu_odist,
                                 otype,
                                 gpu_ostride,
                                 gpu_odist);
    if(verbose > 1)
    {
        std::cout << "L2 diff: " << linfl2diff.first << "\n";
        std::cout << "Linf diff: " << linfl2diff.second << "\n";
    }

    auto total_length = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());

    // TODO: handle case where norm is zero?
    EXPECT_TRUE(linfl2diff.first / (CPU_output_L2Linfnorm.first * log(total_length))
                < type_epsilon(precision))
        << "Linf test failed.  Linf:" << linfl2diff.first << "\tnormalized Linf: "
        << linfl2diff.first / (CPU_output_L2Linfnorm.first * log(total_length))
        << "\tepsilon: " << type_epsilon(precision);
    EXPECT_TRUE(linfl2diff.second / (CPU_output_L2Linfnorm.second * sqrt(log(total_length)))
                < type_epsilon(precision))
        << "L2 test failed. L2: " << linfl2diff.second << "\tnormalized L2: "
        << linfl2diff.second / (CPU_output_L2Linfnorm.second * sqrt(log(total_length)))
        << "\tepsilon: " << type_epsilon(precision);

    rocfft_plan_destroy(gpu_plan);
    gpu_plan = NULL;
    rocfft_plan_description_destroy(desc);
    desc = NULL;
    rocfft_execution_info_destroy(info);
    info = NULL;
    if(wbuffer)
    {
        hipFree(wbuffer);
        wbuffer = NULL;
    }
    for(auto& buf : ibuffer)
    {
        hipFree(buf);
        buf = NULL;
    }
    if(place != rocfft_placement_inplace)
    {
        for(auto& buf : obuffer)
        {
            hipFree(buf);
            buf = NULL;
        }
    }
}
