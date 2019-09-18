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

#include <algorithm>
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <unistd.h>
#include <vector>
#include <complex>

#include "fftw_transform.h"
#include "rocfft.h"
#include "rocfft_against_fftw.h"

using ::testing::ValuesIn;

// Set parameters

static std::vector<std::vector<size_t>> pow2_range
    = {{2, 4}, {8, 16}, {32, 128}, {256, 512}, {1024, 2048}, {4096, 8192}};
static std::vector<std::vector<size_t>> pow3_range  = {{3, 9}, {27, 81}, {243, 729}, {2187, 6561}};
static std::vector<std::vector<size_t>> pow5_range  = {{5, 25}, {125, 625}, {3125, 15625}};
static std::vector<std::vector<size_t>> prime_range = {
    {7, 25}, {11, 625}, {13, 15625}, {1, 11}, {11, 1}, {8191, 243}, {7, 11}, {7, 32}, {1009, 1009}};

static size_t batch_range[] = {1};

static size_t stride_range[] = {1}; // 1: assume packed data

static rocfft_result_placement placeness_range[]
    = {rocfft_placement_notinplace, rocfft_placement_inplace};

// Real/complex transform test framework is only set up for out-of-place transforms:
// TODO: fix the test suite and add coverage for in-place transforms.
static rocfft_result_placement rc_placeness_range[] = {rocfft_placement_notinplace, rocfft_placement_inplace};

static rocfft_transform_type transform_range[]
    = {rocfft_transform_type_complex_forward, rocfft_transform_type_complex_inverse};

static data_pattern pattern_range[] = {sawtooth};

// Test suite classes:

class accuracy_test_complex_2D : public ::testing::TestWithParam<std::tuple<std::vector<size_t>,
                                                                            size_t,
                                                                            rocfft_result_placement,
                                                                            size_t,
                                                                            data_pattern,
                                                                            rocfft_transform_type>>
{
protected:
    void SetUp() override
    {
        rocfft_setup();
    }
    void TearDown() override
    {
        rocfft_cleanup();
    }
};
class accuracy_test_real_2D
    : public ::testing::TestWithParam<
          std::tuple<std::vector<size_t>, size_t, rocfft_result_placement, size_t, data_pattern>>
{
protected:
    void SetUp() override
    {
        rocfft_setup();
    }
    void TearDown() override
    {
        rocfft_cleanup();
    }
};

//  Complex to complex

// Templated test function for complex to complex:
template <typename Tfloat>
void normal_2D_complex_interleaved_to_complex_interleaved(std::vector<size_t>     length,
                                                          size_t                  batch,
                                                          rocfft_result_placement placeness,
                                                          rocfft_transform_type   transform_type,
                                                          size_t                  stride,
                                                          data_pattern            pattern)
{
    size_t total_size
        = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
    if(total_size * sizeof(Tfloat) * 2 >= 2e8)
    {
        // printf("No test is really launched; MB byte size = %f is too big; will
        // return \n", total_size * sizeof(Tfloat) * 2/1e6);
        return; // memory size over 200MB is too big
    }
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(stride);
    output_strides.push_back(stride);
    for(int i = 1; i < length.size(); i++)
    {
        input_strides.push_back(input_strides[i - 1] * length[i - 1]);
        output_strides.push_back(output_strides[i - 1] * length[i - 1]);
    }

    size_t            idist          = 0;
    size_t            odist          = 0;
    rocfft_array_type in_array_type  = rocfft_array_type_complex_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_complex_interleaved;

    complex_to_complex<Tfloat>(pattern,
                               transform_type,
                               length,
                               batch,
                               input_strides,
                               output_strides,
                               idist,
                               odist,
                               in_array_type,
                               out_array_type,
                               placeness);
}

// Implemetation of complex-to-complex tests for float and double:

TEST_P(accuracy_test_complex_2D,
       normal_2D_complex_interleaved_to_complex_interleaved_single_precision)
{
    std::vector<size_t>     length        = std::get<0>(GetParam());
    size_t                  batch          = std::get<1>(GetParam());
    rocfft_result_placement placeness      = std::get<2>(GetParam());
    size_t                  stride         = std::get<3>(GetParam());
    data_pattern            pattern        = std::get<4>(GetParam());
    rocfft_transform_type   transform_type = std::get<5>(GetParam());

    try
    {
        normal_2D_complex_interleaved_to_complex_interleaved<float>(
            length, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_complex_2D,
       normal_2D_complex_interleaved_to_complex_interleaved_double_precision)
{
    std::vector<size_t>     length        = std::get<0>(GetParam());
    size_t                  batch          = std::get<1>(GetParam());
    rocfft_result_placement placeness      = std::get<2>(GetParam());
    size_t                  stride         = std::get<3>(GetParam());
    data_pattern            pattern        = std::get<4>(GetParam());
    rocfft_transform_type   transform_type = std::get<5>(GetParam());

    try
    {
        normal_2D_complex_interleaved_to_complex_interleaved<double>(
            length, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Populate test cases from parameter combinations:

// Real to complex

// Templated test function for real to complex:
template <typename Tfloat>
void normal_2D_real_to_complex_interleaved(std::vector<size_t>     length,
                                           size_t                  batch,
                                           rocfft_result_placement placeness,
                                           rocfft_transform_type   transform_type,
                                           size_t                  stride,
                                           data_pattern            pattern)
{
    using fftw_complex_type = typename fftw_trait<Tfloat>::fftw_complex_type;
    
    const size_t Nx = length[1];
    const size_t Ny = length[0];
    const bool   inplace = placeness == rocfft_placement_inplace;

    // TODO: add coverage for non-unit stride.
    ASSERT_TRUE(stride == 1) << "Failure: test assumes contiguous data:";
    
    // TODO: add logic to deal with discontiguous data in Nystride
    const size_t Nycomplex = Ny / 2 + 1;
    const size_t Nystride = inplace ? 2 * Nycomplex : Ny;

    // Dimension configuration:
    std::array<fftw_iodim64, 2> dims;
    dims[1].n  = Ny;
    dims[1].is = stride;
    dims[1].os = stride;
    dims[0].n  = Nx;
    dims[0].is = Nystride * dims[1].is;
    dims[0].os = (dims[1].n/2 + 1) * dims[1].os;

    const size_t isize = dims[0].n * dims[0].is;
    const size_t osize = dims[0].n * dims[0].os;

    if(inplace)
        std::cout << "in-place\n";
    else
        std::cout << "out-of-place\n";
    for (int i = 0; i < dims.size(); ++i) {
        std::cout << "dim " << i << std::endl;
        std::cout << "\tn: " << dims[i].n << std::endl;
        std::cout << "\tis: " << dims[i].is << std::endl;
        std::cout << "\tos: " << dims[i].os << std::endl;
    }
    std::cout << "isize: " << isize << "\n";
    std::cout << "osize: " << osize << "\n";
    
    // Batch configuration:
    std::array<fftw_iodim64, 1> howmany_dims;
    howmany_dims[0].n = batch;
    howmany_dims[0].is = isize;
    howmany_dims[0].os = osize;

    // Set up buffers:
    // Local data buffer:
    Tfloat* cpu_in = fftw_alloc_type<Tfloat>(isize);
    // Output buffer
    std::complex<Tfloat>* cpu_out = inplace ? (std::complex<Tfloat>*) cpu_in : 
        fftw_alloc_type<std::complex<Tfloat>>(osize);

    Tfloat* gpu_in = NULL;
    hipError_t hip_status = hipSuccess;
    hip_status = hipMalloc(&gpu_in, isize * sizeof(Tfloat));
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
    std::complex<Tfloat>* gpu_out = inplace ? (std::complex<Tfloat>*)gpu_in : NULL;
    if(!inplace)
    {
        hip_status = hipMalloc(&gpu_out, osize * sizeof(std::complex<Tfloat>));
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
    }

    // Set up the CPU plan:
    auto cpu_plan = fftw_plan_guru64_r2c<Tfloat>(dims.size(),
                                                 dims.data(),
                                                 howmany_dims.size(),
                                                 howmany_dims.data(),
                                                 cpu_in,
                                                 reinterpret_cast<fftw_complex_type*>(cpu_out),
                                                 FFTW_ESTIMATE);
    ASSERT_TRUE(cpu_plan != NULL) << "FFTW plan creation failure";
    
    // Set up the GPU plan:
    rocfft_status fft_status = rocfft_status_success;
    rocfft_plan forward = NULL;
    fft_status = rocfft_plan_create(&forward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_real_forward,
                                precision_selector<Tfloat>(),
                                2, // Dimensions
                                length.data(), // lengths
                                1, // Number of transforms
                                NULL); // Description  // FIXME:
                                       // needed for strides!
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan creation failure";

    // The real-to-complex transform uses work memory, which is passed
    // via a rocfft_execution_info struct.
    rocfft_execution_info forwardinfo = NULL;
    fft_status = rocfft_execution_info_create(&forwardinfo);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT execution info creation failure";
    size_t forwardworkbuffersize = 0;
    fft_status = rocfft_plan_get_work_buffer_size(forward, &forwardworkbuffersize);
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT get buffer size get failure";
    void* forwardwbuffer = NULL;
    if(forwardworkbuffersize > 0)
    {
        hip_status = hipMalloc(&forwardwbuffer, forwardworkbuffersize);
        ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
        fft_status = rocfft_execution_info_set_work_buffer(forwardinfo, forwardwbuffer,
                                                       forwardworkbuffersize);
        ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT set work buffer failure";
    }
    
    // Set up the data:
    std::fill(cpu_in, cpu_in + isize, 0.0);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            // TODO: make pattern variable?
            Tfloat val = i + j;
            cpu_in[i * Nystride + j] = val;
        }
    }

    // for(int i = 0; i < isize; ++i) {
    //     std::cout << cpu_in[i] << " ";
    // }
    // std::cout << std::endl;
    // for(size_t i = 0; i < Nx; i++)
    // {
    //     for(size_t j = 0; j < Ny; j++)
    //     {
    //         std::cout << cpu_in[i * Nystride + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    
    hip_status = hipMemcpy(gpu_in, cpu_in, isize * sizeof(Tfloat), hipMemcpyHostToDevice);
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";
    
    // Execute the GPU transform:
    fft_status = rocfft_execute(forward, // plan
                            (void**)&gpu_in, // in_buffer
                            (void**)&gpu_out, // out_buffer
                            forwardinfo); // execution info
    ASSERT_TRUE(fft_status == rocfft_status_success) << "rocFFT plan execution failure";

    // Execute the CPU transform:
    fftw_execute_type<Tfloat>(cpu_plan);

    // std::cout << "cpu_out:\n";
    // for(size_t i = 0; i < Nx; i++)
    // {
    //     for(size_t j = 0; j < Nycomplex; j++)
    //     {
    //         std::cout << cpu_out[i * Nycomplex + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    
    // Copy the data back and compare:
    fftw_vector<std::complex<Tfloat>> gpu_out_comp(osize);
    hip_status = hipMemcpy(gpu_out_comp.data(), gpu_out, osize * sizeof(std::complex<Tfloat>),
                           hipMemcpyDeviceToHost);
    ASSERT_TRUE(hip_status == hipSuccess) << "hipMalloc failure";

    // std::cout << "gpu_out:\n";
    // for(size_t i = 0; i < Nx; i++)
    // {
    //     for(size_t j = 0; j < Nycomplex; j++)
    //     {
    //         std::cout << gpu_out_comp[i * Nycomplex + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    Tfloat L2diff = 0.0;
    Tfloat Linfdiff = 0.0;
    Tfloat L2norm = 0.0;
    Tfloat Linfnorm = 0.0;
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Nycomplex; j++)
        {
            const int pos = i * Nycomplex + j;
            Tfloat diff= abs(cpu_out[pos] - gpu_out_comp[pos]);
            Linfnorm = std::max(std::abs(cpu_out[pos]), Linfnorm);
            L2norm +=  std::abs(cpu_out[pos]) * std::abs(cpu_out[pos]);
            Linfdiff = std::max(diff, Linfdiff);
            L2diff += diff * diff;
        }
    }
    L2norm = sqrt(L2norm);
    L2diff = sqrt(L2diff);

    Tfloat L2error = L2diff / (L2norm * sqrt(log(Nx * Ny)));
    Tfloat Linferror = Linfdiff / (Linfnorm * log(Nx * Ny));
    std::cout << "relative L2 error: " << L2error << std::endl;
    std::cout << "relative Linf error: " << Linferror << std::endl;

    EXPECT_TRUE(L2error < type_epsilon<Tfloat>())
        << "Tolerance failure: L2error" << L2error << ", tolerance: " << type_epsilon<Tfloat>(); 
    EXPECT_TRUE(Linferror < type_epsilon<Tfloat>())
        << "Tolerance failure: Linferror" << Linferror << ", tolerance: " << type_epsilon<Tfloat>();
    
    // Free GPU memory:
    hipFree(gpu_in);
    fftw_free(cpu_in);
    if(!inplace)
    {
        hipFree(gpu_out);
        fftw_free(cpu_out);
    }
    if (forwardwbuffer != NULL)
    {
        hipFree(forwardwbuffer);
    }
    
    // Destroy plans:
    rocfft_plan_destroy(forward);
    fftw_destroy_plan_type(cpu_plan);
}

// Implemetation of real-to-complex tests for float and double:

TEST_P(accuracy_test_real_2D, normal_2D_real_to_complex_interleaved_single_precision)
{
    std::vector<size_t>     length        = std::get<0>(GetParam());
    size_t                  batch          = std::get<1>(GetParam());
    rocfft_result_placement placeness      = std::get<2>(GetParam());
    size_t                  stride         = std::get<3>(GetParam());
    data_pattern            pattern        = std::get<4>(GetParam());
    rocfft_transform_type   transform_type = rocfft_transform_type_real_forward;

    try
    {
        normal_2D_real_to_complex_interleaved<float>(
            length, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real_2D, normal_2D_real_to_complex_interleaved_double_precision)
{
    std::vector<size_t>     length        = std::get<0>(GetParam());
    size_t                  batch          = std::get<1>(GetParam());
    rocfft_result_placement placeness      = std::get<2>(GetParam());
    size_t                  stride         = std::get<3>(GetParam());
    data_pattern            pattern        = std::get<4>(GetParam());
    rocfft_transform_type   transform_type = rocfft_transform_type_real_forward;

    try
    {
        normal_2D_real_to_complex_interleaved<double>(
            length, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Complex to real

// Templated test function for complex to real:
template <typename Tfloat>
void normal_2D_complex_interleaved_to_real(std::vector<size_t>     length,
                                           size_t                  batch,
                                           rocfft_result_placement placeness,
                                           rocfft_transform_type   transform_type,
                                           size_t                  stride,
                                           data_pattern            pattern)
{
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(stride);
    output_strides.push_back(stride);

    size_t            idist          = 0; // 0 means the data are densely packed
    size_t            odist          = 0; // 0 means the data are densely packed
    rocfft_array_type in_array_type  = rocfft_array_type_hermitian_interleaved;
    rocfft_array_type out_array_type = rocfft_array_type_real;

    complex_to_real<Tfloat>(pattern,
                            transform_type,
                            length,
                            batch,
                            input_strides,
                            output_strides,
                            idist,
                            odist,
                            in_array_type,
                            out_array_type,
                            placeness);
}

// Implemetation of real-to-complex tests for float and double:

TEST_P(accuracy_test_real_2D, normal_2D_complex_interleaved_to_real_single_precision)
{
    std::vector<size_t>     length   = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    rocfft_result_placement placeness = std::get<2>(GetParam());
    size_t                  stride    = std::get<3>(GetParam());
    data_pattern            pattern   = std::get<4>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_inverse; // must be real inverse

    try
    {
        normal_2D_complex_interleaved_to_real<float>(
            length, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

TEST_P(accuracy_test_real_2D, normal_2D_complex_interleaved_to_real_double_precision)
{
    std::vector<size_t>     length   = std::get<0>(GetParam());
    size_t                  batch     = std::get<1>(GetParam());
    rocfft_result_placement placeness = std::get<2>(GetParam());
    size_t                  stride    = std::get<3>(GetParam());
    data_pattern            pattern   = std::get<4>(GetParam());
    rocfft_transform_type   transform_type
        = rocfft_transform_type_real_inverse; // must be real inverse

    try
    {
        normal_2D_complex_interleaved_to_real<double>(
            length, batch, placeness, transform_type, stride, pattern);
    }
    catch(const std::exception& err)
    {
        handle_exception(err);
    }
}

// Complex-to-complex:
INSTANTIATE_TEST_CASE_P(rocfft_pow2_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(pow2_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(pow3_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range),
                                           ValuesIn(transform_range)));

INSTANTIATE_TEST_CASE_P(rocfft_prime_2D,
                        accuracy_test_complex_2D,
                        ::testing::Combine(ValuesIn(prime_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range),
                                           ValuesIn(transform_range)));

// Complex to real and real-to-complex:
INSTANTIATE_TEST_CASE_P(rocfft_pow2_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow2_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(rc_placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow3_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow3_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(rc_placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_pow5_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(rc_placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));

INSTANTIATE_TEST_CASE_P(rocfft_prime_2D,
                        accuracy_test_real_2D,
                        ::testing::Combine(ValuesIn(pow5_range),
                                           ValuesIn(batch_range),
                                           ValuesIn(rc_placeness_range),
                                           ValuesIn(stride_range),
                                           ValuesIn(pattern_range)));
