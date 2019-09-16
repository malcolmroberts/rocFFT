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

// The even-length c2r fails 4096x8192.
// TODO: make test precision vary with problem size, then re-enable.
static std::vector<std::vector<size_t>> pow2_range_c2r
    = {{2, 4}, {8, 16}, {32, 128}, {256, 512}, {1024, 2048}};
// Real/complex transform test framework is only set up for out-of-place transforms:
static rocfft_result_placement rc_placeness_range[] = {rocfft_placement_notinplace};

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
};
class accuracy_test_real_2D
    : public ::testing::TestWithParam<
          std::tuple<std::vector<size_t>, size_t, rocfft_result_placement, size_t, data_pattern>>
{
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
    rocfft_setup(); // TODO: move to gtest setup?
    
    std::vector<size_t> input_strides;
    std::vector<size_t> output_strides;
    input_strides.push_back(stride);
    output_strides.push_back(stride);

    
    
    const size_t Nx = length[1];
    const size_t Ny = length[0];
    const bool   inplace = placeness == rocfft_placement_inplace;

    // TODO: add logic to deal with discontiguous data in Nystride
    const size_t Nycomplex = Ny / 2 + 1;
    const size_t Nystride = inplace ? 2 * Nycomplex : Ny;

    // Local data buffer: (TODO: assumes contiguous data).
    std::vector<Tfloat> cx(Nx * Nystride);
    // Output buffer
    std::vector<std::complex<Tfloat>> cy(Nx * Nycomplex);

    Tfloat* x = NULL;
    hipMalloc(&x, cx.size() * sizeof(typename decltype(cx)::value_type));
    hipMemcpy(x, cx.data(), cx.size() * sizeof(typename decltype(cx)::value_type),
              hipMemcpyHostToDevice);
    std::complex<Tfloat>* y = inplace ? (std::complex<Tfloat>*)x : NULL;
    if(!inplace)
    {
        hipMalloc(&y, cy.size() * sizeof(typename decltype(cy)::value_type));
    }
        
    // Dimension configuration:
    std::array<fftw_iodim64, 2> dims;
    dims[1].n  = Ny;
    dims[1].is = stride;
    dims[1].os = stride;
    dims[0].n  = Nx;
    dims[0].is = dims[1].n * dims[1].is;
    dims[0].os = dims[1].n * dims[1].os;

    const size_t isize = dims[0].n * dims[0].is;
    const size_t osize = dims[0].n * dims[0].os;
    
    // Batch configuration:
    std::array<fftw_iodim64, 1> howmany_dims;
    howmany_dims[0].n = batch;
    howmany_dims[0].is = isize; // FIXME: placement
    howmany_dims[0].os = osize;

    // Set up buffers:
    Tfloat* idata = fftw_alloc_real_type<Tfloat>(isize);
    using fftw_complex_type = typename fftw_trait<Tfloat>::fftw_complex_type;
    fftw_complex_type* odata = inplace
        ? (fftw_complex_type*)idata : fftw_alloc_complex_type<Tfloat>(osize);
    
    auto cpu_plan = fftw_plan_guru64_r2c<Tfloat>(dims.size(),
                                                 dims.data(),
                                                 howmany_dims.size(),
                                                 howmany_dims.data(),
                                                 idata,
                                                 odata,
                                                 FFTW_ESTIMATE);

    // Set up the GPU plan:
    rocfft_status status = rocfft_status_success;
    rocfft_plan forward = NULL;
    status = rocfft_plan_create(&forward,
                                inplace ? rocfft_placement_inplace : rocfft_placement_notinplace,
                                rocfft_transform_type_real_forward,
                                precision_selector<Tfloat>(),
                                2, // Dimensions
                                length.data(), // lengths
                                1, // Number of transforms
                                NULL); // Description
    assert(status == rocfft_status_success);

    // The real-to-complex transform uses work memory, which is passed
    // via a rocfft_execution_info struct.
    rocfft_execution_info forwardinfo = NULL;
    status = rocfft_execution_info_create(&forwardinfo);
    assert(status == rocfft_status_success);
    size_t forwardworkbuffersize = 0;
    status = rocfft_plan_get_work_buffer_size(forward, &forwardworkbuffersize);
    assert(status == rocfft_status_success);
    void* forwardwbuffer = NULL;
    if(forwardworkbuffersize > 0)
    {
        hipMalloc(&forwardwbuffer, forwardworkbuffersize);
        status = rocfft_execution_info_set_work_buffer(forwardinfo, forwardwbuffer,
                                                       forwardworkbuffersize);
        assert(status == rocfft_status_success);
    }
       
    // Set up the data:
    std::fill(cx.begin(), cx.end(), 0.0);
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            cx[i * Nystride + j] = i + j;
        }
    }
    // FIXME: copy data to device

    // Execute the GPU transform:
    status = rocfft_execute(forward, // plan
                            (void**)&x, // in_buffer
                            (void**)&y, // out_buffer
                            forwardinfo); // execution info
    assert(status == rocfft_status_success);

    // Execute the CPU transform:
    fftw_execute_type<Tfloat>(cpu_plan);

    // FIXME: copy data back and compare:

    // FREE memory:
    fftw_free(idata);
    hipFree(x);
    if(!inplace)
    {
        hipFree(y);
        fftw_free(odata);
    }
    if (forwardwbuffer != NULL)
    {
        hipFree(forwardwbuffer);
    }
    
    // Delete plans:
    rocfft_plan_destroy(forward);
    fftw_destroy_plan_type<Tfloat>(cpu_plan);
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
                        ::testing::Combine(ValuesIn(pow2_range_c2r),
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
