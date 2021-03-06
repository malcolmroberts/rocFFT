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

#pragma once

#ifndef ACCURACY_TEST
#define ACCURACY_TEST

#include "../client_utils.h"
#include "rocfft.h"
#include <vector>

void print_params(const std::vector<size_t>&    length,
                  const size_t                  istride0,
                  const size_t                  ostride0,
                  const size_t                  nbatch,
                  const rocfft_result_placement place,
                  const rocfft_precision        precision,
                  const rocfft_transform_type   transformType,
                  const rocfft_array_type       itype,
                  const rocfft_array_type       otype);

// Base gtest class for comparison with FFTW.
class accuracy_test
    : public ::testing::TestWithParam<std::tuple<std::vector<size_t>, // length
                                                 size_t, // istride
                                                 size_t, // ostride
                                                 size_t, // batch
                                                 rocfft_precision,
                                                 std::tuple<rocfft_transform_type,
                                                            rocfft_array_type,
                                                            rocfft_array_type,
                                                            rocfft_result_placement>>>
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

// C2C direct data layout options
const static std::vector<
    std::
        tuple<rocfft_transform_type, rocfft_array_type, rocfft_array_type, rocfft_result_placement>>
    c2c_direct_range{std::make_tuple<rocfft_transform_type,
                                     rocfft_array_type,
                                     rocfft_array_type,
                                     rocfft_result_placement>(rocfft_transform_type_complex_forward,
                                                              rocfft_array_type_complex_interleaved,
                                                              rocfft_array_type_complex_interleaved,
                                                              rocfft_placement_inplace),
                     std::make_tuple<rocfft_transform_type,
                                     rocfft_array_type,
                                     rocfft_array_type,
                                     rocfft_result_placement>(rocfft_transform_type_complex_forward,
                                                              rocfft_array_type_complex_interleaved,
                                                              rocfft_array_type_complex_interleaved,
                                                              rocfft_placement_notinplace),

                     std::make_tuple<rocfft_transform_type,
                                     rocfft_array_type,
                                     rocfft_array_type,
                                     rocfft_result_placement>(rocfft_transform_type_complex_forward,
                                                              rocfft_array_type_complex_planar,
                                                              rocfft_array_type_complex_interleaved,
                                                              rocfft_placement_notinplace),
                     std::make_tuple<rocfft_transform_type,
                                     rocfft_array_type,
                                     rocfft_array_type,
                                     rocfft_result_placement>(rocfft_transform_type_complex_forward,
                                                              rocfft_array_type_complex_interleaved,
                                                              rocfft_array_type_complex_planar,
                                                              rocfft_placement_notinplace),

                     std::make_tuple<rocfft_transform_type,
                                     rocfft_array_type,
                                     rocfft_array_type,
                                     rocfft_result_placement>(rocfft_transform_type_complex_forward,
                                                              rocfft_array_type_complex_planar,
                                                              rocfft_array_type_complex_planar,
                                                              rocfft_placement_inplace),
                     std::make_tuple<rocfft_transform_type,
                                     rocfft_array_type,
                                     rocfft_array_type,
                                     rocfft_result_placement>(rocfft_transform_type_complex_forward,
                                                              rocfft_array_type_complex_planar,
                                                              rocfft_array_type_complex_planar,
                                                              rocfft_placement_notinplace)};
// C2C inverse data layout options
const static std::vector<
    std::
        tuple<rocfft_transform_type, rocfft_array_type, rocfft_array_type, rocfft_result_placement>>
    c2c_inverse_range{
        std::make_tuple<rocfft_transform_type,
                        rocfft_array_type,
                        rocfft_array_type,
                        rocfft_result_placement>(rocfft_transform_type_complex_inverse,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_placement_inplace),
        std::make_tuple<rocfft_transform_type,
                        rocfft_array_type,
                        rocfft_array_type,
                        rocfft_result_placement>(rocfft_transform_type_complex_inverse,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_placement_notinplace),

        std::make_tuple<rocfft_transform_type,
                        rocfft_array_type,
                        rocfft_array_type,
                        rocfft_result_placement>(rocfft_transform_type_complex_inverse,
                                                 rocfft_array_type_complex_planar,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_placement_notinplace),
        std::make_tuple<rocfft_transform_type,
                        rocfft_array_type,
                        rocfft_array_type,
                        rocfft_result_placement>(rocfft_transform_type_complex_inverse,
                                                 rocfft_array_type_complex_interleaved,
                                                 rocfft_array_type_complex_planar,
                                                 rocfft_placement_notinplace),

        std::make_tuple<rocfft_transform_type,
                        rocfft_array_type,
                        rocfft_array_type,
                        rocfft_result_placement>(rocfft_transform_type_complex_inverse,
                                                 rocfft_array_type_complex_planar,
                                                 rocfft_array_type_complex_planar,
                                                 rocfft_placement_inplace),

        std::make_tuple<rocfft_transform_type,
                        rocfft_array_type,
                        rocfft_array_type,
                        rocfft_result_placement>(rocfft_transform_type_complex_inverse,
                                                 rocfft_array_type_complex_planar,
                                                 rocfft_array_type_complex_planar,
                                                 rocfft_placement_notinplace)};

// R2C data layout options
const static std::vector<
    std::
        tuple<rocfft_transform_type, rocfft_array_type, rocfft_array_type, rocfft_result_placement>>
    r2c_range{std::make_tuple<rocfft_transform_type,
                              rocfft_array_type,
                              rocfft_array_type,
                              rocfft_result_placement>(rocfft_transform_type_real_forward,
                                                       rocfft_array_type_real,
                                                       rocfft_array_type_hermitian_interleaved,
                                                       rocfft_placement_inplace),
              std::make_tuple<rocfft_transform_type,
                              rocfft_array_type,
                              rocfft_array_type,
                              rocfft_result_placement>(rocfft_transform_type_real_forward,
                                                       rocfft_array_type_real,
                                                       rocfft_array_type_hermitian_interleaved,
                                                       rocfft_placement_notinplace),
              std::make_tuple<rocfft_transform_type,
                              rocfft_array_type,
                              rocfft_array_type,
                              rocfft_result_placement>(rocfft_transform_type_real_forward,
                                                       rocfft_array_type_real,
                                                       rocfft_array_type_hermitian_planar,
                                                       rocfft_placement_notinplace)};
// C2R data layout options
const static std::vector<
    std::
        tuple<rocfft_transform_type, rocfft_array_type, rocfft_array_type, rocfft_result_placement>>
    c2r_range{std::make_tuple<rocfft_transform_type,
                              rocfft_array_type,
                              rocfft_array_type,
                              rocfft_result_placement>(rocfft_transform_type_real_inverse,
                                                       rocfft_array_type_hermitian_interleaved,
                                                       rocfft_array_type_real,
                                                       rocfft_placement_inplace),
              std::make_tuple<rocfft_transform_type,
                              rocfft_array_type,
                              rocfft_array_type,
                              rocfft_result_placement>(rocfft_transform_type_real_inverse,
                                                       rocfft_array_type_hermitian_interleaved,
                                                       rocfft_array_type_real,
                                                       rocfft_placement_notinplace),

              std::make_tuple<rocfft_transform_type,
                              rocfft_array_type,
                              rocfft_array_type,
                              rocfft_result_placement>(rocfft_transform_type_real_inverse,
                                                       rocfft_array_type_hermitian_planar,
                                                       rocfft_array_type_real,
                                                       rocfft_placement_notinplace)};

const static std::vector<size_t> batch_range = {1, 2};

const static std::vector<rocfft_precision> precision_range
    = {rocfft_precision_single, rocfft_precision_double};

// Given a vector of vector of lengths, generate all permutations.
inline std::vector<std::vector<size_t>>
    generate_lengths(const std::vector<std::vector<size_t>>& inlengths)
{
    std::vector<std::vector<size_t>> output;
    if(inlengths.size() == 0)
    {
        return output;
    }
    const size_t        dim = inlengths.size();
    std::vector<size_t> looplength(dim);
    for(int i = 0; i < dim; ++i)
    {
        looplength[i] = inlengths[i].size();
    }
    for(int idx = 0; idx < inlengths.size(); ++idx)
    {
        std::vector<size_t> index(dim);
        do
        {
            std::vector<size_t> length(dim);
            for(int i = 0; i < dim; ++i)
            {
                length[i] = inlengths[i][index[i]];
            }
            output.push_back(length);
        } while(increment_colmajor(index, looplength));
    }
    return output;
}

#endif
