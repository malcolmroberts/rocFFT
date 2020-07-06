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

#include <atomic>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "rocfft.h"

#include "logging.h"
#include "plan.h"
#include "repo.h"
#include "transform.h"

#include "radix_table.h"

#include "kernel_launch.h"

#include "function_pool.h"
#include "ref_cpu.h"

#include "real2complex.h"

//#define TMP_DEBUG
#ifdef TMP_DEBUG
#include "rocfft_hip.h"
#include <fstream>
#include <sstream>
#endif

std::atomic<bool> fn_checked(false);

// This function is called during creation of plan: enqueue the HIP kernels by function
// pointers. Return true if everything goes well. Any internal device memory allocation
// failure returns false right away.
bool PlanPowX(ExecPlan& execPlan)
{
    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        if((execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM)
           || (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
           || (execPlan.execSeq[i]->scheme == CS_KERNEL_STOCKHAM_BLOCK_RC))
        {
            execPlan.execSeq[i]->twiddles = twiddles_create(
                execPlan.execSeq[i]->length[0], execPlan.execSeq[i]->precision, false, false);
            if(execPlan.execSeq[i]->twiddles == nullptr)
                return false;
        }
        else if((execPlan.execSeq[i]->scheme == CS_KERNEL_R_TO_CMPLX)
                || (execPlan.execSeq[i]->scheme == CS_KERNEL_CMPLX_TO_R))
        {
            execPlan.execSeq[i]->twiddles = twiddles_create(
                2 * execPlan.execSeq[i]->length[0], execPlan.execSeq[i]->precision, false, true);
            if(execPlan.execSeq[i]->twiddles == nullptr)
                return false;
        }

        if(execPlan.execSeq[i]->large1D != 0)
        {
            execPlan.execSeq[i]->twiddles_large = twiddles_create(
                execPlan.execSeq[i]->large1D, execPlan.execSeq[i]->precision, true, false);
            if(execPlan.execSeq[i]->twiddles_large == nullptr)
                return false;
        }
    }
    // copy host buffer to device buffer
    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        execPlan.execSeq[i]->devKernArg = kargs_create(execPlan.execSeq[i]->length,
                                                       execPlan.execSeq[i]->inStride,
                                                       execPlan.execSeq[i]->outStride,
                                                       execPlan.execSeq[i]->iDist,
                                                       execPlan.execSeq[i]->oDist);
        if(execPlan.execSeq[i]->devKernArg == nullptr)
            return false;
    }

    if(!fn_checked)
    {
        fn_checked = true;
        function_pool::verify_no_null_functions();
    }

    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {
        DevFnCall ptr = nullptr;
        GridParam gp;
        size_t    bwd, wgs, lds;

        switch(execPlan.execSeq[i]->scheme)
        {
        case CS_KERNEL_STOCKHAM:
        {
            // get working group size and number of transforms
            size_t workGroupSize;
            size_t numTransforms;
            GetWGSAndNT(execPlan.execSeq[i]->length[0], workGroupSize, numTransforms);
            ptr = (execPlan.execSeq[0]->precision == rocfft_precision_single)
                      ? function_pool::get_function_single(
                          std::make_pair(execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM))
                      : function_pool::get_function_double(
                          std::make_pair(execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM));
            size_t batch = execPlan.execSeq[i]->batch;
            for(size_t j = 1; j < execPlan.execSeq[i]->length.size(); j++)
                batch *= execPlan.execSeq[i]->length[j];
            gp.b_x
                = (batch % numTransforms) ? 1 + (batch / numTransforms) : (batch / numTransforms);
            gp.tpb_x = workGroupSize;
        }
        break;
        case CS_KERNEL_STOCKHAM_BLOCK_CC:
            ptr = (execPlan.execSeq[0]->precision == rocfft_precision_single)
                      ? function_pool::get_function_single(std::make_pair(
                          execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_CC))
                      : function_pool::get_function_double(std::make_pair(
                          execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_CC));
            GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
            gp.b_x = (execPlan.execSeq[i]->length[1]) / bwd * execPlan.execSeq[i]->batch;
            if(execPlan.execSeq[i]->length.size() == 3)
            {
                gp.b_x *= execPlan.execSeq[i]->length[2];
            }
            gp.tpb_x = wgs;
            break;
        case CS_KERNEL_STOCKHAM_BLOCK_RC:
            ptr = (execPlan.execSeq[0]->precision == rocfft_precision_single)
                      ? function_pool::get_function_single(std::make_pair(
                          execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_RC))
                      : function_pool::get_function_double(std::make_pair(
                          execPlan.execSeq[i]->length[0], CS_KERNEL_STOCKHAM_BLOCK_RC));
            GetBlockComputeTable(execPlan.execSeq[i]->length[0], bwd, wgs, lds);
            gp.b_x = (execPlan.execSeq[i]->length[1]) / bwd * execPlan.execSeq[i]->batch;
            if(execPlan.execSeq[i]->length.size() == 3)
            {
                gp.b_x *= execPlan.execSeq[i]->length[2];
            }
            gp.tpb_x = wgs;
            break;
        case CS_KERNEL_TRANSPOSE:
        case CS_KERNEL_TRANSPOSE_XY_Z:
        case CS_KERNEL_TRANSPOSE_Z_XY:
            ptr      = &FN_PRFX(transpose_var2);
            gp.tpb_x = (execPlan.execSeq[0]->precision == rocfft_precision_single) ? 32 : 64;
            gp.tpb_y = (execPlan.execSeq[0]->precision == rocfft_precision_single) ? 32 : 16;
            break;
        case CS_KERNEL_COPY_R_TO_CMPLX:
            ptr      = &real2complex;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_COPY_CMPLX_TO_R:
            ptr      = &complex2real;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_COPY_HERM_TO_CMPLX:
            ptr      = &hermitian2complex;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_COPY_CMPLX_TO_HERM:
            ptr      = &complex2hermitian;
            gp.b_x   = (execPlan.execSeq[i]->length[0] - 1) / 512 + 1;
            gp.b_y   = execPlan.execSeq[i]->batch;
            gp.tpb_x = 512;
            gp.tpb_y = 1;
            break;
        case CS_KERNEL_R_TO_CMPLX:
            ptr = &r2c_1d_post;
            // specify grid params only if the kernel from code generator
            break;
        case CS_KERNEL_CMPLX_TO_R:
            ptr = &c2r_1d_pre;
            // specify grid params only if the kernel from code generator
            break;
        case CS_KERNEL_PAIR_UNPACK:
            ptr = &complex2pair_unpack;
            // specify grid params only if the kernel from code generator
            break;
        case CS_KERNEL_PAIR_PACK:
            ptr = &pair2complex_pack;
            // specify grid params only if the kernel from code generator
            break;
        case CS_KERNEL_CHIRP:
            ptr      = &FN_PRFX(chirp);
            gp.tpb_x = 64;
            break;
        case CS_KERNEL_PAD_MUL:
        case CS_KERNEL_FFT_MUL:
        case CS_KERNEL_RES_MUL:
            ptr      = &FN_PRFX(mul);
            gp.tpb_x = 64;
            break;
        default:
            rocfft_cout << "should not be in this case" << std::endl;
            rocfft_cout << "scheme: " << PrintScheme(execPlan.execSeq[i]->scheme) << std::endl;
            assert(false);
        }

        execPlan.devFnCall.push_back(ptr);
        execPlan.gridParam.push_back(gp);
    }
    
    return true;
}

static size_t data_size_bytes(const std::vector<size_t>& lengths,
                              rocfft_precision           precision,
                              rocfft_array_type          type)
{
    // first compute the raw number of elements
    size_t elems = std::accumulate(lengths.begin(), lengths.end(), 1, std::multiplies<size_t>());
    // size of each element
    size_t elemsize = (precision == rocfft_precision_single ? sizeof(float) : sizeof(double));
    switch(type)
    {
    case rocfft_array_type_complex_interleaved:
    case rocfft_array_type_complex_planar:
        // complex needs two numbers per element
        return 2 * elems * elemsize;
    case rocfft_array_type_real:
        // real needs one number per element
        return elems * elemsize;
    case rocfft_array_type_hermitian_interleaved:
    case rocfft_array_type_hermitian_planar:
    {
        // hermitian requires 2 numbers per element, but innermost
        // dimension is cut down to roughly half
        size_t non_innermost = elems / lengths[0];
        return 2 * non_innermost * elemsize * ((lengths[0] / 2) + 1);
    }
    case rocfft_array_type_unset:
        // we should really have an array type at this point
        assert(false);
        return 0;
    }
}

static float execution_bandwidth_GB_per_s(size_t data_size_bytes, float duration_ms)
{
    // divide bytes by (1000000 * milliseconds) to get GB/s
    return static_cast<float>(data_size_bytes) / (1000000.0 * duration_ms);
}

// NOTE: HIP returns the maximum global frequency in kHz, which might
// not be the actual frequency when the transform ran.  This function
// might also return 0.0 if the bandwidth can't be queried.
static float max_memory_bandwidth_GB_per_s()
{
    int deviceid = 0;
    hipGetDevice(&deviceid);
    int max_memory_clock_kHz = 0;
    int memory_bus_width     = 0;
    hipDeviceGetAttribute(&max_memory_clock_kHz, hipDeviceAttributeMemoryClockRate, deviceid);
    hipDeviceGetAttribute(&memory_bus_width, hipDeviceAttributeMemoryBusWidth, deviceid);
    auto max_memory_clock_MHz = static_cast<float>(max_memory_clock_kHz) / 1024.0;
    // multiply by 2.0 because transfer is bidirectional
    // divide by 8.0 because bus width is in bits and we want bytes
    // divide by 1000 to convert MB to GB
    float result = (max_memory_clock_MHz * 2.0 * memory_bus_width / 8.0) / 1000.0;
    return result;
}

// Internal plan executor.
// For in-place transforms, in_buffer == out_buffer.
void TransformPowX(const ExecPlan&       execPlan,
                   void*                 in_buffer[],
                   void*                 out_buffer[],
                   rocfft_execution_info info)
{
    assert(execPlan.execSeq.size() == execPlan.devFnCall.size());
    assert(execPlan.execSeq.size() == execPlan.gridParam.size());

    // we can log profile information if we're on the null stream,
    // since we will be able to wait for the transform to finish
    bool       emit_profile_log = LOG_TRACE_ENABLED() && !info->rocfft_stream;
    float      max_memory_bw    = 0.0;
    hipEvent_t start, stop;
    if(emit_profile_log)
    {
        hipEventCreate(&start);
        hipEventCreate(&stop);
        max_memory_bw = max_memory_bandwidth_GB_per_s();
    }
    for(size_t i = 0; i < execPlan.execSeq.size(); i++)
    {        
        DeviceCallIn data;
        data.node          = execPlan.execSeq[i];
        data.rocfft_stream = (info == nullptr) ? 0 : info->rocfft_stream;

        // Size of complex type
        const size_t complexTSize
            = (data.node->precision == rocfft_precision_single)
            ? sizeof(float) * 2
            : sizeof(double) * 2;

        //if(data.node->parent->scheme == CS_REAL_TRANSFORM_PAIR && i == 0)
        // if(execPlan.rootPlan->inArrayType == rocfft_array_type_real &&
        //    (data.node->inArrayType == rocfft_array_type_complex_planar ||
        //     data.node->outArrayType == rocfft_array_type_complex_planar)
        //     )
        if(data.node->parent->scheme == CS_REAL_TRANSFORM_PAIR)
        {
            std::cout << "sneaky!" << std::endl;
            // assert(data.node->parent->scheme == CS_REAL_TRANSFORM_PAIR);
            
            // We conclude that we are performing real/complex paired transform, where the real
            // values are treated as the real and complex parts of a complex/complex transform in
            // planar format.

            // FIXME: use planar_offset != 0 as a test instead?
            
            // We impose that the c2c transform be in-place on the input buffer.
            // assert(data.node->obIn == OB_USER_IN);
            // assert(data.node->obOut == OB_USER_IN);
            
            // Size of real type
            const size_t realTSize  = (data.node->precision == rocfft_precision_single)
                ? sizeof(float)
                : sizeof(double);

            // FIXME: what about inverse transforms?

            // Calculate the pointer to the planar format when using the paired real/complex
            // method.
            const ptrdiff_t offset
                = (execPlan.rootPlan->batch % 2 == 0)
                ? realTSize * execPlan.rootPlan->iDist
                : realTSize * execPlan.rootPlan->inStride[data.node->pairdim];

            assert(offset != 0);

            std::cout << "offset: " << offset << std::endl;
            
            data.bufIn[0] = in_buffer[0];
            data.bufIn[1] = (void*)((char*)data.bufIn[0] + offset);

            if(data.node->outArrayType == rocfft_array_type_complex_planar)
            {
                // We impose that this transform is in-place.
                assert(data.node->obIn == OB_USER_IN);
                data.bufOut[0] =  data.bufIn[0];
                data.bufOut[1] =  data.bufIn[1];
            }
            else
            {
                switch(data.node->obIn)
                {
                case OB_USER_IN:
                    data.bufIn[0] = in_buffer[0];
                    break;
                case OB_USER_OUT:
                    data.bufIn[0] = out_buffer[0];
                    break;
                case OB_TEMP:
                    data.bufIn[0] = info->workBuffer;
                    break;
                default:
                    std::cerr << "Error: operating buffer not specified for kernel!\n";
                    assert(false);
                }
            }
            
        }
        else
        {
            std::cout << "not sneaky!" << std::endl;
        
                
            switch(data.node->obIn)
            {
            case OB_USER_IN:
                data.bufIn[0] = in_buffer[0];
                if(data.node->inArrayType == rocfft_array_type_complex_planar
                   || data.node->inArrayType == rocfft_array_type_hermitian_planar)
                {
                    data.bufIn[1] = in_buffer[1];
                }
                break;
            case OB_USER_OUT:
                data.bufIn[0] = out_buffer[0];
                if(data.node->inArrayType == rocfft_array_type_complex_planar
                   || data.node->inArrayType == rocfft_array_type_hermitian_planar)
                {
                    data.bufIn[1] = out_buffer[1];
                }
                break;
            case OB_TEMP:
                data.bufIn[0] = info->workBuffer;
                if(data.node->inArrayType == rocfft_array_type_complex_planar
                   || data.node->inArrayType == rocfft_array_type_hermitian_planar)
                {
                    // Assume planar using the same extra size of memory as
                    // interleaved format, and we just need to split it for
                    // planar.
                    data.bufIn[1]
                        = (void*)((char*)info->workBuffer + execPlan.workBufSize * complexTSize / 2);
                }
                break;
            case OB_TEMP_CMPLX_FOR_REAL:
                data.bufIn[0] = (void*)((char*)info->workBuffer + execPlan.tmpWorkBufSize * complexTSize);
                break;
            case OB_TEMP_BLUESTEIN:
                data.bufIn[0] = (void*)((char*)info->workBuffer
                                        + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize
                                           + data.node->iOffset)
                                        * complexTSize);
                break;
            case OB_UNINIT:
                std::cerr << "Error: operating buffer not initialized for kernel!\n";
                assert(data.node->obIn != OB_UNINIT);
            default:
                std::cerr << "Error: operating buffer not specified for kernel!\n";
                assert(false);
            }

            switch(data.node->obOut)
            {
            case OB_USER_IN:
                data.bufOut[0] = in_buffer[0];
                if(data.node->outArrayType == rocfft_array_type_complex_planar
                   || data.node->outArrayType == rocfft_array_type_hermitian_planar)
                {
                    data.bufOut[1] = in_buffer[1];
                }
                break;
            case OB_USER_OUT:
                data.bufOut[0] = out_buffer[0];
                if(data.node->outArrayType == rocfft_array_type_complex_planar
                   || data.node->outArrayType == rocfft_array_type_hermitian_planar)
                {
                    data.bufOut[1] = out_buffer[1];
                }
                break;
            case OB_TEMP:
                data.bufOut[0] = info->workBuffer;
                if(data.node->outArrayType == rocfft_array_type_complex_planar
                   || data.node->outArrayType == rocfft_array_type_hermitian_planar)
                {
                    // assume planar using the same extra size of memory as
                    // interleaved format, and we just need to split it for
                    // planar.
                    data.bufOut[1]
                        = (void*)((char*)info->workBuffer + execPlan.workBufSize * complexTSize / 2);
                }
                break;
            case OB_TEMP_CMPLX_FOR_REAL:
                data.bufOut[0] = (void*)((char*)info->workBuffer + execPlan.tmpWorkBufSize * complexTSize);
                break;
            case OB_TEMP_BLUESTEIN:
                data.bufOut[0] = (void*)((char*)info->workBuffer
                                         + (execPlan.tmpWorkBufSize + execPlan.copyWorkBufSize
                                            + data.node->oOffset)
                                         * complexTSize);
                break;
            default:
                assert(false);
        }
        }
            
        data.gridParam = execPlan.gridParam[i];

#ifdef TMP_DEBUG
        //TODO:
        // - move the below into DeviceCallIn
        // - fix it for real data

        rocfft_cout << "--- --- scheme " << PrintScheme(data.node->scheme) << std::endl;

        const size_t in_size = data.node->iDist * data.node->batch;
        size_t       base_type_size
            = (data.node->precision == rocfft_precision_double) ? sizeof(double) : sizeof(float);
        base_type_size *= 2;

        size_t in_size_bytes = in_size * base_type_size;
        void*  dbg_in        = malloc(in_size_bytes);
        hipDeviceSynchronize();
        std::stringstream ss;
        std::ofstream     realPart, imagPart;

        ss.str("");
        ss << "kernel_" << i << "_input_real.bin";
        realPart.open(ss.str().c_str(), std::ios::out | std::ios::binary);
        ss.str("");
        ss << "kernel_" << i << "_input_imag.bin";
        imagPart.open(ss.str().c_str(), std::ios::out | std::ios::binary);

        if(data.node->inArrayType == rocfft_array_type_complex_planar
           || data.node->inArrayType == rocfft_array_type_hermitian_planar)
        {
            hipMemcpy(dbg_in, data.bufIn[0], in_size_bytes / 2, hipMemcpyDeviceToHost);
            hipMemcpy((void*)((char*)dbg_in + in_size_bytes / 2),
                      data.bufIn[1],
                      in_size_bytes / 2,
                      hipMemcpyDeviceToHost);

            realPart.write((char*)dbg_in, in_size_bytes / 2);
            imagPart.write((char*)dbg_in + in_size_bytes / 2, in_size_bytes / 2);
        }
        else
        {
            hipMemcpy(dbg_in, data.bufIn[0], in_size_bytes, hipMemcpyDeviceToHost);

            for(size_t ii = 0; ii < in_size_bytes; ii += base_type_size)
            {
                realPart.write((char*)dbg_in + ii, base_type_size / 2);
                imagPart.write((char*)dbg_in + ii + base_type_size / 2, base_type_size / 2);
            }
        }
        realPart.close();
        imagPart.close();

        const size_t out_size       = data.node->oDist * data.node->batch;
        const size_t out_size_bytes = out_size * base_type_size;
        void*        dbg_out        = malloc(out_size_bytes);

        //rocfft_cout << "data.node->iDist " << data.node->iDist << ", data.node->batch " << data.node->batch << std::endl;
        //rocfft_cout << "in_size " << in_size << ", out_size " << out_size << std::endl;
        //rocfft_cout << "in_size_bytes " << in_size_bytes << ", out_size_bytes " << out_size_bytes << std::endl;
        // memset(dbg_out, 0x40, out_size_bytes);
        // if(data.node->placement != rocfft_placement_inplace)
        // {
        //     hipDeviceSynchronize();
        //     hipMemcpy(data.bufOut[0], dbg_out, out_size_bytes, hipMemcpyHostToDevice);
        // }
        rocfft_cout << "attempting kernel: " << i << std::endl;
#endif

        DevFnCall fn = execPlan.devFnCall[i];
        if(fn)
        {
#ifdef REF_DEBUG
            rocfft_cout << "\n---------------------------------------------\n";
            rocfft_cout << "\n\nkernel: " << i << std::endl;
            rocfft_cout << "\tscheme: " << PrintScheme(execPlan.execSeq[i]->scheme) << std::endl;
            rocfft_cout << "\titype: " << execPlan.execSeq[i]->inArrayType << std::endl;
            rocfft_cout << "\totype: " << execPlan.execSeq[i]->outArrayType << std::endl;
            rocfft_cout << "\tlength: ";
            for(const auto& i : execPlan.execSeq[i]->length)
            {
                rocfft_cout << i << " ";
            }
            rocfft_cout << std::endl;
            rocfft_cout << "\tbatch:   " << execPlan.execSeq[i]->batch << std::endl;
            rocfft_cout << "\tidist:   " << execPlan.execSeq[i]->iDist << std::endl;
            rocfft_cout << "\todist:   " << execPlan.execSeq[i]->oDist << std::endl;
            rocfft_cout << "\tistride:";
            for(const auto& i : execPlan.execSeq[i]->inStride)
            {
                rocfft_cout << " " << i;
            }
            rocfft_cout << std::endl;
            rocfft_cout << "\tostride:";
            for(const auto& i : execPlan.execSeq[i]->outStride)
            {
                rocfft_cout << " " << i;
            }
            rocfft_cout << std::endl;

            RefLibOp refLibOp(&data);
#endif

            // execution kernel:
            if(emit_profile_log)
                hipEventRecord(start);
            DeviceCallOut back;
            fn(&data, &back);
            if(emit_profile_log)
                hipEventRecord(stop);

            // If we were on the null stream, measure elapsed time
            // and emit profile logging.  If a stream was given, we
            // can't wait for the transform to finish, so we can't
            // emit any information.
            if(emit_profile_log)
            {
                hipEventSynchronize(stop);
                size_t in_size_bytes = data_size_bytes(
                    data.node->length, data.node->precision, data.node->inArrayType);
                size_t out_size_bytes = data_size_bytes(
                    data.node->length, data.node->precision, data.node->outArrayType);
                size_t total_size_bytes = in_size_bytes + out_size_bytes;

                float duration_ms = 0.0f;
                hipEventElapsedTime(&duration_ms, start, stop);
                auto exec_bw
                    = execution_bandwidth_GB_per_s(in_size_bytes + out_size_bytes, duration_ms);
                auto efficiency_pct = 0.0;
                if(max_memory_bw != 0.0)
                    efficiency_pct = 100.0 * exec_bw / max_memory_bw;
                log_profile(__func__,
                            "scheme",
                            PrintScheme(execPlan.execSeq[i]->scheme),
                            "duration_ms",
                            duration_ms,
                            "in_size",
                            std::make_pair(static_cast<const size_t*>(data.node->length.data()),
                                           data.node->length.size()),
                            "total_size_bytes",
                            total_size_bytes,
                            "exec_GB_s",
                            exec_bw,
                            "max_mem_GB_s",
                            max_memory_bw,
                            "bw_efficiency_pct",
                            efficiency_pct);
            }

#ifdef REF_DEBUG
            refLibOp.VerifyResult(&data);
#endif
        }
        else
        {
            rocfft_cout << "null ptr function call error\n";
        }

#ifdef TMP_DEBUG
        hipError_t err = hipPeekAtLastError();
        if(err != hipSuccess)
        {
            rocfft_cout << "Error: " << hipGetErrorName(err) << ", " << hipGetErrorString(err)
                        << std::endl;
            exit(-1);
        }
        hipDeviceSynchronize();
        rocfft_cout << "executed kernel: " << i << std::endl;

        ss.str("");
        ss << "kernel_" << i << "_output_real.bin";
        realPart.open(ss.str().c_str(), std::ios::out | std::ios::binary);
        ss.str("");
        ss << "kernel_" << i << "_output_imag.bin";
        imagPart.open(ss.str().c_str(), std::ios::out | std::ios::binary);

        if(data.node->outArrayType == rocfft_array_type_complex_planar
           || data.node->outArrayType == rocfft_array_type_hermitian_planar)
        {
            hipMemcpy(dbg_out, data.bufOut[0], out_size_bytes / 2, hipMemcpyDeviceToHost);
            hipMemcpy((void*)((char*)dbg_out + out_size_bytes / 2),
                      data.bufOut[1],
                      out_size_bytes / 2,
                      hipMemcpyDeviceToHost);

            realPart.write((char*)dbg_out, out_size_bytes / 2);
            imagPart.write((char*)dbg_out + out_size_bytes / 2, out_size_bytes / 2);
        }
        else
        {
            hipMemcpy(dbg_out, data.bufOut[0], out_size_bytes, hipMemcpyDeviceToHost);

            for(size_t ii = 0; ii < out_size_bytes; ii += base_type_size)
            {
                realPart.write((char*)dbg_out + ii, base_type_size / 2);
                imagPart.write((char*)dbg_out + ii + base_type_size / 2, base_type_size / 2);
            }
        }

        realPart.close();
        imagPart.close();

        rocfft_cout << "copied from device\n";

        // temporary print out the kernel output
        // rocfft_cout << "input:" << std::endl;
        // for(size_t i = 0; i < data.node->iDist * data.node->batch; i++)
        // {
        //     rocfft_cout << f_in[i].x << " " << f_in[i].y << "\n";
        // }
        // rocfft_cout << "output:" << std::endl;
        // for(size_t i = 0; i < data.node->oDist * data.node->batch; i++)
        // {
        //     rocfft_cout << f_out[i].x << " " << f_out[i].y << "\n";
        // }
        // rocfft_cout << "\n---------------------------------------------\n";
        free(dbg_out);
        free(dbg_in);
#endif
    }
    if(emit_profile_log)
    {
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }
}
