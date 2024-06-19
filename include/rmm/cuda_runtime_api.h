// MIT License
// 
// Copyright (c) 2023 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <hip/hip_runtime_api.h>

#define CUDART_VERSION 0

// types
using cudaError_t = hipError_t;
using cudaEvent_t = hipEvent_t;
using cudaMemPool_t = hipMemPool_t;
using cudaStream_t = hipStream_t;
using cudaMemPoolAttr = hipMemPoolAttr;
using cudaMemPoolProps = hipMemPoolProps;
using cudaMemAllocationHandleType = hipMemAllocationHandleType;
using cudaPointerAttributes = hipPointerAttribute_t;
// macros, enum constant definitions
constexpr cudaStream_t cudaStreamLegacy = nullptr;
#define cudaStreamPerThread hipStreamPerThread
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaDevAttrMemoryPoolSupportedHandleTypes hipDevAttrMemoryPoolSupportedHandleTypes
#define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
#define cudaDevAttrL2CacheSize hipDeviceAttributeL2CacheSize
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaSuccess hipSuccess
#define cudaMemAllocationTypePinned hipMemAllocationTypePinned
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold
#define cudaMemHandleTypeNone hipMemHandleTypeNone
#define cudaMemLocationTypeDevice hipMemLocationTypeDevice
#define cudaMemPoolReuseAllowOpportunistic hipMemPoolReuseAllowOpportunistic
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemoryTypeManaged hipMemoryTypeManaged
// functions
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cudaDeviceSynchronize hipDeviceSynchronize

#define cudaDriverGetVersion hipDriverGetVersion

#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize

#define cudaFree hipFree
#define cudaFreeAsync hipFreeAsync
#define cudaFreeHost hipHostFree

#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount

#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

#define cudaMallocAsync hipMallocAsync
#define cudaMalloc hipMalloc
#define cudaMallocFromPoolAsync hipMallocFromPoolAsync
#define cudaMallocHost hipHostMalloc
#define cudaMallocManaged hipMallocManaged

#define cudaMemGetInfo hipMemGetInfo
#define cudaMemPoolCreate hipMemPoolCreate
#define cudaMemPoolDestroy hipMemPoolDestroy
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute

#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemsetAsync hipMemsetAsync

#define cudaSetDevice hipSetDevice

#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize

#define cudaStreamWaitEvent(a,b,c) hipStreamWaitEvent(a,b,c)
#define cudaEventCreate hipEventCreate
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaEventElapsedTime hipEventElapsedTime

#define cudaStreamQuery hipStreamQuery