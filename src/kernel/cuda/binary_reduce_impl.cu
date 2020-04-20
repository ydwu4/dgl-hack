/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_impl.cu
 * \brief Binary reduce implementation on cuda.
 */
#include <algorithm>
#include <cuda_runtime.h>

#include "../binary_reduce_impl.h"
#include "../csr_interface.h"

using dgl::runtime::NDArray;

namespace dgl {
namespace kernel {

template <typename DType>
__device__ DType gatLeakyReluExp(DType val, DType slope) {
    return val > 0 ? exp(val) : exp(slope * val);
}

template <typename Idx, typename DType>
__global__ void gatExpLeakyReluSumKernel(GatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
    Idx stride_y = blockIdx.y*gridDim.y;
    Idx dst_vid = ty;
    DType e_xlen = gdata.e_xlen;
    Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
    Idx stride_x = blockIdx.x * gridDim.x;
    while (dst_vid < csr.row_offsets.length) {
        Idx feat_idx = tx;
        Idx start_off = *(csr.row_offsets.data + dst_vid);
        Idx end_off = *(csr.row_offsets.data + dst_vid);
        while (feat_idx < e_xlen) {
            DType sum = 0.;
            for (Idx eid=start_off; eid<end_off; ++eid) {
                Idx src_id = *(csr.column_indices.data + eid);
                Idx feat_off = src_id * e_xlen + feat_idx;
                DType tmp = gatLeakyReluExp(gdata.el[feat_off] + gdata.er[feat_off], gdata.leaky_relu_slope);
                gdata.exp[Idx(eid * e_xlen) + feat_idx] = tmp;
                sum += tmp;
            }
            gdata.sum[Idx(dst_vid*e_xlen) + feat_idx] = sum;
            feat_idx += stride_x;
        }
        dst_vid += stride_y;
    }
}

void FusedGatKernelImpl(
    const CSRWrapper& graph,
    runtime::NDArray feat_src,
    runtime::NDArray el,
    runtime::NDArray er,
    runtime::NDArray sum,
    runtime::NDArray exp,
    runtime::NDArray ret,
    float slope) {
        typedef int32_t Idx;
        typedef float DType;
        const Idx MAX_NBLKS = 65535;
        // zero out ret, and packing feat_src, el, er, ret, graph together into one struct using raw float pointers
        // get csr matrix
        GatFusedData<Idx, DType> gdata;
        int64_t el_xlen =  utils::ComputeXLength(el);
        int64_t feat_src_xlen =  utils::ComputeXLength(feat_src);
        int64_t ret_len =  utils::ComputeXLength(ret);
        gdata.feat_src = static_cast<DType*>(feat_src->data);
        gdata.el = static_cast<DType*>(el->data);
        gdata.er = static_cast<DType*>(er->data);
        gdata.sum = static_cast<DType*>(sum->data);
        gdata.exp = static_cast<DType*>(exp->data);
        gdata.ret = static_cast<DType*>(ret->data);
        gdata.leaky_relu_slope = slope;
        gdata.n = el.GetSize()/sizeof(DType)/el_xlen; 
        gdata.e_xlen = el_xlen;
        gdata.feat_src_xlen =  feat_src_xlen;
        gdata.ret_xlen = ret_len;
        auto incsr = graph.GetInCSRMatrix();
        minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
        // write a device function and call it from here
        LOG(INFO) << "Within Fused Gat Kernel Impl." << "feat_src_dim:" << feat_src.GetSize()/sizeof(DType)/feat_src_xlen << "*" << feat_src_xlen 
            <<" el_dim:" << el.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen  << " ret_dim:" << ret.GetSize()/sizeof(DType)/ret_len <<"*" << ret_len
            << " graph csr row_offset length:" <<csr.row_offsets.length << " graph csr column indices length:" << csr.column_indices.length;

        // Configure kernel launch parameters. Reference DGL heuristics.
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        const int min_xlen = std::min(feat_src_xlen, el_xlen);
        const int nthrs_x = utils::FindNumThreads(min_xlen, 64);
        const int nthrs_y = 1024/nthrs_x;
        const int nblks_x = (min_xlen + nthrs_x-1)/(nthrs_x);
        const int nblks_y = std::min(gdata.n, MAX_NBLKS);
        const dim3 nblks(nblks_x, nblks_y);
        const dim3 nthrs(nthrs_x, nthrs_y);
        LOG(INFO) << "blk dim:" << nblks_x << "*" <<nblks_y << " nthrs:" <<nthrs_x << "*" << nthrs_y;
        gatExpLeakyReluSumKernel<<<nblks, nthrs, 0, thr_entry->stream>>>(gdata, csr);
    }

template void BinaryReduceImpl<kDLGPU>(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping);

template void BinaryReduceBcastImpl<kDLGPU>(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping);

template void BackwardBinaryReduceImpl<kDLGPU>(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_mapping, NDArray rhs_mapping, NDArray out_mapping,
    NDArray lhs_data, NDArray rhs_data, NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_lhs_data, NDArray grad_rhs_data);

template void BackwardBinaryReduceBcastImpl<kDLGPU>(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs_tgt, binary_op::Target rhs_tgt,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs, runtime::NDArray rhs, runtime::NDArray out, runtime::NDArray grad_out,
    runtime::NDArray grad_lhs, runtime::NDArray grad_rhs);

}  // namespace kernel
}  // namespace dgl
