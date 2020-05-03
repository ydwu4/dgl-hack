/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_impl.cu
 * \brief Binary reduce implementation on cuda.
 */
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>

#include "../binary_reduce_impl.h"
#include "../csr_interface.h"

using dgl::runtime::NDArray;

namespace dgl {
namespace kernel {

template <typename Idx>
std::string print_csr(const minigun::Csr<Idx>& csr);

template <typename Idx, typename DType>
std::string print_gdata2d(runtime::NDArray a, Idx dim1, Idx dim2);

template <typename Idx, typename DType>
void print_gdata(runtime::NDArray feat_src,
    runtime::NDArray el,
    runtime::NDArray er,
    runtime::NDArray sum,
    runtime::NDArray exp,
    runtime::NDArray ret,
    const minigun::Csr<Idx> &csr,
    Idx el_xlen,
    Idx feat_src_xlen);

template <typename DType>
__device__ DType gatLeakyReluExp(DType val, DType slope) {
    return val > 0 ? exp(val) : exp(slope * val);
}

template <typename Idx, typename DType>
__global__ void gatExpLeakyReluSumKernel(GatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    //extern __shared__ DType er[];
    Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
    Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
    Idx stride_x = blockDim.x * gridDim.x;
    Idx stride_y = blockDim.y * gridDim.y;
    Idx dst_vid = ty;
    Idx e_xlen = gdata.e_xlen;
    while (dst_vid < csr.row_offsets.length) {
        Idx start_off = *(csr.row_offsets.data + dst_vid);
        Idx end_off = *(csr.row_offsets.data + dst_vid + 1);
        Idx feat_idx = tx;
        while (feat_idx < e_xlen) {
            // 1. Load dstnation vertex into shared memory
            Idx feat_off_dst = dst_vid * e_xlen + feat_idx;
            //er[threadIdx.x] = gdata.er[feat_off_dst];
            //__syncthreads();
            // 2. Do the computation
            DType sum = 0.;
            for (Idx eid=start_off; eid<end_off; ++eid) {
                Idx src_id = *(csr.column_indices.data + eid);
                Idx feat_off_src = src_id * e_xlen + feat_idx;
                //DType tmp = gatLeakyReluExp(gdata.el[feat_off_src] + er[threadIdx.x], gdata.leaky_relu_slope);
                DType tmp = gatLeakyReluExp(gdata.el[feat_off_src] + gdata.er[feat_off_dst], gdata.leaky_relu_slope);
                gdata.exp[Idx(gdata.eids[eid] * e_xlen) + feat_idx] = tmp;
                sum += tmp;
            }
            gdata.sum[Idx(dst_vid*e_xlen) + feat_idx] = sum;
            feat_idx += stride_x;
        }
        dst_vid += stride_y;
    }
}

template <typename Idx, typename DType>
__global__ void gatSumProdZipDivKernel(GatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    Idx dst_vid = blockIdx.y;
    Idx stride_vid =  gridDim.y;
    Idx stride_head = blockDim.x * gridDim.x;
    Idx e_xlen = gdata.e_xlen;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    while (dst_vid < csr.row_offsets.length-1) {
        Idx start_off = *(csr.row_offsets.data + dst_vid);
        Idx end_off = *(csr.row_offsets.data + dst_vid + 1);
        Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (head_idx < e_xlen) {
            Idx feat_idx = threadIdx.y;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                for (Idx eid=start_off; eid<end_off; eid++) {
                    Idx src_vid = csr.column_indices.data[eid];
                    s +=  gdata.exp[gdata.eids[eid] * e_xlen + head_idx] / gdata.sum[dst_vid*e_xlen + head_idx] 
                                        * gdata.feat_src[src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx];
                }
                gdata.ret[dst_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx] = s;
                feat_idx += blockDim.y;
            }
            head_idx += stride_head;
        }
        dst_vid += stride_vid;
    }
}

/*** Implement the logic of computing grad_feat_src.
    feat_src is of dimension: N * num_heads * num_hidden
    exp is of dimension: M * num_heads
    sum is of dimension: N * num_heads
    * means element-wise mutliplication
    In forward computation: out = sum([feat_src[e.src] * exp[e.eid]/sum[curnode] for e in curnode.inedges]),
    In backward computation: grad_feat_src[curnode] = sum([grad_out[e.dst] * exp[e.eid]/sum[e.dst] for e in curnode.outedges])
***/
template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradFeatSrc(BackwardGatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    Idx src_vid = blockIdx.y;
    Idx stride_vid = gridDim.y;
    Idx e_xlen = gdata.e_xlen;
    Idx stride_head = blockDim.x * gridDim.x;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    while (src_vid < csr.row_offsets.length -1) {
        Idx start_off = csr.row_offsets.data[src_vid];
        Idx end_off = csr.row_offsets.data[src_vid+1];
        Idx head_idx = blockIdx.x * blockDim.x  + threadIdx.x;
        while (head_idx < e_xlen) {
            Idx feat_idx = threadIdx.y;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                for (Idx e=start_off; e<end_off; ++e) {
                    Idx eid = gdata.eids[e];
                    Idx dst_id = csr.column_indices.data[e];
                    // TODO: maybe it's better to cache exp/sum to reduce mem traffic as well as redundant computation?
                    s += gdata.exp[eid*e_xlen + head_idx] / gdata.sum[dst_id*e_xlen + head_idx]
                        * gdata.grad_out[dst_id*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx];
                }
                gdata.grad_feat_src[src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx] = s;
                feat_idx += blockDim.y;
            }
            head_idx += stride_head;
        }
        src_vid += stride_vid;
    }
}
/***
Implement the logic of computing grad_el. 
Dimension of grad_out: N * num_heads * num_hidden
             grad_el:  N * num_heads
             grad_er:  N * num_heads
             el:       N * num_heads
             er:       N * num_heads
             exp:      M * num_heads
             sum:      N * num_heads
             feat_src: N * num_heads * num_hidden 

In forward computation: gdata.exp = [exp(leaky_relu(e.el[src] + e.el[dst])) for e in curnode.inedges]
                        gdata.sum[curnode] = sum([exp[e.eid] for e in curnode.inedges])
                        out[curnode] = sum([gdata.exp[e.eid] / gdata.sum[curnode] * gdata.feat_src[e.src] for e in curnode.inedges])
In backward computation:
                        grad_er = sum([grad_exp[e.eid] * exp(leaky_relu(gdata.el[src]+ gdata.er[dst])) * grad_leaky_relu(gdata.el[src] + gdata.er[dst]) for e in curnode.inedges])
                        grad_el = sum([grad_exp[e.eid] * leaky_relu(gdata.el[src] + gdata.er[dst]) * grad_leaky_relu(gdata.el[src] + gdata.er[dst]) for e in curnode.outedges])
                        grad_exp = [grad_out[e.dst] * (feat_src[e.src] - out[e.dst])/sum[e.dst] for e in outedges]
***/
//template <typename Idx, typename DType>
//__global__ void fusedGatBackwardGradElEr(BackwardGatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
//    Idx src_vid = blockIdx.y;
//    Idx stride_vid = gridDim.y;
//    Idx e_xlen = gdata.e_xlen;
//    Idx stride_head = blockDim.x * gridDim.x;
//    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
//    while (src_vid < csr.row_offsets.length -1) {
//        Idx start_off = csr.row_offsets.data[src_vid];
//        Idx end_off = csr.row_offsets.data[src_vid+1];
//        Idx head_idx = blockIdx.x * blockDim.x  + threadIdx.x;
//        while (head_idx < e_xlen) {
//            Idx feat_idx = threadIdx.y;
//            while (feat_idx < hidden_xlen) {
//                DType s = 0.;
//                Idx feat_src_offset = src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx;
//                for (Idx e=start_off; e<end_off; ++e) {
//                    Idx eid = gdata.eids[e];
//                    Idx dst_vid = csr.column_indices.data[e];
//                }
//                feat_idx += blockDim.y;
//            }
//            head_idx += stride_head;
//        }
//        src_vid += stride_vid;
//    }
//}
template <typename DType>
__device__ DType gradLeaky(DType val, DType slope) {
    return val > 0 ? 1 : slope;
}

template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradElEr(BackwardGatFusedData<Idx, DType> gdata, minigun::Csr<Idx> csr) {
    Idx src_vid = blockIdx.y;
    Idx stride_vid = gridDim.y;
    Idx e_xlen = gdata.e_xlen;
    Idx stride_head = blockDim.x * gridDim.x;
    Idx hidden_xlen = gdata.feat_src_xlen/e_xlen;
    while (src_vid < csr.row_offsets.length -1) {
        Idx start_off = csr.row_offsets.data[src_vid];
        Idx end_off = csr.row_offsets.data[src_vid+1];
        Idx head_idx = blockIdx.x * blockDim.x  + threadIdx.x;
        while (head_idx < e_xlen) {
            Idx feat_idx = threadIdx.y;
            while (feat_idx < hidden_xlen) {
                DType s = 0.;
                Idx feat_src_offset = src_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx;
                Idx src_node_feat_offset = src_vid*e_xlen + head_idx;
                for (Idx e=start_off; e<end_off; ++e) {
                    Idx edge_offset = gdata.eids[e] * e_xlen + head_idx;
                    Idx dst_vid = csr.column_indices.data[e];
                    Idx dst_node_feat_offset = dst_vid*e_xlen + head_idx;
                    Idx dst_out_offset = dst_vid*gdata.feat_src_xlen + head_idx*hidden_xlen + feat_idx;
                    DType grad_exp = gdata.grad_out[dst_out_offset]* (gdata.feat_src[feat_src_offset]- gdata.ret[dst_out_offset])/gdata.sum[dst_node_feat_offset] ;
                    DType tmp_sum = gdata.el[src_node_feat_offset] + gdata.er[dst_node_feat_offset];
                    DType tmp2 = grad_exp * gdata.exp[edge_offset] * gradLeaky(tmp_sum, gdata.leaky_relu_slope);
                    s += tmp2;
                    atomicAdd(gdata.grad_er + dst_node_feat_offset, tmp2);
                }
                atomicAdd(gdata.grad_el + src_node_feat_offset , s);
                feat_idx += blockDim.y;
            }
            head_idx += stride_head;
        }
        src_vid += stride_vid;
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
        const Idx MAX_NTHRS = 1024;
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
        gdata.feat_src_hidden = feat_src_xlen/el_xlen;
        gdata.ret_xlen = ret_len;
        auto incsr = graph.GetInCSRMatrix();
        minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
        gdata.eids = static_cast<Idx*>(incsr.data->data);
        // write a device function and call it from here
        //LOG(INFO) << "Within Fused Gat Kernel Impl." << "feat_src_dim:" << feat_src.GetSize()/sizeof(DType)/feat_src_xlen << "*" << feat_src_xlen 
        //    <<" el_dim:" << el.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen  << " ret_dim:" << ret.GetSize()/sizeof(DType)/ret_len <<"*" << ret_len
        //    <<" sum_dim:" << sum.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    <<" exp_dim:" << exp.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    << " graph csr row_offset length:" <<csr.row_offsets.length << " graph csr column indices length:" << csr.column_indices.length;

        // Configure kernel launch parameters.
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        int nthrs_x = 32;
        int nthrs_y = 1;
        int nblks_x = (el_xlen + nthrs_x-1)/(nthrs_x);
        int nblks_y = std::min(gdata.n, MAX_NBLKS);
        const dim3 nblks(nblks_x, nblks_y);
        const dim3 nthrs(nthrs_x, nthrs_y);
        //LOG(INFO) << "kernel1 blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:" <<nthrs_x << "*" << nthrs_y;

        //print_gdata<Idx, DType>(feat_src,el,er,sum,exp,ret,csr,el_xlen, feat_src_xlen);
        //gatExpLeakyReluSumKernel<<<nblks, nthrs, el_xlen*sizeof(DType), thr_entry->stream>>>(gdata, csr);
        gatExpLeakyReluSumKernel<<<nblks, nthrs, 0, thr_entry->stream>>>(gdata, csr);
        //print_gdata<Idx, DType>(feat_src,el,er,sum,exp,ret,csr,el_xlen, feat_src_xlen);

        nthrs_x = utils::FindNumThreads(el_xlen, 64);
        nthrs_y = utils::FindNumThreads(gdata.feat_src_hidden, MAX_NTHRS/nthrs_x);
        nblks_x = 1;
        nblks_y = std::min(gdata.n, MAX_NBLKS);
        const dim3 nthrs2(nthrs_x, nthrs_y);
        const dim3 nblks2(nblks_x, nblks_y);
        //LOG(INFO) << "kernel2 blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:" <<nthrs_x << "*" << nthrs_y;
        gatSumProdZipDivKernel<<<nblks2, nthrs2, 0, thr_entry->stream>>>(gdata, csr);
}

void BackwardFusedGatKernelImpl(
    const CSRWrapper& graph,
    runtime::NDArray feat_src,
    runtime::NDArray el,
    runtime::NDArray er,
    runtime::NDArray sum,
    runtime::NDArray exp,
    runtime::NDArray ret,
    runtime::NDArray grad_out,
    runtime::NDArray grad_feat_src,
    runtime::NDArray grad_el,
    runtime::NDArray grad_er,
    float slope) {
        typedef int32_t Idx;
        typedef float DType;
        const Idx MAX_NBLKS = 65535;
        const Idx MAX_NTHRS = 1024;
        // zero out ret, and packing feat_src, el, er, ret, graph together into one struct using raw float pointers
        // get csr matrix
        BackwardGatFusedData<Idx, DType> gdata;
        int64_t el_xlen =  utils::ComputeXLength(el);
        int64_t feat_src_xlen =  utils::ComputeXLength(feat_src);
        gdata.feat_src = static_cast<DType*>(feat_src->data);
        gdata.el = static_cast<DType*>(el->data);
        gdata.er = static_cast<DType*>(er->data);
        gdata.sum = static_cast<DType*>(sum->data);
        gdata.exp = static_cast<DType*>(exp->data);
        gdata.ret = static_cast<DType*>(ret->data);
        gdata.grad_out= static_cast<DType*>(grad_out->data);
        gdata.grad_feat_src = static_cast<DType*>(grad_feat_src->data);
        gdata.grad_el = static_cast<DType*>(grad_el->data);
        gdata.grad_er = static_cast<DType*>(grad_er->data);
        gdata.leaky_relu_slope = slope;
        gdata.n = el.GetSize()/sizeof(DType)/el_xlen; 
        gdata.e_xlen = el_xlen;
        gdata.feat_src_xlen =  feat_src_xlen;
        gdata.feat_src_hidden = feat_src_xlen/el_xlen;
        auto outcsr = graph.GetOutCSRMatrix();
        minigun::Csr<Idx> ocsr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
        gdata.eids = static_cast<Idx*>(outcsr.data->data);
        // write a device function and call it from here
        //LOG(INFO) << "Within Fused Gat Kernel Impl." << "feat_src_dim:" << feat_src.GetSize()/sizeof(DType)/feat_src_xlen << "*" << feat_src_xlen 
        //    <<" el_dim:" << el.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen  << " ret_dim:" << ret.GetSize()/sizeof(DType)/ret_len <<"*" << ret_len
        //    <<" sum_dim:" << sum.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    <<" exp_dim:" << exp.GetSize()/sizeof(DType)/el_xlen << "*" << el_xlen
        //    << " graph csr row_offset length:" <<csr.row_offsets.length << " graph csr column indices length:" << csr.column_indices.length;
        //print_gdata<Idx, DType>(feat_src,el,er,sum,exp,grad_out,ocsr,el_xlen, feat_src_xlen);
        // Configure kernel launch parameters.
        auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
        int nthrs_x = utils::FindNumThreads(el_xlen, 64);
        int nthrs_y = utils::FindNumThreads(gdata.feat_src_hidden, MAX_NTHRS/nthrs_x);
        int nblks_x = 1;
        int nblks_y = std::min(gdata.n, MAX_NBLKS);
        const dim3 nthrs(nthrs_x, nthrs_y);
        const dim3 nblks(nblks_x, nblks_y);
        LOG(INFO) << "GradFeatSrc kernel blk dim:" << nblks_x << "*" <<nblks_y << " thr dim:" <<nthrs_x << "*" << nthrs_y;
        fusedGatBackwardGradFeatSrc<<<nblks, nthrs, 0, thr_entry->stream>>>(gdata, ocsr);
        fusedGatBackwardGradElEr<<<nblks, nthrs, 0, thr_entry->stream>>>(gdata, ocsr);
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

template <typename Idx>
std::string print_csr(const minigun::Csr<Idx>& csr) {
    Idx row_len = csr.row_offsets.length;
    Idx col_ind_len = csr.column_indices.length;
    Idx* row_cpu = (Idx*) malloc(sizeof(Idx)*csr.row_offsets.length);
    Idx* col_ind_cpu = (Idx*) malloc(sizeof(Idx)*csr.column_indices.length);
    cudaMemcpy(row_cpu, csr.row_offsets.data, sizeof(Idx)*csr.row_offsets.length, cudaMemcpyDeviceToHost);
    cudaMemcpy(col_ind_cpu, csr.column_indices.data, sizeof(Idx)*csr.column_indices.length, cudaMemcpyDeviceToHost);
    std::string tmp = "";
    tmp += "row_offsets:\n";
    for (Idx i=0; i<row_len;++i) {
        tmp += std::to_string(row_cpu[i]) + ", ";
    }
    tmp += "\ncol_indices:\n";
    for (Idx i=0; i<col_ind_len; ++i) {
        tmp += std::to_string(col_ind_cpu[i])  + ", ";
    }
    tmp+= "\n";
    free(row_cpu);
    free(col_ind_cpu);
    return tmp;
}

template <typename Idx, typename DType>
std::string print_gdata2d(runtime::NDArray a, Idx dim1, Idx dim2) {
    if (a->ctx.device_type != kDLGPU) {
        LOG(FATAL) << "Tensor is not on GPU it is on:" << a->ctx.device_type;
    }
    Idx size = a.GetSize();
    DType* vals = (DType*)malloc(size);
    memset(vals, 0, size);
    cudaError_t err = cudaMemcpy(vals, a->data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG(FATAL) << std::string(cudaGetErrorString(err));
    }
    Idx n = size/sizeof(DType);
    if (n != dim1 * dim2) {
        LOG(FATAL) << "dim1 * dim2 != n, dim1:" << dim1 << " dim2:" << dim2 << " n" << n;
    }
    std::string str = "[";
    for (Idx i=0; i<dim1; i++) {
        str += "[";
        for (Idx j=0; j<dim2; j++) {
            str += std::to_string(vals[i*dim2+j]) + ", ";
        }
        str += "]\n";
    }
    str += "]\n";
    free(vals);
    return str;
}

template <typename Idx, typename DType>
void print_gdata(runtime::NDArray feat_src,
    runtime::NDArray el,
    runtime::NDArray er,
    runtime::NDArray sum,
    runtime::NDArray exp,
    runtime::NDArray ret,
    const minigun::Csr<Idx> &csr,
    Idx el_xlen,
    Idx feat_src_xlen) {
        std::string str_csr = print_csr<Idx>(csr);
        std::string str_el = print_gdata2d<Idx, DType>(el, csr.row_offsets.length-1, el_xlen);
        std::string str_er = print_gdata2d<Idx, DType>(er, csr.row_offsets.length-1, el_xlen);
        std::string str_feat_src= print_gdata2d<Idx, DType>(feat_src, csr.row_offsets.length-1, feat_src_xlen);
        std::string str_exp = print_gdata2d<Idx, DType>(exp, csr.column_indices.length, el_xlen);
        std::string str_sum = print_gdata2d<Idx, DType>(sum, csr.row_offsets.length-1, el_xlen);
        std::string str_ret = print_gdata2d<Idx, DType>(ret, csr.row_offsets.length-1, feat_src_xlen);
        LOG(INFO) << "csr " << str_csr << " feat_src "<< str_feat_src << " el "<< str_el << " er " << str_er << "exp " << str_exp << "sum " <<str_sum << " ret" << str_ret;
}

}  // namespace kernel
}  // namespace dgl
