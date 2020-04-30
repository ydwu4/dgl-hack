import argparse
import torch as th
from torch import nn
from torch.autograd import grad
import dgl
import dgl.backend as B
from dgl import function as fn


def main(args):
    NUM_NODES = args.num_nodes
    NUM_HEADS = args.num_heads
    NUM_HIDDEN = args.num_hidden
    negative_slope = args.negative_slope
    dropout_ratio = args.attn_drop
    th.cuda.set_device(args.gpu)
    
    g = dgl.DGLGraph()
    g.add_nodes(NUM_NODES)
    g.add_edges(0, [i for i in range(NUM_NODES)])
    g.add_edges(1, [i for i in range(NUM_NODES)])
    feat_src = th.ones((NUM_NODES, NUM_HEADS, NUM_HIDDEN))
    feat_src.requires_grad = True
    el = th.ones((NUM_NODES, NUM_HEADS, 1))
    el.requires_grad = True
    er = th.ones((NUM_NODES, NUM_HEADS, 1))
    er.requires_grad = True
    
    feat_src = feat_src.cuda()
    el = el.cuda()
    er = er.cuda()
    
    leaky_relu = nn.LeakyReLU(negative_slope)
    attn_drop = nn.Dropout(dropout_ratio)
    
    def expected_output():
        g.srcdata.update({'ft': feat_src, 'el': el})
        g.dstdata.update({'er': er})
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = leaky_relu(g.edata.pop('e'))
        g.edata['out'] = th.exp(e)
        g.update_all(fn.copy_e('out', 'm'), fn.sum('m', 'out_sum'))
        g.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
        # Omit attn_drop for deterministic execution
        g.edata['a'] = (g.edata['out'])
        # message passing
        g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = g.dstdata['ft']
        return rst
    
    r1 = expected_output()
    r2 = B.fused_gat(g, feat_src, el, er, negative_slope)
    print('expected ret', r1, 'gatfused ret', r2, " close enough?", th.allclose(r1, r2))
    grad_out = th.ones_like(r1)
    #grad_feat_src1 = grad(outputs=r1, inputs=feat_src, grad_outputs=grad_out, retain_graph=True)
    #grad_feat_src2 = grad(outputs=r2, inputs=feat_src, grad_outputs=grad_out, retain_graph=True)
    #print('expected grad_feat_src', grad_feat_src1, 'gatfused grad_feat_src', grad_feat_src2[0], 'close enough?',
    #        th.allclose(grad_feat_src1[0], grad_feat_src2[0]))

    grad_el1 = grad(outputs=r1, inputs=el, grad_outputs=grad_out, retain_graph=True)
    grad_el2 = grad(outputs=r2, inputs=el, grad_outputs=grad_out, retain_graph=True)
    print('expected grad_el', grad_el1[0], '\ngatfused grad_el', grad_el2[0], 'close enough?',
            th.allclose(grad_el1[0], grad_el2[0]))

    #grad_er1 = grad(outputs=r1, inputs=er, grad_outputs=grad_out, retain_graph=True)
    #grad_er2 = grad(outputs=r2, inputs=er, grad_outputs=grad_out, retain_graph=True)
    #print('expected grad_er', grad_er1[0], 'gatfused grad_er', grad_er2[0], 'close enough?',
    #        th.allclose(grad_er1[0], grad_er2[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fusedGatUnitTest')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-nodes", type=int, default=65536,
                        help="number of nodes of synthetic graph")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    args = parser.parse_args()
    print(args)
    main(args)
