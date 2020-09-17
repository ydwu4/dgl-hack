import argparse, time, math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import dgl.function as fn
from dgl.data import register_data_args, load_data
from egl import CtxManager
from torch.autograd import grad

class EglGCNConvTest(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(EglGCNConvTest, self).__init__()
        self.g = g
        self.norm = self.g.ndata['norm']
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()
        self.cm = CtxManager(dgl.backend.run_egl)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        def dgl_compute(h):
            h = h * self.g.ndata['norm']
            self.g.ndata['h'] = h
            self.g.update_all(fn.copy_src(src='h', out='m'),
                              fn.sum(msg='m', out='h'))
            h = self.g.ndata.pop('h')
            # normalization by square root of dst degree
            h = h * self.g.ndata['norm']
            #if self.bias is not None:
            #    h = h + self.bias
            #if self.activation:
            #    h = self.activation(h)
            return h
            
        def egl_compute(h):
            dgl_context = dgl.utils.to_dgl_context(h.device)
            graph = self.g._graph.get_immutable_gidx(dgl_context)
            @self.cm.zoomIn(nspace=[self, torch])
            def nb_compute(v):
                h = sum([nb.h*nb.norm for nb in v.innbs])
                h = h * v.norm
                # bias
                #if self.bias is not None:
                    #h = h + self.bias
                #if self.activation:
                #    h = self.activation(h)
                return h
            return nb_compute(g=graph, n_feats={'norm': self.norm, 'h' : h})

        h1 = dgl_compute(h)
        h2 = egl_compute(h)
        grad_out = torch.ones_like(h1)
        print('Forward rst Close enough?', torch.allclose(h1, h2), '\ndgl_out:', h1, '\negl_out:', h2)
        dgl_grad_weight = grad(h1, self.weight, grad_out, retain_graph=True)
        egl_grad_weight = grad(h2, self.weight, grad_out, retain_graph=True)
        print('grad_weight Close enough?', torch.allclose(dgl_grad_weight[0], egl_grad_weight[0]), '\ndgl_out:', dgl_grad_weight, '\negl_out:', egl_grad_weight)
        dgl_grad_h = grad(h1, h, grad_out, retain_graph=True)
        egl_grad_h = grad(h2, h, grad_out, retain_graph=True)
        print('grad_h Close enough?', torch.allclose(dgl_grad_h[0], egl_grad_h[0]), '\ndgl_h:', dgl_grad_h, '\negl_h:', egl_grad_h)
        for k, v in self.cm._ctx_map.items():
            print('tensor map ', k , v._executor_cache.ts.tensor_map)
        #dgl_grad_bias = grad(h1, self.bias, grad_out, retain_graph=True)
        #egl_grad_bias = grad(h2, self.bias, grad_out, retain_graph=True)
        #print('grad_bias Close enough?', torch.allclose(dgl_grad_bias[0], egl_grad_bias[0]), '\ndgl_out:', dgl_grad_bias, '\negl_out:', egl_grad_bias)
        return h1, h2

def main(args):
    NUM_NODES = args.num_nodes
    NUM_HIDDEN = args.num_hidden
    IN_FEATS=args.in_feats
    torch.cuda.set_device(args.gpu)
    
    g = DGLGraph()
    g.add_nodes(NUM_NODES)
    g.add_edges([i for i in range(NUM_NODES)], 0)
    g.add_edges([i for i in range(NUM_NODES)], 1)
    norm = torch.rand((NUM_NODES, 1)).cuda()
    g.ndata['norm'] = norm
    feat_src = torch.rand((NUM_NODES, IN_FEATS))
    feat_src.requires_grad = True
    feat_src = feat_src.cuda()
    conv_test = EglGCNConvTest(
                 g,
                 IN_FEATS,
                 NUM_HIDDEN,
                 activation=torch.nn.functional.relu,
                 dropout=args.dropout,
                 bias=True)
    conv_test.cuda()
    dgl_rst, egl_rst = conv_test.forward(feat_src)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--num_nodes", type=int, default=5,
            help="num nodes")
    parser.add_argument("--in_feats", type=int, default=1,
            help="in feats")
    parser.add_argument("--num_hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)