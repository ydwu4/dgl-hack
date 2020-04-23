import torch as th
from torch import nn
import dgl
import dgl.backend as B
from dgl import function as fn

NUM_NODES=102400
NUM_HEADS=64
NUM_HIDDEN=64
negative_slope = 0.2
dropout_ratio = 0.6
th.cuda.set_device(0)

g = dgl.DGLGraph()
g.add_nodes(NUM_NODES)
g.add_edges(0, [i for i in range(NUM_NODES)])
g.add_edges(1, [i for i in range(NUM_NODES)])
feat_src = th.rand((NUM_NODES, NUM_HEADS, NUM_HIDDEN))
el = th.rand((NUM_NODES, NUM_HEADS, 1))
er = th.rand((NUM_NODES, NUM_HEADS, 1))

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
    g.edata['a'] = (g.edata['out'])
    # message passing
    g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                     fn.sum('m', 'ft'))
    rst = g.dstdata['ft']
    return rst

r1 = expected_output()
r2 = B.fused_gat(g, feat_src, el, er, negative_slope)
print('expected ret', r1, 'gatfused ret', r2, " close enough?", th.allclose(r1, r2))
