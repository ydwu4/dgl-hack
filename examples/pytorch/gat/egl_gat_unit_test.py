import argparse
import torch as th
from torch.autograd import grad
from dgl import DGLGraph
from dgl import utils
from dgl import backend as B
from dgl import function as fn
from torch import nn

from egl import ContextManager

# pylint: enable=W0235
class EglGATConvTest(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(EglGATConvTest, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(
            self._in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.negative_slope = negative_slope
        self.res_fc = nn.Linear(
                    self._in_feats, num_heads * out_feats, bias=False)
        self.reset_parameters()
        self.activation = activation
        self.cm = ContextManager(B.run_egl)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        graph = graph.local_var()
        h_src = h_dst = self.feat_drop(feat)
        feat = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
        ell = (self.attn_l * feat).sum(dim=-1, keepdim=True) 
        err = (self.attn_r * feat).sum(dim=-1, keepdim=True)
        g = graph
        g.srcdata.update({'ft': feat , 'el': ell})
        g.dstdata.update({'er': err})
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(g.edata.pop('e'))
        g.edata['out'] = th.exp(e)
        g.update_all(fn.copy_e('out', 'm'), fn.sum('m', 'out_sum'))
        g.apply_edges(fn.e_div_v('out', 'out_sum', 'out1'))
        # Omit attn_drop for deterministic execution
        g.edata['a'] = g.edata['out1']
        # message passing
        g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        dglrst = g.dstdata['ft']
        fusedrst = B.fused_gat(g, feat, ell, err, self.negative_slope)

        dgl_context = utils.to_dgl_context(feat.device)
        graph = graph._graph.get_immutable_gidx(dgl_context)
        with self.cm.zoomIn(namespace=[self, th], graph=graph, node_feats={'f' : h_src}, edge_feats={}) as v:
            feat_src = [self.fc(n.f).view(self._num_heads, self._out_feats) for n in v.innbs]
            el = [(nf * self.attn_l).sum(dim=-1, keepdim=True) for nf in feat_src]
            er = (self.fc(v.f).view(self._num_heads, self._out_feats) * self.attn_r).sum(dim=-1, keepdim=True)
            coeff = [th.exp(self.leaky_relu(l + er)) for l in el]
            s = sum(coeff)
            alpha = [c/s for c in coeff]
            rst = sum([ef[0]*ef[1] for ef in zip(alpha, feat_src)])
            v.collect_output(rst)
        rst = self.cm.zoomOut()
        grad_out = th.ones_like(rst)
        egl_graer= grad(outputs=rst, inputs=self.cm._executor.ts.tensor_map['v7'], grad_outputs=grad_out, retain_graph=True)
        egl_grael= grad(outputs=rst, inputs=self.cm._executor.ts.tensor_map['v3'], grad_outputs=grad_out, retain_graph=True)
        dgl_graer = grad(outputs=dglrst, inputs=err, grad_outputs=grad_out, retain_graph=True)
        dgl_grael = grad(outputs=dglrst, inputs=ell, grad_outputs=grad_out, retain_graph=True)
        fused_graer = grad(outputs=fusedrst, inputs=err, grad_outputs=grad_out, retain_graph=True)
        fused_grael = grad(outputs=fusedrst, inputs=ell, grad_outputs=grad_out, retain_graph=True)
        print('rst close?', th.allclose(rst, dglrst), rst)
        #print('exp', g.edata['out'], 'div', g.edata['a'], 'rst', dglrst, 'feat', feat, 'ell', ell, 'err', err)
        print('\negl_graer', egl_graer,'\ndgl_graer', dgl_graer, '\nfused_graer', fused_graer, 'egl close with dgl?', th.allclose(egl_graer[0], dgl_graer[0]))
        print('\negl_grael', egl_grael,'\ndgl_grael', dgl_grael, '\nfused_grael', fused_grael, 'egl close with dgl?', th.allclose(egl_grael[0], dgl_grael[0]))
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst, dglrst



def main(args):
    NUM_NODES = args.num_nodes
    NUM_HEADS = args.num_heads
    NUM_HIDDEN = args.num_hidden
    negative_slope = args.negative_slope
    dropout_ratio = args.attn_drop
    th.cuda.set_device(args.gpu)
    
    IN_FEATS= 1
    g = DGLGraph()
    g.add_nodes(NUM_NODES)
    g.add_edges([i for i in range(NUM_NODES)], 0)
    g.add_edges([i for i in range(NUM_NODES)], 1)
    feat_src = th.rand((NUM_NODES, IN_FEATS))
    feat_src.requires_grad = True
    feat_src = feat_src.cuda()
    conv_test = EglGATConvTest(
                 IN_FEATS,
                 NUM_HIDDEN,
                 num_heads=NUM_HEADS,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=negative_slope,
                 residual=False,
                 activation=None)
    conv_test.cuda()
    rst, dgl_rst = conv_test.forward(g, feat_src)
    #grad_attnl_egl = grad(outputs=rst, inputs=conv_test.attn_l, grad_outputs=grad_out, retain_graph=True)
    #grad_attnl_dgl = grad(outputs=dgl_rst, inputs=conv_test.attn_l, grad_outputs=grad_out, retain_graph=True)
    #print('expected grad_attnl_egl', grad_attnl_egl[0], '\ngrad_attnl_dgl', grad_attnl_dgl[0] , 'close enough?',
    #         th.allclose(grad_attnl_egl[0], grad_attnl_dgl[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fusedGatUnitTest')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-nodes", type=int, default=5,
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
