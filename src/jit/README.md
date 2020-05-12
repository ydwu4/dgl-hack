# Programming Model and JIT Technical Documentation
This document serves as the design doc as well as implementation specification.
## Objective
We propose a new vertex-centric shared memory model and implement auto-differentiation and JIT.
## Motivation
DGL (and all existing GNN systems) adopt a *dataflow-centric* programming model in order to leverage the automatic differentiation capability of backend DL systems. Specifically, graph propogation operations such as aggregating features of neighbors are implemented as operators. There are two drawbacks of this design: *high comprehension burden* of writing algorithms and *low computation and memory efficiency*. 

To understand the problem of *high comprehension burden*, we consider a simple operation from an existing GNN model DGMG which concates for each edge the src node's feature, edge's feature and dst node's feature together. We attribute this burden to the *dataflow-centric* programming model where users are forced to program using node and edge feature *tensor*.

The problem of *low computation and memory efficiency* is also due to *dataflow-centric* programming model. To conduct message propogation in this model, some systems(PyG, NeuGraph) use simple edge materialization to prepare for edge-wise computation then aggregation. Specifically, the scatter operation copies from the original node feature tnesor into an new tensor according to the outgoing edges incident to this node. Then edge-wise operations are performed on this new tensor. This creates huge amount of intermediate results whose sizes are proportional to the number of edges. DGL takes one step further, it provides built-in functions such as copyFromSrc and sumByDst as arguments of an operator updateAll. The system can automatically fuse these two operations into one operation thus saving both momeory usage and traffic. However this approach is not perfect as the graph computation is a seperate operator from the rest the data-flow programming model prohibiting further holistic optimizations.

## Design
### Programming model 
The basic idea is that we factor out the node dimension and let user focuse on the computation patterns around a vertex's neighborhood. Such a design is possible based on the fact (or assumption?) that 1. different nodes share the common set of model parameters 2. operators usually apply on feature dimension and repeat across the node (in other words: batch) dimension. Therefore the same deep learning model is still valid (in the sense that the shape of tensors flowing through operator chains remains coherent) even if we factor out node dimension (0-th dimension). This factoriation may appear uesless for traditional DL models such as CNN but can be critical for GNN in terms of lowering the comprehension burden.

We use GATConv as an illustration. 
TODO: 1. Add more examples e.g. GCNConv, ChebConv, AGNNConv, GeomGCN.
      2. Add examples that use conditionals, for loops, heterogeneous edges.
```python
def forward(self, graph, nfeat):
    # Global scope. Users are allowed to do any pre-transformations before zoomIn
    # ...
    # ... 
    # Local scope.
    with zoomIn(graph=graph, node_feat=[nfeat]) as v:
        '''Use list comprehension and redudcion to do computation'''
        el = [self.fc(self.feat_drop(n.nfeat)).view(self.num_heads, self.out_feats) * self.attn_l.sum(dim=-1) for n in v.innbs]
        er = self.fc(self.feat_drop(v.nfeat)).view(self.num_heads, self.out_feats) * self.attn_r.sum(dim=-1)
        coeff = [self.leaky_relu(th.exp(l + er)) for l in el]
        s = sum(coeff)
        alpha = [c/s for c in coeff]
        rst = sum([ef[0]*ef[1] for ef in zip(alpha, feat_src)])
    # Global scope. Users are allowed to do any pre-transformations after zoomIn.
    return zoomOut(rst)
```

### Front End
The objetive of front end is to construct an intermediate representation in c++ space for the python programs written in local scope. This representation must 1. precisely describe what need to be done 2. easily support auto-differentitation, operator fusion, memory planning and code generations.

What to represent? We need to construct a representation for the whole forward function (instead of statements only within zoomIn). Reason: 1. allows us to explore opportunities beyond graph proporgation (to avoid performance dips due to users' misuse) 2. For backward propagation, we need to know whose gradient will be generated then we need to know which variable whithin zoomIn are used outside and has gradient.

How to generate such representation? As in Pytorch, we can have 1. tracing 2. scripting. The third option is lazy evaluation.
1. Tracing means we generate representations as python interpreter goes through the statements. 
pros: Easy to implement.
cons: Cannot support data-dependent control flow. Need to construct a sample input to do the tracing.

2. Scripting means we analyze the python abastrat syntax tree and then generate representations accordingly.
pros: Can support data-dependent control flow.
cons: Complex to implement.

3. Tracing within context. The first time entering the zoomIn contex, the operators are monkey patched then executed in a **symbolic** manner which generates an intermediate representation. When zoomOut is called, the compilation/execution is triggered.
pros: Easiest to implement, clean user interface.
cons: cannot capture operations before (after) the zoomIn (zoomOut). This can be addressed from users' perspective.

What's the detailed specification of intermediate representation?


### Back End

## Implementations
0. The folder and file names
1. Declarations for all classes, procedures, and global/class variables
2. Descriptions of all procedures, including the purpose of the procedure, an explanation of how it works and/or pseudocode

## Tests