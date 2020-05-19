# Programming Model and JIT Technical Documentation
This document serves as the design doc as well as implementation specification.
## Objective
We propose a new vertex-centric shared memory model and implement auto-differentiation and JIT.
## Motivation
DGL (and all existing GNN systems) adopt a *dataflow-centric* programming model in order to leverage the automatic differentiation capability of backend DL systems. Specifically, graph propogation operations such as aggregating features of neighbors are implemented as operators. There are two drawbacks of this design: *high comprehension burden* of writing algorithms and *low computation and memory efficiency*. 

To understand the problem of *high comprehension burden*, we consider a simple operation from an existing GNN model DGMG which concates for each edge the src node's feature, edge's feature and dst node's feature together. We attribute this burden to the *dataflow-centric* programming model where users are forced to program using node and edge feature *tensor*.

The problem of *low computation and memory efficiency* is also due to *dataflow-centric* programming model. To conduct message propogation in this model, some systems(PyG, NeuGraph) use simple edge materialization to prepare for edge-wise computation then aggregation. Specifically, the scatter operation copies from the original node feature tnesor into an new tensor according to the outgoing edges incident to this node. Then edge-wise operations are performed on this new tensor. This creates huge amount of intermediate results whose sizes are proportional to the number of edges. DGL takes one step further, it provides built-in functions such as copyFromSrc and sumByDst as arguments of an operator updateAll. The system can automatically fuse these two operations into one operation thus saving both momeory usage and traffic. However this approach is not perfect as the graph computation is a seperate operator from the rest the data-flow programming model prohibiting further *holistic optimizations*.

## Design
### Programming model 
The basic idea is that we factor out the node dimension and let user focuse on the computation patterns around a vertex's neighborhood. Such a design is possible based on the fact (or assumption?) that 1. different nodes share the common set of model parameters 2. operators usually apply on feature dimension and repeat across the node (in other words: batch) dimension. Therefore the same deep learning model is still valid (in the sense that the shape of tensors flowing through operator chains remains coherent) even if we factor out node dimension (0-th dimension). This factoriation may appear uesless for traditional DL models such as CNN but can be critical for GNN in terms of lowering the comprehension burden.

We use GATConv as an illustration: 
TODO: 1. Add more examples e.g. GeomGCN, SageConv, GCNConv, ChebConv, AGNNConv, RGCN.
      2. Add examples that use conditionals, for loops, heterogeneous edges.
```python
def forward(self, graph, nfeat):
    # Global scope. Users are allowed to do any pre-transformations before zoomIn
    # ...
    # ... 
    # Local scope.
    with zoomIn(graph=graph, node_feat=[nfeat]) as v:
        '''Use list comprehension and redudcion to do computation'''
        feat_src = [self.fc(self.feat_drop(n.nfeat)).view(self.num_heads, self.out_feats) for n in v.innbs]
        el = [ f * self.attn_l.sum(dim=-1) for f in feat_src]
        er = self.fc(self.feat_drop(v.nfeat)).view(self.num_heads, self.out_feats) * self.attn_r.sum(dim=-1)
        coeff = [th.exp(self.leaky_relu(l + er)) for l in el]
        s = sum(coeff)
        alpha = [c/s for c in coeff]
        rst = sum([ef[0]*ef[1] for ef in zip(alpha, feat_src)])
    # Global scope. Users are allowed to do any post-transformations after zoomIn.
    return zoomOut(rst)
```


### Front End
The objetive of front end is to construct an intermediate representation in c++ space according to the python programs written in local scope. This representation must 1. precisely describe what need to be done 2. can support auto-differentitation, operator fusion, memory planning and code generations.

#### What to represent?
To simplify developement, we construct a representation for statements only within zoomIn. Notes: 1. users may not fully exploit the benefit of JIT due to failing to include more computation in the zoomIn context. 2. For backward propagation, we need to know which variable whithin zoomIn are used outside and contains gradients.

#### How to generate such representation?
 As in Pytorch, we can have 1. tracing 2. scripting. The third and fourth option is modified according to our need.
1. Tracing means we generate representations as python interpreter goes through the statements. 
pros: Easy to implement.
cons: Cannot support data-dependent control flow: conditional statement will be executed directly creating a linear graph without branching. Need to construct a sample input to do the tracing, which somewhat complicates the interface.

2. Scripting means we analyze the python abastrat syntax tree and then generate representations accordingly.
pros: Can support data-dependent control flow.
cons: Complex to implement.

3. Tracing within context. The first time entering the zoomIn contex, the operators are monkey patched then executed in a **symbolic** manner which generates an intermediate representation. When zoomOut is called, the compilation/execution is triggered.
pros: Easiest to implement, clean user interface.
cons: cannot capture operations before (after) the zoomIn (zoomOut). This can be addressed from users' perspective, though. Cannot support data-dependent control flow: conditional statement will be executed directly creating a linear graph without branching.

4. Scripting within context. The first time entering the zoomIn context, we analyze the source code.
pros: similar to 3
cons: cannot know which outputs are used later at the time of compilation

#### What's the detailed specification of intermediate representation?
We adopt a similar IR as in pytorch, which takes a Static Single Assignment(SSA) form: var_id:type = operator_id(var_id1, var_id2).
**Differences**:
We add *type scope* to denote whether a variable is a 
1. innbs(similar for outnbs if its needed
.): inneighbor's variable
2. v:     current node's varialbe
3. e:     edge variable
4. n:     node variable

We add *operator scope*:
1. node:    node specific operators;
2. edge:    functions conducted on edges, which can be implemented directly using cuda;
3. agg:     neighborhood aggregation operators(Note:leave advanced aggregation operators such as LSTM as future work); 
4. prim:    primitives such as constants;
5. layout:  e.g.use zip to access muliple feature list e.g. (a[0],b[0]), (a[1],b[1]) ... of the same dimension

The usage of type and operator annotations will be made clear when we conduct optimizations on this IR. We use GAT as an example.
```
graph(%num_heads: Int,
      %out_feats: Int,
      %drop_out_rate: Float,
      %attn_l : ParamTensor,
      %attn_r : ParamTensor,
      %v: CurNode):
    # feat_src = [self.fc(self.feat_drop(n.nfeat)).view(self.num_heads, self.out_feats) for n in v.innbs]
    %1 : innbs::Unknown = node::drop_out(%v.innbs.nfeat, %drop_out_rate) 
    %2 : innbs::Unknown = node::linear(%1)
    %feat_src : innbs::Unknown = node::view(%2, %num_heads, %out_feats)
    # el = [ f * self.attn_l.sum(dim=-1) for f in feat_src]
    %4 : innbs::Unknown = node::mul(%feat_src, %attn_l)
    %5 : innbs::Unknown = prim::Constant(-1)
    %el : innbs::Unknown = node::sum(%4, dim=%5) 

    # er = self.fc(self.feat_drop(v.nfeat)).view(self.num_heads, self.out_feats) * self.attn_r.sum(dim=-1)
    %6 : v::Unknown = node::drop_out(%v.nfeat, %drop_out_rate) 
    %7 : v::Unknown = node::linear(%1)
    %8 : v::Unknown = node::view(%7, %num_heads, %out_feats)
    %9 : v::Unknown = node::mul(%8, %attn_r)
    %10 : v::Unknown = prim::Constant(-1)
    %er : v::Unknown = node::sum(%9, dim=%10) 

    # coeff = [self.leaky_relu(th.exp(l + er)) for l in el] 
    %11 : e::Unknown = edge::add(%el, %er)
    %12 : e::Unknown = edge::leaky_relu(%11)
    %coeff: e::Unknown = edge::exp(%12)

    # s = sum(coeff) 
    %s : v::Unknown = agg::sum(%coeff)
      
    # alpha = [c/s for c in coeff]
    %alpha: e::Unknown = edge::div(%coeff, %s)

    # rst = sum([ef[0]*ef[1] for ef in zip(alpha, feat_src)])
    %13 : e::Unknown, %14 : e::Unknown = layout::zip(%alpha, %feat_src)
    %15 : e::Unknown = edge::mul(%13, %14)
    %rst : v::Unknown = agg::sum(%15)
```
#### Variable and Operation Type Scoping Propagation
Simple rules to determine a variable scope:
0. if a variable is in arguments of the graph, its type scoping is none. None scope is considered the same with all scopes;
1. v.innbs.* belong to innbs scope, v.feat belong to v scope, v.inedges.* belong to e scope; 
2. The return value of a function which takes one or more variables of the same scope will be the same type scope;
3. A function which takes different type scope will be in edge scope its return value is edge scope. e.g. edge::sum(%el, %er);
4. agg scope functions accepts edge scope variables and returns node scope variables.

#### Operation Batching 
The purporse for this step is to transform the local node-centric operations to graph-level operations. The reason is that code for node-centric IR has extremely low efficiency due to redundant computations and many smalll kernel launches.

Basically what this step does is upcasting the scope of variable v and its innbs to be n. Meaning that operations that uses/produces variables in the scope of v and n as input/output will instead use node feature tensor. The casting from v to n introduces additional computation and intermediate states for nodes that have no incoming edges. The casting from innbs to n introduces additional computation and intermediate states for nodes have no outgoing edges. However, considering the overhead of copying and maintaining additional tensors for 1. nodes that contain at least one incoming edges and 2. ndoes that contain at least one outgoing edges is large in most times, currently we simply do the upcasting for both nodes. It might be interesting to have a special extraction operation to allow more efficient computation on special graphs. Note that for both casting operations, we end up with computing the **superset** of what need to done in local IR therefore all node features that are required in later edge-wise computation are still present.

This lowering process creates the following IR.
```
graph(%num_heads: Int,
      %out_feats: Int,
      %drop_out_rate: Float,
      %attn_l : ParamTensor,
      %attn_r : ParamTensor,
      %nfeat: NodeFeatTensor):
    # feat_src = [self.fc(self.feat_drop(n.nfeat)).view(self.num_heads, self.out_feats) for n in v.innbs]
    %1 : n::Unknown = node::drop_out(%nfeat, %drop_out_rate) 
    %2 : n::Unknown = node::linear(%1)
    %feat_src : n::Unknown = node::view(%2, %num_heads, %out_feats)
    # el = [ f * self.attn_l.sum(dim=-1) for f in feat_src]
    %4 : n::Unknown = node::mul(%feat_src, %attn_l)
    %5 : n::Unknown = prim::Constant(-1)
    %el : n::Unknown = node::sum(%4, dim=%5) 

    # er = self.fc(self.feat_drop(v.nfeat)).view(self.num_heads, self.out_feats) * self.attn_r.sum(dim=-1)
    %6 : n::Unknown = node::drop_out(%nfeat, %drop_out_rate) 
    %7 : n::Unknown = node::linear(%1)
    %8 : n::Unknown = node::view(%7, %num_heads, %out_feats)
    %9 : n::Unknown = node::mul(%8, %attn_r)
    %10 : n::Unknown = prim::Constant(-1)
    %er : n::Unknown = node::sum(%9, dim=%10) 

    # coeff = [self.leaky_relu(th.exp(l + er)) for l in el] 
    %11 : e::Unknown = edge::add(%er, %el)
    %12 : e::Unknown = edge::leaky_relu(%11)
    %coeff: e::Unknown = edge::exp(%12)

    # s = sum(coeff) 
    %s : n::Unknown = agg::sum(%coeff)
      
    # alpha = [c/s for c in coeff]
    %alpha: e::Unknown = edge::div(%coeff, %s)

    # rst = sum([ef[0]*ef[1] for ef in zip(alpha, feat_src)])
    %13, %14 : e::Unknown = layout::zip(%alpha, %feat_src)
    %15 : e::Unknown = edge::mul(%13, %14)
    %rst : n::Unknown = agg::sum(%15)
```
**TODO: How to prove it's correct? Will this step break some semantic in forward or backward computation?**
For forward computation:
Not breaking the semantic of forward computation means that both program *can* produce the same result. We only need to prove that batching produces a superset of results. To see this, we can show that inbbs and v are both subsets of node scope feature tensor therefore operations conducted on each individual nodes will not create results beyond the results computed based on all nodes. 

### Back End
Up until now, we have created an IR for the original program that's ready for optimiation and code generation. The objective of the backend is to apply various transformations to the global-level representation generated by front end, which include auto-differentiation, operator fusion, memory planning etc then conduct code generation.

#### Common Subexpression Elimination (CSE)
The Operation Batching step may create duplicated sub-expressions. This naturally lead us to CSE. We show the expected result of this pass:
```
graph(%num_heads: Int,
      %out_feats: Int,
      %drop_out_rate: Float,
      %attn_l : ParamTensor,
      %attn_r : ParamTensor,
      %nfeat: NodeFeatTensor):
    # feat_src = [self.fc(self.feat_drop(n.nfeat)).view(self.num_heads, self.out_feats) for n in v.innbs]
    %1 : n::Unknown = node::drop_out(%nfeat, %drop_out_rate) 
    %2 : n::Unknown = node::linear(%1)
    %feat_src : n::Unknown = node::view(%2, %num_heads, %out_feats)

    # el = [ f * self.attn_l.sum(dim=-1) for f in feat_src]
    %3 : n::Unknown = node::mul(%feat_src, %attn_l)
    %4 : n::Unknown = prim::Constant(-1)
    %el : n::Unknown = node::sum(%3, dim=%4) 

    # er = self.fc(self.feat_drop(v.nfeat)).view(self.num_heads, self.out_feats) * self.attn_r.sum(dim=-1)
    %5 : n::Unknown = node::mul(%feat_src, %attn_r)
    %er : n::Unknown = node::sum(%5, dim=%4) 

    # coeff = [self.leaky_relu(th.exp(l + er)) for l in el] 
    %6 : e::Unknown = edge::add(%er, %el)
    %7 : e::Unknown = edge::leaky_relu(%6)
    %coeff : e::Unknown = edge::exp(%7)

    # s = sum(coeff) 
    %s : n::Unknown = agg::sum(%coeff)
      
    # alpha = [c/s for c in coeff]
    %alpha : e::Unknown = edge::div(%coeff, %s)

    # rst = sum([ef[0]*ef[1] for ef in zip(alpha, feat_src)])
    %8, %9 : e::Unknown = layout::zip(%alpha, %feat_src)
    %10 : e::Unknown = edge::mul(%8, %9)
    %rst : n::Unknown = agg::sum(%10)
```
#### Auto differentiation
**TODO: Add data type info. handling bcast.**
The scope of auto-differentiation is the same as code generation. Other parts will be delegated to the auto-diff of DL framework.

We need to first figure out which ParamTensor requires gradient and which variable in zoomIn() are used after the context is closed. Variables are not used can be eilimitated. To know which variables are used and may have gradients, we create an autograd function zoomOut(), which takes a variadic number of symbolic variables used whithin zoomIn() context and returns the associated tensors.

In GAT, zoomOut is triggered for rst. Rst has gradient gradOut which is of same dimension as rst. attn_l, attn_r and linear all contains parameters. Therfore the auto-diff boundary for gat will be starting from %6 = ... util %rst = ....

Backward IR for edge computation. Here the sum will be aggregated on source nodes(it seems necessary to provide two name scopes for aggregation: srcagg: aggregation on src nodes and dstagg: aggregation on destination nodes). 
```
coeff : e:Unknown = edge::exp(%7) =>
gradv7 : e:Unknown = gradcoeff * coeff

%7 : e::Unknown = edge::leaky_relu(%6) => 
gradv6 : e:Unkown = gradv7 * edge::backward_leaky_relu(%6)

%6 : e::Unknown = edge::add(%el, %er) => 
gradel: n:Unkown = srcagg::sum(gradv6)
grader: n:Unkown = dstagg::sum(gradv6)
```

Backward IR for sum aggregation 
```
%s : n::Unknown = agg::sum(%coeff) =>
gradcoeff =  srcagg::sum(grads)
```

#### Operator Fusion
**TODO: Add backward fusion**
Create fusion groups. Operations inside a group will be compiled to a single GPU kernel meaning that the result of producer operator can be immediately consumed by consumer. 

We decide fusion boundaries using a two-direction greedy expansion strategy. We start from the beginning of the IR, when we encounter an edge function, we create a fusion group that include the current edge function then check whether it's possible to fuse the operations previous and next (based on input-output dependency) to it and expand the fusion group if they are fusable until the fusion group can no longer expand. 

How to determine whether it's fusable? Several heuristics we have:
0. consecutive element-wise operators in the same scope can be fused together;
1. agg operations can be fused with its upstream edge scoped element-wise operators;
2. agg operations can be fused with its downstream node scope element-wise operators.

Applying this heuristic to GAT forward we get the following:
```
graph(%num_heads: Int,
      %out_feats: Int,
      %drop_out_rate: Float,
      %attn_l : ParamTensor,
      %attn_r : ParamTensor,
      %nfeat: NodeFeatTensor):
    # feat_src = [self.fc(self.feat_drop(n.nfeat)).view(self.num_heads, self.out_feats) for n in v.innbs]
    %1 : n::Unknown = node::drop_out(%nfeat, %drop_out_rate) 
    %2 : n::Unknown = node::linear(%1)
    %feat_src : n::Unknown = node::view(%2, %num_heads, %out_feats)

    # el = [ f * self.attn_l.sum(dim=-1) for f in feat_src]
    %3 : n::Unknown = node::mul(%feat_src, %attn_l)
    %4 : n::Unknown = prim::Constant(-1)
    %el : n::Unknown = node::sum(%3, dim=%4) 

    # er = self.fc(self.feat_drop(v.nfeat)).view(self.num_heads, self.out_feats) * self.attn_r.sum(dim=-1)
    %5 : n::Unknown = node::mul(%feat_src, %attn_r)
    %er : n::Unknown = node::sum(%5, dim=%4) 

    %s : n::Unknown, $coeff : e::Unknown = fusedGraph1(%el, %er)

    return fusedGraph2(%s, %coeff)

fusedGraph1(%el : n::Unknown
           %er : n::Unknown):
    # coeff = [self.leaky_relu(th.exp(l + er)) for l in el] 
    %6 : e::Unknown = edge::add(%er, %el)
    %7 : e::Unknown = edge::leaky_relu(%6)
    %coeff : e::Unknown = edge::exp(%7)

    # s = sum(coeff) 
    %s : n::Unknown = agg::sum(%coeff)
    return %s, %coeff

fusedGraph2(%s : n::Unknown
           %coeff: n::Unknown):
    # alpha = [c/s for c in coeff]
    %alpha : e::Unknown = edge::div(%coeff, %s)

    # rst = sum([ef[0]*ef[1] for ef in zip(alpha, feat_src)])
    %8, %9 : e::Unknown = layout::zip(%alpha, %feat_src)
    %10 : e::Unknown = edge::mul(%8, %9)
    %rst : n::Unknown = agg::sum(%10)
    return %rst
```
The end result is a partitioned graph with subGraphs

#### Memory Planning
Memory planning requires knowing the type and shape of each variables, number of edges, number of nodes and etc. We plan which variables will be re-computed and which variables will be materialized. This directly influence how backward computations are carries out. The influence is double-directional meaning that choosing the materialization point will affect how backward computation is conducted. Change of the backward computation can affect the planning objective e.g. amount of data loading.

We may borrow insight from works on recomputation for general nn. The end result is an annotation for each node in the graph whether or not to materiaze its outputs.

#### Code Generation 
We must generate cuda code for edge-scoped and agg-scoped operations as there are no built-in operations in DL systems to support them. We can optionaly generate code for node-scoped element-wise functions that involves no prameters or only constants to increase compute/memory ratio. (We need a good reason of not generating code for operations containing trainable parameters.)

In GNN workload, it's common to see a pattern: an edge-wise operation followed by an aggregation function. We can have three parallelization strategies: src node parallel, edge parallel and dst node parallel. Using edge or src node parallel introduces atomic operations due to the aggregation operation at the destination node. This makes dst node parallel a more plausible solution. We also use incsr format for forward propagation.

We design the following execution template for generating code for GNN workload:
```
tempate <typename Idx, typename DType>
struct GData {
    // input and output datafields
    Idx feat_size;
    // ...
};

// we have two dimensional blocks but only one dimension thread to simplify code generation
template <typename Idx, typename DType>
__global__ void ${kernelName}(GData<Idx, DType> gdata, Graph<Idx> graph) {
    Idx feat_size = gdata.feat_size;
    Idx num_nodes = graph.num_nodes;
    Idx feat_step = gridDim.y*blockDim.y;
    for (Idx dst_id = blockIdx.x; dst_id < num_nodes; dst_id += gridDim.x) {
        for (Idx ty = blockIdx.y * blockDim.y + threadIdx.y; ty < feat_size; feat_size += feat_step) {
            Idx start_off = graph.getInedgesStart(eid);
            Idx end_off = graph.getInedgesEnd(eid);
            ${aggInit}
            for (Idx eid=start_off;eid<end_off;++eid) {
                src_id = graph.getInedges(eid);
                ${edgeCompute} // formed by concating a series of edgeCompute
                ${aggCompute}
            }
            ${aggWrite}
            ${nodeCompute}
        }
    }
}
```

Sample generated cuda code for operation:
```
%6 : e::Unknown = edge::add(%el, %er) => 
edgeCompute= "
    // if not fusion: el = gdata.el[gdata.compute_noff(src_id, ty)];
    // if not fusion: er = gdata.el[gdata.compute_noff(dst_id, ty)]; 
    v6 = el + er;
    // if materialize: gdata.v6[gdata.compute_eoff(eid, ty)] = er;"

%7 : e::Unknown = edge::leaky_relu(%6) => 
edgeCompute= "
    // if not fusion: v6 = gdata.v6[gdata.compute_eoff(eid, ty)];
    v7 = leaky_relu(v6);
    // if materialize: gdata.v7[gdata.compute_eoff(eid, ty)] = v7;"

coeff : e:Unknown = edge::exp(%7) =>
edgeCompute= "
    // if not fusion: v7 = gdata.v7[gdata.compute_eoff(eid, ty)];
    coeff = exp(v7);
    // if materialize: gdata.coeff[gdata.compute_eoff(eid, ty)] = coeff;"
```

Generate cuda code for aggregation operation(only show the fused version):
```
%s : n::Unknown = agg::sum(%coeff) =>
    aggInit = "s = 0.;"
    aggCompute = "s += coeff;"
    aggWrite = "gdata.s[gdata.compute_noff(dst_id, ty)] = s;"
// if materialize: 
```

Generate cuda code for broadcast oeperation:
Definition of broadcast operation: element-wise operation for two tensors, A and B. A and B has different dimensions. For example, A has dimension 5 * 4 and B has dimension 1 * 4. Then mul(A, B) will returns a tensor of diemension 5 * 4.

For operations that are fused into one kernel, it's either an element-wise operation or an broadcast operation or an aggregation among edges therefore the output's dimensions will be larger than any input/intermediate results. We map each feature in output tensor to exactly one thread. To access input, if it has the same dimensions as output, we can directly use thread id (ty). To access inputs of different dimension with output, we need to map ty to the required offsets in that input, the formula is given by first compute the coordinate in output, then map to the coordinate of inputs finally dot product with the strides. We apply the same rule recursively and propogate index information backward to input. Due to broadcasting operation, we may need to generate code starting from the end of the kernel because for operators in the begining, it is not clear how ty is mapped to their accessor index.

To summarize:
For code generation, we are given the fusion groupGraph or normal nodes. we first start from the leaf node (TODO: deal with multiple leaves) and propogate the index accessing information to the inputs (a series of mapping function from ty to feature index). Then starting from first node, we generate edgeCompute one by one. For each statement we will use the option fusion and materialization annotated in pervious passes to generate string arguments as the template argument to the kernel template.

## Implementations
**TODO:**
0. The folder and file names
1. Declarations for all classes, procedures, and global/class variables
2. Descriptions of all procedures, including the purpose of the procedure, an explanation of how it works and/or pseudocode

## Tests