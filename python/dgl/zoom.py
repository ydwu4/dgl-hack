import egl
from .backend.backend import run_egl

def zoomIn(namespace, graph, node_feats, edge_feats):
    return egl.zoomIn(namespace, graph, node_feats, edge_feats)

def zoomOut(vals):
    return egl.zoomOut(vals, run_egl)