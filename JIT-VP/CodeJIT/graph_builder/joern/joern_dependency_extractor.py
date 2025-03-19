import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import networkx as nx

from helpers import get_logger

logger = get_logger(__name__)


def joern_graph_extraction(source_file_path, joern_file_path):
    """Extract relevant components of Code Representation.

    DEBUGGING:
    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/180189.c"
    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/182480.c"

    """
    # cache_name = "_".join(str(filepath).split("/")[-3:])
    # cachefp = svd.get_dir(svd.cache_dir() / "ivdetect_feat_ext") / Path(cache_name).stem
    # try:
    #     with open(cachefp, "rb") as f:
    #         return pkl.load(f)
    # except:
    #     pass
    nodes, edges = get_node_edges(source_file_path, joern_file_path)
    # 1. Generate tokenised subtoken sequences
    # subseq = (
    #     nodes.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    #     .groupby("lineNumber")
    #     .head(1)
    # )
    # subseq = subseq[["lineNumber", "code", "local_type"]].copy()
    # subseq.code = subseq.local_type + " " + subseq.code
    # subseq = subseq.drop(columns="local_type")
    # subseq = subseq[~subseq.eq("").any(1)]
    # subseq = subseq[subseq.code != " "]
    # subseq.lineNumber = subseq.lineNumber.astype(int)
    # subseq = subseq.sort_values("lineNumber")
    # subseq.code = subseq.code.apply(tokenise)
    # subseq = subseq.set_index("lineNumber").to_dict()["code"]

    # 2. Line to AST
    # ast_edges = rdg(edges, "ast")
    # ast_nodes = drop_lone_nodes(nodes, ast_edges)
    # ast_nodes = ast_nodes[ast_nodes.lineNumber != ""]
    # ast_nodes.lineNumber = ast_nodes.lineNumber.astype(int)
    # ast_nodes["lineidx"] = ast_nodes.groupby("lineNumber").cumcount().values
    # ast_edges = ast_edges[ast_edges.line_out == ast_edges.line_in]
    # ast_dict = pd.Series(ast_nodes.lineidx.values, index=ast_nodes.id).to_dict()
    # ast_edges.innode = ast_edges.innode.map(ast_dict)
    # ast_edges.outnode = ast_edges.outnode.map(ast_dict)
    # ast_edges = ast_edges.groupby("line_in").agg({"innode": list, "outnode": list})
    # ast_nodes.code = ast_nodes.code.fillna("").apply(tokenise)
    # nodes_per_line = (
    #     ast_nodes.groupby("lineNumber").agg({"lineidx": list}).to_dict()["lineidx"]
    # )
    # ast_nodes = ast_nodes.groupby("lineNumber").agg({"code": list})
    # ast = ast_edges.join(ast_nodes, how="inner")
    # ast["ast"] = ast.apply(lambda x: [x.outnode, x.innode, x.code], axis=1)
    # ast = ast.to_dict()["ast"]
    #
    # # If it is a lone node (nodeid doesn't appear in edges) or it is a node with no
    # # incoming connections (parent node), then add an edge from that node to the node
    # # with id = 0 (unless it is zero itself).
    # # DEBUG:
    # # import sastvd.helpers.graphs as svdgr
    # # svdgr.simple_nx_plot(ast[20][0], ast[20][1], ast[20][2])
    # for k, v in ast.items():
    #     allnodes = nodes_per_line[k]
    #     outnodes = v[0]
    #     innodes = v[1]
    #     lonenodes = [i for i in allnodes if i not in outnodes + innodes]
    #     parentnodes = [i for i in outnodes if i not in innodes]
    #     for n in set(lonenodes + parentnodes) - set([0]):
    #         outnodes.append(0)
    #         innodes.append(n)
    #     ast[k] = [outnodes, innodes, v[2]]

    # 3. Variable names and types
    # reftype_edges = rdg(edges, "reftype")
    # reftype_nodes = drop_lone_nodes(nodes, reftype_edges)
    # reftype_nx = nx.Graph()
    # reftype_nx.add_edges_from(reftype_edges[["innode", "outnode"]].to_numpy())
    # reftype_cc = list(nx.connected_components(reftype_nx))
    # varnametypes = list()
    # for cc in reftype_cc:
    #     cc_nodes = reftype_nodes[reftype_nodes.id.isin(cc)]
    #     print("-", type(cc), cc, cc_nodes[cc_nodes["_label"] == "TYPE"])
    #     var_type = cc_nodes[cc_nodes["_label"] == "TYPE"].name.item()
    #     for idrow in cc_nodes[cc_nodes["_label"] == "IDENTIFIER"].itertuples():
    #         varnametypes += [[idrow.lineNumber, var_type, idrow.name]]
    # nametypes = pd.DataFrame(varnametypes, columns=["lineNumber", "type", "name"])
    # nametypes = nametypes.drop_duplicates().sort_values("lineNumber")
    # nametypes.type = nametypes.type.apply(tokenise)
    # nametypes.name = nametypes.name.apply(tokenise)
    # nametypes["nametype"] = nametypes.type + " " + nametypes.name
    # nametypes = nametypes.groupby("lineNumber").agg({"nametype": lambda x: " ".join(x)})
    # nametypes = nametypes.to_dict()["nametype"]

    # 4/5. Data dependency / Control dependency context
    # Group nodes into statements
    nodesline = nodes[nodes.lineNumber != ""].copy()
    nodesline.lineNumber = nodesline.lineNumber.astype(int)
    nodesline = (
        nodesline.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
        .groupby("lineNumber")
        .head(1)
    )
    edgesline = edges.copy()
    edgesline.innode = edgesline.line_in
    edgesline.outnode = edgesline.line_out
    nodesline.id = nodesline.lineNumber
    edgesline = rdg(edgesline, "pdg")
    nodesline = drop_lone_nodes(nodesline, edgesline)
    # Drop duplicate edges
    edgesline = edgesline.drop_duplicates(subset=["innode", "outnode", "etype"])
    # REACHING DEF to DDG
    edgesline["etype"] = edgesline.apply(
        lambda x: "DDG" if x.etype == "REACHING_DEF" else x.etype, axis=1
    )
    edgesline = edgesline[edgesline.innode.apply(lambda x: isinstance(x, float))]
    edgesline = edgesline[edgesline.outnode.apply(lambda x: isinstance(x, float))]
    edgesline_reverse = edgesline[["innode", "outnode", "etype"]].copy()
    edgesline_reverse.columns = ["outnode", "innode", "etype"]
    uedge = pd.concat([edgesline, edgesline_reverse])
    uedge = uedge[uedge.innode != uedge.outnode]
    uedge = uedge.groupby(["innode", "etype"]).agg({"outnode": set})
    uedge = uedge.reset_index()
    if len(uedge) > 0:
        uedge = uedge.pivot("innode", "etype", "outnode")
        if "DDG" not in uedge.columns:
            uedge["DDG"] = None
        if "CDG" not in uedge.columns:
            uedge["CDG"] = None
        uedge = uedge.reset_index()[["innode", "CDG", "DDG"]]
        uedge.columns = ["lineNumber", "control", "data"]
        uedge.control = uedge.control.apply(
            lambda x: list(x) if isinstance(x, set) else []
        )
        uedge.data = uedge.data.apply(lambda x: list(x) if isinstance(x, set) else [])
        data = uedge.set_index("lineNumber").to_dict()["data"]
        control = uedge.set_index("lineNumber").to_dict()["control"]
    else:
        data = {}
        control = {}

    # Generate PDG
    pdg_nodes = nodesline.copy()
    pdg_nodes = pdg_nodes[["id"]].sort_values("id")
    # pdg_nodes["subseq"] = pdg_nodes.id.map(subseq).fillna("")
    # pdg_nodes["ast"] = pdg_nodes.id.map(ast).fillna("")
    # pdg_nodes["nametypes"] = pdg_nodes.id.map(nametypes).fillna("")
    pdg_nodes["data"] = pdg_nodes.id.map(data)
    pdg_nodes["control"] = pdg_nodes.id.map(control)
    pdg_edges = edgesline.copy()
    pdg_nodes = pdg_nodes.reset_index(drop=True).reset_index()
    pdg_dict = pd.Series(pdg_nodes.index.values, index=pdg_nodes.id).to_dict()
    pdg_edges.innode = pdg_edges.innode.map(pdg_dict)
    pdg_edges.outnode = pdg_edges.outnode.map(pdg_dict)
    pdg_edges = pdg_edges.dropna()
    pdg_edges = (pdg_edges.outnode.tolist(), pdg_edges.innode.tolist())

    # # Cache
    # with open(cachefp, "wb") as f:
    #     pkl.dump([pdg_nodes, pdg_edges], f)
    return pdg_nodes, pdg_edges


def get_node_edges(source_file_path: str, joern_file_path: str, verbose=0):
    """Get node and edges given filepath (must run after write_joern_parsed_output).

    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/53.c"
    """
    outdir = Path(joern_file_path).parent
    outfile = outdir / Path(joern_file_path).name
    with open(str(outfile) + ".edges.json", "r") as f:
        edges = json.load(f)
        edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow"])
        edges = edges.fillna("")

    with open(str(outfile) + ".nodes.json", "r") as f:
        nodes = json.load(f)
        nodes = pd.DataFrame.from_records(nodes)
        if "controlStructureType" not in nodes.columns:
            nodes["controlStructureType"] = ""
        nodes = nodes.fillna("")
        try:
            nodes = nodes[
                ["id", "_label", "name", "code", "lineNumber", "controlStructureType"]
            ]
        except Exception as E:
            if verbose > 1:
                logger.warning(f"Failed {source_file_path}: {E}")
            return None, None

    # Assign line number to local variables
    with open(source_file_path, "r") as f:
        code = f.readlines()
    lmap = assign_line_num_to_local(nodes, edges, code)
    nodes.lineNumber = nodes.apply(
        lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
    )
    nodes = nodes.fillna("")

    # Assign node name to node code if code is null
    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x["name"], axis=1)

    # Assign node label for printing in the graph
    nodes["node_label"] = (
            nodes._label + "_" + nodes.lineNumber.astype(str) + ": " + nodes.code
    )

    # Filter by node type
    nodes = nodes[nodes._label != "COMMENT"]
    nodes = nodes[nodes._label != "FILE"]

    # Filter by edge type
    edges = edges[edges.etype != "CONTAINS"]
    edges = edges[edges.etype != "SOURCE_FILE"]
    edges = edges[edges.etype != "DOMINATE"]
    edges = edges[edges.etype != "POST_DOMINATE"]

    # Remove nodes not connected to line number nodes (maybe not efficient)
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
        left_on="outnode",
        right_on="id",
    )
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
        left_on="innode",
        right_on="id",
    )
    edges = edges[(edges.line_out != "") | (edges.line_in != "")]

    # Uniquify types
    edges.outnode = edges.apply(
        lambda x: f"{x.outnode}_{x.innode}" if x.line_out == "" else x.outnode, axis=1
    )
    typemap = nodes[["id", "name"]].set_index("id").to_dict()["name"]

    linemap = nodes.set_index("id").to_dict()["lineNumber"]
    for e in edges.itertuples():
        if type(e.outnode) == str:
            lineNum = linemap[e.innode]
            node_label = f"TYPE_{lineNum}: {typemap[int(e.outnode.split('_')[0])]}"
            nodes = nodes.append(
                {"id": e.outnode, "node_label": node_label, "lineNumber": lineNum},
                ignore_index=True,
            )

    return nodes, edges


def neighbour_nodes(nodes, edges, nodeids: list, hop: int = 1, intermediate=True):
    """Given nodes, edges, nodeid, return hop neighbours.

    nodes = pd.DataFrame()

    """
    nodes_new = (
        nodes.reset_index(drop=True).reset_index().rename(columns={"index": "adj"})
    )
    id2adj = pd.Series(nodes_new.adj.values, index=nodes_new.id).to_dict()
    adj2id = {v: k for k, v in id2adj.items()}

    arr = []
    for e in zip(edges.innode.map(id2adj), edges.outnode.map(id2adj)):
        arr.append([e[0], e[1]])
        arr.append([e[1], e[0]])

    arr = np.array(arr)
    shape = tuple(arr.max(axis=0)[:2] + 1)
    coo = sparse.coo_matrix((np.ones(len(arr)), (arr[:, 0], arr[:, 1])), shape=shape)

    def nodeid_neighbours_from_csr(nodeid):
        return [
            adj2id[i]
            for i in csr[
                id2adj[nodeid],
            ]
            .toarray()[0]
            .nonzero()[0]
        ]

    neighbours = defaultdict(list)
    if intermediate:
        for h in range(1, hop + 1):
            csr = coo.tocsr()
            csr **= h
            for nodeid in nodeids:
                neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours
    else:
        csr = coo.tocsr()
        csr **= hop
        for nodeid in nodeids:
            neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours


def rdg(edges, gtype):
    """Reduce graph given type."""
    if gtype == "reftype":
        return edges[(edges.etype == "EVAL_TYPE") | (edges.etype == "REF")]
    if gtype == "ast":
        return edges[(edges.etype == "AST")]
    if gtype == "pdg":
        return edges[(edges.etype == "REACHING_DEF") | (edges.etype == "CDG")]
    if gtype == "cfgcdg":
        return edges[(edges.etype == "CFG") | (edges.etype == "CDG")]
    if gtype == "all":
        return edges[
            (edges.etype == "REACHING_DEF")
            | (edges.etype == "CDG")
            | (edges.etype == "AST")
            | (edges.etype == "EVAL_TYPE")
            | (edges.etype == "REF")
            ]


def assign_line_num_to_local(nodes, edges, code):
    """Assign line number to local variable in CPG."""
    label_nodes = nodes[nodes._label == "LOCAL"].id.tolist()
    onehop_labels = neighbour_nodes(nodes, rdg(edges, "ast"), label_nodes, 1, False)
    twohop_labels = neighbour_nodes(nodes, rdg(edges, "reftype"), label_nodes, 2, False)
    node_types = nodes[nodes._label == "TYPE"]
    id2name = pd.Series(node_types.name.values, index=node_types.id).to_dict()
    node_blocks = nodes[
        (nodes._label == "BLOCK") | (nodes._label == "CONTROL_STRUCTURE")
        ]
    blocknode2line = pd.Series(
        node_blocks.lineNumber.values, index=node_blocks.id
    ).to_dict()
    local_vars = dict()
    local_vars_block = dict()
    for k, v in twohop_labels.items():
        types = [i for i in v if i in id2name and i < 1000]
        if len(types) == 0:
            continue
        assert len(types) == 1, "Incorrect Type Assumption."
        block = onehop_labels[k]
        assert len(block) == 1, "Incorrect block Assumption."
        block = block[0]
        local_vars[k] = id2name[types[0]]
        local_vars_block[k] = blocknode2line[block]
    nodes["local_type"] = nodes.id.map(local_vars)
    nodes["local_block"] = nodes.id.map(local_vars_block)
    local_line_map = dict()
    for row in nodes.dropna().itertuples():
        localstr = "".join((row.local_type + row.name).split()) + ";"
        try:
            ln = ["".join(i.split()) for i in code][int(row.local_block):].index(
                localstr
            )
            rel_ln = row.local_block + ln + 1
            local_line_map[row.id] = rel_ln
        except:
            continue
    return local_line_map


def drop_lone_nodes(nodes, edges):
    """Remove nodes with no edge connections.

    Args:
        nodes (pd.DataFrame): columns are id, node_label
        edges (pd.DataFrame): columns are outnode, innode, etype
    """
    nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]
    return nodes


def tokenise(s):
    """Tokenise according to IVDetect.

    Tests:
    s = "FooBar fooBar foo bar_blub23/x~y'z"
    """
    spec_char = re.compile(r"[^a-zA-Z0-9\s]")
    camelcase = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    spec_split = re.split(spec_char, s)
    space_split = " ".join(spec_split).split()

    def camel_case_split(identifier):
        return [i.group(0) for i in re.finditer(camelcase, identifier)]

    camel_split = [i for j in [camel_case_split(i) for i in space_split] for i in j]
    remove_single = [i for i in camel_split if len(i) > 1]
    return " ".join(remove_single)
