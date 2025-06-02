import argparse
import os

import pandas

from parse_joern import get_node_edges
from queue import Queue

from Joern_Node import NODE
import multiprocessing as mp


def find_root_node(stmt_edges):
    outnodes = stmt_edges["outnode"].tolist()
    innodes = stmt_edges["innode"].tolist()
    for n in outnodes:
        if n not in innodes:
            return n


def forward_slice_graph(nodes, edges, etype):
    changed_nodes = nodes[nodes["ALPHA"] != "REMAIN"]["id"].tolist()
    nodes_in_changed_edges = edges[(edges["change_operation"] != "REMAIN") & (edges["etype"] == etype)][
        "outnode"].tolist()
    for n in nodes_in_changed_edges:
        if n not in changed_nodes:
            changed_nodes.append(n)
    visited = []
    q = Queue()
    for n in changed_nodes:
        q.put(n)
    while not q.empty():
        n_id = q.get()
        visited.append(n_id)
        neighbors = edges[(edges["outnode"] == n_id) & (edges["etype"] == etype)]["innode"].to_list()
        for n in neighbors:
            if n not in visited:
                q.put(n)
    return visited


def backward_slice_graph(nodes, edges, etype):
    changed_nodes = nodes[nodes["ALPHA"] != "REMAIN"]["id"].tolist()
    nodes_in_changed_edges = edges[(edges["change_operation"] != "REMAIN") & (edges["etype"] == etype)][
        "innode"].tolist()
    for n in nodes_in_changed_edges:
        if n not in changed_nodes:
            changed_nodes.append(n)
    visited = []
    q = Queue()
    for n in changed_nodes:
        q.put(n)
    while not q.empty():
        n_id = q.get()
        visited.append(n_id)
        neighbors = edges[(edges["innode"] == n_id) & (edges["etype"] == etype)]["outnode"].to_list()
        for n in neighbors:
            if n not in visited:
                q.put(n)
    return visited


def aggregate_edges(stmt_nodes, edges):
    node_ids = stmt_nodes["id"].to_list()
    stmt_edges = edges[(edges["innode"].isin(node_ids) & edges["outnode"].isin(node_ids)) & (edges["etype"] == "AST")]
    root = find_root_node(stmt_edges)
    if root is not None:
        for n in node_ids:
            if n != root:
                edges.loc[(edges["innode"] == n) & (edges["etype"] != "AST"), "innode"] = root
                edges.loc[(edges["outnode"] == n) & (edges["etype"] != "AST"), "outnode"] = root
    return edges


def generate_node_content(nodes):
    content = []
    for i, n in nodes.iterrows():
        tmp = NODE(nodes.at[i, "id"], nodes.at[i, "_label"], nodes.at[i, "code"],
                   nodes.at[i, "name"])
        content.append(tmp.print_node())
    nodes["node_content"] = content
    return nodes

#edge_dir + "/edge_{}.csv".format(commit_id)
def trim_CTG(row, idx, separate_token, graph_dir):
    commit_id = row["commit_id"]
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    node_dir = os.path.join(graph_dir, "node")
    edge_dir = os.path.join(graph_dir, "edge")
    if not os.path.isdir(node_dir):
        os.makedirs(node_dir)
    if not os.path.isdir(edge_dir):
        os.makedirs(edge_dir)
    if os.path.exists(edge_dir + "/edge_{}.csv".format(commit_id)):
       	print('already trim' + commit_id)
        return  
    sub_graph_nodes = row["nodes"].split(separate_token)
    sub_graph_edges = row["edges"].split(separate_token)
    commit_nodes = []
    commit_edges = []
    for sub_graph_idx in range(0, len(sub_graph_nodes)):
        node_content = sub_graph_nodes[sub_graph_idx].split("_____")[1]
        edge_content = sub_graph_edges[sub_graph_idx].split("_____")[1]

        nodes, edges = get_node_edges(edge_content, node_content)

        nodes = nodes.dropna(subset=['ALPHA'])
        edges = edges.dropna(subset=['change_operation'])

        removed_nodes = nodes[nodes["lineNumber"] == ""]["id"].tolist()
        nodes = nodes[nodes["lineNumber"] != ""]
        edges = edges[~edges["innode"].isin(removed_nodes)]
        edges = edges[~edges["outnode"].isin(removed_nodes)]

        removed_nodes = nodes[nodes["_label"] == "BLOCK"]["id"].tolist()
        nodes = nodes[nodes["_label"] != "BLOCK"]
        edges = edges[~edges["innode"].isin(removed_nodes)]
        edges = edges[~edges["outnode"].isin(removed_nodes)]

        lineNumbers = nodes.lineNumber.unique()
        kept_nodes = set()

        for x in lineNumbers:
            stmt_nodes = nodes[nodes["lineNumber"] == x]
            edges = aggregate_edges(stmt_nodes, edges)
        edges = edges[edges.innode != edges.outnode]
        edges = edges.drop_duplicates(subset=["innode", "outnode", "etype"], keep='first')

        visited = forward_slice_graph(nodes, edges, "CDG")
        kept_nodes.update(visited)
        visited = backward_slice_graph(nodes, edges, "CDG")
        kept_nodes.update(visited)

        visited = forward_slice_graph(nodes, edges, "DDG")
        kept_nodes.update(visited)
        visited = backward_slice_graph(nodes, edges, "DDG")
        kept_nodes.update(visited)

        kept_nodes = list(kept_nodes)
        kept_line_numbers = nodes[nodes["id"].isin(kept_nodes)]["lineNumber"].to_list()
        nodes = nodes[nodes["lineNumber"].isin(kept_line_numbers)]
        kept_nodes = nodes["id"].to_list()
        edges = edges[(edges["innode"].isin(kept_nodes)) | (edges["outnode"].isin(kept_nodes))]

        nodes = generate_node_content(nodes)

        node_list = nodes["id"].tolist()

        edges = edges[edges["outnode"].isin(node_list)]
        edges = edges[edges["innode"].isin(node_list)]

        node_ids = nodes["id"]
        node_ids = [str(sub_graph_idx) + "_" + str(s) for s in node_ids]
        nodes["id"] = node_ids

        node_linenumbers = nodes["lineNumber"]
        node_linenumbers = [sub_graph_nodes[sub_graph_idx].split("_____")[0] + "_____" + str(s) for s in
                            node_linenumbers]
        nodes["lineNumber"] = node_linenumbers

        in_nodes = edges["innode"]
        in_nodes = [str(sub_graph_idx) + "_" + str(s) for s in in_nodes]
        edges["innode"] = in_nodes

        out_nodes = edges["outnode"]
        out_nodes = [str(sub_graph_idx) + "_" + str(s) for s in out_nodes]
        edges["outnode"] = out_nodes
        commit_nodes.append(nodes)
        commit_edges.append(edges)

    all_nodes = pandas.concat(commit_nodes)
    all_edges = pandas.concat(commit_edges)

    all_nodes.to_csv(node_dir + "/node_{}.csv".format(commit_id))
    all_edges.to_csv(edge_dir + "/edge_{}.csv".format(commit_id))

pool = mp.Pool(mp.cpu_count())
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path of the dataset file', default=".")
    parser.add_argument('--graph', type=str, help='dir to save graph after trimming', default="OUTPUT")
    args = parser.parse_args()

    commit_raw_file_path = args.data
    print(commit_raw_file_path)
    df = pandas.read_csv(commit_raw_file_path)
    separate_token = "=" * 100
    graph_dir = args.graph
    for idx, row in df.iterrows():
        commit_id = row["commit_id"]
        try:
            print("trimming graph:" + commit_id)
            #trim_CTG(df, idx, separate_token, graph_dir)
            pool.apply_async(trim_CTG, args=(row, idx, separate_token, graph_dir))

        except:
            print("exception:", commit_id)
    pool.close()
    pool.join()
