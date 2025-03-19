import json
from collections import defaultdict

from pyszz.diff.parse_diff import parse_diff

from config import JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR
from file_manager import is_path_exist, join_path
from helpers import get_logger
from pyszz.ctg.utils import (format_node, get_parent_edges, format_edge,
                             plot_graph, pre_compare)

logger = get_logger(__name__)

BASE_PATH = JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR


def read_json_file(path):
    with open(path) as f:
        return json.load(f)


def get_cpg_before_after(commit, idx):

    base_path = join_path(BASE_PATH, commit)

    a_nodes_path = join_path(base_path, f"after.{idx}.cpp.nodes.json")
    a_edges_path = join_path(base_path, f"after.{idx}.cpp.edges.json")
    
    b_nodes_path = join_path(base_path, f"before.{idx}.cpp.nodes.json")
    b_edges_path = join_path(base_path, f"before.{idx}.cpp.edges.json")
    if not is_path_exist(a_edges_path):
        return None
    a_nodes = read_json_file(a_nodes_path)
    a_edges = read_json_file(a_edges_path)
    a_nodes, a_edges = format_node(a_nodes, a_edges)
    a_edges = format_edge(a_edges)
    
    b_nodes = read_json_file(b_nodes_path)
    b_edges = read_json_file(b_edges_path)
    b_nodes, b_edges = format_node(b_nodes, b_edges)
    b_edges = format_edge(b_edges)
    return a_nodes, a_edges, b_nodes, b_edges


def draw_cpg_orgi(nodes, edges, label="REMAIN"):
    edges = [e for e in edges if e[2] == "AST"]
    for node in nodes:
        node["ALPHA"] = label
    for e in edges:
        e.append(label)
    plot_graph(nodes, edges)


def pp(*args):
    print("".join(args))

def hash_edge(e):
    return hash("__".join([str(e0) for e0 in e[:3]]))

def generate_ctg(commit, idx):
    base_path = join_path(BASE_PATH, commit)
    code_path = join_path(base_path, f"after.{idx}.cpp.nodes.json")
    if not is_path_exist(code_path):
        logger.error(f"No such file : {commit}, {idx}")
        return
    ctg_nodes_path = join_path(base_path, f"ctg.{idx}.cpp.nodes.json")
    ctg_edges_path = join_path(base_path, f"ctg.{idx}.cpp.edges.json")
    # if is_path_exist(ctg_edges_path):
    #     print(f"Generated CTG {commit}, {idx}")
    #     return
    a_n, a_e, b_n, b_e = get_cpg_before_after(commit, idx)
    if len(a_n) > 20000 or len(a_n) < 20 or len(a_e) > 100000:
        print(f"IGNORE: {commit}")
        return
    ctg_nodes = list()
    ctg_edges = list()
    map_b_a = defaultdict()
    map_a_b = defaultdict()
    diff_elements, map_lines_d_a = parse_diff(commit, idx)

    dict_a_to_b = dict()
    dict_b_to_a = dict()
    for el in diff_elements["map_a_to_d"]:
        dict_a_to_b[el[0]] = el[1]
    for el in diff_elements["map_d_to_a"]:
        dict_b_to_a[el[0]] = el[1]
    line_added = set()
    line_deleted = set()
    if diff_elements is not None:
        line_added = set([e["line"] for e in diff_elements["add"]])
        line_deleted = set([e["line"] for e in diff_elements["del"]])
    MAX_LEN = 20
    print("line added  :", line_added)
    print("line deleted: ", line_deleted)

    for n in a_n:
        if "lineNumber" in n and n["lineNumber"] > MAX_LEN:
            MAX_LEN = n["lineNumber"]
    for n in b_n:
        if "lineNumber" in n and n["lineNumber"] > MAX_LEN:
            MAX_LEN = n["lineNumber"]
    MAX_LEN += 2

    line_remain_a = [1 if e in line_added else 0 for e in range(MAX_LEN)]
    line_remain_b = [1 if e in line_deleted else 0 for e in range(MAX_LEN)]
    for i in range(1, MAX_LEN):
        line_remain_a[i] = line_remain_a[i-1] + line_remain_a[i]
        line_remain_b[i] = line_remain_b[i-1] + line_remain_b[i]
    for k, v in map_lines_d_a.items():
        if k >= MAX_LEN - 1 or v >= MAX_LEN - 1:
            continue
        if k - line_remain_b[k] != v - line_remain_a[v]:
            tmp_value = (k - line_remain_b[k]) - (v - line_remain_a[v])
            if tmp_value > 0:
                for i_tmp in range(k, MAX_LEN):
                    line_remain_b[i_tmp] += tmp_value
            else:
                tmp_value = - (k - line_remain_b[k]) + (v - line_remain_a[v])
                for i_tmp in range(v, MAX_LEN):
                    line_remain_a[i_tmp] += tmp_value
    a_nodes_unchange = [
        n for n in a_n if "lineNumber" in n and n["lineNumber"] not in line_added]
    b_nodes_unchange = [
        n for n in b_n if "lineNumber" in n and n["lineNumber"] not in line_deleted]
    a_nodes_unchange_with_columnNumber = list()
    a_nodes_unchange_without_columnNumber = list()
    for n in a_nodes_unchange:
        if "columnNumber" in n:
            a_nodes_unchange_with_columnNumber.append(n)
        else:
            a_nodes_unchange_without_columnNumber.append(n)
    for n_befor in b_nodes_unchange:
        line_before = n_befor["lineNumber"]
        line_before = line_before - line_remain_b[line_before]
        if "columnNumber" not in n_befor:
            for n_after in a_nodes_unchange_without_columnNumber:
                if not pre_compare(n_after, n_befor):
                    continue
                line_after = n_after["lineNumber"]
                line_after = line_after - line_remain_a[line_after-1]
                if line_before == line_after and n_befor["_label"] == n_after["_label"]:
                    map_a_b[n_after["id"]] = n_befor["id"]
                    map_b_a[n_befor["id"]] = n_after["id"]
                    ctg_node = dict(n_befor)
                    ctg_node["ALPHA"] = "REMAIN"
                    ctg_nodes.append(ctg_node)
                    a_nodes_unchange_without_columnNumber.remove(n_after)
                    break
            continue
        for n_after in a_nodes_unchange_with_columnNumber:
            if not pre_compare(n_after, n_befor):
                continue
            line_after = n_after["lineNumber"]
            line_after = line_after - line_remain_a[line_after]
            if n_befor["columnNumber"] == n_after["columnNumber"] and \
                    line_before == line_after and \
                    n_after["_label"] == n_befor["_label"]:
                map_a_b[n_after["id"]] = n_befor["id"]
                map_b_a[n_befor["id"]] = n_after["id"]
                ctg_node = dict(n_befor)
                ctg_node["ALPHA"] = "REMAIN"
                ctg_nodes.append(ctg_node)
                a_nodes_unchange_with_columnNumber.remove(n_after)
                break
    b_ids_unchange = map_b_a.keys()
    a_ids_unchange = map_a_b.keys()
    print("parsed :", len(ctg_nodes))
    a_nodes_change = [n for n in a_n if n["id"] not in a_ids_unchange]
    b_nodes_change = [n for n in b_n if n["id"] not in b_ids_unchange]
    # b_edges_change = [e for e in b_e if e[0]
    #                   not in b_ids_unchange or e[1] not in b_ids_unchange]
    # a_edges_change = [e for e in a_e if e[0]
    #                   not in a_ids_unchange or e[1] not in a_ids_unchange]

    cluster_node_a = defaultdict(lambda: list())
    for node in a_nodes_change:
        cluster_node_a[node["_label"]].append(node)
    b_edges_unchange = [e for e in b_e if e[0]
                        in b_ids_unchange and e[1] in b_ids_unchange]
    a_edges_unchange = [e for e in a_e if e[0]
                        in a_ids_unchange and e[1] in a_ids_unchange]
    for node_b in b_nodes_change:
        label = node_b["_label"]
        
        nodes_a = cluster_node_a[label]
        b_edges = get_parent_edges(node_b["id"], map_b_a.keys(), b_e)
        for node_a in nodes_a:
            
            if not pre_compare(node_a, node_b):
                continue
            a_edges = get_parent_edges(node_a["id"], map_a_b.keys(), a_e)
            if len(b_edges) != len(a_edges):
                continue

            b_ids_parents = set([e[1] for e in b_edges])
            a_ids_parents = set([map_a_b[e[1]] for e in a_edges])
            if len(b_ids_parents) != len(a_ids_parents):
                continue
            for id_pr in b_ids_parents:
                a_ids_parents.discard(id_pr)
            if len(a_ids_parents) > 0:
                continue
            if node_a["id"] in map_a_b.keys():
                continue
            if node_b["id"] in map_b_a.keys():
                continue
            map_a_b[node_a["id"]] = node_b["id"]
            map_b_a[node_b["id"]] = node_a["id"]

            ctg_node = dict(node_b)
            ctg_node["ALPHA"] = "REMAIN"
            ctg_nodes.append(ctg_node)
            cluster_node_a[label].remove(node_a)
            break

    a_ids_unchange = list(map_a_b.keys())
    b_ids_unchange = list(map_b_a.keys())
    b_edges_unchange = [e for e in b_e if e[0]
                        in b_ids_unchange and e[1] in b_ids_unchange]
    a_edges_unchange = [e for e in a_e if e[0]
                        in a_ids_unchange and e[1] in a_ids_unchange]
    a_edges_unchange_n = list()
    for e in a_edges_unchange:
        a_e_uc = list(e)
        a_e_uc[0] = map_a_b[e[0]]
        a_e_uc[1] = map_a_b[e[1]]
        a_edges_unchange_n.append(a_e_uc)
    a_edges_unchange = a_edges_unchange_n
    
    hash_b_edges_uc = [hash_edge(e) for e in b_edges_unchange]
    hash_a_edges_uc = [hash_edge(e) for e in a_edges_unchange]
    for e in b_edges_unchange:
        if hash_edge(e) in hash_a_edges_uc:
            ctg_e = list(e)
            ctg_e.append("REMAIN")
            ctg_edges.append(ctg_e)
        else:
            ctg_e = list(e)
            ctg_e.append("DELETE")
            ctg_edges.append(ctg_e)

    for e in a_edges_unchange:
        if hash_edge(e) not in hash_b_edges_uc:
            ctg_e = list(e)
            ctg_e.append("ADD")
            ctg_edges.append(ctg_e)
    # //end
    a_nodes_change = [n for n in a_n if n["id"] not in a_ids_unchange]
    b_nodes_change = [n for n in b_n if n["id"] not in b_ids_unchange]
    a_ids_change = [n["id"] for n in a_nodes_change]
    b_ids_change = [n["id"] for n in b_nodes_change]
    assert len(a_ids_change) + len(a_ids_unchange) == len(a_n)
    assert len(b_ids_change) + len(b_ids_unchange) == len(b_n)
    b_edges_change = [e for e in b_e if e[0]
                      in b_ids_change or e[1] in b_ids_change]
    a_edges_change_f = [e for e in a_e if e[0]
                      in a_ids_change or e[1] in a_ids_change]
    for n in b_nodes_change:
        ctg_n = dict(n)
        ctg_n["ALPHA"] = "DELETE"
        ctg_nodes.append(ctg_n)
    len_b_nodes = len(b_n)
    for n in a_nodes_change:
        ctg_n = dict(n)
        ctg_n["ALPHA"] = "ADD"
        map_a_b[ctg_n["id"]] = ctg_n["id"]+len_b_nodes
        map_b_a[ctg_n["id"] + len_b_nodes] = ctg_n["id"]
        ctg_n["id"] = ctg_n["id"] + len_b_nodes
        ctg_nodes.append(ctg_n)
    
    for e_a in a_edges_change_f:
        ctg_e = list(e_a)
        ctg_e.append("ADD")
        if ctg_e[0] in map_a_b.keys():
            ctg_e[0] = map_a_b[ctg_e[0]]
        else:
            print("Error: edge", e)
            continue

        if ctg_e[1] in map_a_b.keys():
            ctg_e[1] = map_a_b[ctg_e[1]]
        else:
            print("Error: edge", e)
            continue
        ctg_edges.append(ctg_e)

    for e in b_edges_change:
        ctg_e = list(e)
        ctg_e.append("DELETE")
        ctg_edges.append(ctg_e)

    with open(ctg_nodes_path, "w+") as f:
        json.dump(ctg_nodes, f)
    with open(ctg_edges_path, "w+") as f:
        json.dump(ctg_edges, f)
    
    # draw_cpg_orgi(a_n,a_e,"ADD")
    pp(f"DEBUG FOR CTG {commit} : {idx}")
    pp(f"len ctg nodes :  {len(ctg_nodes)} , edges : {len(ctg_edges)}")
    pp(f"len after nodes :  {len(a_n)} , edges : {len(a_e)}")
    pp(f"len before nodes :  {len(b_n)} , edges : {len(b_e)}")
    pp("="*33)
    
    # # check node
    # ctg_ids = set([n["id"] for n in ctg_nodes])
    # for e in ctg_edges:
    #     if e[0] not in ctg_ids:
    #         raise Exception("not have node for edge: ", e, e[0])
    #     if e[1] not in ctg_ids:
    #         raise Exception("not have node for edge: ", e, e[1])
    # import os
    # os.system(
    #     f"google-chrome data/joern_parsed_output/functions/{commit}/diff.{idx}.html")
    # plot_graph(ctg_nodes, ctg_edges)
    return ctg_nodes_path, ctg_edges_path
