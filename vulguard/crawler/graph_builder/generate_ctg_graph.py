import json
import shutil
import threading

import pandas as pd

from config import JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR
from file_manager import is_path_exist, join_path, mkdir_if_not_exist, unlink
from pyszz.ctg.generate_ctg import generate_ctg
from pyszz.ctg.utils import *

# CSV_PATH = "data/extracted_output/project/FFmpeg_function_vtc.csv"
#CSV_PATH = "data/jit_vul_dataset/vul_triggering_commit_data_function_level.csv"
#CSV_PATH = "data/jit_vul_dataset/vtc_old.csv"
NUMBER_THREAD = 64

def print_progressbar(percent):
    print("[%-100s] %d%%" % ('='*percent, percent))


SPLIT = 1
PART = 0
TOKEN_SPLIT = "="*100
REMOVE_ERR_FILE = True

threads = list()


def gen_ctg(cm_id, idx):
    t1 = threading.Thread(target=generate_ctg,
                          args=(cm_id, idx))
    t1.start()
    threads.append(t1)
    if len(threads) > NUMBER_THREAD:
        for thread in threads:
            thread.join()
        threads.clear()


def main():    
    datax = pd.read_csv(CSV_PATH, low_memory=False)
    print(datax.shape)
    code_before = 'function_before'
    code_after = 'function_after'
    data = datax[datax[code_before].notna()]
    data = data[data[code_after].notna()]
    print(data.shape)
    commits = set(datax["commit_id"])
    size = len(commits)
    threads = list()
    count = 0
    percent = 0
    print(size)
    for cm_id in list(commits)[::1]:
        count += 1
        if int(100 * count/size) > percent:
            percent += 1
            print_progressbar(percent)
        rows = data.loc[data["commit_id"] == cm_id].reset_index()
        rows2 = datax.loc[datax["commit_id"] == cm_id].reset_index()
        c = 0
        pad = rows.shape[0]
        for idx, row in rows.iterrows():
            t1 = threading.Thread(target=generate_ctg,
                                  args=(row["commit_id"], idx))
            t1.start()
            threads.append(t1)
            if len(threads) > NUMBER_THREAD:
                for thread in threads:
                    thread.join()
                threads.clear()
        for idx, row in rows2.iterrows():
            before = row[code_before]
            after = row[code_after]
            if isinstance(before, str) and isinstance(after, str):
                continue
            idx = pad + c
            c += 1
            t1 = threading.Thread(target=generate_ctg,
                                  args=(row["commit_id"], idx))
            t1.start()
            threads.append(t1)
            if len(threads) > NUMBER_THREAD:
                for thread in threads:
                    thread.join()
                threads.clear()

    # free
    for thread in threads:
        thread.join()


def move_file(cm, idx, dst):
    dst = join_path(dst, cm)
    mkdir_if_not_exist(dst)
    base_path = join_path(JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR, cm)
    edges_path = join_path(base_path, f"ctg.{idx}.cpp.edges.json")
    nodes_path = join_path(base_path, f"ctg.{idx}.cpp.nodes.json")
    if not is_path_exist(join_path(dst, f"ctg.{idx}.cpp.edges.json")) and is_path_exist(edges_path):
        shutil.copy(edges_path, dst)
    if not is_path_exist(join_path(dst, f"ctg.{idx}.cpp.nodes.json")) and is_path_exist(nodes_path):
        shutil.copy(nodes_path, dst)
    pass


OUT_PUT = "functions"


def extract_ctg():
    mkdir_if_not_exist(OUT_PUT)
    data = pd.read_csv(CSV_PATH, low_memory=False)
    print(data.shape)
    data = data[data['before'].notna()]
    data = data[data['after'].notna()]
    commits = set(data["commit_id"])
    size = len(commits)
    threads = list()

    count = 0
    percent = 0
    for cm_id in list(commits)[::1]:
        count += 1
        if int(100 * count/size) > percent:
            percent += 1
            print_progressbar(percent)
        rows = data.loc[data["commit_id"] == cm_id].reset_index()
        for idx, row in rows.iterrows():
            t1 = threading.Thread(target=move_file,
                                  args=(cm_id, idx, OUT_PUT))
            t1.start()
            threads.append(t1)
            if len(threads) > NUMBER_THREAD:
                for thread in threads:
                    thread.join()
                threads.clear()
    for thread in threads:
        thread.join()


def unlink_cpg(cm_id, idx):
    base_path = join_path(JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR, cm_id)
    types = ["ctg", "before", "after"]
    types_file = ["nodes", "edges"]
    for t in types:
        for tf in types_file:
            path_f = join_path(base_path, f"{t}.{idx}.cpp.{tf}.json")
            if is_path_exist(path_f):
                unlink(path_f)


def ctg_to_csv(file):
    datax = pd.read_csv(CSV_PATH, low_memory=False)
    print(datax.shape)
    code_before = 'function_before'
    code_after = 'function_after'
    data = datax[datax[code_before].notna()]
    data = data[data[code_after].notna()]
    print(data.shape)
    commits = set(datax["commit_id"])
    size = len(commits)
    print(size)
    result = list()
    count = 0
    percent = 0
    for cm_id in list(commits)[::1]:
        count += 1
        if int(100 * count/size) > percent:
            percent += 1
            print_progressbar(percent)
        rows = data.loc[data["commit_id"] == cm_id].reset_index()
        rows2 = datax.loc[datax["commit_id"] == cm_id].reset_index()
        c = 0
        pad = rows.shape[0]
        base_path = join_path(JOERN_PARSED_FUNCTIONS_OUTPUTS_DIR, cm_id)
        tmp_n_list = list()
        tmp_e_list = list()
        tmp_bl_list = list()
        bl_path = ""
        for idx, row in rows.iterrows():
            edges_path = join_path(base_path, f"ctg.{idx}.cpp.edges.json")
            nodes_path = join_path(base_path, f"ctg.{idx}.cpp.nodes.json")
            if not is_path_exist(edges_path) or not is_path_exist(nodes_path):
                continue
            with open(nodes_path) as f:
                nodes = json.load(f)
            with open(edges_path) as f:
                edges = json.load(f)

            e_change = [e for e in edges if e[4] == "ADD" or e[4] == "DELETE"]
            if len(e_change) <= 0 or (len(nodes) < 20 or len(edges) < 20) and len(nodes) > 1:
                if not REMOVE_ERR_FILE:
                    continue
                unlink(edges_path)
                unlink(nodes_path)
                cpg_e_path = join_path(
                    base_path, f"before.{idx}.cpp.edges.json")
                cpg_n_path = join_path(
                    base_path, f"before.{idx}.cpp.nodes.json")
                unlink(cpg_e_path)
                unlink(cpg_n_path)
                cpg_n_path = join_path(
                    base_path, f"after.{idx}.cpp.edges.json")
                cpg_e_path = join_path(
                    base_path, f"after.{idx}.cpp.nodes.json")
                unlink(cpg_e_path)
                unlink(cpg_n_path)
                continue
            bl_path = join_path(base_path, f"blame.{idx}.txt")
            if is_path_exist(bl_path):
                with open(bl_path) as ff:
                    tmp_bl_list.append(f"{idx}_____"+ff.read().strip())
            tmp_n_list.append(f"{idx}_____"+json.dumps(nodes))
            tmp_e_list.append(f"{idx}_____"+json.dumps(edges))

        for idx, row in rows2.iterrows():
            before = row[code_before]
            after = row[code_after]
            if isinstance(before, str) and isinstance(after, str):
                continue
            idx = pad + c
            c += 1
            edges_path = join_path(base_path, f"ctg.{idx}.cpp.edges.json")
            nodes_path = join_path(base_path, f"ctg.{idx}.cpp.nodes.json")
            if not is_path_exist(edges_path) or not is_path_exist(nodes_path):
                continue
            with open(nodes_path) as f:
                nodes = json.load(f)
            with open(edges_path) as f:
                edges = json.load(f)
            e_change = [e for e in edges if e[4] == "ADD" or e[4] == "DELETE"]
            if len(e_change) <= 0 or (len(nodes) < 20 or len(edges) < 20) and len(nodes) > 1:
                continue
            if is_path_exist(bl_path):
                with open(bl_path) as ff:
                    tmp_bl_list.append(f"{idx}_____"+ff.read().strip())
            tmp_n_list.append(f"{idx}_____"+json.dumps(nodes))
            tmp_e_list.append(f"{idx}_____"+json.dumps(edges))

        if len(tmp_e_list) > 0 and len(tmp_n_list) > 0:
            result.append([cm_id, TOKEN_SPLIT.join(tmp_n_list),
                           TOKEN_SPLIT.join(tmp_e_list), TOKEN_SPLIT.join(tmp_bl_list)])
    df_s = pd.DataFrame(
        result, columns=["commit_id", "nodes", "edges", "vul_lines"])
    df_s.to_csv(file, index=False)


def slice(nodes, edges):
    e_a = [e for e in edges if e[4] == "ADD"]
    e_d = [e for e in edges if e[4] == "DELETE"]
    n_c = [n["id"]
           for n in nodes if n["ALPHA"] == "ADD" or n['ALPHA'] == "DELETE"]
    change_node_id = list()
    for n in n_c:
        tmp = [n, -1, 'CDG']
        tmp2 = [n, -1, 'DDG']
        tmp3 = [n, 0, 'AST']
        if tmp not in change_node_id:
            change_node_id.append(tmp)
        if tmp2 not in change_node_id:
            change_node_id.append(tmp2)
        if tmp3 not in change_node_id:
            change_node_id.append(tmp3)
    for e_c in e_a + e_d:
        tmp = [e_c[0], 1, e_c[2]]
        tmp2 = [e_c[1], 0, e_c[2]]
        if tmp not in change_node_id and e_c[2] != 'AST':
            change_node_id.append(tmp)
        if tmp2 not in change_node_id:
            change_node_id.append(tmp2)
    sliced_nodes_id = set()
    while len(change_node_id) > 0:
        node_id = change_node_id.pop(0)
        parents = get_all_parents(
            node_id[0], edges, node_id[1], node_id[2], ["AST", "DDG", "CDG"])
        for p_id in parents:
            if p_id not in sliced_nodes_id:
                change_node_id.append(p_id)
                sliced_nodes_id.add(p_id)
    sliced_nodes_id = set([el[0] for el in sliced_nodes_id])
    n_nodes = [n for n in nodes if n["id"] in sliced_nodes_id]
    n_edges = [e for e in edges if e[0]
               in sliced_nodes_id and e[1] in sliced_nodes_id]

    print(
        f"After slice remain: {len(n_nodes)/len(nodes)*100}% nodes, {len(n_edges)/len(edges)*100}% edges")
    print("-"*33)
    if len(n_nodes) == 0 or len(n_edges) == 0:
        return nodes, edges
    return n_nodes, n_edges


def slice_func(row, result):
    print(row["commit_id"])
    nodes = row["nodes"].split(TOKEN_SPLIT)
    edges = row["edges"].split(TOKEN_SPLIT)
    sliced_nodes = list()
    sliced_edges = list()
    assert len(nodes) == len(edges)
    for index, node in enumerate(nodes):
        edge_element = edges[index].split('_____')
        node_element = node.split('_____')
        edge = json.loads(edge_element[1])
        node = json.loads(node_element[1])
        n, e = slice(node, edge)
        ctg_id = edge_element[0]
        sliced_edges.append(f"{ctg_id}_____"+json.dumps(e))
        sliced_nodes.append(f"{ctg_id}_____"+json.dumps(n))
    result.append([row["commit_id"], TOKEN_SPLIT.join(sliced_nodes),
                  TOKEN_SPLIT.join(sliced_edges), row["vul_lines"]])


def slice_ctg(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    result = list()
    print("---")
    threads = list()
    for idx, row in df.iterrows():
        t1 = threading.Thread(target=slice_func,
                              args=(row, result))
        t1.start()
        threads.append(t1)
        if len(threads) > NUMBER_THREAD:
            for thread in threads:
                thread.join()
            threads.clear()
    for thread in threads:
        thread.join()
    df_s = pd.DataFrame(
        result, columns=["commit_id", "nodes", "edges", "vul_lines"])
    df_s.to_csv(f'{csv_path[:-4]}_sliced.csv', index=False)


if __name__ == "__main__":
    # read file csv
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--file", type=str)
    parser.add_argument("--project", type=str)
    p = parser.parse_args()
    
    CSV_PATH = f"data/extracted_output/project/{p.project}_function_{p.file}.csv"
    
    main()
    #slice_ctg('ctg_data/test.csv')
    # extract_ctg()
    ctg_to_csv(f"{p.file}_ctg.csv")
