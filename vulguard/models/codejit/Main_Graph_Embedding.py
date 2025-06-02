import argparse
import os
from os import listdir
from os.path import isfile, join
import torch
import pandas
import json
from graph_embedding.relational_graph import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_graph_dir', type=str, help='dir of the node files')
    parser.add_argument('--edge_graph_dir', type=str, help='dir of the edge files')
    parser.add_argument('--embedding_graph_dir', type=str, help='dir to save embedding graph')
    parser.add_argument('--label', type=int, help='label of the commits, 1 if the commits are buggy, 0 otherwise')
    parser.add_argument('--label_file', type=str, help='file contains all commit_id')
    args = parser.parse_args()

    node_graph_dir = args.node_graph_dir
    edge_graph_dir = args.edge_graph_dir
    embedding_graph_dir = args.embedding_graph_dir
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    node_files = [f for f in listdir(node_graph_dir) if isfile(join(node_graph_dir, f))]
    label = int(args.label)
    if label == 1:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VTC")
    else:
        embedding_graph_dir = os.path.join(embedding_graph_dir, "VFC")
    if not os.path.isdir(embedding_graph_dir):
        os.makedirs(embedding_graph_dir)
    
    commit_id_list = []
    with open(args.label_file, "r") as f:
        for line in f:
            commit = json.loads(line)
            commit_id_list.append(commit["commit_id"])
        
    for f in node_files:
        try:
            commit_id = f.split(".")[0].split("_")[-1]
            if commit_id not in commit_id_list:
                continue
            
            node_info = pandas.read_csv(join(node_graph_dir, f))
            edge_info = pandas.read_csv(join(edge_graph_dir, "edge_" + commit_id + ".csv"))
            edge_info = edge_info[edge_info["etype"] != "CFG"]
            data = embed_graph(commit_id, label, node_info,  edge_info)
            torch.save(data, os.path.join(embedding_graph_dir, "data_{}.pt".format(commit_id)))
        except:
            print("exception:" + commit_id)