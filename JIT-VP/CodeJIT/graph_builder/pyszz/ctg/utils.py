import matplotlib.pyplot as plt
import networkx as nx


def plot_sub_graph(nodes, edges):
    G = nx.Graph()
    ids = set()
    c = 0
    ll = ([], [], [])
    colors = list()
    for node in nodes:
        if node["ALPHA"] == "REMAIN":
            ll[0].append(node["id"])
        elif node["ALPHA"] == "DELETE":
            ll[1].append(node["id"])
        elif node["ALPHA"] == "ADD":
            ll[2].append(node["id"])
        if 'content' not in node:
            if 'code' in node:
                node['content'] = node['code']
            else:
                node['content'] = node['id']
    for e in edges:
        if e[4] == "REMAIN":
            color = 'orange'
            G.add_edge(e[0], e[1], color='orange', weight=c)
            pass
        elif e[4] == "ADD":
            color = 'green'
            G.add_edge(e[0], e[1], color='green', weight=c)
            pass
        elif e[4] == "DELETE":
            color = 'RED'
            G.add_edge(e[0], e[1], color='RED', weight=c)
            pass
        # colors.append(color)
        c += 0.1
        ids.add(e[0])
        ids.add(e[1])
    colors = nx.get_edge_attributes(G, 'color').values()
    values = ["orange" if node in ll[0] else "RED" if node in ll[1]
              else "green" for node in G.nodes()]
    print(len(values))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, font_color="blue", edge_color=colors, node_color=values, font_size=9,
            labels={n["id"]: n["content"] for n in nodes if n["id"] in ids})
    nx.draw_networkx_edge_labels(
        G, pos, font_size=7, edge_labels={(e[0], e[1]): e[2].lower() for e in edges})


def plot_graph(nodes, edges):
    edges = [e for e in edges if e[2] == "AST"]
    plot_sub_graph(nodes, edges)
    plt.show()


def remove_egdes(nodes, edges):
    ids = [n["id"] for n in nodes]
    edges = [e for e in edges if e[0] not in ids and e[1] not in ids]
    return edges


def format_node(nodes, edges):
    i_nodes = list()
    r_nodes = list()
    for node in nodes:
        label = node["_label"]
        if label == "META_DATA":
            # ignore
            node["_label"] = "IGNORE"
            pass
        elif label == "NAMESPACE_BLOCK":
            node["content"] = "NAMESPACE_BLOCK"
            pass
        elif label == "TYPE_DECL":
            node["content"] = "DECL"
            pass
        elif label == "BLOCK":
            node["content"] = "block"
            pass
        elif label == "BINDING":
            node["content"] = node["signature"]
            pass
        elif label == "METHOD_RETURN":
            node["content"] = node["typeFullName"]
            pass
        elif label == "TYPE":
            node["content"] = node["name"]
            pass
        elif label == "FILE":
            node["_label"] = "IGNORE"
            pass
        elif label == "LOCAL":
            node["content"] = node["code"]
            pass
        elif label == "CALL":
            node["content"] = node["name"]
            pass
        elif label == "IDENTIFIER":
            node["content"] = node["name"]
            pass
        elif label == "FIELD_IDENTIFIER":
            if "name" not in node:
                node['content'] = node["canonicalName"]
            else:
                node["content"] = node["name"]
            pass
        elif label == "CONTROL_STRUCTURE":
            node["content"] = "CONTROL_STRUCTURE"
            pass
        elif label == "LITERAL":
            node["content"] = node["code"]
            pass
        elif label == "METHOD":
            node["content"] = "METHOD"
            pass
        elif label == "METHOD_PARAMETER_IN":
            node["content"] = node["typeFullName"]
            pass
        elif label == "METHOD_PARAMETER_OUT":
            node["content"] = node["typeFullName"]
        elif label == "RETURN":
            node["content"] = "RETURN"
        elif label == "NAMESPACE":
            node["content"] = "NAMESPACE"
        elif label == "UNKNOWN":
            if node["code"] == "<empty>" or node["code"] == '':
                node["_label"] = "IGNORE"
            node["content"] = node["code"]
        else:
            node["content"] = node["code"]
        if node["_label"] == "IGNORE":
            i_nodes.append(node)
        else:
            r_nodes.append(node)
            if len(node["content"]) == 0:
                print("no content", node)
    edges = remove_egdes(i_nodes, edges)
    return r_nodes, edges


def format_edge(edges):
    only_edges = ["AST", "CDG", "REACHING_DEF", "CFG"]
    result = list()
    for e in edges:
        if e[2] not in only_edges:
            continue
        if e[2] == "REACHING_DEF":
            e[2] = "DDG"
        result.append(e)
    return result


def get_parent_edges(node_id, parents, edges):
    only_edges = ["AST"]
    result = list()
    for e in edges:
        if e[0] != node_id:
            continue
        if parents and e[1] not in parents:
            continue
        if e[2] not in only_edges:
            continue
        if e[0] < e[1]:
            continue
        result.append(e)
    return result


def get_all_parents(node_id, edges, index=-1, type_edge=None, types_only = None):
    result = list()
    for e in edges:
        if types_only is not None and e[2] not in types_only:
            continue
        if type_edge is not None and e[2] != type_edge:
            continue
        if index == 0 and e[0] != node_id:
            continue
        elif index == 1 and e[1] != node_id:
            continue
        if e[0] == node_id:
            result.append((e[1], 0, e[2]))
        if e[1] == node_id:
            result.append((e[0], 1, e[2]))
    return result


def is_notin(list1, list2):
    for e1 in list1:
        if e1[1] not in list2:
            return True
    return False


def pre_compare(node_a, node_b):
    if len(node_a.keys()) != len(node_b.keys()):
        return False
    if node_a["content"] != node_b["content"]:
        return False
    keys_of_b = [[0, k] for k in node_b.keys()]
    if is_notin(keys_of_b, node_a.keys()):
        return False
    if "name" in node_a and node_a["name"] != node_b["name"]:
        return False
    if "astParentType" in node_a and node_a["astParentType"] != node_b["astParentType"]:
        return False
    return True
