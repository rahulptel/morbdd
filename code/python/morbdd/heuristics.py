import networkx as nx
import time
import numpy as np
# import gurobipy as gp
from pathlib import Path
import pandas as pd


# gp.setParam("Threads", 1)
# # Focus on feasibility
# gp.setParam("MIPFocus", 1)
# gp.setParam("TimeLimit", 300)


def get_node_resistance(pred_score, threshold=0.5, round_upto=1):
    return 0 \
        if np.round(pred_score, round_upto) >= threshold \
        else threshold - pred_score


def switch_on_node(node, threshold=0.5):
    node["prev_pred"] = float(node["pred"])
    node["pred"] = float(threshold + 0.001)
    return node


def get_active_layers(bdd, lidx, lookahead):
    layers = []
    for i in range(lidx - 1, lidx + lookahead):
        if i == -1:
            layers.append([{"pred": 1, "conn": 1}])
        else:
            layers.append(bdd[i])

    return layers


def initialize_paths(parent_layer, threshold=0.5, round_upto=1):
    return [[node_idx]
            for node_idx, node in enumerate(parent_layer)
            if np.round(node["pred"], round_upto) >= threshold and "conn" in node]


def extend_paths(layer, partial_paths):
    new_partial_paths = []
    for node_idx, node in enumerate(layer):
        for parent_idx in node["op"]:
            for path in partial_paths:
                if path[-1] == parent_idx:
                    new_path = path[:]
                    new_path.append(node_idx)
                    new_partial_paths.append(new_path)

        for parent_idx in node["zp"]:
            for path in partial_paths:
                if path[-1] == parent_idx:
                    new_path = path[:]
                    new_path.append(node_idx)
                    new_partial_paths.append(new_path)

    return new_partial_paths


def get_paths_with_min_resistance(resistances, paths):
    # Sort paths based on resistance and select the one offering minimum resistance
    resistances, paths = zip(*sorted(zip(resistances, paths), key=lambda x: x[0]))
    k = 1
    for r in resistances[1:]:
        if r > resistances[0]:
            break
        else:
            k += 1

    return paths[:k]


def calculate_path_resistance(path, layers, threshold=0.5, round_upto=1):
    resistance = 0
    for node_idx, layer in zip(path[1:], layers[1:]):
        pred_score = layer[node_idx]["pred"]
        resistance += get_node_resistance(pred_score,
                                          threshold=threshold,
                                          round_upto=round_upto)

    return resistance


def generate_resistance_graph(bdd, threshold, round_upto):
    g = nx.DiGraph()
    root, terminal = "0-0", f"{len(bdd) + 1}-0"

    edges = []
    # From root to penultimate layer
    for lidx, layer in enumerate(bdd):
        for nidx, node in enumerate(layer):
            parent_pre = f"{lidx}"
            node_name = f"{lidx + 1}-{nidx}"
            resistance = get_node_resistance(node["pred"],
                                             threshold=threshold,
                                             round_upto=round_upto)

            for op in node['op']:
                edges.append((f"{parent_pre}-{op}", node_name, resistance))

            for zp in node['zp']:
                edges.append((f"{parent_pre}-{zp}", node_name, resistance))

    # From penultimate layer to terminal node
    for nidx, _ in enumerate(bdd[-1]):
        edges.append((f"{len(bdd)}-{nidx}", terminal, 0))

    g.add_weighted_edges_from(edges)

    return g, root, terminal


def switch_on_nodes_in_shortest_path(path, bdd, threshold, round_upto):
    for node_name in enumerate(path):
        lidx, nidx = list(map(int, node_name.split("-")))
        if 0 < lidx < len(bdd) + 1:
            lidx -= 1
            node = bdd[lidx][nidx]
            if np.round(node["pred"], round_upto) < threshold:
                bdd[lidx][nidx] = switch_on_node(node, threshold)
                bdd[lidx][nidx]["conn"] = True
    return bdd


def generate_min_resistance_mip_model(bdd, threshold=0.5, round_upto=1, profile=None):
    node_vars, outgoing_arcs = [], {}

    m = gp.Model("Min-resistance Graph")
    # From root to penultimate layer
    for lidx, layer in enumerate(bdd):
        layer_node_vars = []
        for nidx, node in enumerate(layer):
            node_name = f"{lidx}-{nidx}"
            resistance = get_node_resistance(node["pred"],
                                             threshold=threshold,
                                             round_upto=round_upto)
            # Select nodes above threshold
            lb = 1 if resistance == 0 else 0
            node_var = m.addVar(vtype="B", name=node_name, obj=resistance, lb=lb)

            # node_vars.append(node_var)
            layer_node_vars.append(node_var)

            if lidx > 0:
                parent_prefix = f"{lidx - 1}"
                incoming_arcs = []

                for op in node['op']:
                    parent_node_name = f"{parent_prefix}-{op}"
                    arc_name = f"{parent_node_name}-{node_name}-1"
                    arc = m.addVar(vtype="B", name=arc_name, obj=0)
                    incoming_arcs.append(arc)

                    if parent_node_name not in outgoing_arcs:
                        outgoing_arcs[parent_node_name] = []
                    outgoing_arcs[parent_node_name].append(arc)

                for zp in node['zp']:
                    parent_node_name = f"{parent_prefix}-{zp}"
                    arc_name = f"{parent_node_name}-{node_name}-1"
                    arc = m.addVar(vtype="B", name=arc_name, obj=0)
                    incoming_arcs.append(arc)

                    if parent_node_name not in outgoing_arcs:
                        outgoing_arcs[parent_node_name] = []

                    outgoing_arcs[parent_node_name].append(arc)

                # Select node if at least one incoming arc is selected
                m.addConstr(gp.quicksum(incoming_arcs) <= len(incoming_arcs) * node_var)
                # Don't select node if none of the incoming arcs are selected
                m.addConstr(node_var <= gp.quicksum(incoming_arcs))

        node_vars.append(layer_node_vars)
        if profile is None:
            # For the first (root + 1) and the last (terminal - 1) layers, select at least one node
            if lidx == 0 or lidx == len(bdd) - 1:
                m.addConstr(gp.quicksum(layer_node_vars) >= 1)
        else:
            m.addConstr(gp.quicksum(layer_node_vars) >= np.ceil(profile[lidx] * len(bdd[lidx])))

        # Select at least one outgoing arc if a node is selected
        if lidx > 0:
            for nidx, node_var in enumerate(node_vars[lidx - 1]):
                parent_node_name = f"{lidx - 1}-{nidx}"
                if parent_node_name not in outgoing_arcs:
                    print("Parent node not found!")
                else:
                    m.addConstr(gp.quicksum(outgoing_arcs[parent_node_name]) >= node_var)

    m._node_vars = node_vars
    return m


def get_mip_solution(mip):
    node_vars = mip._node_vars
    solution = []
    for layer in node_vars:
        _sol = []
        for node in layer:
            _sol.append(np.round(node.x + 0.5))
        solution.append(_sol)

    return solution


def switch_on_nodes_in_mip_solution(bdd, sol, threshold=0.5, round_upto=1):
    for bdd_layer, sol_layer in zip(bdd, sol):
        for bdd_node, is_selected in zip(bdd_layer, sol_layer):
            if is_selected and np.round(bdd_node["pred"], round_upto) < threshold:
                bdd_node = switch_on_node(bdd_node, threshold=0.5)
                bdd_node["conn"] = True

    return bdd


def run_select_all(bdd, lidx, threshold=0.5, round_upto=1):
    time_stitching = time.time()
    for nidx, node in enumerate(bdd[lidx]):
        node["conn"] = True
        if np.round(node["pred"], round_upto) < threshold:
            bdd[lidx][nidx] = switch_on_node(node, threshold)

    time_stitching = time.time() - time_stitching

    return bdd, time_stitching


def run_lookahead(bdd, lidx, lookahead, threshold=0.5, round_upto=1):
    time_stitching = time.time()
    layers = get_active_layers(bdd, lidx, lookahead)
    paths = initialize_paths(layers[0],
                             threshold=threshold,
                             round_upto=round_upto)
    for layer in layers[1:]:
        paths = extend_paths(layer, paths)

    resistances = [calculate_path_resistance(path, layers,
                                             threshold=threshold,
                                             round_upto=round_upto)
                   for path in paths]
    min_resistance_paths = get_paths_with_min_resistance(resistances, paths)
    # Switch on the nodes in the minimum resistance paths
    for path in min_resistance_paths:
        for node_idx, layer in zip(path[1:], layers[1:]):
            node = layer[node_idx]
            node["conn"] = True
            node = switch_on_node(node, threshold)

    time_stitching = time.time() - time_stitching

    return bdd, time_stitching


def run_shortest_path(bdd, threshold=0.5, round_upto=1):
    time_stitching = time.time()

    resistance_graph, root, terminal = generate_resistance_graph(bdd, threshold, round_upto)
    sp = nx.shortest_path(resistance_graph, root, terminal)
    bdd = switch_on_nodes_in_shortest_path(sp, bdd, threshold, round_upto)
    time_stitching = time.time() - time_stitching

    return bdd, time_stitching


def run_shortest_path_up_down(bdd, lidx, threshold=0.5, round_upto=1):
    time_stitching = time.time()
    # If the first layer is disconnected, run shortest path from root to terminal node
    if lidx == 0:
        bdd, time_stitching = run_shortest_path(bdd, threshold=threshold, round_upto=round_upto)
        return bdd, time_stitching

    resistance_graph, root, terminal = generate_resistance_graph(bdd, threshold, round_upto)
    # DOWN stitching
    # Turn on nodes in the shortest paths starting from connected node in the previous layer
    # to the terminal layer.
    for nidx, node in bdd[lidx - 1]:
        if node["conn"]:
            start_node = f"{lidx}-{nidx}"
            sp = nx.shortest_path(resistance_graph, start_node, terminal)
            bdd = switch_on_nodes_in_shortest_path(sp, bdd, threshold, round_upto)
    # UP Stitching
    # Turn on nodes in the shortest paths starting from the root to the high scoring nodes
    # in the current layer
    for nidx, node in bdd[lidx]:
        if np.round(node["pred"], round_upto) >= threshold:
            end_node = f"{lidx + 1}-{nidx}"
            sp = nx.shortest_path(resistance_graph, root, end_node)
            bdd = switch_on_nodes_in_shortest_path(sp, bdd, threshold, round_upto)

    time_stitching = time.time() - time_stitching

    return bdd, time_stitching


def run_mip(bdd, threshold=0.5, round_upto=1, profile=None):
    incs = None
    # incs = []

    # def callback(model, where):
    #     if where == gp.GRB.Callback.MIPSOL:
    #         time = model.cbGet(gp.GRB.Callback.RUNTIME)
    #         best = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST)
    #         incs.append([time, best])

    time_mip = time.time()
    mip = generate_min_resistance_mip_model(bdd,
                                            threshold=threshold,
                                            round_upto=round_upto,
                                            profile=profile)
    time_mip = time.time() - time_mip

    time_stitching = time.time()
    # mip.optimize(callback)
    mip.optimize()
    sol = get_mip_solution(mip)
    bdd = switch_on_nodes_in_mip_solution(bdd, sol)
    time_stitching = time.time() - time_stitching

    return bdd, time_stitching, time_mip, incs


def stitch(problem, cfg, bdd, lidx, total_time_stitching):
    # If BDD is disconnected on the first layer, select both nodes.
    time_stitching, time_mip = None, None
    invalid_heuristic = False
    actual_lidx = lidx + 1

    if actual_lidx < cfg.deploy.select_all_upto:
        bdd, time_stitching = run_select_all(bdd, lidx,
                                             threshold=cfg.deploy.threshold,
                                             round_upto=cfg.deploy.round_upto)

    elif cfg.deploy.stitching_heuristic == "min_resistance":
        bdd, time_stitching = run_lookahead(bdd, lidx,
                                            cfg.deploy.lookahead,
                                            threshold=cfg.deploy.threshold,
                                            round_upto=cfg.deploy.round_upto)

    elif cfg.deploy.stitching_heuristic == "shortest_path":
        bdd, time_stitching = run_shortest_path(bdd,
                                                threshold=cfg.deploy.threshold,
                                                round_upto=cfg.deploy.round_upto)

    elif cfg.deploy.stitching_heuristic == "shortest_path_up_down":
        bdd, time_stitching = run_shortest_path_up_down(bdd,
                                                        lidx,
                                                        threshold=cfg.deploy.threshold,
                                                        round_upto=cfg.deploy.round_upto)

    elif cfg.deploy.stitching_heuristic == "mip":
        # return flag_stitched_layer, time_stitching, bdd
        # profile = pd.read_csv(
        #     Path(__file__).parent / f"dist/{cfg.problem.name}/"
        #                             f"{cfg.problem.n_objs}-{cfg.problem.n_vars}.csv")
        # bdd, time_stitching, time_mip = run_mip(bdd,
        #                                         threshold=cfg.threshold,
        #                                         round_upto=cfg.round_upto,
        #                                         profile=profile["8"].values)
        bdd, time_stitching, time_mip, incs = run_mip(bdd,
                                                      threshold=cfg.deploy.threshold,
                                                      round_upto=cfg.deploy.round_upto)

    else:
        invalid_heuristic = True

    if invalid_heuristic:
        raise ValueError("Invalid heuristic!")

    total_time_stitching += time_stitching
    return bdd, total_time_stitching, time_mip
