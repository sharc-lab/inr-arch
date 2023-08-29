import os
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx



color_map = {
    "input": "lightgreen",
    "output": "lightblue",
    None: "white",
}

def plot_graph(G: nx.DiGraph, figure_fp: Optional[Path] = None):
    fig, ax = plt.subplots(1,1, figsize=(20, 10))
    pos = nx.kamada_kawai_layout(G, scale=10)
    colors = [color_map[G.nodes[node]["type"]] for node in G.nodes]
    nx.draw(G, pos, with_labels=True, node_shape="s", font_size=10, ax=ax, node_color=colors, edgecolors="black")
    if figure_fp:
        os.makedirs(figure_fp.parent, exist_ok=True)
        plt.savefig(figure_fp, dpi=300)
    else:
        plt.show()


def plot_grad_graph(G_grad: nx.DiGraph, figure_fp: Optional[Path] = None):

    G = G_grad.copy()

    # set all nodes as unlabeled
    for node in G.nodes:
        G.nodes[node]["type"] = None
    
    dag_inputs = [n for n, d in G.in_degree() if d == 0]
    dag_outputs = [n for n, d in G.out_degree() if d == 0]
    for n in dag_inputs:
        G.nodes[n]['type'] = 'input'
    for n in dag_outputs:
        G.nodes[n]['type'] = 'output'

    plot_graph(G, figure_fp)