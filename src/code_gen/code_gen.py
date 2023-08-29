import os
from pathlib import Path
import time
from functools import partial


from rich import print
import tqdm
import jinja2

import numpy as np
import networkx as nx

import torch
import torch.nn as nn
from torch.nn import Sequential

from src.profiling import profileit


def read_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


def write_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)

CURRENT_SCRIPT_DIR = Path(__file__).parent

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(CURRENT_SCRIPT_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
    cache_size=0,
    auto_reload=True,
)


MODEL_H_TEMPLATE = partial(jinja_env.get_template, "templates/model.h.jinja")
MODEL_CPP_TEMPLATE = partial(jinja_env.get_template, "templates/model.cpp.jinja")
MAKEFILE_TEMPLATE = partial(jinja_env.get_template, "templates/makefile.jinja")
RUN_HLS_TEMPLATE = partial(jinja_env.get_template, "templates/run_hls.tcl.jinja")

LIB_FP = CURRENT_SCRIPT_DIR.parent / "inr_hw_lib" / "inr_hw_lib.h"





def legalize_name(fn_name: str):
    fn_name = fn_name.replace(".", "_")
    return fn_name


VALID_FUNCTIONS = [
    'Permute',
    'Cos',
    'Mm',
    'Select',
    'Sin',
    'Unsqueeze',
    'T',
    'Mul',
    'Add',
    'Neg'
]

def compute_graph_data(G_in):

    topo_sort = list(nx.lexicographical_topological_sort(G_in))
    nodes_topo = [(n, G_in.nodes[n]["type"]) for n in topo_sort]

    nodes_input = [ n for n, n_type in nodes_topo if n_type == "input" ]
    nodes_output = [ n for n, n_type in nodes_topo if n_type == "output" ]
    nodes_fn = [ n for n, n_type in nodes_topo if n_type == "fn" ]

    nodes_input_data = [ G_in.nodes[n] for n in nodes_input ]
    nodes_output_data = [ G_in.nodes[n] for n in nodes_output ]
    nodes_fn_data = [ G_in.nodes[n] for n in nodes_fn ]

    edges = G_in.edges()
    edges_data = [ G_in.edges[e] for e in edges ]

    graph_data = {
        "nodes_input": nodes_input,
        "nodes_output": nodes_output,
        "nodes_fn": nodes_fn,
        "nodes_input_data": nodes_input_data,
        "nodes_output_data": nodes_output_data,
        "nodes_fn_data": nodes_fn_data,
        "edges": edges,
        "edges_data": edges_data,
    }

    return graph_data

def gen_cosnt_array_2d(tensor: torch.Tensor, name: str, type: str):
    code =""
    # code += f"static const {type} {name}[] = {{\n"
    code += f"{type} {name}"
    for dim_size in tensor.shape:
        code += f"[{dim_size}]"
    code += " = {\n"
    for i in range(tensor.shape[0]):
        code += "    {"
        for j in range(tensor.shape[1]):
            code += f"{float(tensor[i][j])}"
            if j != tensor.shape[1] - 1:
                code += ", "
        # code += "},\n"
        if i != tensor.shape[0] - 1:
            code += "},\n"
        else:
            code += "}\n"
    code += "};\n"
    code += "\n"
    return code


def gen_buffer_copy_code(n, n_data, graph_data, G_in):

    # topo_sort = list(nx.lexicographical_topological_sort(G_in))
    # nodes_topo = [(n, G_in.nodes[n]["type"]) for n in topo_sort]

    # nodes_input = [ n for n, n_type in nodes_topo if n_type == "input" ]
    # nodes_output = [ n for n, n_type in nodes_topo if n_type == "output" ]
    # nodes_fn = [ n for n, n_type in nodes_topo if n_type == "fn" ]

    # nodes_input_data = [ G_in.nodes[n] for n in nodes_input ]
    # nodes_output_data = [ G_in.nodes[n] for n in nodes_output ]
    # nodes_fn_data = [ G_in.nodes[n] for n in nodes_fn ]

    # edges = G_in.edges()
    # edges_data = [ G_in.edges[e] for e in edges ]

    # graph_data = compute_graph_data(G_in)

    nodes_input = graph_data["nodes_input"]
    nodes_output = graph_data["nodes_output"]
    nodes_fn = graph_data["nodes_fn"]

    nodes_input_data = graph_data["nodes_input_data"]
    nodes_output_data = graph_data["nodes_output_data"]
    nodes_fn_data = graph_data["nodes_fn_data"]

    edges = graph_data["edges"]
    edges_data = graph_data["edges_data"]

    code = ""


    if n in nodes_output:
        raise Exception("No buffer copy code for output nodes, how did you get here?")
    node_name = n

    in_edges = [ e for e in edges if e[1] == node_name ]
    out_edges = [ e for e in edges if e[0] == node_name ]

    if n in nodes_input:
        out_edge_src_buffer = f"{legalize_name(n_data['name'])}"
    else:
        out_edge_src_buffer = f"out_buffer___{legalize_name(node_name)}"
    # print(out_edge_src_buffer)
    out_edges_destination_buffers = []
    for e in out_edges:
        src = e[0]
        dst = e[1]

        if dst in nodes_output:
            out_edges_destination_buffers.append(f"{legalize_name(dst)}")
        else:
            # print(f"src: {src}, dst: {dst}")
            dst_in_edges = [ e for e in edges if e[1] == dst ]
            det_in_edges_pos = [ G_in.edges[e]["edge_pos"] for e in dst_in_edges ]
            dst_idx = dst_in_edges.index(e)
            dst_pos = det_in_edges_pos[dst_idx]
            # print(f"dst_in_edges: {dst_in_edges}")
            # print()
            out_edges_destination_buffers.append(f"in_buffer_{dst_pos}_{legalize_name(dst)}")
    # print(out_edges_destination_buffers)

    out_edges_shape = [ G_in.edges[e]["shape"] for e in out_edges ]
    out_shape = out_edges_shape[0]

    if len(out_shape) == 1:
        code += f"Broadcast_1d<F_TYPE, {out_shape[0]}> broadcast_{out_edge_src_buffer};\n"
        code += f"broadcast_{out_edge_src_buffer}.broadcast_1d_{len(out_edges_destination_buffers)}(\n"
        code += f"    {out_edge_src_buffer},\n"
        for i, out_edges_destination_buffer in enumerate(out_edges_destination_buffers):
            code += f"    {out_edges_destination_buffer}"
            if i < len(out_edges_destination_buffers) - 1:
                code += ","
            code += "\n"
        code += ");\n"
        code += "\n"
    elif len(out_shape) == 2:
        code += f"Broadcast_2d<F_TYPE, {out_shape[0]}, {out_shape[1]}> broadcast_{out_edge_src_buffer};\n"
        code += f"broadcast_{out_edge_src_buffer}.broadcast_2d_{len(out_edges_destination_buffers)}(\n"
        code += f"    {out_edge_src_buffer},\n"
        for i, out_edges_destination_buffer in enumerate(out_edges_destination_buffers):
            code += f"    {out_edges_destination_buffer}"
            if i < len(out_edges_destination_buffers) - 1:
                code += ","
            code += "\n"
        code += ");\n"
        code += "\n"
    elif len(out_shape) == 3:
        code += f"Broadcast_3d<F_TYPE, {out_shape[0]}, {out_shape[1]}, {out_shape[2]}> broadcast_{out_edge_src_buffer};\n"
        code += f"broadcast_{out_edge_src_buffer}.broadcast_3d_{len(out_edges_destination_buffers)}(\n"
        code += f"    {out_edge_src_buffer},\n"
        for i, out_edges_destination_buffer in enumerate(out_edges_destination_buffers):
            code += f"    {out_edges_destination_buffer}"
            if i < len(out_edges_destination_buffers) - 1:
                code += ","
            code += "\n"
        code += ");\n"
        code += "\n"
    else:
        raise Exception("Invalid shape")

    return code


def gen_fn_code(fn, fn_data, G_in):
    ...


def code_gen_from_graph(G_in: nx.DiGraph, dataflow:bool=False, n_jobs:int=1):

    code = ""

    code += "#include \"model.h\"\n"
    code += "\n"

    t_start = time.perf_counter()

    # topo_sort = list(nx.lexicographical_topological_sort(G_in))
    # nodes_topo = [(n, G_in.nodes[n]["type"]) for n in topo_sort]

    # nodes_input = [ n for n, n_type in nodes_topo if n_type == "input" ]
    # nodes_output = [ n for n, n_type in nodes_topo if n_type == "output" ]
    # nodes_fn = [ n for n, n_type in nodes_topo if n_type == "fn" ]

    # nodes_input_data = [ G_in.nodes[n] for n in nodes_input ]
    # nodes_output_data = [ G_in.nodes[n] for n in nodes_output ]
    # nodes_fn_data = [ G_in.nodes[n] for n in nodes_fn ]

    # edges = G_in.edges()
    # edges_data = [ G_in.edges[e] for e in edges ]

    graph_data = compute_graph_data(G_in)

    nodes_input = graph_data["nodes_input"]
    nodes_output = graph_data["nodes_output"]
    nodes_fn = graph_data["nodes_fn"]

    nodes_input_data = graph_data["nodes_input_data"]
    nodes_output_data = graph_data["nodes_output_data"]
    nodes_fn_data = graph_data["nodes_fn_data"]

    edges = graph_data["edges"]
    edges_data = graph_data["edges_data"]

    t_end = time.perf_counter()
    print(f"Time to process graph: {t_end - t_start}")

    for n in nodes_input_data:
        n_name = n["name"]
        n_shape = n["shape"]

        array_initilization = ""
        array_initilization += f"F_TYPE {legalize_name(n_name)}"
        for s in n_shape:
            array_initilization += f"[{s}]"
        array_initilization += ";"
        array_initilization += "\n"

        code += array_initilization

    code += "\n"

    for n, n_data in zip(nodes_output, nodes_output_data):

        n_name = n
        n_shape = n_data["shape"]

        array_initilization = ""
        array_initilization += f"F_TYPE {legalize_name(n_name)}"
        for s in n_shape:
            array_initilization += f"[{s}]"
        array_initilization += ";"
        array_initilization += "\n"
        
        code += array_initilization
    
    code += "\n"

    for fn, fn_data in list(zip(nodes_fn, nodes_fn_data)):
        node_name = fn

        in_edges = [ e for e in edges if e[1] == node_name ]
        out_edges = [ e for e in edges if e[0] == node_name ]

        in_edges_shape = [ G_in.edges[e]["shape"] for e in in_edges ]
        out_edges_shape = [ G_in.edges[e]["shape"] for e in out_edges ]

        # assert that all out edges have the same shape
        if not len(set(out_edges_shape)) == 1:
            raise ValueError("All out edges must have the same shape")

        out_shape = out_edges_shape[0]

        in_edges_pos = [ G_in.edges[e]["edge_pos"] for e in in_edges ]
        out_edges_pos = [ G_in.edges[e]["edge_pos"] for e in out_edges ]

        for e, s, p in zip(in_edges, in_edges_shape, in_edges_pos):
            array_initilization = ""
            array_initilization += f"F_TYPE in_buffer_{p}_{legalize_name(node_name)}"
            for s_i in s:
                array_initilization += f"[{s_i}]"
            array_initilization += ";"
            array_initilization += "\n"
            code += array_initilization

        array_initilization = ""
        array_initilization += f"F_TYPE out_buffer___{legalize_name(node_name)}"
        for s in out_shape:
            array_initilization += f"[{s}]"
        array_initilization += ";"
        array_initilization += "\n"
        code += array_initilization

    code += "\n"


    code += "void main_code_region() {\n"
    code += "\n"
    
    if dataflow:
        code += "#pragma HLS DATAFLOW\n"
        code += "\n"

    for n, n_data in zip(nodes_input, nodes_input_data):
        code += gen_buffer_copy_code(n, n_data, graph_data, G_in)
        code += "\n"


    for fn, fn_data in tqdm.tqdm(list(zip(nodes_fn, nodes_fn_data))):
        node_name = fn
        fn_type = fn_data["fn"]

        # print(f"fn_name: {fn_name}")
        # print(f"fn_type: {fn_type}")
        # print(F"fn_data: {fn_data}")

        in_edges = [ e for e in edges if e[1] == node_name ]
        out_edges = [ e for e in edges if e[0] == node_name ]

        in_edges_shape = [ G_in.edges[e]["shape"] for e in in_edges ]
        out_edges_shape = [ G_in.edges[e]["shape"] for e in out_edges ]

        # assert that all out edges have the same shape
        if not len(set(out_edges_shape)) == 1:
            raise ValueError("All out edges must have the same shape")

        out_shape = out_edges_shape[0]

        in_edges_pos = [ G_in.edges[e]["edge_pos"] for e in in_edges ]
        out_edges_pos = [ G_in.edges[e]["edge_pos"] for e in out_edges ]


        code += f'printf("Executing {node_name}...\\n");\n'

        if fn_type not in VALID_FUNCTIONS:
            raise Exception(f"Invalid function {fn_type}")

        # permute
        if fn_type == "Permute":
            # code += f"// Permute: {fn_name}\n"
            raise NotImplementedError("Permute not implemented")
        
        # select
        if fn_type == "Select":
            code += f"// Select: {node_name}\n"

            in_shape = in_edges_shape[0]
            ndim = len(in_shape)
            if ndim not in (2, 3):
                raise NotImplementedError(f"Select not implemented for {ndim} dimensions")

            select_dim = fn_data["attrs"]["dim"]
            select_index = fn_data["attrs"]["index"]
            code += f"// select_dim: {select_dim}\n"
            code += f"// select_index: {select_index}\n"

            # print(fn_name)
            # print(fn_data)

            n_dims = len(fn_data["attrs"]["self_sizes"])
            # print(f"n_dims: {n_dims}")

            in_shape = in_edges_shape[0]
            # print(f"in_shape: {in_shape}")

            if n_dims == 2:
                # print("2 -> 41")
                code += f"Select_2_1<F_TYPE, {in_shape[0]}, {in_shape[1]}, {select_dim}, {select_index}> select_{legalize_name(node_name)};\n"
                code += f"select_{legalize_name(node_name)}.forward_dim_{select_dim}(\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += "\n"

            if n_dims == 3:
                raise NotImplementedError("3 dims -> 2 dims select not implemented")

        
        
        # unsqueeze
        if fn_type == "Unsqueeze":
            code += f"// Unsqueeze: {node_name}\n"
            # print(fn_data)
            # print(in_edges_shape)
            # print(out_shape)

            in_shape = in_edges_shape[0]

            dim = fn_data["attrs"]["dim"]
            if dim < -2 or dim > 1:
                raise ValueError(f"Invalid dim: {dim}")
            
            if len(in_shape) == 1:
                code += f"Unsqueeze_1_2<F_TYPE, {in_shape[0]}, {dim}> unsqueeze_{legalize_name(node_name)};\n"
                if dim == -2:
                    code += f"unsqueeze_{legalize_name(node_name)}.forward_dim_neg_2(\n"
                    code += f"    in_buffer_0_{legalize_name(node_name)},\n"
                    code += f"    out_buffer___{legalize_name(node_name)}\n"
                    code += f");\n"
                    code += f"\n"
                elif dim == -1:
                    raise NotImplementedError("Unsqueeze dim -1 not implemented")
                elif dim == 0:
                    raise NotImplementedError("Unsqueeze dim 0 not implemented")
                elif dim == 1:
                    raise NotImplementedError("Unsqueeze dim 1 not implemented")
            else:
                raise ValueError(f"Invalid input shape: {in_shape}")
                    
        # transpose
        if fn_type == "T":
            code += f"// T: {node_name}\n"

            code += f"Transpose_2<F_TYPE, {in_edges_shape[0][0]}, {in_edges_shape[0][1]}> transpose_{legalize_name(node_name)};\n"
            code += f"transpose_{legalize_name(node_name)}.forward(\n"
            code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
            code += f"    out_buffer___{legalize_name(node_name)}\n"
            code += f");\n"
            code += f"\n"
        
        # mul
        if fn_type == "Mul":
            code += f"// Mul: {node_name}\n"


            if fn_data["attrs"]["self"] != None:
                # print(fn_name)
                # print(fn_data)
                # print("two input add")

                # sort by in_edges_pos
                in_edges_sorted = [ e for _, e in sorted(zip(in_edges_pos, in_edges)) ]
                in_edges_shape_sorted = [ e for _, e in sorted(zip(in_edges_pos, in_edges_shape)) ]

                in_shape = in_edges_shape[0]
                n_dims = len(in_shape)
                
                if n_dims == 2:
                    code += f"ElementWiseMul2D<F_TYPE, {in_shape[0]}, {in_shape[1]}> mul_{legalize_name(node_name)};\n"
                    code += f"mul_{legalize_name(node_name)}.forward(\n"
                    code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                    code += f"    in_buffer_{in_edges_pos[1]}_{legalize_name(node_name)},\n"
                    code += f"    out_buffer___{legalize_name(node_name)}\n"
                    code += f");\n"
                    code += f"\n"
                else:
                    raise NotImplementedError(f"Mul with {n_dims} dimensions not implemented")

            else:
                const_val = fn_data["attrs"]["other"]["value"]
                const_val = float(const_val)
                in_shape = in_edges_shape[0]
                n_dims = len(in_shape)
                if n_dims == 2:
                    code += f"ElementWiseMul2DConst<F_TYPE, {in_shape[0]}, {in_shape[1]}> mul_{legalize_name(node_name)};\n"
                    code += f"mul_{legalize_name(node_name)}.forward(\n"
                    code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                    code += f"    out_buffer___{legalize_name(node_name)},\n"
                    code += f"    F_TYPE({const_val})\n"
                    code += f");\n"
                    code += f"\n"
                else:
                    raise NotImplementedError(f"Const mul with {n_dims} dimensions not implemented")
                

        # add
        if fn_type == "Add":
            code += f"// Add: {node_name}\n"
            # print(fn_name)
            # print(fn_data)
            # print(in_edges_shape)
            # print(in_edges_pos)

            if len(set(in_edges_shape)) != 1:
                n_dims = len(in_edges_shape[0])
                if n_dims == 2:
                    if in_edges_shape[1][0] == 1:
                        # print("cast on dim_0")
                        code += f"ElementWiseAdd2DCasting<F_TYPE, {in_edges_shape[0][0]}, {in_edges_shape[0][1]}> add_{legalize_name(node_name)};\n"
                        code += f"add_{legalize_name(node_name)}.forward_cast_dim_0(\n"
                        code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                        code += f"    in_buffer_{in_edges_pos[1]}_{legalize_name(node_name)},\n"
                        code += f"    out_buffer___{legalize_name(node_name)}\n"
                        code += f");\n"
                        code += f"\n"
                    else:
                        # print("cast on dim_1")
                        code += f"ElementWiseAdd2DCasting<F_TYPE, {in_edges_shape[0][0]}, {in_edges_shape[0][1]}> add_{legalize_name(node_name)};\n"
                        code += f"add_{legalize_name(node_name)}.forward_cast_dim_1(\n"
                        code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                        code += f"    in_buffer_{in_edges_pos[1]}_{legalize_name(node_name)},\n"
                        code += f"    out_buffer___{legalize_name(node_name)}\n"
                        code += f");\n"
                        code += f"\n"
                else:
                    raise NotImplementedError(f"Add with {n_dims} dimensions not implemented")
            else:
                # print(fn_data)
                # print(in_edges_shape)
                # print(in_edges_pos)
                n_dims = len(in_edges_shape[0])
                if n_dims == 2:
                    code += f"ElementWiseAdd2D<F_TYPE, {in_edges_shape[0][0]}, {in_edges_shape[0][1]}> add_{legalize_name(node_name)};\n"
                    code += f"add_{legalize_name(node_name)}.forward(\n"
                    code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                    code += f"    in_buffer_{in_edges_pos[1]}_{legalize_name(node_name)},\n"
                    code += f"    out_buffer___{legalize_name(node_name)}\n"
                    code += f");\n"
                    code += f"\n"
                else:
                    raise NotImplementedError(f"Add with {n_dims} dimensions not implemented")

            
            

            
        # neg
        if fn_type == "Neg":
            code += f"// Neg: {node_name}\n"
            n_dims = len(in_edges_shape[0])
            if n_dims == 1:
                code += f"Neg1D<F_TYPE, {in_edges_shape[0][0]}> neg_{legalize_name(node_name)};\n"
                code += f"neg_{legalize_name(node_name)}.forward(\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"
            elif n_dims == 2:
                code += f"Neg2D<F_TYPE, {in_edges_shape[0][0]}, {in_edges_shape[0][1]}> neg_{legalize_name(node_name)};\n"
                code += f"neg_{legalize_name(node_name)}.forward(\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"
            elif n_dims == 3:
                code += f"Neg3D<F_TYPE, {in_edges_shape[0][0]}, {in_edges_shape[0][1]}, {in_edges_shape[0][2]}> neg_{legalize_name(node_name)};\n"
                code += f"neg_{legalize_name(node_name)}.forward(\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"
            else:
                raise NotImplementedError(f"Neg with {n_dims} dimensions not implemented")
            
        # mm
        if fn_type == "Mm":
            code += f"// Mm: {node_name}\n"
         
            if len(in_edges_shape) != 2:
                # print(fn_name)
                # print(fn_data)
                # print(in_edges_shape)
                # print("ERROR: mm with != 2 inputs")
                const_tensor = fn_data["attrs"]["self"]["contents"]
                code_static_init_array = gen_cosnt_array_2d(const_tensor, f"mm_{legalize_name(node_name)}_const_input_0", "F_TYPE")
                code += code_static_init_array

                # print(in_edges)
                # print(in_edges_shape)

                self_shape = const_tensor.shape
                other_shape = in_edges_shape[0]
                
                code += f"MM<F_TYPE, {self_shape[0]}, {self_shape[1]}, {other_shape[1]}> mm_{legalize_name(node_name)};\n"
                code += f"mm_{legalize_name(node_name)}.forward(\n"
                code += f"    mm_{legalize_name(node_name)}_const_input_0,\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"

            else:
                in_edges_shapes_sorted = [ e for _, e in sorted(zip(in_edges_pos, in_edges_shape)) ]

                code += f"MM<F_TYPE, {in_edges_shapes_sorted[0][0]}, {in_edges_shapes_sorted[0][1]}, {in_edges_shapes_sorted[1][1]}> mm_{legalize_name(node_name)};\n"
                code += f"mm_{legalize_name(node_name)}.forward(\n"
                code += f"    in_buffer_0_{legalize_name(node_name)},\n"
                code += f"    in_buffer_1_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"

            # print(in_edges_shape[in_edges_pos[0]])
            # print(in_edges_shape[in_edges_pos[1]])


        # sin
        if fn_type == "Sin":
            code += f"// Sin: {node_name}\n"
            # if output shape is 2 dims
            if len(out_shape) == 1:
                code += f"Activation_1d<F_TYPE, {out_shape[0]}, activation_sin<F_TYPE>> sin_{legalize_name(node_name)};\n"
                code += f"sin_{legalize_name(node_name)}.forward(\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"
            elif len(out_shape) == 2:
                code += f"Activation_2d<F_TYPE, {out_shape[0]}, {out_shape[1]}, activation_sin<F_TYPE>> sin_{legalize_name(node_name)};\n"
                code += f"sin_{legalize_name(node_name)}.forward(\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"
            elif len(out_shape) == 3:
                code += f"Activation_3d<F_TYPE, {out_shape[0]}, {out_shape[1]}, {out_shape[2]}, activation_sin<F_TYPE>> sin_{legalize_name(node_name)};\n"
                code += f"sin_{legalize_name(node_name)}.forward(\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"
            else:
                raise Exception("Invalid shape")
        
        # cos
        if fn_type == "Cos":
            code += f"// Cos: {node_name}\n"      

            if len(out_shape) == 1:
                code += f"Activation_1d<F_TYPE, {out_shape[0]}, activation_cos<F_TYPE>> cos_{legalize_name(node_name)};\n"
                code += f"cos_{legalize_name(node_name)}.forward(\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"
            elif len(out_shape) == 2:
                code += f"Activation_2d<F_TYPE, {out_shape[0]}, {out_shape[1]}, activation_cos<F_TYPE>> cos_{legalize_name(node_name)};\n"
                code += f"cos_{legalize_name(node_name)}.forward(\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"
            elif len(out_shape) == 3:
                code += f"Activation_3d<F_TYPE, {out_shape[0]}, {out_shape[1]}, {out_shape[2]}, activation_cos<F_TYPE>> cos_{legalize_name(node_name)};\n"
                code += f"cos_{legalize_name(node_name)}.forward(\n"
                code += f"    in_buffer_{in_edges_pos[0]}_{legalize_name(node_name)},\n"
                code += f"    out_buffer___{legalize_name(node_name)}\n"
                code += f");\n"
                code += f"\n"
            else:
                raise Exception("Invalid shape")

        code += "\n"

        code += gen_buffer_copy_code(fn, fn_data, graph_data, G_in)

        code += "\n"  


        
    code += "}\n"

    code += "\n"

    code += f"void copy_top_in(\n"
    for i, n in enumerate(nodes_input_data):
        code += f"    F_TYPE {legalize_name(n['name'])}_top_in"
        for s in n["shape"]:
            code += f"[{s}]"
        if i != len(nodes_input_data) - 1:
            code += ",\n"
        else:
            code += "\n"
    code += f")\n"
    code += f"{{\n"
    code += f"    #pragma HLS INLINE off\n"
    for n in nodes_input_data:
        if len(n['shape']) == 1:
            code += f"    copy_1d<{n['shape'][0]}>({legalize_name(n['name'])}_top_in, {legalize_name(n['name'])});\n"
        elif len(n['shape']) == 2:
            code += f"    copy_2d<{n['shape'][0]}, {n['shape'][1]}>({legalize_name(n['name'])}_top_in, {legalize_name(n['name'])});\n"
        elif len(n['shape']) == 3:
            code += f"    copy_3d<{n['shape'][0]}, {n['shape'][1]}, {n['shape'][2]}>({legalize_name(n['name'])}_top_in, {legalize_name(n['name'])});\n"
        else:
            raise Exception("Invalid shape")
    code += f"}}\n"

    code += "\n"


    code += f"void copy_top_out(\n"
    for i, (n, n_data) in enumerate(zip(nodes_output, nodes_output_data)):
        code += f"    F_TYPE {legalize_name(n)}_top_out"
        for s in n_data["shape"]:
            code += f"[{s}]"
        if i != len(nodes_output_data) - 1:
            code += ",\n"
        else:
            code += "\n"
    code += f")\n"
    code += f"{{\n"
    code += f"    #pragma HLS INLINE off\n"
    for n, n_data in zip(nodes_output, nodes_output_data):
        if len(n_data['shape']) == 1:
            code += f"    copy_1d<{n_data['shape'][0]}>({legalize_name(n)}, {legalize_name(n)}_top_out);\n"
        elif len(n_data['shape']) == 2:
            code += f"    copy_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}, {legalize_name(n)}_top_out);\n"
        elif len(n_data['shape']) == 3:
            code += f"    copy_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}, {legalize_name(n)}_top_out);\n"
        else:
            raise Exception("Invalid shape")
    code += f"}}\n"

    code += "\n"

    code += f"void model_top(\n"
    for n in nodes_input_data:
        code += f"    F_TYPE {legalize_name(n['name'])}_top_in"
        for s in n["shape"]:
            code += f"[{s}]"
        code += f",\n"
    for i, (n, n_data) in enumerate(zip(nodes_output, nodes_output_data)):
        code += f"    F_TYPE {legalize_name(n)}_top_out"
        for s in n_data["shape"]:
            code += f"[{s}]"
        if i != len(nodes_output) - 1:
            code += ",\n"
        else:
            code += "\n"
        
    
    code += f")\n"
    code += f"{{\n"

    code += f"    copy_top_in(\n"
    for i, n in enumerate(nodes_input_data):
        code += f"        {legalize_name(n['name'])}_top_in"
        if i != len(nodes_input_data) - 1:
            code += ",\n"
        else:
            code += "\n"
    code += f"    );\n"

    code += f"\n"
    code += f"    main_code_region();\n"
    code += f"\n"

    code += f"    copy_top_out(\n"
    for i, (n, n_data) in enumerate(zip(nodes_output, nodes_output_data)):
        code += f"        {legalize_name(n)}_top_out"
        if i != len(nodes_output) - 1:
            code += ",\n"
        else:
            code += "\n"
    code += f"    );\n"

    code += f"}}\n"
    code += "\n"

    
    return code


def model_top_declaration_gen(G_in: nx.DiGraph):

    # topo_sort = list(nx.lexicographical_topological_sort(G_in))
    # nodes_topo = [(n, G_in.nodes[n]["type"]) for n in topo_sort]

    # nodes_input = [ n for n, n_type in nodes_topo if n_type == "input" ]
    # nodes_output = [ n for n, n_type in nodes_topo if n_type == "output" ]
    # nodes_fn = [ n for n, n_type in nodes_topo if n_type == "fn" ]

    # nodes_input_data = [ G_in.nodes[n] for n in nodes_input ]
    # nodes_output_data = [ G_in.nodes[n] for n in nodes_output ]
    # nodes_fn_data = [ G_in.nodes[n] for n in nodes_fn ]

    graph_data = compute_graph_data(G_in)
    
    nodes_input = graph_data["nodes_input"]
    nodes_output = graph_data["nodes_output"]
    nodes_fn = graph_data["nodes_fn"]

    nodes_input_data = graph_data["nodes_input_data"]
    nodes_output_data = graph_data["nodes_output_data"]
    nodes_fn_data = graph_data["nodes_fn_data"]

    code = ""
    code += f"extern \"C\" {{\n"

    code += f"void model_top(\n"
    for n in nodes_input_data:
        code += f"    F_TYPE {legalize_name(n['name'])}_top_in"
        for s in n["shape"]:
            code += f"[{s}]"
        code += f",\n"
    for i, (n, n_data) in enumerate(zip(nodes_output, nodes_output_data)):
        code += f"    F_TYPE {legalize_name(n)}_top_out"
        for s in n_data["shape"]:
            code += f"[{s}]"
        if i != len(nodes_output) - 1:
            code += ",\n"
        else:
            code += "\n"

    code += f");\n"
    code += f"}}\n"
    code += "\n"

    return code


def testbench_gen(G_in: nx.DiGraph):

    graph_data = compute_graph_data(G_in)

    nodes_input = graph_data["nodes_input"]
    nodes_output = graph_data["nodes_output"]
    nodes_fn = graph_data["nodes_fn"]

    nodes_input_data = graph_data["nodes_input_data"]
    nodes_output_data = graph_data["nodes_output_data"]
    nodes_fn_data = graph_data["nodes_fn_data"]

    code = ""
    code += "#include \"model.h\"\n"
    code += "\n"

    for n, n_data in zip(nodes_input, nodes_input_data):
        code += f"float {legalize_name(n)}_top_in_float"
        for s in n_data["shape"]:
            code += f"[{s}]"
        code += f";\n"

        code += f"F_TYPE {legalize_name(n)}_top_in"
        for s in n_data["shape"]:
            code += f"[{s}]"
        code += f";\n"

    for n, n_data in zip(nodes_output, nodes_output_data):
        code += f"float {legalize_name(n)}_top_out_float"
        for s in n_data["shape"]:
            code += f"[{s}]"
        code += f";\n"

        code += f"F_TYPE {legalize_name(n)}_top_out"
        for s in n_data["shape"]:
            code += f"[{s}]"
        code += f";\n"

        code += f"float {legalize_name(n)}_top_out_cast_back"
        for s in n_data["shape"]:
            code += f"[{s}]"
        code += f";\n"
    
    code += "\n"

    code += f"int main()\n"
    code += f"{{\n"

    for n, n_data in zip(nodes_input, nodes_input_data):
        if len(n_data['shape']) == 1:
            code += f"load_data_1d<{n_data['shape'][0]}>(\"./testbench_data/{n_data['name']}.bin\", {legalize_name(n)}_top_in_float);\n"
        elif len(n_data['shape']) == 2:
            code += f"load_data_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>(\"./testbench_data/{n_data['name']}.bin\", {legalize_name(n)}_top_in_float);\n"
        elif len(n_data['shape']) == 3:
            code += f"load_data_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>(\"./testbench_data/{n_data['name']}.bin\", {legalize_name(n)}_top_in_float);\n"
        else:
            raise Exception("Invalid shape")

    code += f"\n"

    for n, n_data in zip(nodes_output, nodes_output_data):
        if len(n_data['shape']) == 1:
            code += f"load_data_1d<{n_data['shape'][0]}>(\"./testbench_data/{n}.bin\", {legalize_name(n)}_top_out_float);\n"
        elif len(n_data['shape']) == 2:
            code += f"load_data_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>(\"./testbench_data/{n}.bin\", {legalize_name(n)}_top_out_float);\n"
        elif len(n_data['shape']) == 3:
            code += f"load_data_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>(\"./testbench_data/{n}.bin\", {legalize_name(n)}_top_out_float);\n"
        else:
            raise Exception("Invalid shape")
        
    code += f"\n"


    # cast to F_TYPE
    for n, n_data in zip(nodes_input, nodes_input_data):
        if len(n_data['shape']) == 1:
            code += f"cast_1d<{n_data['shape'][0]}>({legalize_name(n)}_top_in_float, {legalize_name(n)}_top_in);\n"
        elif len(n_data['shape']) == 2:
            code += f"cast_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}_top_in_float, {legalize_name(n)}_top_in);\n"
        elif len(n_data['shape']) == 3:
            code += f"cast_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}_top_in_float, {legalize_name(n)}_top_in);\n"
        else:
            raise Exception("Invalid shape")

    code += f"\n"

    code += f"model_top(\n"
    for n, n_data in zip(nodes_input, nodes_input_data):
        code += f"    {legalize_name(n)}_top_in"
        code += f",\n"
    for i, (n, n_data) in enumerate(zip(nodes_output, nodes_output_data)):
        code += f"    {legalize_name(n)}_top_out"
        if i != len(nodes_output) - 1:
            code += ",\n"
        else:
            code += "\n"
    code += f");\n"

    code += f"\n"

    for n, n_data in zip(nodes_output, nodes_output_data):
        if len(n_data['shape']) == 1:
            code += f"cast_1d<{n_data['shape'][0]}>({legalize_name(n)}_top_out, {legalize_name(n)}_top_out_cast_back);\n"
        elif len(n_data['shape']) == 2:
            code += f"cast_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}_top_out, {legalize_name(n)}_top_out_cast_back);\n"
        elif len(n_data['shape']) == 3:
            code += f"cast_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}_top_out, {legalize_name(n)}_top_out_cast_back);\n"
        else:
            raise Exception("Invalid shape")
    
    code += f"\n"

    code += f"bool pass = true;\n"
    code += f"float eps = 1e-3;\n"
    code += f"\n"
    for n, n_data in zip(nodes_output, nodes_output_data):
        if len(n_data['shape']) == 1:
            code += f"pass &= compare_data_1d<{n_data['shape'][0]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back, eps);\n"
        elif len(n_data['shape']) == 2:
            code += f"pass &= compare_data_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back, eps);\n"
        elif len(n_data['shape']) == 3:
            code += f"pass &= compare_data_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back, eps);\n"
        else:
            raise Exception("Invalid shape") 
    code += f"\n"

    code += f"if (pass) {{\n"
    code += f"    printf(\"PASS\\n\");\n"
    code += f"}} else {{\n"
    code += f"    printf(\"FAIL\\n\");\n"
    code += f"}}\n"

    code += f"\n"

    # print the kernel output and the actual output
    for n, n_data in zip(nodes_output, nodes_output_data):
        code += f'printf("======================================\\n");\n'

        code += f'printf("{legalize_name(n)}_top_out_float = ");\n'
        if len(n_data['shape']) == 1:
            code += f"print_array_1d({legalize_name(n)}_top_out_float);\n"
        elif len(n_data['shape']) == 2:
            code += f"print_array_2d({legalize_name(n)}_top_out_float);\n"
        else:
            raise Exception("Invalid shape")
        
        code += f'printf("{legalize_name(n)}_top_out_cast_back = ");\n'
        if len(n_data['shape']) == 1:
            code += f"print_array_1d({legalize_name(n)}_top_out_cast_back);\n"
        elif len(n_data['shape']) == 2:
            code += f"print_array_2d({legalize_name(n)}_top_out_cast_back);\n"
        else:
            raise Exception("Invalid shape")
        
        code += f'printf("======================================\\n");\n'
        code += f"\n"

    code += f"\n"

    code += f"float testbench_mae = 0.0;\n"
    code += f"\n"
    for n, n_data in zip(nodes_output, nodes_output_data):
        if len(n_data['shape']) == 1:
            code += f"testbench_mae += compute_mae_1d<{n_data['shape'][0]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back);\n"
        elif len(n_data['shape']) == 2:
            code += f"testbench_mae += compute_mae_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back);\n"
        elif len(n_data['shape']) == 3:
            code += f"testbench_mae += compute_mae_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back);\n"
        else:
            raise Exception("Invalid shape")
    code += f"\n"

    code += f"FILE *fp = fopen(\"./testbench_mae.txt\", \"w\");\n"
    code += f"fprintf(fp, \"testbench_mae %.9f\\n\", testbench_mae);\n"
    code += f"fclose(fp);\n"
    code += f"\n"

    code += f"}}\n"
    code += "\n"

    return code
