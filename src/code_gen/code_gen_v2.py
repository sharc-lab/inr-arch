from pathlib import Path
from functools import partial, cache
import jinja2
import networkx as nx
import torch


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

LIB_FP = CURRENT_SCRIPT_DIR.parent / "inr_hw_lib" / "inr_hw_lib_v2.h"


@cache
def legalize_name(fn_name: str, suffix: str = ""):
    if "." in fn_name:
        fn_name = fn_name.replace(".", "_")
    fn_name += suffix
    return fn_name


@cache
def array_str(name: str, shape: torch.Size, suffix: str = ""):
    s = ""
    s += f"F_TYPE {legalize_name(name)}{suffix}"
    for dim in shape:
        s += f"[{dim}]"
    return s


@cache
def array_stream_type_str(name: str, shape: torch.Size, block_size: int = 1):
    s = ""
    s_dims = ", ".join([str(dim) for dim in shape])
    s += f"array_stream<F_TYPE, array_shape<{s_dims}>, {block_size}>"
    return s


@cache
def array_stream_str(
    name: str, shape: torch.Size, block_size: int = 1, suffix: str = "_stream"
):
    s = ""
    s_dims = ", ".join([str(dim) for dim in shape])
    s += f"{array_stream_type_str(name, shape, block_size=block_size)} {legalize_name(name)}{suffix}"
    return s


@cache
def array_stream_typedef_str(
    name: str, shape: torch.Size, block_size: int = 1, suffix: str = "_stream_T"
):
    s = ""
    s_dims = ", ".join([str(dim) for dim in shape])
    s += f"typedef {array_stream_type_str(name, shape, block_size=block_size)} {legalize_name(name)}{suffix}"
    return s


def gen_cosnt_array_2d(tensor: torch.Tensor, name: str, type: str):
    code = ""
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


VALID_FUNCTIONS = [
    "Permute",
    "Cos",
    "Mm",
    "Select",
    "Sin",
    "Unsqueeze",
    "T",
    "Mul",
    "Add",
    "Neg",
]


NL = "\n"
TAB = "    "

ERROR_UNSUPPORTED_ARRAY_SHAPE = NotImplementedError(
    "Only 1D, 2D, 3D arrays are supported"
)


@cache
def compute_graph_data(G_in: nx.DiGraph):
    topo_sort = list(nx.lexicographical_topological_sort(G_in))
    nodes_topo = [(n, G_in.nodes[n]["type"]) for n in topo_sort]

    nodes_input = [n for n, n_type in nodes_topo if n_type == "input"]
    nodes_output = [n for n, n_type in nodes_topo if n_type == "output"]
    nodes_fn = [n for n, n_type in nodes_topo if n_type == "fn"]

    nodes_input_data = [G_in.nodes[n] for n in nodes_input]
    nodes_output_data = [G_in.nodes[n] for n in nodes_output]
    nodes_fn_data = [G_in.nodes[n] for n in nodes_fn]

    edges = G_in.edges()
    edges_data = [G_in.edges[e] for e in edges]

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


def gen_model_cpp(
    model_name: str,
    model_graph: nx.DiGraph,
    fifo_depths: dict[str, int] = {},
    mm_p: int = 1
):
    graph_data = compute_graph_data(model_graph)

    code = ""

    code += f'#include "model.h"' + NL
    code += NL

    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        code += array_str(n, n_data["shape"]) + ";" + NL
    code += NL

    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        code += array_str(n, n_data["shape"]) + ";" + NL
    code += NL

    code += "void main_dataflow_region(){" + NL
    code += NL
    code += "#pragma HLS DATAFLOW" + NL
    code += NL

    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        code += array_stream_typedef_str(n, n_data["shape"], block_size=1) + ";" + NL
        code += (
            f"CSIM_STATIC {legalize_name(n)}_stream_T {legalize_name(n)}_stream;" + NL
        )
    code += NL

    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        code += array_stream_typedef_str(n, n_data["shape"], block_size=1) + ";" + NL
        code += (
            f"CSIM_STATIC {legalize_name(n)}_stream_T {legalize_name(n)}_stream;" + NL
        )
    code += NL

    for n, n_data in zip(graph_data["nodes_fn"], graph_data["nodes_fn_data"]):
        code += (
            array_stream_typedef_str(
                n, n_data["shape"], block_size=1, suffix="_out_stream_T"
            )
            + ";"
            + NL
        )
        code += (
            f"CSIM_STATIC {legalize_name(n)}_out_stream_T {legalize_name(n)}__out_stream;"
            + NL
        )
    code += NL

    for e, e_data in zip(graph_data["edges"], graph_data["edges_data"]):
        n_to = e[1]
        edge_pos = e_data["edge_pos"]
        shape = e_data["shape"]

        if n_to in graph_data["nodes_fn"]:
            code += (
                array_stream_typedef_str(
                    n_to, shape, block_size=1, suffix=f"_in_{edge_pos}_stream_T"
                )
                + ";"
                + NL
            )
            code += (
                f"CSIM_STATIC {legalize_name(n_to)}_in_{edge_pos}_stream_T {legalize_name(n_to)}__in_{edge_pos}_stream;"
                + NL
            )

    code += NL
    for fifo_name, fifo_size in fifo_depths.items():
        code += f"#pragma HLS STREAM variable={fifo_name}.data depth={fifo_size}" + NL
    code += NL


    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        if len(n_data["shape"]) == 1:
            code += (
                f"array_1d_to_array_stream<F_TYPE, {n_data['shape'][0]}>({legalize_name(n)}, {legalize_name(n, suffix='_stream')});"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"array_2d_to_array_stream<F_TYPE, {n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}, {legalize_name(n, suffix='_stream')});"
                + NL
            )
        elif len(n_data["shape"]) == 3:
            code += (
                f"array_3d_to_array_stream<F_TYPE, {n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}, {legalize_name(n, suffix='_stream')});"
                + NL
            )
        else:
            raise ERROR_UNSUPPORTED_ARRAY_SHAPE

        out_edges = model_graph.out_edges(n)
        edge_data = [model_graph.edges[e] for e in out_edges]

        destinations = [e[1] for e in out_edges]
        destination_indexes = [e_data["edge_pos"] for e_data in edge_data]
        dest_zipped = zip(destinations, destination_indexes)

        destination_streams = []
        for d, i in dest_zipped:
            if d in graph_data["nodes_output"]:
                destination_streams.append(legalize_name(d, suffix=f"_stream"))
            else:
                # TODO: index i might not be correct
                destination_streams.append(legalize_name(d, suffix=f"__in_{i}_stream"))
        out_args = f"{', '.join(destination_streams)}"
        code += f"copy_stream({legalize_name(n, suffix='_stream')}, {out_args});" + NL

    code += NL * 2

    for n, n_data in zip(graph_data["nodes_fn"], graph_data["nodes_fn_data"]):
        fn_name = n
        fn_type = n_data["fn"]
        code += f"// {fn_name}" + NL

        if fn_type == "Cos":
            code += (
                f"elementwise_cos({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                + NL
            )
            code += NL

        if fn_type == "Sin":
            code += (
                f"elementwise_sin({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                + NL
            )
            code += NL

        if fn_type == "Neg":
            code += (
                f"elementwise_negate({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                + NL
            )
            code += NL

        if fn_type == "Add":
            # in_edges = model_graph.in_edges(n, data=True)
            # in_edges_sorted = sorted(in_edges, key=lambda x: x[2]["edge_pos"])
            # edge_shapes = [e[2]["shape"] for e in in_edges_sorted]
            # print(edge_shapes)
            n_dims = len(n_data["shape"])
            if n_dims == 2:
                in_edges = model_graph.in_edges(n, data=True)
                in_edges_sorted = sorted(in_edges, key=lambda x: x[2]["edge_pos"])
                edge_shapes = [e[2]["shape"] for e in in_edges_sorted]

                M_a, N_a = edge_shapes[0]
                M_b, N_b = edge_shapes[1]
                M_c, N_c = n_data["shape"]

                same_dim = (M_a == M_b) and (N_a == N_b)
                singleton_N = (M_a == M_b) and ((N_a == 1) or (N_b == 1))
                singleton_M = (N_a == N_b) and ((M_a == 1) or (M_b == 1))

                assert same_dim or singleton_N or singleton_M

                if same_dim:
                    code += (
                        f"elementwise_add({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix=f'__in_1_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                        + NL
                    )
                else:
                    code += (
                        f"{legalize_name(fn_name, suffix='_out_stream_T')} {legalize_name(fn_name, suffix='__temp_stream')};"
                        + NL
                    )
                    if singleton_M:
                        if M_a == 1:
                            # <dim, num>
                            code += (
                                f"repeat_singleton_dim_2d<0, {M_b}>({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix='__temp_stream')});"
                                + NL
                            )
                            code += (
                                f"elementwise_add({legalize_name(fn_name, suffix='__temp_stream')}, {legalize_name(fn_name, suffix=f'__in_1_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                                + NL
                            )
                        if M_b == 1:
                            code += (
                                f"repeat_singleton_dim_2d<0, {M_a}>({legalize_name(fn_name, suffix=f'__in_1_stream')}, {legalize_name(fn_name, suffix='__temp_stream')});"
                                + NL
                            )
                            code += (
                                f"elementwise_add({legalize_name(fn_name, suffix='__temp_stream')}, {legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                                + NL
                            )
                    if singleton_N:
                        if N_a == 1:
                            code += (
                                f"repeat_singleton_dim_2d<1, {N_b}>({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix='__temp_stream')});"
                                + NL
                            )
                            code += (
                                f"elementwise_add({legalize_name(fn_name, suffix='__temp_stream')}, {legalize_name(fn_name, suffix=f'__in_1_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                                + NL
                            )
                        if N_b == 1:
                            code += (
                                f"repeat_singleton_dim_2d<1, {N_a}>({legalize_name(fn_name, suffix=f'__in_1_stream')}, {legalize_name(fn_name, suffix='__temp_stream')});"
                                + NL
                            )
                            code += (
                                f"elementwise_add({legalize_name(fn_name, suffix='__temp_stream')}, {legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                                + NL
                            )
            else:
                code += (
                    f"elementwise_add({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix=f'__in_1_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                    + NL
                )
            code += NL

        if fn_type == "Mul":
            if "value" in n_data["attrs"]["other"].keys():
                code += (
                    f"elementwise_mul_const({legalize_name(fn_name, suffix=f'__in_0_stream')}, F_TYPE({n_data['attrs']['other']['value']}), {legalize_name(fn_name, suffix='__out_stream')});"
                    + NL
                )
            else:
                code += (
                    f"elementwise_mul({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix=f'__in_1_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                    + NL
                )
            code += NL

        if fn_type == "T":
            code += (
                f"transpose_2d({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                + NL
            )
            code += NL

        if fn_type == "Unsqueeze":
            dim = n_data["attrs"]["dim"]
            if dim < 0:
                dim += len(n_data["shape"])
            code += (
                f"unsqueeze({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix='__out_stream')}, {dim});"
                + NL
            )
            code += NL

        if fn_type == "Select":
            dim = n_data["attrs"]["dim"]
            index = n_data["attrs"]["index"]
            code += (
                f"select<{dim}, {index}>({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                + NL
            )
            code += NL

        if fn_type == "Mm":
            # print(n)
            # print(model_graph.in_edges(n))
            # print([model_graph.get_edge_data(*e)["shape"] for e in model_graph.in_edges(n)])
            # print()
            if "contents" in n_data["attrs"]["self"].keys():
                const_array_str = gen_cosnt_array_2d(
                    n_data["attrs"]["self"]["contents"],
                    f"{legalize_name(fn_name, suffix=f'__in_0')}",
                    "F_TYPE",
                )
                shape = n_data["attrs"]["self"]["shape"]
                code += const_array_str + NL
                code += NL

                code += array_stream_typedef_str(f"{legalize_name(fn_name, suffix=f'_in_0')}", shape, block_size=1) + ";" + NL
                code += "CSIM_STATIC " + legalize_name(fn_name, suffix=f'_in_0_stream') + "_T" + " " + legalize_name(fn_name, suffix=f'__in_0_stream') + ";" + NL
                code += (
                    f"array_2d_to_array_stream<F_TYPE, {shape[0]}, {shape[1]}>({legalize_name(fn_name, suffix=f'__in_0')}, {legalize_name(fn_name, suffix=f'__in_0_stream')});"
                    + NL
                )
                code += NL

            template_args = [
                legalize_name(fn_name, suffix='_in_0_stream_T'),
                legalize_name(fn_name, suffix='_in_1_stream_T'),
                legalize_name(fn_name, suffix='_out_stream_T'),
                str(mm_p)
            ]
            template_args_str = ", ".join(template_args)
            code += (
                f"mm_v2<{template_args_str}>({legalize_name(fn_name, suffix=f'__in_0_stream')}, {legalize_name(fn_name, suffix=f'__in_1_stream')}, {legalize_name(fn_name, suffix='__out_stream')});"
                + NL
            )
            code += NL

        out_edges = model_graph.out_edges(n)
        edge_data = [model_graph.edges[e] for e in out_edges]

        destinations = [e[1] for e in out_edges]
        destination_indexes = [e_data["edge_pos"] for e_data in edge_data]
        dest_zipped = zip(destinations, destination_indexes)
        # template <typename T_array, int n_streams>
        # void copy_stream(T_array &input, T_array output[n_streams])


        destination_streams = []
        for d, i in dest_zipped:
            if d in graph_data["nodes_output"]:
                destination_streams.append(legalize_name(d, suffix=f"_stream"))
            else:
                # TODO: index i might not be correct
                destination_streams.append(legalize_name(d, suffix=f"__in_{i}_stream"))

        out_args = f"{', '.join(destination_streams)}"

        # FIXME: handle broadcasting
        copy_stream_code = ""
        copy_stream_code += (
            f"copy_stream({legalize_name(fn_name, suffix=f'__out_stream')}, {out_args});"
            + NL
        )
        copy_stream_code += NL
        code += copy_stream_code

        code += "/" * 16 + NL
        code += NL * 2

    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        if len(n_data["shape"]) == 1:
            code += (
                f"array_stream_to_array_1d<F_TYPE, {n_data['shape'][0]}>({legalize_name(n, suffix='_stream')}, {legalize_name(n)});"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"array_stream_to_array_2d<F_TYPE, {n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n, suffix='_stream')}, {legalize_name(n)});"
                + NL
            )
        elif len(n_data["shape"]) == 3:
            code += (
                f"array_stream_to_array_3d<F_TYPE, {n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n, suffix='_stream')}, {legalize_name(n)});"
                + NL
            )
        else:
            raise ERROR_UNSUPPORTED_ARRAY_SHAPE
    code += NL * 2

    code += "}" + NL
    code += NL

    # code += "void copy_inputs_top(){" + NL
    code += "void copy_inputs(" + NL
    args = []
    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        args.append(array_str(n, n_data["shape"], suffix="_top"))
    code += f",{NL}".join(args) + NL
    code += "){" + NL

    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        if len(n_data["shape"]) == 1:
            code += (
                f"copy_1d<F_TYPE, {n_data['shape'][0]}>({legalize_name(n, suffix='_top')}, {legalize_name(n)});"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"copy_2d<F_TYPE, {n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n, suffix='_top')}, {legalize_name(n)});"
                + NL
            )
        elif len(n_data["shape"]) == 3:
            code += (
                f"copy_3d<F_TYPE, {n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n, suffix='_top')}, {legalize_name(n)});"
                + NL
            )
        else:
            raise ERROR_UNSUPPORTED_ARRAY_SHAPE
    code += "}" + NL
    code += NL

    # code += "void copy_outputs_top(){" + NL
    code += "void copy_outputs(" + NL
    args = []
    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        args.append(array_str(n, n_data["shape"], suffix="_top"))
    code += f",{NL}".join(args) + NL
    code += "){" + NL

    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        if len(n_data["shape"]) == 1:
            code += (
                f"copy_1d<F_TYPE, {n_data['shape'][0]}>({legalize_name(n)}, {legalize_name(n, suffix='_top')});"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"copy_2d<F_TYPE, {n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}, {legalize_name(n, suffix='_top')});"
                + NL
            )
        elif len(n_data["shape"]) == 3:
            code += (
                f"copy_3d<F_TYPE, {n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}, {legalize_name(n, suffix='_top')});"
                + NL
            )
        else:
            raise ERROR_UNSUPPORTED_ARRAY_SHAPE
    code += "}"
    code += NL

    top_array_str = []
    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        top_array_str.append(array_str(n, n_data["shape"], suffix="_top"))
    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        top_array_str.append(array_str(n, n_data["shape"], suffix="_top"))

    code += "void model_top(" + NL
    code += f",{NL}".join(top_array_str) + NL
    code += "){" + NL
    code += NL

    # code += "copy_inputs_top();" + NL
    code += "copy_inputs(" + NL
    args = []
    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        args.append(legalize_name(n, suffix="_top"))
    code += f",{NL}".join(args) + NL
    code += ");" + NL
    code += NL

    code += "main_dataflow_region();" + NL
    code += NL

    # code += "copy_outputs_top();" + NL
    code += "copy_outputs(" + NL
    args = []
    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        args.append(legalize_name(n, suffix="_top"))
    code += f",{NL}".join(args) + NL
    code += ");" + NL
    code += NL

    code += "}"
    code += NL

    return code


def gen_model_declearation(model_name: str, model_graph: nx.DiGraph):
    graph_data = compute_graph_data(model_graph)

    code = ""
    code += f'extern "C" {{' + NL

    top_array_str = []
    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        top_array_str.append(array_str(n, n_data["shape"], suffix="_top"))
    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        top_array_str.append(array_str(n, n_data["shape"], suffix="_top"))

    code += "void model_top(" + NL
    code += f",\n{TAB}".join(top_array_str)
    code += ");"

    code += "};" + NL
    code += NL

    return code


def gen_model_h(model_name: str, model_graph: nx.DiGraph):
    ...


def gen_testbench(model_name: str, model_graph: nx.DiGraph):
    graph_data = compute_graph_data(model_graph)

    code = ""
    code += '#include "model.h"' + NL
    code += NL

    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        shape_str = "".join([f"[{s}]" for s in n_data["shape"]])
        code += f"float {legalize_name(n)}_top_in_float{shape_str};" + NL
        code += f"F_TYPE {legalize_name(n)}_top_in{shape_str};" + NL
        code += NL

    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        shape_str = "".join([f"[{s}]" for s in n_data["shape"]])
        code += f"float {legalize_name(n)}_top_out_float{shape_str};" + NL
        code += f"F_TYPE {legalize_name(n)}_top_out{shape_str};" + NL
        code += f"float {legalize_name(n)}_top_out_cast_back{shape_str};" + NL
        code += NL

    code += NL * 4

    code += "int main(){" + NL
    code += NL

    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        if len(n_data["shape"]) == 1:
            code += (
                f"load_data_1d<{n_data['shape'][0]}>(\"./testbench_data/{n_data['name']}.bin\", {legalize_name(n)}_top_in_float);"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"load_data_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>(\"./testbench_data/{n_data['name']}.bin\", {legalize_name(n)}_top_in_float);"
                + NL
            )
        elif len(n_data["shape"]) == 3:
            code += (
                f"load_data_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>(\"./testbench_data/{n_data['name']}.bin\", {legalize_name(n)}_top_in_float);"
                + NL
            )
        else:
            raise Exception("Invalid shape")
    code += NL

    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        if len(n_data["shape"]) == 1:
            code += (
                f"load_data_1d<{n_data['shape'][0]}>(\"./testbench_data/{n}.bin\", {legalize_name(n)}_top_out_float);"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"load_data_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>(\"./testbench_data/{n}.bin\", {legalize_name(n)}_top_out_float);"
                + NL
            )
        elif len(n_data["shape"]) == 3:
            code += (
                f"load_data_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>(\"./testbench_data/{n}.bin\", {legalize_name(n)}_top_out_float);"
                + NL
            )
        else:
            raise Exception("Invalid shape")
    code += NL

    for n, n_data in zip(graph_data["nodes_input"], graph_data["nodes_input_data"]):
        if len(n_data["shape"]) == 1:
            code += (
                f"cast_1d<{n_data['shape'][0]}>({legalize_name(n)}_top_in_float, {legalize_name(n)}_top_in);"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"cast_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}_top_in_float, {legalize_name(n)}_top_in);"
                + NL
            )
        elif len(n_data["shape"]) == 3:
            code += (
                f"cast_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}_top_in_float, {legalize_name(n)}_top_in);"
                + NL
            )
        else:
            raise Exception("Invalid shape")
    code += NL

    code += "model_top(" + NL
    input_array_args = [f"{legalize_name(n)}_top_in" for n in graph_data["nodes_input"]]
    output_array_args = [
        f"{legalize_name(n)}_top_out" for n in graph_data["nodes_output"]
    ]
    inout_array_args = input_array_args + output_array_args
    code += f",{NL}{TAB}".join(inout_array_args)
    code += ");" + NL
    code += NL

    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        if len(n_data["shape"]) == 1:
            code += (
                f"cast_1d<{n_data['shape'][0]}>({legalize_name(n)}_top_out, {legalize_name(n)}_top_out_cast_back);"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"cast_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}_top_out, {legalize_name(n)}_top_out_cast_back);"
                + NL
            )
        elif len(n_data["shape"]) == 3:
            code += (
                f"cast_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}_top_out, {legalize_name(n)}_top_out_cast_back);"
                + NL
            )
        else:
            raise Exception("Invalid shape")
    code += NL

    code += f"bool pass = true;" + NL
    code += f"float eps = 1e-3;" + NL
    code += NL
    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        if len(n_data["shape"]) == 1:
            code += (
                f"pass &= compare_data_1d<{n_data['shape'][0]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back, eps);"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"pass &= compare_data_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back, eps);"
                + NL
            )
        elif len(n_data["shape"]) == 3:
            code += (
                f"pass &= compare_data_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back, eps);"
                + NL
            )
        else:
            raise Exception("Invalid shape")
    code += NL

    code += f"if (pass) {{" + NL
    code += f'{TAB}printf("PASS\\n");' + NL
    code += f"}} else {{" + NL
    code += f'{TAB}printf("FAIL\\n");' + NL
    code += f"}}" + NL
    code += NL

    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        code += 'printf("======================================\\n");' + NL

        code += f'printf("{legalize_name(n)}_top_out_float = ");' + NL
        if len(n_data["shape"]) == 1:
            # code += f"print_array_1d({legalize_name(n)}_top_out_float);" + NL
            code += (
                f"print_array_1d<{n_data['shape'][0]}>({legalize_name(n)}_top_out_float);"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"print_array_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}_top_out_float);"
                + NL
            )
        else:
            raise Exception("Invalid shape")

        code += f'printf("{legalize_name(n)}_top_out_cast_back = ");' + NL
        if len(n_data["shape"]) == 1:
            code += (
                f"print_array_1d<{n_data['shape'][0]}>({legalize_name(n)}_top_out_cast_back);"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"print_array_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}_top_out_cast_back);"
                + NL
            )
        else:
            raise Exception("Invalid shape")

        code += 'printf("======================================\\n");' + NL

    code += NL

    code += "float testbench_mae = 0.0;" + NL
    code += NL

    for n, n_data in zip(graph_data["nodes_output"], graph_data["nodes_output_data"]):
        if len(n_data["shape"]) == 1:
            code += (
                f"testbench_mae += compute_mae_1d<{n_data['shape'][0]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back);"
                + NL
            )
        elif len(n_data["shape"]) == 2:
            code += (
                f"testbench_mae += compute_mae_2d<{n_data['shape'][0]}, {n_data['shape'][1]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back);"
                + NL
            )
        elif len(n_data["shape"]) == 3:
            code += (
                f"testbench_mae += compute_mae_3d<{n_data['shape'][0]}, {n_data['shape'][1]}, {n_data['shape'][2]}>({legalize_name(n)}_top_out_float, {legalize_name(n)}_top_out_cast_back);"
                + NL
            )
        else:
            raise Exception("Invalid shape")
    code += NL

    code += 'FILE *fp = fopen("./testbench_mae.txt", "w");' + NL
    code += 'fprintf(fp, "testbench_mae %.9f\\n", testbench_mae);' + NL
    code += "fclose(fp);" + NL
    code += NL

    code += "return 0;" + NL

    code += f"}}" + NL
    code += NL

    return code
