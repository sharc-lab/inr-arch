import networkx as nx
from tqdm import tqdm
from typing import Any, Dict, NamedTuple, Sequence, TypedDict
from .latency.engine import DFGBuilder, run
from .latency.lib import (
    ArrayStream,
    sequential,
    rebatch,
    select,
    repeat_singleton_dim_2d,
)


class _NodeDataBase(TypedDict):
    type: str
    shape: Sequence[int]


class NodeData(_NodeDataBase, total=False):
    block_size: int
    fn: str
    attrs: Dict[str, Any]


class _EdgeDataBase(TypedDict):
    edge_pos: int


class EdgeData(_EdgeDataBase, total=False):
    stream: ArrayStream


class Predecessor(NamedTuple):
    node: NodeData
    stream: ArrayStream


def legalize_name(fn_name: str, suffix: str = ""):
    return fn_name.replace(".", "_") + suffix


def update_node(G: nx.DiGraph, dfg: DFGBuilder, node: Any):
    data: NodeData = G.nodes[node]
    node_type = data["type"]
    shape = data["shape"]
    preds: Dict[int, Predecessor] = {
        edge_data["edge_pos"]: Predecessor(G.nodes[pred], edge_data["stream"])
        for pred, edge_data in G.pred[node].items()
    }

    def broadcast(array: ArrayStream) -> ArrayStream:
        if shape == array.shape:
            return array

        assert len(shape) == 2 and len(array.shape) == 2
        m_src, n_src = array.shape
        m_dst, n_dst = shape
        name = legalize_name(node, suffix="__temp_stream")
        broadcasted = ArrayStream(shape, array.block_size, name=name)

        if m_src == m_dst and n_src == 1:
            run(dfg, repeat_singleton_dim_2d(broadcasted, array, 1, n_dst))
            return broadcasted
        if n_src == n_dst and m_src == 1:
            run(dfg, repeat_singleton_dim_2d(broadcasted, array, 0, m_dst))
            return broadcasted
        raise ValueError(f"cannot broadcast shape {array.shape} to {shape}")

    output = None
    match node_type:
        case "input":
            block_size = data.setdefault("block_size", 1)
            name = legalize_name(node, suffix="_stream")
            output = ArrayStream(shape, block_size, name=name)
            run(dfg, output.write_all())

        case "output":
            data["block_size"] = preds[0].stream.block_size
            run(dfg, preds[0].stream.read_all())

        case "fn":
            assert "fn" in data
            assert "attrs" in data
            fn = data["fn"]
            match fn:
                case "Add" | "Cos" | "Sin" | "Neg" | "Mul" | "Unsqueeze" | "Select":
                    block_size = next(pred.stream.block_size for pred in preds.values())
                    assert all(
                        pred.stream.block_size == block_size for pred in preds.values()
                    )
                    data["block_size"] = block_size
                case "T" | "Mm":
                    block_size = data.setdefault(
                        "block_size",
                        next(pred.stream.block_size for pred in preds.values()),
                    )
                case _:
                    raise ValueError(f"unknown fn: {fn!r}")

            if fn == "Mm" and "contents" in data["attrs"]["self"]:
                const_shape = data["attrs"]["self"]["shape"]
                name = legalize_name(node, suffix="__in_0_stream")
                const_stream = ArrayStream(const_shape, block_size, name=name)
                run(dfg, const_stream.write_all())
                preds[0] = Predecessor(
                    {"type": "input", "shape": const_shape, "block_size": block_size},
                    const_stream,
                )

            name = legalize_name(node, suffix="__out_stream")
            output = ArrayStream(shape, block_size, name=name)

            match fn:
                case "Add" | "Cos" | "Sin" | "Neg" | "Mul":
                    run(
                        dfg,
                        output.write_all(
                            *(broadcast(pred.stream) for pred in preds.values())
                        ),
                    )
                case "Unsqueeze":
                    run(dfg, output.write_all(preds[0].stream))
                case "Select":
                    assert "attrs" in data
                    dim = data["attrs"]["dim"]
                    index = data["attrs"]["index"]
                    run(dfg, select(output, preds[0].stream, dim, index))
                case "T":
                    input_stream = preds[0].stream
                    run(
                        dfg,
                        sequential(output, input_stream, delay=input_stream.n),
                    )
                case "Mm":
                    input_1 = preds[0].stream
                    input_2 = preds[1].stream
                    m, n = input_1.shape
                    _, o = input_2.shape
                    run(
                        dfg,
                        sequential(output, input_1, input_2, delay=m * n * o),
                    )
                case _:
                    raise ValueError(f"unknown fn: {fn!r}")

        case "rebatch":
            try:
                out_block_size = data["block_size"]
            except KeyError:
                raise ValueError("rebatch node missing block_size") from None
            name = legalize_name(node, suffix="__out_stream")
            output = ArrayStream(shape, out_block_size, name=name)
            run(dfg, rebatch(output, preds[0].stream))

        case _:
            raise ValueError(f"unknown node type: {node_type!r}")

    if output is not None:
        outputs = [
            edge_data.setdefault(
                "stream",
                ArrayStream(
                    output.shape,
                    output.block_size,
                    name=name,
                ),
            )
            for edge_data, name in (
                (
                    edge_data,
                    legalize_name(succ, suffix=f"__in_{edge_data['edge_pos']}_stream"),
                )
                for succ, edge_data in G.succ[node].items()
            )
        ]
        run(dfg, output.read_all(*outputs))


def update_graph(G: nx.DiGraph):
    dfg_builder = DFGBuilder()
    total = len(G.nodes)
    for i, node in tqdm(
        enumerate(nx.algorithms.dag.topological_sort(G)),
        desc="Updating nodes",
        total=total,
    ):
        update_node(G, dfg_builder, node)
    print("Building the graph. Just a moment...")
    dfg = dfg_builder.build()
    print("Done!")
    return dfg


def set_batch_size(G: nx.DiGraph, node: Any, batch_size: int):
    data: NodeData = G.nodes[node]
    node_type = data["type"]
    assert node_type != "output", "cannot set batch size on output node"

    if node_type == "rebatch":
        (pred,) = G.pred[node]
        return set_batch_size(G, pred, batch_size)

    try:
        (rebatch,) = G.succ[node]
    except ValueError:
        pass
    else:
        if G.nodes[rebatch]["type"] == "rebatch":
            for succ, edge_data in G.succ[rebatch].items():
                G.add_edge(node, succ, **edge_data)
            G.remove_node(rebatch)

    if data.get("block_size") != batch_size:
        if node_type == "input" or (
            node_type == "fn" and data.get("fn") in ("T", "Mm")
        ):
            # rebatch node not needed
            data["block_size"] = batch_size
        else:
            # rebatch node needed
            rebatch = f"{node}_rebatch"
            G.add_node(
                rebatch,
                type="rebatch",
                shape=data["shape"],
                block_size=batch_size,
            )
            for succ, edge_data in G.succ[node].items():
                G.add_edge(rebatch, succ, **edge_data)
            G.remove_edges_from(G.out_edges(node))
            G.add_edge(node, rebatch, edge_pos=0)

    return update_graph(G)
