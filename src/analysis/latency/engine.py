import numpy as np
import numpy.typing as npt
from enum import IntEnum
from scipy.sparse import coo_array
from typing import (
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    NewType,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
    Union,
    overload,
)
from .util import connected_components, shortest_path

DFGNode: TypeAlias = Union["DFGEndpoint", "StreamIO"]
DFGEndpoint = NewType("DFGEndpoint", object)
DFG_START = DFGEndpoint(object())
DFG_END = DFGEndpoint(object())


class DFGBuilder:
    def __init__(self):
        self.streams: Set[Stream] = set()
        self.u: List[DFGNode] = []
        self.v: List[DFGNode] = []
        self.delay: List[int] = []

    def add_node(self, stream: "Stream"):
        self.streams.add(stream)

    def add_edge(self, u: DFGNode, v: DFGNode, delay: int = 0):
        self.u.append(u)
        self.v.append(v)
        self.delay.append(delay)

    def build(self):
        nodes = DFGNodeTable(self.streams)
        u = np.concatenate(
            (
                np.fromiter(
                    map(nodes.lookup, self.u), dtype=np.int64, count=len(self.u)
                ),
                *(nodes.range(stream, StreamIOType.WRITE) for stream in nodes.streams),
            )
        )
        v = np.concatenate(
            (
                np.fromiter(
                    map(nodes.lookup, self.v), dtype=np.int64, count=len(self.v)
                ),
                *(nodes.range(stream, StreamIOType.READ) for stream in nodes.streams),
            )
        )
        delay = np.concatenate(
            (
                np.array(self.delay, dtype=np.int64),
                np.ones(
                    (sum(stream.num_writes for stream in nodes.streams),),
                    dtype=np.int64,
                ),
            )
        )
        return DFG(coo_array((delay, (u, v)), shape=(len(nodes), len(nodes))), nodes)


class DFG:
    def __init__(self, graph: coo_array, nodes: "DFGNodeTable"):
        self.graph = graph
        self.nodes = nodes

    def with_depths(self, depths: Dict[str, int]):
        u = np.concatenate(
            (
                self.graph.row,
                *(
                    self.nodes.range(
                        stream,
                        StreamIOType.READ,
                        stream.num_writes - depths.get(stream.name, 2),
                    )
                    for stream in self.nodes.streams
                ),
            )
        )
        v = np.concatenate(
            (
                self.graph.col,
                *(
                    self.nodes.range(
                        stream,
                        StreamIOType.WRITE,
                        depths.get(stream.name, 2),
                        stream.num_writes,
                    )
                    for stream in self.nodes.streams
                ),
            )
        )
        delay = np.concatenate(
            (
                self.graph.data,
                np.ones(
                    (
                        sum(
                            max(stream.num_writes - depths.get(stream.name, 2), 0)
                            for stream in self.nodes.streams
                        ),
                    ),
                    dtype=np.int64,
                ),
            )
        )

        # for stream in self.nodes.streams:
        #     depth = depths.get(stream.name, 2)
        #     if depth <= 2:
        #         # implemented as shift register
        #         w2r_delay = 1
        #         r2w_delay = 1
        #     else:
        #         # implemented as RAM
        #         w2r_delay = 2
        #         r2w_delay = 1

        #     write_start = self.nodes.forward_table[stream][StreamIOType.WRITE]
        #     write_end = write_start + stream.num_writes
        #     read_start = self.nodes.forward_table[stream][StreamIOType.READ]
        #     read_end = read_start + stream.num_writes

        #     if w2r_delay != 1:
        #         w2r_mask = (
        #             (u >= write_start)
        #             & (u < write_end)
        #             & (v >= read_start)
        #             & (v < read_end)
        #         )
        #         delay[w2r_mask] = w2r_delay

        #     if r2w_delay != 1:
        #         r2w_mask = (
        #             (u >= read_start)
        #             & (u < read_end)
        #             & (v >= write_start)
        #             & (v < write_end)
        #         )
        #         delay[r2w_mask] = r2w_delay

        return DFG(coo_array((delay, (u, v)), shape=self.graph.shape), self.nodes)

    def has_cycle(self):
        num_components = connected_components(
            self.graph, connection="strong", return_labels=False
        )
        return num_components < len(self.nodes)

    def get_cycle_participants(self):
        _, labels = connected_components(self.graph, connection="strong")
        _, unique_inverse, label_counts = np.unique(
            labels, return_inverse=True, return_counts=True
        )
        (participant_ids,) = np.nonzero(label_counts[unique_inverse] > 1)
        return self.nodes.lookup_reverse(participant_ids)

    def get_latency(self):
        negative_weighted_graph = coo_array(
            (-self.graph.data, (self.graph.row, self.graph.col)),
            shape=self.graph.shape,
        )
        start = self.nodes.lookup(DFG_START)
        end = self.nodes.lookup(DFG_END)
        distances = shortest_path(
            negative_weighted_graph,
            method="J",
            indices=start,
        )
        return -int(distances[end])


class DFGNodeTable:
    def __init__(self, streams: Iterable["Stream"]):
        self.streams = list(streams)
        self.forward_table: Dict[Stream, Tuple[int, ...]] = {
            stream: () for stream in self.streams
        }
        self.reverse_table = np.zeros(
            (len(self.streams), len(StreamIOType)), dtype=np.int64
        )
        offset = 2
        for i, stream in enumerate(self.streams):
            assert stream.num_writes == stream.num_reads, (
                f"stream {stream.name} has mismatched read/write counts "
                f"({stream.num_writes} writes vs. {stream.num_reads} reads)"
            )
            offsets_by_type = [0 for _ in StreamIOType]
            for io_type in StreamIOType:
                offsets_by_type[io_type] = offset
                offset += stream.num_writes
            self.forward_table[stream] = tuple(offsets_by_type)
            self.reverse_table[i, :] = offsets_by_type
        self.count = offset

    def lookup(self, node: DFGNode):
        match node:
            case StreamIO(stream=stream, type=io_type, index=index):
                return self.forward_table[stream][io_type] + index
            case _ if node is DFG_START:
                return 0
            case _ if node is DFG_END:
                return 1
            case _:
                raise ValueError(f"invalid DFG node: {node!r}")

    @overload
    def lookup_reverse(self, node_id: int) -> "StreamIO":
        ...

    @overload
    def lookup_reverse(self, node_id: npt.NDArray[np.int64]) -> List["StreamIO"]:
        ...

    def lookup_reverse(self, node_id: Union[int, npt.NDArray[np.int64]]):
        # TODO: support DFG_START and DFG_END
        i = np.searchsorted(self.reverse_table.ravel(), node_id, side="right") - 1
        if np.any(i < 0):
            raise ValueError(f"node ID out of range")
        stream_idx, io_type_idx = np.unravel_index(i, self.reverse_table.shape)
        index = node_id - self.reverse_table[stream_idx, io_type_idx]
        try:
            zip_ = zip(stream_idx, io_type_idx, index)
        except TypeError:
            # Pylance doesn't understand that (stream_idx, io_type_idx, index) are ints if node_id is an int
            return StreamIO(self.streams[stream_idx], StreamIOType(io_type_idx), index)  # type: ignore
        else:
            return [
                StreamIO(self.streams[stream_idx], StreamIOType(io_type_idx), index)
                for stream_idx, io_type_idx, index in zip_
            ]

    def range(self, stream: "Stream", io_type: "StreamIOType", *start_stop_step: int):
        offset = self.forward_table[stream][io_type]
        start = 0
        stop = stream.num_writes
        step = 1
        match start_stop_step:
            case ():
                pass
            case (stop,):
                pass
            case (start, stop):
                pass
            case (start, stop, step):
                pass
            case _:
                raise ValueError("expected 0-3 arguments for start, stop, step")
        return np.arange(offset + start, offset + stop, step, dtype=np.int64)

    def __len__(self):
        return self.count

    def __getitem__(self, item: Union[DFGNode, int, npt.NDArray[np.int64]]):
        match item:
            case StreamIO() as node:
                return self.lookup(node)
            case int() as node_id:
                return self.lookup_reverse(node_id)
            case np.ndarray() as node_ids:
                return self.lookup_reverse(node_ids)
            case _ if item in (DFG_START, DFG_END):
                return self.lookup(item)
            case _:
                raise TypeError(f"expected StreamIO or int, got {type(item)}")


class Stream:
    def __init__(self, name: str):
        self.name = name
        self.num_writes = 0
        self.num_reads = 0

    def read(self):
        index = self.num_reads
        self.num_reads += 1
        return StreamIO(self, StreamIOType.READ, index)

    def write(self):
        index = self.num_writes
        self.num_writes += 1
        return StreamIO(self, StreamIOType.WRITE, index)


class StreamIO(NamedTuple):
    stream: Stream
    type: "StreamIOType"
    index: int


class StreamIOType(IntEnum):
    READ = 0
    WRITE = 1


class Sleep(NamedTuple):
    duration: int


class UnboundEdge(NamedTuple):
    node: DFGNode
    delay: int


Process: TypeAlias = Generator["ProcessYieldType", "ProcessSendType", None]
ProcessHandle: TypeAlias = Sequence[UnboundEdge]
ProcessYieldType: TypeAlias = Union[StreamIO, Sleep, Process, ProcessHandle]
ProcessSendType: TypeAlias = Union[None, ProcessHandle]


def run(dfg: DFGBuilder, process: Process, parent: ProcessHandle = ()) -> ProcessHandle:
    is_top = not parent
    next_edges = list(parent)
    send_val: ProcessSendType = None

    if is_top:
        next_edges.append(UnboundEdge(DFG_START, 0))

    while True:
        try:
            yield_val = process.send(send_val)
        except StopIteration:
            break

        send_val = None
        match yield_val:
            case StreamIO(stream=stream) as node:
                dfg.add_node(stream)
                for source, delay in next_edges:
                    dfg.add_edge(source, node, delay=delay)
                next_edges = [UnboundEdge(node, 0)]

            case Sleep(duration):
                if duration != 0:
                    next_edges = [
                        UnboundEdge(node, delay + duration)
                        for node, delay in next_edges
                    ]

            case [*handle]:
                next_edges.extend(handle)

            case subprocess:
                send_val = run(dfg, subprocess, next_edges)

    if is_top:
        for edge in next_edges:
            dfg.add_edge(edge.node, DFG_END, delay=edge.delay)
        next_edges = ()

    return tuple(next_edges)
