import numpy as np
import numpy.typing as npt
from enum import IntEnum
from itertools import tee
from typing import Dict, List, NamedTuple
from lightningsim.trace_file import (
    FIFOIOMetadata,
    ResolvedBlock,
    ResolvedEvent,
    ResolvedTrace,
    SubcallMetadata,
    Stream,
)
from scipy.sparse import coo_array, csc_array
from tqdm.auto import tqdm
from .util import connected_components, topological_sort


def build_dfg(trace: ResolvedTrace):
    u: List[int] = []
    v: List[int] = []
    delays: List[int] = []
    streams = list(trace.channel_depths.keys())
    io_by_stream: Dict[Stream, List[List[int]]] = {
        stream: [[] for io_type in range(max(StreamIOType) + 1)] for stream in streams
    }
    num_nodes = max(DFGEndpoint) + 1

    def get_subcall_init_stage(event: ResolvedEvent) -> int:
        return event.start_stage

    def get_stall_stage(event: ResolvedEvent) -> int:
        if event.type in ("axi_readreq",):
            return event.start_stage
        return event.end_stage

    def process(
        trace: List[ResolvedBlock],
        parent: int = DFGEndpoint.START,
        delay: int = 0,
    ) -> int:
        nonlocal num_nodes

        stage = 0
        subcalls: Dict[int, int] = {}
        end_stage = max(entry.end_stage for entry in trace)

        events_iter1, events_iter2 = tee(
            event for entry in trace for event in entry.events
        )
        subcall_inits = sorted(
            (event for event in events_iter1 if event.type == "call"),
            key=get_subcall_init_stage,
        )
        subcall_inits_ptr = 0
        stalls = sorted(
            (
                event
                for event in events_iter2
                if event.type in ("call", "fifo_write", "fifo_read")
            ),
            key=get_stall_stage,
        )
        stalls_ptr = 0
        done = False

        while not done:
            stalls_range_start = stalls_ptr
            if stalls_range_start < len(stalls):
                stalls_range_end = stalls_range_start + 1
                stall_stage = get_stall_stage(stalls[stalls_range_start])
                while stalls_range_end < len(stalls):
                    if get_stall_stage(stalls[stalls_range_end]) != stall_stage:
                        break
                    stalls_range_end += 1
            else:
                stalls_range_end = stalls_range_start
                stall_stage = end_stage
                done = True

            while subcall_inits_ptr < len(subcall_inits):
                subcall_init = subcall_inits[subcall_inits_ptr]
                subcall_init_stage = get_subcall_init_stage(subcall_init)
                if subcall_init_stage > stall_stage:
                    break
                region = subcall_init.instruction.basic_block.region
                start_delay = 1 if region.ii is None and region.dataflow is None else 0
                assert isinstance(subcall_init.metadata, SubcallMetadata)
                subcalls[id(subcall_init)] = process(
                    subcall_init.metadata.trace,
                    parent=parent,
                    delay=delay + subcall_init_stage - stage + start_delay,
                )
                subcall_inits_ptr += 1

            node = num_nodes
            num_nodes += 1
            u.append(parent)
            v.append(node)
            delays.append(delay + stall_stage - stage)
            for stall_idx in range(stalls_range_start, stalls_range_end):
                stall = stalls[stall_idx]
                if stall.type == "call":
                    u.append(subcalls.pop(id(stall)))
                    v.append(node)
                    delays.append(0)
                if stall.type == "fifo_write":
                    assert isinstance(stall.metadata, FIFOIOMetadata)
                    io_by_stream[stall.metadata.fifo][StreamIOType.WRITE].append(node)
                if stall.type == "fifo_read":
                    assert isinstance(stall.metadata, FIFOIOMetadata)
                    io_by_stream[stall.metadata.fifo][StreamIOType.READ].append(node)
            parent = node
            delay = 0

            stalls_ptr = stalls_range_end
            stage = stall_stage

        return parent

    u.append(process(trace.trace))
    v.append(DFGEndpoint.END)
    delays.append(0)
    num_edges = len(u)
    num_stream_writes = sum(
        len(stream_io[StreamIOType.WRITE]) for stream_io in io_by_stream.values()
    )

    for stream, stream_io in io_by_stream.items():
        if len(stream_io[StreamIOType.WRITE]) != len(stream_io[StreamIOType.READ]):
            raise ValueError(
                f"stream {stream} has "
                f"{len(stream_io[StreamIOType.WRITE])} writes and "
                f"{len(stream_io[StreamIOType.READ])} reads"
            )

    stream_io_arrays = {
        stream: np.array(stream_io) for stream, stream_io in io_by_stream.items()
    }
    u_array = np.concatenate(
        (
            np.array(u),
            *(stream_io_arrays[stream][StreamIOType.WRITE, :] for stream in streams),
        )
    )
    v_array = np.concatenate(
        (
            np.array(v),
            *(stream_io_arrays[stream][StreamIOType.READ, :] for stream in streams),
        )
    )
    delay_array = np.concatenate(
        (
            delays,
            np.ones((num_stream_writes,), dtype=np.int_),
        )
    )

    graph = coo_array((delay_array, (u_array, v_array)), shape=(num_nodes, num_nodes))
    w2r_offset = num_edges
    r2w_offset = num_edges + num_stream_writes

    return DFG(
        graph=graph,
        streams=streams,
        stream_io=stream_io_arrays,
        w2r_offset=w2r_offset,
        r2w_offset=r2w_offset,
    )


class DFGEndpoint(IntEnum):
    START = 0
    END = 1


class StreamIOType(IntEnum):
    WRITE = 0
    READ = 1


class StreamIO(NamedTuple):
    stream: Stream
    type: StreamIOType
    index: int


class DFG:
    def __init__(
        self,
        graph: coo_array,
        streams: List[Stream],
        stream_io: Dict[Stream, npt.NDArray[np.int_]],
        w2r_offset: int,
        r2w_offset: int,
    ):
        self.graph = graph
        self.streams = streams
        self.stream_io = stream_io
        self.w2r_offset = w2r_offset
        self.r2w_offset = r2w_offset

    def with_depths(self, depths: Dict[Stream, int]):
        u = np.concatenate(
            (
                self.graph.row[: self.r2w_offset],
                *(
                    self.stream_io[stream][StreamIOType.READ, : -depths[stream]]
                    for stream in self.streams
                ),
            )
        )
        v = np.concatenate(
            (
                self.graph.col[: self.r2w_offset],
                *(
                    self.stream_io[stream][StreamIOType.WRITE, depths[stream] :]
                    for stream in self.streams
                ),
            )
        )
        delay = np.concatenate(
            (
                self.graph.data,
                np.ones(
                    (
                        sum(
                            max(self.stream_io[stream].shape[1] - depths[stream], 0)
                            for stream in self.streams
                        ),
                    ),
                    dtype=np.int_,
                ),
            )
        )

        w2r_start = self.w2r_offset
        r2w_start = self.r2w_offset
        for stream in self.streams:
            depth = depths[stream]
            _, num_writes = self.stream_io[stream].shape
            w2r_end = w2r_start + num_writes
            r2w_end = r2w_start + max(num_writes - depth, 0)

            if depth <= 2:
                # implemented as shift register
                w2r_delay = 1
                r2w_delay = 1
            else:
                # implemented as RAM
                w2r_delay = 2
                r2w_delay = 1

            delay[w2r_start:w2r_end] = w2r_delay
            delay[r2w_start:r2w_end] = r2w_delay

            w2r_start = w2r_end
            r2w_start = r2w_end

        graph = coo_array((delay, (u, v)), shape=self.graph.shape)
        return DFG(
            graph=graph,
            streams=self.streams,
            stream_io=self.stream_io,
            w2r_offset=self.w2r_offset,
            r2w_offset=self.r2w_offset,
        )

    def has_cycle(self):
        num_nodes, _ = self.graph.shape
        num_components = connected_components(
            self.graph, connection="strong", return_labels=False
        )
        return num_components < num_nodes
    
    def has_cycle_approx(self):
        num_nodes, _ = self.graph.shape
        num_components = connected_components(
            self.graph, connection="strong", return_labels=False
        )
        return (num_components < num_nodes), (num_nodes-num_components)

    def get_cycle_participants(self) -> List[List[StreamIO]]:
        num_nodes, _ = self.graph.shape
        reverse_lookup_table: List[List[StreamIO]] = [[] for _ in range(num_nodes)]
        for stream in self.streams:
            for io_type in StreamIOType:
                for index, node in enumerate(self.stream_io[stream][io_type, :]):
                    reverse_lookup_table[node].append(StreamIO(stream, io_type, index))

        _, labels = connected_components(self.graph, connection="strong")
        _, unique_inverse, label_counts = np.unique(
            labels, return_inverse=True, return_counts=True
        )
        (participant_ids,) = np.nonzero(label_counts[unique_inverse] > 1)
        return [reverse_lookup_table[node] for node in participant_ids]

    def get_latency(self, show_progress=False):
        num_nodes, _ = self.graph.shape
        latency = np.zeros((num_nodes,), dtype=np.int_)

        if show_progress:
            print("(1/3) Converting to CSC format...")
        csc = csc_array(self.graph)

        if show_progress:
            print("(2/3) Computing topological sort...")
        sorted_nodes = topological_sort(
            self.graph,
            indices=DFGEndpoint.START,
            show_progress=show_progress,
        )

        if show_progress:
            print("(3/3) Computing latency...")
            sorted_nodes = tqdm(sorted_nodes)
        for node in sorted_nodes:
            pred_start = csc.indptr[node]
            pred_end = csc.indptr[node + 1]
            pred_latency = latency[csc.indices[pred_start:pred_end]]
            pred_delay = csc.data[pred_start:pred_end]
            latency[node] = np.amax(pred_latency + pred_delay, initial=0)

        return latency
