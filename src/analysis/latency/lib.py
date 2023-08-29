from math import prod
from typing import List, Optional, Sequence

from .engine import Process, ProcessHandle, Sleep, Stream


class ArrayStream(Stream):
    def __init__(self, shape: Sequence[int], block_size=1, *, name: str):
        super().__init__(name)
        self.shape = shape
        self.block_size = block_size

    @property
    def n(self):
        return prod(self.shape)

    @property
    def num_blocks(self):
        return (self.n + self.block_size - 1) // self.block_size

    def write_all(self, *from_inputs: Stream) -> Process:
        for i in range(self.num_blocks):
            for input in from_inputs:
                yield input.read()
            yield Sleep(1)
            yield self.write()

    def read_all(self, *to_outputs: Stream) -> Process:
        for i in range(self.num_blocks):
            yield self.read()
            yield Sleep(1)
            for output in to_outputs:
                yield output.write()

    def __repr__(self):
        return f"<ArrayStream {self.name!r} {'x'.join(str(dim) for dim in self.shape)}\u00F7{self.block_size}>"


class AutoStreamWriter:
    def __init__(self, output: ArrayStream):
        self.output = output
        self.num_buffered = 0

    def write(self):
        self.num_buffered += 1
        if self.num_buffered == self.output.block_size:
            yield from self.flush()

    def flush(self):
        yield self.output.write()
        self.num_buffered = 0

    def flush_if_not_empty(self):
        if self.num_buffered > 0:
            yield from self.flush()


def sequential(output: ArrayStream, *inputs: ArrayStream, delay: int = 0) -> Process:
    subcalls: List[ProcessHandle] = []
    for input in inputs:
        subcall = yield input.read_all()
        assert subcall is not None
        subcalls.append(subcall)
    for subcall in subcalls:
        yield subcall

    yield Sleep(delay)
    subcall = yield output.write_all()
    assert subcall is not None
    yield subcall


def select(output: ArrayStream, input: ArrayStream, dim: int, index: int) -> Process:
    chunk_size = prod(input.shape[dim + 1 :])
    start_index = index * chunk_size
    chunk_stride = prod(input.shape[dim:])

    current_linear_index = 0
    total_output_count = 0
    asw = AutoStreamWriter(output)
    for i in range(input.num_blocks):
        yield input.read()
        for j in range(input.block_size):
            yield Sleep(1)
            if total_output_count < output.n:
                past_start_index = current_linear_index >= start_index
                i = (current_linear_index - start_index) // chunk_stride
                chunk_start = start_index + i * chunk_stride
                chunk_end = chunk_start + chunk_size
                inside_chunk = (chunk_start <= current_linear_index) and (
                    current_linear_index < chunk_end
                )

                if past_start_index and inside_chunk:
                    yield from asw.write()
                    total_output_count += 1
                current_linear_index += 1

    yield Sleep(1)
    yield from asw.flush_if_not_empty()


def rebatch(output: ArrayStream, input: ArrayStream) -> Process:
    asw = AutoStreamWriter(output)
    for i in range(input.num_blocks):
        yield input.read()
        for j in range(input.block_size):
            yield Sleep(1)
            yield from asw.write()
    yield Sleep(1)
    yield from asw.flush_if_not_empty()


def repeat_singleton_dim_2d(
    output: ArrayStream, input: ArrayStream, dim: int, num: int
) -> Process:
    non_singleton_dim = 1 if dim == 0 else 0
    non_singleton_dim_size = input.shape[non_singleton_dim]

    yield from input.read_all()

    asw = AutoStreamWriter(output)
    for i in range(num):
        for j in range(non_singleton_dim_size):
            yield from asw.write()
            yield Sleep(1)
    yield from asw.flush_if_not_empty()
