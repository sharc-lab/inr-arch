import numpy as np
import numpy.typing as npt
from contextlib import nullcontext
from scipy.sparse import spmatrix, csr_array, csc_array
from typing import Literal, Tuple, Union, overload
from tqdm.auto import tqdm


# region typefix for scipy.sparse.csgraph.connected_components
@overload
def connected_components(
    csgraph: spmatrix,
    directed: bool = ...,
    connection: Literal["weak", "strong"] = ...,
    return_labels: Literal[True] = ...,
) -> Tuple[int, npt.NDArray[np.intc]]:
    ...


@overload
def connected_components(
    csgraph: spmatrix,
    directed: bool = ...,
    connection: Literal["weak", "strong"] = ...,
    return_labels: Literal[False] = ...,
) -> int:
    ...


def connected_components(
    csgraph, directed=..., connection=..., return_labels=...
) -> Union[Tuple[int, npt.NDArray[np.intc]], int]:
    # prevents Pylance from complaining about missing implementation
    assert False, "connected_components should be imported from scipy.sparse.csgraph"


from scipy.sparse.csgraph import connected_components

# endregion


# region typefix for scipy.sparse.csgraph.shortest_path
@overload
def shortest_path(
    csgraph: spmatrix,
    method: Literal["auto", "FW", "D", "BF", "J"] = ...,
    directed: bool = ...,
    return_predecessors: Literal[False] = ...,
    unweighted: bool = ...,
    overwrite: bool = ...,
    indices: Union[npt.NDArray, int, None] = ...,
) -> npt.NDArray:
    ...


@overload
def shortest_path(
    csgraph: spmatrix,
    method: Literal["auto", "FW", "D", "BF", "J"] = ...,
    directed: bool = ...,
    return_predecessors: Literal[True] = ...,
    unweighted: bool = ...,
    overwrite: bool = ...,
    indices: Union[npt.NDArray, int, None] = ...,
) -> Tuple[npt.NDArray, npt.NDArray]:
    ...


def shortest_path(
    csgraph: spmatrix,
    method: Literal["auto", "FW", "D", "BF", "J"] = "auto",
    directed: bool = True,
    return_predecessors: bool = False,
    unweighted: bool = False,
    overwrite: bool = False,
    indices: Union[npt.NDArray, int, None] = None,
) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    # prevents Pylance from complaining about missing implementation
    assert False, "shortest_path should be imported from scipy.sparse.csgraph"


from scipy.sparse.csgraph import shortest_path

# endregion


def topological_sort(
    csgraph: spmatrix,
    indices: Union[npt.NDArray[np.intp], int, None] = None,
    show_progress: bool = False,
):
    match indices:
        case int():
            indices = np.array([indices], dtype=np.intp)
        case None:
            (indices,) = np.nonzero(np.diff(csc_array(csgraph).indptr) == 0)

    n, _ = csgraph.shape
    assert (
        connected_components(
            csgraph,
            directed=True,
            connection="strong",
            return_labels=False,
        )
        == n
    ), "graph contains a cycle"

    csr = csr_array(csgraph)
    visited = np.zeros((n,), dtype=np.bool_)
    sorted_indices = np.zeros((n,), dtype=np.intp)
    num_sorted = 0

    stack_size = indices.size
    stack_indices = np.concatenate(
        (indices, np.zeros_like(indices, shape=(n - stack_size,)))
    )
    stack_is_postorder = np.zeros_like(stack_indices, dtype=np.bool_)

    with tqdm(total=n) if show_progress else nullcontext() as progress:
        while stack_size:
            stack_size -= 1
            node = stack_indices[stack_size]
            is_postorder = stack_is_postorder[stack_size]

            if not is_postorder:
                if visited[node]:
                    continue

                stack_indices[stack_size] = node
                stack_is_postorder[stack_size] = True
                stack_size += 1

                for neighbor in csr.indices[csr.indptr[node] : csr.indptr[node + 1]]:
                    stack_indices[stack_size] = neighbor
                    stack_is_postorder[stack_size] = False
                    stack_size += 1

            else:
                visited[node] = True
                sorted_indices[num_sorted] = node
                num_sorted += 1
                if progress is not None:
                    progress.update(1)

    return np.flip(sorted_indices[:num_sorted], axis=0)


__all__ = ["connected_components", "shortest_path"]
