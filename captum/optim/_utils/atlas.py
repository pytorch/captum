from typing import List, Tuple

import torch


def grid_indices(
    tensor: torch.Tensor,
    size: Tuple[int, int] = (8, 8),
    x_extent: Tuple[float, float] = (0.0, 1.0),
    y_extent: Tuple[float, float] = (0.0, 1.0),
) -> List[List[torch.Tensor]]:
    """
    Create grid cells of a specified size for an irregular grid.
    """

    assert tensor.dim() == 2 and tensor.size(1) == 2
    x_coords = ((tensor[:, 0] - x_extent[0]) / (x_extent[1] - x_extent[0])) * size[1]
    y_coords = ((tensor[:, 1] - y_extent[0]) / (y_extent[1] - y_extent[0])) * size[0]

    x_list = []
    for x in range(size[1]):
        y_list = []
        for y in range(size[0]):
            in_bounds_x = torch.logical_and(x <= x_coords, x_coords <= x + 1)
            in_bounds_y = torch.logical_and(y <= y_coords, y_coords <= y + 1)
            in_bounds_indices = torch.where(
                torch.logical_and(in_bounds_x, in_bounds_y)
            )[0]
            y_list.append(in_bounds_indices)
        x_list.append(y_list)
    return x_list


def normalize_grid(
    x: torch.Tensor,
    min_percentile: float = 0.01,
    max_percentile: float = 0.99,
    relative_margin: float = 0.1,
) -> torch.Tensor:
    """
    Remove outliers and rescale grid to [0,1].
    """

    assert x.dim() == 2 and x.size(1) == 2
    mins = torch.quantile(x, min_percentile, dim=0)
    maxs = torch.quantile(x, max_percentile, dim=0)

    # add margins
    mins = mins - relative_margin * (maxs - mins)
    maxs = maxs + relative_margin * (maxs - mins)

    clipped = torch.max(torch.min(x, maxs), mins)
    clipped = clipped - clipped.min(0)[0]
    return clipped / clipped.max(0)[0]


def extract_grid_vectors(
    grid: List[List[torch.Tensor]],
    activations: torch.Tensor,
    size: Tuple[int, int] = (8, 8),
    min_density: int = 8,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Create direction vectors.
    """

    cell_coords = []
    average_activations = []
    for x in range(size[1]):
        for y in range(size[0]):
            indices = grid[x][y]
            if len(indices) >= min_density:
                average_activations.append(torch.mean(activations[indices], 0))
                cell_coords.append((x, y))
    return torch.stack(average_activations), cell_coords


def create_atlas_vectors(
    tensor: torch.Tensor,
    activations: torch.Tensor,
    size: Tuple[int, int] = (8, 8),
    min_density: int = 8,
    normalize: bool = True,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Create direction vectors by splitting an irregular grid into cells.
    """

    assert tensor.dim() == 2 and tensor.size(1) == 2
    if normalize:
        tensor = normalize_grid(tensor)
    indices = grid_indices(tensor, size)
    grid_vecs, vec_coords = extract_grid_vectors(
        indices, activations, size, min_density
    )
    return grid_vecs, vec_coords


def create_atlas(
    cells: List[torch.Tensor],
    coords: List[List[torch.Tensor]],
    grid_size: Tuple[int, int] = (8, 8),
) -> torch.Tensor:
    cell_h, cell_w = cells[0].shape[2:]
    canvas = torch.ones(1, 3, cell_h * grid_size[0], cell_w * grid_size[1])
    for i, img in enumerate(cells):
        y = int(coords[i][0])
        x = int(coords[i][1])
        canvas[
            ...,
            (grid_size[0] - x - 1) * cell_h : (grid_size[0] - x) * cell_h,
            y * cell_w : (y + 1) * cell_w,
        ] = img
    return canvas
