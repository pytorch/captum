from typing import List, Tuple

import torch


def grid_indices(
    tensor: torch.Tensor,
    grid_size: Tuple[int, int],
    x_extent: Tuple[float, float] = (0.0, 1.0),
    y_extent: Tuple[float, float] = (0.0, 1.0),
) -> List[List[torch.Tensor]]:
    """
    Create grid cells of a specified size for an irregular grid.

    Args:
        tensor (torch.tensor): xy coordinate tensor to extract grid
            indices from.
        grid_size (Tuple[int, int]): The size of grid cells to use.
        x_extent (Tuple[float, float], optional): The x extent to use.
        y_extent (Tuple[float, float], optional): The y extent to use.
    Returns:
        indices (list of list of tensor): Grid cell indices for the
            irregular grid.
    """

    assert tensor.dim() == 2 and tensor.size(1) == 2
    x_coords = ((tensor[:, 0] - x_extent[0]) / (x_extent[1] - x_extent[0])) * grid_size[
        1
    ]
    y_coords = ((tensor[:, 1] - y_extent[0]) / (y_extent[1] - y_extent[0])) * grid_size[
        0
    ]

    x_list = []
    for x in range(grid_size[1]):
        y_list = []
        for y in range(grid_size[0]):
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
    Remove outliers and rescale tensor to [0,1].

    Args:
        x (torch.tensor): Tensor to normalize.
        min_percentile (float, optional): The minamum percentile to
            use when normalizing the tensor.
        max_percentile (float, optional): The maximum percentile to
            use when normalizing the tensor.
        relative_margin (float, optional): The relative margin to use
            when normalizing the tensor.
    Returns:
        clipped (torch.tensor): A normalized tensor.
    """

    assert x.dim() == 2 and x.size(1) == 2
    mins = torch.quantile(x, min_percentile, dim=0)
    maxs = torch.quantile(x, max_percentile, dim=0)

    mins = mins - relative_margin * (maxs - mins)
    maxs = maxs + relative_margin * (maxs - mins)

    clipped = torch.max(torch.min(x, maxs), mins)
    clipped = clipped - clipped.min(0)[0]
    return clipped / clipped.max(0)[0]


def extract_grid_vectors(
    grid: List[List[torch.Tensor]],
    activations: torch.Tensor,
    grid_size: Tuple[int, int],
    min_density: int = 8,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Create direction vectors for activation samples and grid indices.

    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/

    Args:
        grid (torch.tensor): List of lists of grid indices to use.
        activations (torch.tensor): Raw activation samples.
        grid_size (Tuple[int, int]): The grid_size of grid cells to use.
        min_density (int, optional): The minamum number of points for a
            cell to counted.
    Returns:
        cells (torch.tensor): A tensor containing all the direction vector
            that were created.
        cell_coords (List[Tuple[int, int]]): List of coordinates for each of
            the direction vectors.
    """

    assert activations.dim() == 2

    cell_coords = []
    average_activations = []
    for x in range(grid_size[1]):
        for y in range(grid_size[0]):
            indices = grid[x][y]
            if len(indices) >= min_density:
                average_activations.append(torch.mean(activations[indices], 0))
                cell_coords.append((x, y))
    return torch.stack(average_activations), cell_coords


def create_atlas_vectors(
    tensor: torch.Tensor,
    activations: torch.Tensor,
    grid_size: Tuple[int, int],
    min_density: int = 8,
    normalize: bool = True,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Create direction vectors by splitting an irregular grid of activation samples
    into cells.

    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/

    Args:
        tensor (torch.tensor): The dimensionality reduced activation samples.
        activations (torch.tensor): Raw activation samples.
        grid_size (Tuple[int, int]): The size of grid cells to use.
        min_density (int, optional): The minamum number of points for a cell to counted.
        normalize (bool, optional): Whether to normalize the dimensionality
            reduced activation samples to between [0,1] & to remove outliers.
    Returns:
        grid_vecs (torch.tensor): A tensor containing all the direction vector
            that were created.
        vec_coords (List[Tuple[int, int]]): List of coordinates for each of
            the direction vectors.
    """

    assert tensor.dim() == 2 and tensor.size(1) == 2
    assert activations.dim() == 2
    assert activations.shape[0] == tensor.shape[0]

    if normalize:
        tensor = normalize_grid(tensor)
    indices = grid_indices(tensor, grid_size)
    grid_vecs, vec_coords = extract_grid_vectors(
        indices, activations, grid_size, min_density
    )
    return grid_vecs, vec_coords


def create_atlas(
    cells: List[torch.Tensor],
    coords: List[Tuple[int, int]],
    grid_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Create atlas grid from visualization imags with coordinates.

    Args:
        cells (list of tensor): A list of visualizations made using atlas
            direction vectors.
        coords (list of Tuple[int, int]): A list of coordinates to use for
            the visualizations.
        grid_size (Tuple[int, int]): The size of grid cells used to create
            the visualization direction vectors.
    Returns:
        canvas (torch.tensor): The full activation atlas visualization.
    """

    assert sum([c.dim() for c in cells]) == 4 * len(cells)
    assert all([c.shape == cells[0].shape for c in cells])
    assert len(cells) == len(coords)

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
