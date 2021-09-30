from typing import Callable, List, Tuple, Union, cast

import torch


def normalize_grid(
    xy_grid: torch.Tensor,
    min_percentile: float = 0.01,
    max_percentile: float = 0.99,
    relative_margin: float = 0.1,
) -> torch.Tensor:
    """
    Remove outliers from an xy coordinate grid tensor, and rescale it to [0, 1].

    Args:

        xy_grid (torch.tensor): The xy coordinate grid tensor to normalize,
            with a shape of: [n_points, n_axes].
        min_percentile (float, optional): The minimum percentile to use when
            normalizing the tensor. Value must be in the range [0, 1].
        max_percentile (float, optional): The maximum percentile to use when
            normalizing the tensor. Value must be in the range [0, 1].
        relative_margin (float, optional): The relative margin to use when
            normalizing the tensor.

    Returns:
        normalized_grid (torch.tensor): A normalized xy coordinate grid tensor.
    """

    assert xy_grid.dim() == 2
    assert 0.0 <= min_percentile <= 1.0
    assert 0.0 <= max_percentile <= 1.0

    mins = torch.quantile(xy_grid, min_percentile, dim=0)
    maxs = torch.quantile(xy_grid, max_percentile, dim=0)

    mins = mins - relative_margin * (maxs - mins)
    maxs = maxs + relative_margin * (maxs - mins)

    normalized_grid = torch.max(torch.min(xy_grid, maxs), mins)
    normalized_grid = normalized_grid - normalized_grid.min(0)[0]
    return normalized_grid / normalized_grid.max(0)[0]


def calc_grid_indices(
    xy_grid: torch.Tensor,
    grid_size: Tuple[int, int],
    x_extent: Tuple[float, float] = (0.0, 1.0),
    y_extent: Tuple[float, float] = (0.0, 1.0),
) -> List[List[torch.Tensor]]:
    """
    Create sets of grid cell indices of a specified size for an irregular grid.

    Args:

        xy_grid (torch.tensor): The xy coordinate grid activation samples, with a shape
            of: [n_points, 2].
        grid_size (Tuple[int, int]): The grid_size of grid cells to use. The grid_size
            variable should be in the format of: [width, height].
        x_extent (Tuple[float, float], optional): The x axis range to use.
        y_extent (Tuple[float, float], optional): The y axis range to use.

    Returns:
        indices (list of list of tensor): Grid cell indices for the irregular grid.
    """

    assert xy_grid.dim() == 2 and xy_grid.size(1) == 2

    #  Convert coordinates to bins
    x_bin = ((xy_grid[:, 0] - x_extent[0]) / (x_extent[1] - x_extent[0])) * grid_size[0]
    y_bin = ((xy_grid[:, 1] - y_extent[0]) / (y_extent[1] - y_extent[0])) * grid_size[1]

    indices: List[List[torch.Tensor]] = []
    for x in range(grid_size[0]):
        indice_bounds: List[torch.Tensor] = []
        for y in range(grid_size[1]):
            in_bounds_x = torch.logical_and(x <= x_bin, x_bin <= x + 1)
            in_bounds_y = torch.logical_and(y <= y_bin, y_bin <= y + 1)
            in_bounds_indices = torch.where(
                torch.logical_and(in_bounds_x, in_bounds_y)
            )[0]
            indice_bounds.append(in_bounds_indices)
        indices.append(indice_bounds)
    return indices


def extract_grid_vectors(
    grid_indices: List[List[torch.Tensor]],
    raw_activations: torch.Tensor,
    grid_size: Tuple[int, int],
    min_density: int = 8,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    """
    Create direction vectors for activation samples and grid indices. Grid cells
    without the minimum number of points as specified by min_density will be
    ignored.

    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/

    Args:

        grid_indices (list of list of torch.tensor): List of lists of grid indices to
            use.
        raw_activations (torch.tensor): Raw unmodified activation samples, with a shape
            of: [n_samples, n_channels].
        grid_size (Tuple[int, int]): The grid_size of grid cells to use. The grid_size
            variable should be in the format of: [width, height].
        min_density (int, optional): The minimum number of points for a cell to be
            counted.

    Returns:
        cells (torch.tensor): A tensor containing all the direction vector that were
            created.
        cell_coords (list of Tuple[int, int, int]): List of coordinates for grid
            spatial positions of each direction vector, and the number of samples used
            for the cell. The list for each cell is in the format of:
            [x_coord, y_coord, number_of_samples_used].
    """

    assert raw_activations.dim() == 2

    cell_coords: List[Tuple[int, int, int]] = []
    average_activations: List[torch.Tensor] = []
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            indices = grid_indices[x][y]
            if len(indices) >= min_density:
                average_activations.append(torch.mean(raw_activations[indices], 0))
                cell_coords.append((x, y, len(indices)))
    assert len(cell_coords) > 0, "No grid vectors were able to be created."
    return torch.stack(average_activations), cell_coords


def create_atlas_vectors(
    xy_grid: torch.Tensor,
    raw_activations: torch.Tensor,
    grid_size: Tuple[int, int],
    min_density: int = 8,
    normalize: bool = True,
    x_extent: Tuple[float, float] = (0.0, 1.0),
    y_extent: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    """
    Create direction vectors by splitting an irregular grid of activation samples into
    cells. Grid cells without the minimum number of points as specified by min_density
    will be ignored.

    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/

    Args:

        xy_grid (torch.tensor): The xy coordinate grid activation samples, with a shape
            of: [n_points, 2].
        raw_activations (torch.tensor): Raw unmodified activation samples, with a shape
            of: [n_samples, n_channels].
        grid_size (Tuple[int, int]): The size of grid cells to use. The grid_size
            variable should be in the format of: [width, height].
        min_density (int, optional): The minimum number of points for a cell to be
            counted.
        normalize (bool, optional): Whether or not to remove outliers from an xy
            coordinate grid tensor, and rescale it to [0, 1].
        x_extent (Tuple[float, float], optional): The x axis range to use.
        y_extent (Tuple[float, float], optional): The y axis range to use.

    Returns:
        grid_vecs (torch.tensor): A tensor containing all the direction vector that
            were created, stacked along the batch dimension.
        cell_coords (list of Tuple[int, int, int]): List of coordinates for grid
            spatial positions of each direction vector, and the number of samples used
            for the cell. The list for each cell is in the format of:
            [x_coord, y_coord, number_of_samples_used].
    """

    assert xy_grid.dim() == 2 and xy_grid.size(1) == 2
    assert raw_activations.dim() == 2

    if normalize:
        xy_grid = normalize_grid(xy_grid)
    indices = calc_grid_indices(
        xy_grid, grid_size, x_extent=x_extent, y_extent=y_extent
    )
    grid_vecs, vec_coords = extract_grid_vectors(
        indices, raw_activations, grid_size, min_density
    )
    return grid_vecs, vec_coords


def create_atlas(
    cells: Union[torch.Tensor, List[torch.Tensor]],
    coords: List[Tuple[int, int]],
    grid_size: Tuple[int, int],
    base_tensor: Callable = torch.ones,
) -> torch.Tensor:
    """
    Create an NCHW atlas grid image tensor from a set of NCHW image tensors and their
    corresponding grid coordinates.

    Args:

        cells (list of tensor or tensor): A list or stack of image tensors made with
            atlas direction vectors.
        coords (list of Tuple[int, int] or list of Tuple[int, int, int]): A list of
            coordinates to use for the atlas image tensors. The first 2 values in each
            coordinate list should be: [x, y, ...].
        grid_size (Tuple[int, int]): The size of grid cells to use. The grid_size
            variable should be in the format of: [width, height].
        base_tensor (Callable, optional): What to use for the atlas base tensor. Basic
            choices are: torch.ones or torch.zeros.

    Returns:
        atlas_canvas (torch.tensor): The full activation atlas visualization.
    """

    if torch.is_tensor(cells):
        assert cast(torch.Tensor, cells).dim() == 4
        cells = [c.unsqueeze(0) for c in cells]

    assert len(cells) == len(coords)
    assert all([c.shape == cells[0].shape for c in cells])
    assert all([c.device == cells[0].device for c in cells])
    assert cells[0].dim() == 4

    cell_b, cell_c, cell_h, cell_w = cells[0].shape
    atlas_canvas = base_tensor(
        cell_b,
        cell_c,
        cell_h * grid_size[1],
        cell_w * grid_size[0],
        device=cells[0].device,
    )
    for i, img in enumerate(cells):
        y = int(coords[i][1])
        x = int(coords[i][0])
        atlas_canvas[
            ...,
            (grid_size[1] - y - 1) * cell_h : (grid_size[1] - y) * cell_h,
            (grid_size[0] - x - 1) * cell_w : (grid_size[0] - x) * cell_w,
        ] = img.flip([3])
    return torch.flip(atlas_canvas, [3])


__all__ = [
    "normalize_grid",
    "calc_grid_indices",
    "extract_grid_vectors",
    "create_atlas_vectors",
    "create_atlas",
]
