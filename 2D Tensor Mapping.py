import torch
from matplotlib import pyplot as plt


def min_dist(points, res):
    """Calculates a 2D tensor with the minimum distance from each pixel to data.

    Inputs:
      * points: a python list of 2D coordinates, normalized in the range [0,1]
      * res: the resolution of the output tensor.
    Returns:
      A res x res square tensor with floating point values corresponding to the
      Euclidean distance to the closest point in points.
    """
    # Convert the points to a tensor
    points = torch.tensor(points, dtype=torch.float32)

    # Create a grid of coordinates (resolution x resolution)
    grid_x = torch.linspace(0, 1, res).repeat(res, 1)  # (res, res)
    grid_y = torch.linspace(0, 1, res).repeat(res, 1).T  # (res, res)

    # Stack the grid coordinates into a 2D tensor (res x res x 2)
    grid = torch.stack([grid_x, grid_y], dim=-1)

    # Expand points to the same shape as the grid
    # (num_points, res, res, 2)
    expanded_points = points.unsqueeze(1).unsqueeze(1).expand(-1, res, res, -1)

    # Compute the Euclidean distance between each pixel and each point
    diff = grid.unsqueeze(0) - expanded_points  # (num_points, res, res, 2)
    dist = torch.norm(diff, dim=-1)  # (num_points, res, res)

    # Find the minimum distance to any point for each pixel
    min_dist = torch.min(dist, dim=0)[0]  # (res, res)

    return min_dist

# Case 1:
distance_to_data = min_dist([[0.4, 0.3], [0.6, 0.7]], 20)
plt.figure(figsize=(8, 8))
plt.imshow(distance_to_data)

# Case 2:
distance_to_data = min_dist([[0.4, 0.3], [0.6, 0.7], [0.3, 0.8], [0.5, 0.2]], 256)
plt.figure(figsize=(8, 8))
plt.imshow(distance_to_data)

