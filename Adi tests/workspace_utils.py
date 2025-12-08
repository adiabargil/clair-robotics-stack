import numpy as np


stack_position_r2frame = (-0.3614, 0.1927)
# corner of tale of ur5e_2, where I usually stack blocks for collection by the robot

workspace_x_lims_default = (-0.9, -0.54)
workspace_y_lims_default = (-1.0, -0.55)

goal_tower_position = [-0.45, -1.15]


def valid_position(x, y, block_positions, min_dist):
    """
    Check if the position x, y is valid, i.e. not too close to any of the block_positions
    """
    if x is None or y is None:
        return False
    for block_pos in block_positions:
        if np.linalg.norm(np.array(block_pos) - np.array([x, y])) < min_dist:
            return False
    return True


def sample_block_positions_uniform(n_blocks, workspace_x_lims=workspace_x_lims_default,
                                   workspace_y_lims=workspace_y_lims_default, min_dist=0.07):
    """
    sample n_blocks positions within the workspace limits, spaced at least by 0.05m in each axis
    """
    block_positions = []
    for i in range(n_blocks):
        x = None
        y = None
        while valid_position(x, y, block_positions, min_dist=min_dist) is False:
            x = np.random.uniform(*workspace_x_lims)
            y = np.random.uniform(*workspace_y_lims)
        block_positions.append([x, y])

    return block_positions


def sample_block_positions_from_dists(blocks_dist, min_dist=0.07):
    """
    sample n_blocks positions within the workspace limits, spaced at least by 0.05m in each axis
    """
    block_positions = []
    for b in blocks_dist:
        x = None
        y = None
        while valid_position(x, y, block_positions, min_dist=min_dist) is False:
            x, y = b.sample(1)[0]

        block_positions.append([x, y])

    return block_positions
