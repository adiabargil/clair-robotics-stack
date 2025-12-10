import numpy as np
import typer
import time
import matplotlib.pyplot as plt
from typing import Optional
from clair_robotics_stack.ur.lab_setup.manipulation.manipulation_controller_2fg import ManipulationController2FG
from clair_robotics_stack.camera.realsense_camera import RealsenseCamera
from clair_robotics_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from clair_robotics_stack.ur.lab_setup.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from clair_robotics_stack.vision.utils import *
from clair_robotics_stack.vision.object_detection import ObjectDetection
from klampt.math import se3


# --- Constants ---
WORKSPACE_X_LIMS_DEFAULT = (-0.9, -0.54)
WORKSPACE_Y_LIMS_DEFAULT = (-1.0, -0.55)

# Robot motion parameters
DEFAULT_SPEED = 2.0
DEFAULT_ACCELERATION = 1.5
START_HEIGHT = 0.1
DEPTH_WINDOW_SIZE = 2
CAMERA_STABILIZATION_DELAY = 0.5  # seconds to wait after camera movement before capturing

Q_FOR_DETECTION = np.array(
    [0.839488685131073, -0.9910658162883301, 1.5102737585650843, -2.756538530389303, -2.0820935408221644,
     0.6501701474189758])


FIXED_PICKUP_POINT = [0.04, -0.7, 0.1]
FIXED_PUTDOWN_POINT_ROBOT_FRAME = [-0.32, -0.4, 0.1]
CLASSES = ['wooden cube', 'wooden block', 'wooden box']

ROUND_Q = np.round(Q_FOR_DETECTION, 3)

app = typer.Typer()
PLOT = False

def get_world_position_from_robot_relative(grasp_bot,
                                           relative_position: list[float],
                                           robot_name: str = None) -> list[float]:
    """
    Converts a 3D point from a robot's base frame to the shared world frame.

    Args:
        grasp_bot: The main manipulation controller object, used to access the world model.
        relative_position: A list [x, y, z] representing the point in the robot's base frame.
        robot_name: The name of the robot (e.g., 'ur5e_2'). If None, it will use the
                    default robot name from the grasp_bot object.

    Returns:
        A list [x, y, z] representing the point in the shared world frame.
    """
    # If no specific robot name is given, use the one from the grasp_bot
    if robot_name is None:
        robot_name = grasp_bot.robot_name

    # 1. Get the robot model from the world by its name
    robot_model = grasp_bot.motion_planner.world.robot(robot_name)
    if not robot_model:
        raise ValueError(f"Robot '{robot_name}' not found in the world model.")

    # 2. Get the transformation of the robot's base relative to the world
    # This transform (T) contains the rotation and translation of the robot's base.
    T_world_from_robot = robot_model.link(0).getTransform()

    # 3. Apply the transform to convert the relative point to the world frame
    world_position = se3.apply(T_world_from_robot, relative_position)

    return world_position


def setup_robots():
    """Initialize robot controllers with default speed and acceleration."""
    camera_bot = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_1["ip"], ur5e_1["name"])
    grasp_bot = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])

    camera_bot.speed, camera_bot.acceleration = DEFAULT_SPEED, DEFAULT_ACCELERATION
    grasp_bot.speed, grasp_bot.acceleration = DEFAULT_SPEED, DEFAULT_ACCELERATION

    return camera_bot, grasp_bot


def compute_ik_config(grasp_bot, world_position: list[float]):
    """Compute IK solution for a world position with gripper facing down."""
    ready_position = list(world_position).copy()
    ready_position[2] = START_HEIGHT
    ee_pose = grasp_bot.gt.get_gripper_facing_downwards_6d_pose_robot_frame(
        grasp_bot.robot_name, ready_position, rz=0
    )
    return grasp_bot.find_ik_solution(ee_pose), ready_position


def execute_pick_and_place(grasp_bot, pick_config, place_config,
                           pick_pos: list[float], place_pos: list[float]) -> bool:
    """
    Execute a pick-and-place operation.

    Returns:
        True if successful, False if grasp failed.
    """
    grasp_bot.plan_and_moveJ(pick_config, visualise=False)
    grasp_success = grasp_bot.pick_up(pick_pos[0], pick_pos[1], rz=0, start_height=START_HEIGHT)

    if not grasp_success:
        print("Grasp failed!")
        return False

    grasp_bot.plan_and_moveJ(place_config, visualise=False)
    grasp_bot.put_down(place_pos[0], place_pos[1], rz=0, start_height=START_HEIGHT)
    return True


def detect_objects(camera, detector, plot: bool = False):
    """Capture image and detect objects."""
    im, depth_im = camera.get_frame_rgb()
    depth_im = np.clip(depth_im, 0, 1.0)

    if plot:
        plt.imshow(im)
        plt.show()
        plt.imshow(depth_im, cmap='hot')
        plt.colorbar()
        plt.show()

    bboxes, confidences, results = detector.detect_objects([im])
    bboxes, confidences, results = bboxes[0], confidences[0], results[0]

    if plot:
        im_annotated = detector.get_annotated_images(results)
        plt.imshow(im_annotated)
        plt.show()

    return im, depth_im, bboxes, confidences


def get_depth_at_bbox_center(depth_im, bbox) -> Optional[tuple[list[float], float]]:
    """
    Get the depth value at the center of a bounding box.

    Returns:
        Tuple of (center coordinates, depth value) or None if no valid depth.
    """
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    center_int = [int(center[0]), int(center[1])]

    window = depth_im[
        center_int[1] - DEPTH_WINDOW_SIZE:center_int[1] + DEPTH_WINDOW_SIZE,
        center_int[0] - DEPTH_WINDOW_SIZE:center_int[0] + DEPTH_WINDOW_SIZE
    ]
    valid_depths = window[window > 0]

    if len(valid_depths) == 0:
        return None

    return center, np.mean(valid_depths)


@app.command()
def main(run_fixed_points: bool = False) -> None:
    """Main robot manipulation workflow with fixed points and vision-based pickup."""
    # --- Initialize robots and components ---
    camera_bot, grasp_bot = setup_robots()
    camera = RealsenseCamera()
    ImPE = ImageBlockPositionEstimator(WORKSPACE_X_LIMS_DEFAULT, WORKSPACE_Y_LIMS_DEFAULT, camera_bot.gt)
    detector = ObjectDetection(classes=CLASSES, min_confidence=0.05)

    grasp_bot.move_home(DEFAULT_SPEED, DEFAULT_ACCELERATION)
    camera_bot.move_home(DEFAULT_SPEED, DEFAULT_ACCELERATION)

    # --- Compute IK for fixed points ---
    world_pickup_point = get_world_position_from_robot_relative(grasp_bot, FIXED_PICKUP_POINT)
    pick_up_config, ready_to_pickup = compute_ik_config(grasp_bot, world_pickup_point)

    world_putdown_point = get_world_position_from_robot_relative(grasp_bot, FIXED_PUTDOWN_POINT_ROBOT_FRAME)
    put_down_config, ready_to_put_down = compute_ik_config(grasp_bot, world_putdown_point)

    # Validate IK solutions
    if pick_up_config is None:
        print("No valid IK solution for fixed pickup point!")
        return
    if put_down_config is None:
        print("No valid IK solution for fixed putdown point!")
        return

    if run_fixed_points:
        # --- Pickup and Put-down with fixed points ---
        if not execute_pick_and_place(grasp_bot, pick_up_config, put_down_config,
                                       ready_to_pickup, ready_to_put_down):
            print("Fixed point grasp failed! Stopping execution.")
            grasp_bot.move_home(DEFAULT_SPEED, DEFAULT_ACCELERATION)
            return

        grasp_bot.move_home(DEFAULT_SPEED, DEFAULT_ACCELERATION)

        # --- Return cube to original position ---
        execute_pick_and_place(grasp_bot, put_down_config, pick_up_config,
                               ready_to_put_down, ready_to_pickup)
        grasp_bot.move_home(DEFAULT_SPEED, DEFAULT_ACCELERATION)

    # --- Vision-based Detection and Pickup ---

    if not np.array_equal(np.round(camera_bot.getActualQ(), 3), ROUND_Q):
        camera_bot.moveJ(Q_FOR_DETECTION, DEFAULT_SPEED, DEFAULT_ACCELERATION)

    # Wait for the camera to stabilize before capturing
    time.sleep(CAMERA_STABILIZATION_DELAY)

    im, depth_im, bboxes, _ = detect_objects(camera, detector, plot=PLOT)

    if len(bboxes) == 0:
        print("No objects detected.")
        camera_bot.move_home(DEFAULT_SPEED, DEFAULT_ACCELERATION)
        return

    bbox = bboxes[0].cpu().numpy()
    depth_result = get_depth_at_bbox_center(depth_im, bbox)

    if depth_result is None:
        print("No valid depth pixels found in window.")
        camera_bot.move_home(DEFAULT_SPEED, DEFAULT_ACCELERATION)
        return

    # Use ImPE for getting positions
    positions, _ = ImPE.get_block_positions_depth([im], [depth_im], [Q_FOR_DETECTION])
    camera_bot.move_home(DEFAULT_SPEED, DEFAULT_ACCELERATION)

    if not positions or len(positions) == 0:
        print("Could not estimate block position.")
        return

    # --- Vision-based pickup ---
    pickup_point = list(positions[0]).copy()
    pick_up_config_vision, ready_to_pickup_vision = compute_ik_config(grasp_bot, pickup_point)

    if pick_up_config_vision is None:
        print("No valid IK solution for vision-detected object.")
        return

    if not execute_pick_and_place(grasp_bot, pick_up_config_vision, put_down_config,
                               pickup_point, ready_to_put_down):
        print("Vision grasp failed. Skipping put-down.")
        grasp_bot.move_home(DEFAULT_SPEED, DEFAULT_ACCELERATION)
        return

    # --- Return cube to original position for next experiment ---
    execute_pick_and_place(grasp_bot, put_down_config, pick_up_config_vision,
                           ready_to_put_down, pickup_point)

    grasp_bot.move_home(DEFAULT_SPEED, DEFAULT_ACCELERATION)

    

if __name__ == "__main__":
    app()
