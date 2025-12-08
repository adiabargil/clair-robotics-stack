import numpy as np
import typer
import matplotlib.pyplot as plt
from klampt import vis
from numpy.f2py.crackfortran import endifs
from clair_robotics_stack.ur.lab_setup.manipulation.manipulation_controller_2fg import ManipulationController2FG
from clair_robotics_stack.planning.motion.motion_planner import MotionPlanner
from clair_robotics_stack.planning.motion.geometry_and_transforms import GeometryAndTransforms
from clair_robotics_stack.camera.realsense_camera import RealsenseCamera
from clair_robotics_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator
from clair_robotics_stack.ur.lab_setup.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from clair_robotics_stack.vision.utils import *
from clair_robotics_stack.vision.object_detection import ObjectDetection
from clair_robotics_stack.camera.configurations_and_params import color_camera_intrinsic_matrix

# --- Constants ---
WORKSPACE_X_LIMS_DEFAULT = (-0.9, -0.54)
WORKSPACE_Y_LIMS_DEFAULT = (-1.0, -0.55)

Q_FOR_DETECTION = np.array(
    [0.839488685131073, -0.9910658162883301, 1.5102737585650843, -2.756538530389303, -2.0820935408221644,
     0.6501701474189758])

# NOTE: Q_for_pickup was defined but not used in the original code, keeping it for now if needed, or else remove.
Q_FOR_PICKUP = np.array(
    [1.5422165393829346, -0.6949203771403809, 0.9701255003558558, 1.3190886217304687, 1.5162320137023926,
     -0.025219265614644826])

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


@app.command()
def main(n_blocks: int = 4, depth_option: bool = True):
    camera_bot = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_1["ip"], ur5e_1["name"])
    grasp_bot = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])
    camera = RealsenseCamera()
    ImPE = ImageBlockPositionEstimator(WORKSPACE_X_LIMS_DEFAULT, WORKSPACE_Y_LIMS_DEFAULT, camera_bot.gt)
    
    camera_bot.speed, camera_bot.acceleration = 2, 1.3
    grasp_bot.speed, grasp_bot.acceleration = 2, 1.3
    
    detector = ObjectDetection(classes=CLASSES, min_confidence=0.05)

    grasp_bot.move_home(2, 1.5)
    camera_bot.move_home(2, 1.5)

    # --- Pickup with fixed point ---
    world_frame_pickup_point = get_world_position_from_robot_relative(grasp_bot, FIXED_PICKUP_POINT)
    ready_to_pickup = world_frame_pickup_point.copy()
    ee_pose_pickup = grasp_bot.gt.get_gripper_facing_downwards_6d_pose_robot_frame(grasp_bot.robot_name, ready_to_pickup, rz=0)
    pick_up_config = grasp_bot.find_ik_solution(ee_pose_pickup)
    grasp_bot.plan_and_moveJ(pick_up_config)
    grasp_bot.pick_up(ready_to_pickup[0], ready_to_pickup[1], rz=0, start_height=0.1)

    # --- Putdown with fixed point ---
    world_frame_put_down_point = get_world_position_from_robot_relative(grasp_bot, FIXED_PUTDOWN_POINT_ROBOT_FRAME)
    ready_to_put_down = world_frame_put_down_point.copy()
    ee_pose_putdown = grasp_bot.gt.get_gripper_facing_downwards_6d_pose_robot_frame(grasp_bot.robot_name, ready_to_put_down, rz=0)
    put_down_config = grasp_bot.find_ik_solution(ee_pose_putdown)
    grasp_bot.plan_and_moveJ(put_down_config)
    grasp_bot.put_down(ready_to_put_down[0], ready_to_put_down[1], rz=0, start_height=0.1)


    grasp_bot.move_home(2, 1.5)

    # --- Place the cube at the original point ---
    grasp_bot.plan_and_moveJ(put_down_config)
    grasp_bot.pick_up(ready_to_put_down[0], ready_to_put_down[1], rz=0, start_height=0.1)
    grasp_bot.plan_and_moveJ(pick_up_config)
    grasp_bot.put_down(ready_to_pickup[0], ready_to_pickup[1], rz=0, start_height=0.1)

    grasp_bot.move_home(2, 1.5)

    # --- Vision and Detection ---
    if not np.array_equal(np.round(camera_bot.getActualQ(), 3), ROUND_Q):
        camera_bot.moveJ(Q_FOR_DETECTION, 2, 1.3)
        
    im, depth_im = camera.get_frame_rgb()
    depth_im = np.clip(depth_im, 0, 1.0)
    
    if PLOT:
        plt.imshow(im)
        plt.show()
        plt.imshow(depth_im, cmap='hot')
        plt.colorbar()
        plt.show()

    im_batch = [im]
    bboxes, confidences, results = detector.detect_objects(im_batch)

    # result is also returned as batch, we have only 1 element in the batch
    bboxes, confidences, results = bboxes[0], confidences[0], results[0]
    im_annotated = detector.get_annotated_images(results)
    
    if PLOT:
        plt.imshow(im_annotated)
        plt.show()

    if len(bboxes) > 0:
        bbox = bboxes[0].cpu().numpy()
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        center_int = [int(center[0]), int(center[1])]

        window_in_depth = depth_im[center_int[1] - 2:center_int[1] + 2, center_int[0] - 2:center_int[0] + 2]
        # if some depth is zero, this is not a valid pixel:
        window_in_depth = window_in_depth[window_in_depth > 0]
        
        if len(window_in_depth) > 0:
            depth = np.mean(window_in_depth)

            fx = color_camera_intrinsic_matrix[0, 0]
            fy = color_camera_intrinsic_matrix[1, 1]
            ppx = color_camera_intrinsic_matrix[0, 2]
            ppy = color_camera_intrinsic_matrix[1, 2]

            x_cam = (center[0] - ppx) * depth / fx
            y_cam = (center[1] - ppy) * depth / fy
            z_cam = depth
            p_cam = [x_cam, y_cam, z_cam]

            # Use ImPE for getting positions
            positions, annotations = ImPE.get_block_positions_depth([im], [depth_im], [Q_FOR_DETECTION])

            camera_bot.move_home(1.5, 1)

            if positions and len(positions) > 0:
                pickup_point = positions[0].copy()
                ready_to_pickup_vision = positions[0].copy()
                ready_to_pickup_vision[2] = 0.1
                
                ee_pose_vision = grasp_bot.gt.get_gripper_facing_downwards_6d_pose_robot_frame(grasp_bot.robot_name, ready_to_pickup_vision, rz=0)
                pick_up_config_vision = grasp_bot.find_ik_solution(ee_pose_vision)
                
                if pick_up_config_vision:
                    grasp_bot.plan_and_moveJ(pick_up_config_vision)
                    grasp_bot.pick_up(pickup_point[0], pickup_point[1], rz=0, start_height=0.1)
                    grasp_bot.plan_and_moveJ(put_down_config)
                    grasp_bot.put_down(ready_to_put_down[0], ready_to_put_down[1], rz=0, start_height=0.1)
                    grasp_bot.move_home(2, 1.5)
                else:
                    print("No valid pick-up configuration found for the object.")
            else:
                print("Could not estimate block position.")
        else:
            print("No valid depth pixels found in window.")
    else:
        print("No objects detected.")
    
    

if __name__ == "__main__":
    app()






    """
    import klampt.math.se3 as se3
# 1. Get the correct robot model from the world using its name
robot_model = grasp_bot.motion_planner.world.robot(grasp_bot.robot_name)
# 2. Get the transformation of THIS robot's base relative to the world
T_world_from_robot = robot_model.link(0).getTransform()
# 3. Get the TCP position in the robot's frame
tcp_pos_in_robot_frame = grasp_bot.getTargetTCPPose()[:3]
# 4. Apply the transform to convert the TCP position to the world frame
tcp_pos_in_world_frame = se3.apply(T_world_from_robot, tcp_pos_in_robot_frame)
print(f"Correct robot is: '{grasp_bot.robot_name}'")
print(f"Robot Base is at: {T_world_from_robot[1]}")
print(f"TCP position (in Robot Frame): {tcp_pos_in_robot_frame}")
print(f"TCP position (in World Frame): {tcp_pos_in_world_frame}")
print("\nThis should be close to your target:")
print(f"positions[0] = {positions[0]}")
Correct robot is: 'ur5e_2'
Robot Base is at: [-0.76, -1.33, 0.0]
TCP position (in Robot Frame): [0.12843400245304534, -0.8166320847117635, 0.1999147192222231]
TCP position (in World Frame): [-0.8884340023797174, -0.513367915276704, 0.1999147192222231]
This should be close to your target:
positions[0] = [-0.8884400918931376, -0.5133913323021623, -0.00908216698745623]
"""