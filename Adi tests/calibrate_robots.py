import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from clair_robotics_stack.ur.lab_setup.manipulation.manipulation_controller_2fg import ManipulationController2FG
from clair_robotics_stack.camera.realsense_camera import RealsenseCamera
from clair_robotics_stack.ur.lab_setup.robot_inteface.robots_metadata import ur5e_1
from clair_robotics_stack.planning.motion.geometry_and_transforms import GeometryAndTransforms
from clair_robotics_stack.planning.motion.motion_planner import MotionPlanner

# --- Constants ---
# Use the same detection configuration as in test_2.py
Q_FOR_DETECTION = np.array(
    [0.839488685131073, -0.9910658162883301, 1.5102737585650843, -2.756538530389303, -2.0820935408221644,
     0.6501701474189758])

DEFAULT_SPEED = 1.0
DEFAULT_ACCELERATION = 0.5
CAMERA_STABILIZATION_DELAY = 1.0

# Approximate expected location of the corner in World Frame (from klampt_world.xml)
# Table2 is at -0.805, -0.615. Size x=0.84, y=1.85.
# Bounds X: [-1.225, -0.385]
# Bounds Y: [-1.54, 0.31]
# The corner visible in the workspace (NEGATIVE X, NEGATIVE Y) is likely the "Top Right" corner of the table
# relative to its own center, but in world frame it's the corner closest to origin?
# Actually, let's assume we are looking for the corner at approx (-0.385, -0.615 +...)
# Wait, workspace limits are X: (-0.9, -0.54), Y: (-1.0, -0.55).
# The corner INSIDE or NEAR this workspace would be X=-0.385? No, -0.385 is outside (-0.9, -0.54).
# Maybe the corner is detecting is the one at X=-0.385 (right edge) and some Y?
# Or X=-1.225 (left edge)?
# The workspace is "on" the table.
# A visible "corner" might be the corner of the table surface itself if the camera sees the edge.
# Let's assume the user wants to detect the corner at (max_x, max_y) of the table slab that is within view?
# Actually, the user mentioned "Corner of the table".
# Let's write the vision code to find *the most prominent corner* in the depth map (depth discontinuity).

def get_depth_image(camera):
    """Capture average depth image for stability."""
    frames = 30
    depth_accum = None
    for _ in range(frames):
        _, depth_im = camera.get_frame_rgb()
        if depth_accum is None:
            depth_accum = depth_im
        else:
            depth_accum += depth_im
        time.sleep(0.01)
    return depth_accum / frames

def detect_table_corner(depth_im, intrinsic_matrix, plot=True) -> Optional[Tuple[int, int]]:
    """
    Finds the table corner in the depth image.
    Assumes the table is a flat surface and we look for the corner of that surface.
    """
    # 1. Threshold depth to find objects at table height (approx 0.5m to 1.5m from camera?)
    # We need to know the camera Z to filter.
    # For simplicity, we'll use a relative threshold or Canny edges.
    
    # Normalize depth for visualization and processing
    depth_vis = (depth_im / np.max(depth_im) * 255).astype(np.uint8)
    
    # Simple edge detection on depth
    # edges = cv2.Canny(depth_vis, 50, 150)
    
    # Better approach: Find the mask of valid depth that represents the table.
    # We assume the table is the large object in the center.
    # 0 is invalid depth in realsense usually (or noise).
    mask = (depth_im > 0.2) & (depth_im < 2.0) # Clip range
    mask = mask.astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found in depth image.")
        return None
        
    # Assume the largest contour is the table
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Find the corner that is most "top-right" or "bottom-right" in the image?
    # This largely depends on the camera orientation.
    # In the Q_FOR_DETECTION:
    # We need to visualize to be sure.
    
    if plot:
        im_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(im_color, [approx], -1, (0, 255, 0), 2)
        
        # Draw points
        for i, p in enumerate(approx):
            cv2.circle(im_color, tuple(p[0]), 5, (0, 0, 255), -1)
            cv2.putText(im_color, str(i), tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        plt.imshow(im_color)
        plt.title("Detected Table Contour & Corners")
        plt.show()
        
    # Heuristic: The corner we want is likely the one with the sharpest angle or specific Image coordinates?
    # For now, return the point in 'approx' that has the highest X (rightmost) + highest Y (bottom)?
    # Or ask logic.
    # Let's return the simplified polygon points and let the main logic decide/convert all to 3D.
    return approx.reshape(-1, 2)


def main():
    print("--- Robot setup ---")
    # Initialize only the camera robot (ur5e_1)
    camera_bot = ManipulationController2FG.build_from_robot_name_and_ip(ur5e_1["ip"], ur5e_1["name"])
    camera_bot.speed = DEFAULT_SPEED
    camera_bot.acceleration = DEFAULT_ACCELERATION
    
    camera = RealsenseCamera()
    
    mp = MotionPlanner()
    gt = GeometryAndTransforms(mp)
    
    print("--- Moving to detection pose ---")
    if not np.array_equal(np.round(camera_bot.getActualQ(), 3), np.round(Q_FOR_DETECTION, 3)):
        camera_bot.moveJ(Q_FOR_DETECTION)
        
    time.sleep(CAMERA_STABILIZATION_DELAY)
    
    print("--- Capturing depth ---")
    depth_im = get_depth_image(camera)
    
    # Get intrinsic matrix
    intrinsics = camera.get_intrinsics() # This usually returns an object or dict
    # We can reconstruct it or import it if needed. 
    # For now assuming 'dt.intrinsic_camera_matrix' from configurations if available or use camera methods.
    # The 'RealsenseCamera' class might not expose get_intrinsics directly as a matrix.
    # Let's look at `configurations_and_params.py`: `depth_camera_intrinsic_matrix`.
    from clair_robotics_stack.camera.configurations_and_params import depth_camera_intrinsic_matrix
    
    print("--- Detecting corner ---")
    # Corners in pixels
    corners_px = detect_table_corner(depth_im, depth_camera_intrinsic_matrix)
    
    if corners_px is None:
        print("Could not detect corners.")
        return
        
    # Convert all corners to 3D World coordinates
    corners_3d = []
    
    current_config = camera_bot.getActualQ()
    
    # Depth at these pixels?
    # We should sample depth at the corner pixel. 
    # Since edges are noisy in depth, we might want to sample slightly "inside" the table mask,
    # or just use the pixel coordinates and trace the ray to the Z-plane of the table (if Z is known).
    # But we are calibrating XY mainly.
    # If we assume Table Z is roughly 0 (or known), we can do plane intersection.
    # Let's try to get the depth from the image first.
    
    for px, py in corners_px:
        # naive depth fetch
        d = depth_im[int(py), int(px)]
        if d <= 0:
            # try neighbor
             d = np.max(depth_im[max(0, int(py)-5):int(py)+5, max(0, int(px)-5):int(px)+5])
             
        if d > 0:
            # Reproject to 3D camera
            # (u - cx) * Z / fx
            fx = depth_camera_intrinsic_matrix[0,0]
            fy = depth_camera_intrinsic_matrix[1,1]
            cx = depth_camera_intrinsic_matrix[0,2]
            cy = depth_camera_intrinsic_matrix[1,2]
            
            x_cam = (px - cx) * d / fx
            y_cam = (py - cy) * d / fy
            z_cam = d
            
            
            point_cam = [x_cam, y_cam, z_cam]
            
            # To World
            point_world = gt.point_camera_to_world(point_cam, "ur5e_1", current_config)
            corners_3d.append(point_world)
            
    print("\n--- Detected 3D Corners (World Frame) ---")
    
    # Expected corners of table2
    # table_center = np.array([-0.805, -0.615, 0])
    # table_dims = np.array([0.84, 1.85, 0.01])
    # half_dims = table_dims / 2
    
    # x_min, x_max = -0.805 - 0.42, -0.805 + 0.42  # [-1.225, -0.385]
    # y_min, y_max = -0.615 - 0.925, -0.615 + 0.925 # [-1.54, 0.31]
    
    expected_corners = [
        np.array([-1.225, -1.54, 0]),
        np.array([-1.225, 0.31, 0]),
        np.array([-0.385, -1.54, 0]),
        np.array([-0.385, 0.31, 0])
    ]
    
    best_match_idx = -1
    min_dist = float('inf')
    best_detected_point = None
    best_expected_point = None
    
    for i, pt in enumerate(corners_3d):
        print(f"Detected Point {i}: {np.round(pt, 4)}")
        # Check distance to all expected corners (ignoring Z for match, assuming flat)
        for expected in expected_corners:
            dist = np.linalg.norm(pt[:2] - expected[:2])
            if dist < min_dist:
                min_dist = dist
                best_match_idx = i
                best_detected_point = pt
                best_expected_point = expected
                
    print("\n-------------------------------------------------------------")
    if best_detected_point is not None and min_dist < 0.5: # 0.5m threshold
        print(f"Closest match found!")
        print(f"Detected: {np.round(best_detected_point, 4)}")
        print(f"Expected: {best_expected_point}")
        print(f"Distance: {min_dist:.4f} m")
        
        # Calculate offset
        # Observed = Expected_Original + Offset
        # Offset = Observed - Expected_Original
        # Actually, the 'Expected' values come from the OLD XML.
        # So the New Position = Old Position + Offset
        
        offset = best_detected_point - best_expected_point
        print(f"Estimated Shift (Offset): {np.round(offset, 4)}")
        
        # Current ur5e_2 position from XML
        current_base_pos = np.array([-0.76, -1.33, 0.0])
        new_base_pos = current_base_pos + offset
        
        print("\n--- RECOMMENDED ACTION ---")
        print(f"Update 'clair_robotics_stack/planning/motion/klampt_world.xml':")
        print(f"Old ur5e_2 position: \"-0.76 -1.33 0\"")
        print(f"New ur5e_2 position: \"{new_base_pos[0]:.4f} {new_base_pos[1]:.4f} {new_base_pos[2]:.4f}\"")
        
        print(f"\nAlso likely need to update 'table2' position in XML by the same offset.")
        current_table_pos = np.array([-0.805, -0.615, 0.0])
        new_table_pos = current_table_pos + offset
        print(f"New table2 position: \"{new_table_pos[0]:.4f} {new_table_pos[1]:.4f} {new_table_pos[2]:.4f}\"")
        
    else:
        print("No detected point was close enough to any expected corner (thresh 0.5m).")
        print("Please check if the robot is actually looking at a table corner.")
        print("Try inspecting the plot window.")

if __name__ == "__main__":
    main()
