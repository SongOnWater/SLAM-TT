import os
import sys
import json
from collections import deque
import cv2
import numpy as np
import torch
import time

sys.path.append('./')

from data_process.ttnet_video_loader import TTNet_Video_Loader
from models.model_utils import create_model, load_pretrained_model
from config.config import parse_configs
from utils.post_processing import post_processing
from utils.misc import time_synchronized

TABLE_LENGTH = 2.74  # meters
TABLE_WIDTH = 1.525  # meters

def extract_table_mask(seg_img):
    """Extract the table mask from the segmentation image."""
    # Assuming blue index represents the table, extract only the blue channel
    blue_channel = seg_img[:, :, 0]  # Assuming blue is the first channel in BGR
    table_mask = (blue_channel > 0).astype(np.uint8) * 255

    return table_mask

def find_table_bounds(seg_img):
    """Find table bounds from the segmentation image."""
    seg_img = (seg_img > 0).astype(np.uint8)  # Convert to binary mask
    contours, _ = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours were found
    if len(contours) == 0:
        raise ValueError("No contours found in the segmentation mask. Could not detect the table.")
    
    # Find the largest contour (assume it's the table)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a polygon (4 corners for the table)
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        raise ValueError("Could not detect exactly 4 corners for the table.")

    # Draw the detected bounds on the image
    bounds_img = cv2.cvtColor(seg_img * 255, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(bounds_img, [approx], -1, (0, 255, 0), 2)
    cv2.imshow("Table Bounds", bounds_img)
    cv2.waitKey(0)  # Wait for user input to close the visualization
    cv2.destroyAllWindows()  # Close all OpenCV windows

    return np.squeeze(approx)  # Return as a 4x2 array

def compute_homography(table_corners_img):
    """Compute homography matrix given table corners with top-left-origin coordinates."""

    # Real-world coordinates of a ping pong table (length 2.74m, width 1.525m)
    
    table_corners_world = np.array([
        [0, 0],
        [TABLE_LENGTH, 0],
        [TABLE_LENGTH, TABLE_WIDTH],
        [0, TABLE_WIDTH]
    ], dtype=np.float32)
    
    H, _ = cv2.findHomography(table_corners_img, table_corners_world)
    return H

def map_bounce_to_real_world(H, bounce_point_img):
    bounce_point_img = np.array([bounce_point_img[0], bounce_point_img[1], 1]).reshape((3, 1))
    bounce_point_world = np.dot(H, bounce_point_img)

    # Normalize to get real-world coordinates
    bounce_point_world = bounce_point_world / bounce_point_world[2]
    return bounce_point_world[0:2].flatten()

def merge_bounce_events(bounces, frame_threshold=5):
    """Merge bounce events that are within a certain number of frames."""
    if not bounces:
        return []

    merged_bounces = [bounces[0]]
    for bounce in bounces[1:]:
        last_bounce = merged_bounces[-1]
        if bounce['frame'] - last_bounce['frame'] <= frame_threshold:
            # Average the position
            avg_position = [
                (last_bounce['position'][0] + bounce['position'][0]) / 2,
                (last_bounce['position'][1] + bounce['position'][1]) / 2
            ]
            merged_bounces[-1] = {
                "frame": bounce['frame'],  # TODO: set based on highest bounce confidence
                "position": avg_position
            }
        else:
            merged_bounces.append(bounce)

    return merged_bounces

def visualize_bounces(bounces):
    """Visualize the table and mapped bounce points."""
    # Scale factor for visualization (converting meters to centimeters)
    scale_factor = 100

    # Create a blank white image
    visualization_img = np.ones((int(TABLE_LENGTH * scale_factor), int(TABLE_WIDTH * scale_factor), 3), dtype=np.uint8) * 255

    # Draw the table outline (scaled)
    table_top_left = (0, 0)
    table_bottom_right = (int(TABLE_LENGTH * scale_factor), int(TABLE_WIDTH * scale_factor))
    cv2.rectangle(visualization_img, table_top_left, table_bottom_right, (0, 0, 255), 3)

    # Draw bounce points
    for bounce in bounces:
        point_x = int(bounce['position'][0] * scale_factor)
        point_y = int(bounce['position'][1] * scale_factor)
        cv2.circle(visualization_img, (point_x, point_y), 5, (255, 0, 0), -1)  # Blue point

    # Display the visualization
    cv2.imshow("Table Bounces Visualization", visualization_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(configs):
    # Load model
    model = create_model(configs)
    model.cuda()
    assert configs.pretrained_path is not None, "Need to load the pre-trained model"
    model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx, configs.overwrite_global_2_local)
    model.eval()

    middle_idx = int(configs.num_frames_sequence / 2)
    queue_frames = deque(maxlen=middle_idx + 1)
    frame_idx = 0
    w_original, h_original = 1920, 1080
    w_resize, h_resize = 320, 128
    w_ratio = w_original / w_resize
    h_ratio = h_original / h_resize

    video_loader = TTNet_Video_Loader(configs.video_path, configs.input_size, configs.num_frames_sequence)
    frame_idx = 0
    w_original, h_original = 1920, 1080

    homography_saved = None
    bounces = []

    with torch.no_grad():
        for count, resized_imgs in video_loader:
            # Run inference
            img = cv2.resize(resized_imgs[3 * middle_idx: 3 * (middle_idx + 1)].transpose(1, 2, 0), (w_original, h_original))
            resized_imgs = torch.from_numpy(resized_imgs).to(configs.device, non_blocking=True).float().unsqueeze(0)

            t1 = time_synchronized()
            pred_ball_global, pred_ball_local, pred_events, pred_seg = model.run_demo(resized_imgs)
            t2 = time_synchronized()
            prediction_global, prediction_local, prediction_seg, prediction_events = post_processing(
                pred_ball_global, pred_ball_local, pred_events, pred_seg, configs.input_size[0],
                configs.thresh_ball_pos_mask, configs.seg_thresh, configs.event_thresh)
            prediction_ball_final = [
                int(prediction_global[0] * w_ratio + prediction_local[0] - w_resize / 2),
                int(prediction_global[1] * h_ratio + prediction_local[1] - h_resize / 2)
            ]

            if homography_saved is None:
                table_mask = extract_table_mask(prediction_seg)
                table_mask = cv2.resize(table_mask, (w_original, h_original))

                try:
                    table_corners = find_table_bounds(table_mask)
                    homography_saved = compute_homography(table_corners)
                    print("Homography Matrix:")
                    print(homography_saved)
                except ValueError as e:
                    print(f"Warning: {e}. Could not compute homography for the {frame_idx} frame.")
            
            if homography_saved is not None:
                if prediction_events[0] > configs.bounce_thresh:
                    real_world_position = map_bounce_to_real_world(homography_saved, prediction_ball_final)
                    print("Real-world coordinates of the ball (top-left origin):", real_world_position)
                    bounces.append({
                        "frame": frame_idx,
                        "position": real_world_position.tolist(),
                    })

            frame_idx += 1

    # Merge bounce events that are within 5 frames of each other
    merged_bounces = merge_bounce_events(bounces)

    # Save bounce information to a JSON file
    with open('bounce_positions.json', 'w') as f:
        output = {
            "bounces": merged_bounces
        }
        json.dump(output, f, indent=2)
    print("Saved output to bounce_positions.json")

    # Visualize the bounces
    visualize_bounces(merged_bounces)

if __name__ == '__main__':
    configs = parse_configs()
    configs.video_path = "../dataset/test/videos/test_1_trimmed.mp4"
    configs.gpu_idx = 0
    configs.pretrained_path = "../checkpoints/ttnet.pth"
    configs.show_image = False
    configs.save_demo_output = True
    configs.no_seg = True
    configs.bounce_thresh = 0.7 # Threshold for detecting a bounce
    process_video(configs=configs)
