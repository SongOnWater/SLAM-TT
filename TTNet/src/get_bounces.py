import os
import sys
import json
from collections import deque
import cv2
import numpy as np
import torch
import time

sys.path.append('./')

from data_process.ttnet_video_loader import TTNet_Video_Loader, TTNet_Video_Loader_V2
from models.model_utils import create_model, load_pretrained_model
from config.config import parse_configs
from utils.post_processing import post_processing
from utils.misc import time_synchronized
from utils.homography import extract_table_mask, find_table_bounds, compute_homography, map_bounce_to_real_world2, is_ball_in_bounds

TABLE_WIDTH = 2740  # mm
TABLE_LENGTH = 1525  # mm

def merge_bounce_events_highest_confidence(bounces, max_frame_diff=9):
    # Sort bounces by frame to make it easier to merge
    bounces = sorted(bounces, key=lambda x: x['frame'])
    
    # Initialize the list for merged bounces
    merged_bounces = []
    
    # Iterate through the sorted bounces and merge them based on the frame difference
    i = 0
    while i < len(bounces):
        # Start with the current bounce as the candidate for merging
        current_bounce = bounces[i]
        i += 1

        # Compare with subsequent bounces within the frame window
        while i < len(bounces) and (bounces[i]['frame'] - current_bounce['frame']) <= max_frame_diff:
            # Keep the bounce with the highest confidence
            if float(bounces[i]['bounce_confidence']) > float(current_bounce['bounce_confidence']):
                current_bounce = bounces[i]
            i += 1
        
        # Add the selected bounce to the merged list
        merged_bounces.append(current_bounce)
    
    return merged_bounces

def merge_bounce_events(bounces, max_frame_diff=9):
    # Keeps the first detected frame as the bounce point
    
    bounces = sorted(bounces, key=lambda x: x['frame'])
    merged_bounces = []
    
    i = 0
    while i < len(bounces):
        current_bounce = bounces[i]
        i += 1

        while i < len(bounces) and (bounces[i]['frame'] - current_bounce['frame']) <= max_frame_diff:
            i += 1
        
        merged_bounces.append(current_bounce)
    
    return merged_bounces

def visualize_bounces(bounces, enable_labels=False): 
    """
    Visualize ball bounces on a table using OpenCV.

    Parameters:
        bounces (list): List of bounces, where each bounce is a dictionary with "position": [x, y].
        enable_labels (bool): Whether to enable labels on the visualization.
    """
    RESCALED_WIDTH = 800
    RESCALED_HEIGHT = 400

    # Create a blank image (black background) with rescaled dimensions
    table_image = np.zeros((RESCALED_HEIGHT, RESCALED_WIDTH, 3), dtype=np.uint8)

    # Draw the table outline (white rectangle)
    cv2.rectangle(table_image, (0, 0), (RESCALED_WIDTH - 1, RESCALED_HEIGHT - 1), (255, 255, 255), 2)

    # Calculate the scaling factors for x and y coordinates
    scale_x = RESCALED_WIDTH / TABLE_WIDTH
    scale_y = RESCALED_HEIGHT / TABLE_LENGTH

    # Draw each bounce
    for i, bounce in enumerate(bounces):
        # Real-world coordinates of the bounce
        x, y = bounce["position"]
        # Scale the coordinates to fit the rescaled image
        x_rescaled = int(x * scale_x)
        y_rescaled = int(y * scale_y)

        # Draw a circle at the bounce position (red)
        cv2.circle(table_image, (x_rescaled, y_rescaled), 8, (0, 0, 255), -1)  # Red circle

        if enable_labels:
            cv2.putText(table_image, str(i + 1), (x_rescaled + 10, y_rescaled - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Green labels

    # Display the visualization
    cv2.imshow("Table with Bounces", table_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_detection(img, ball_pos, events):
    """Show the predicted information in the image."""
    img = cv2.circle(img, tuple(ball_pos), 5, (255, 0, 255), -1)
    
    event_name = 'is bounce: {:.2f}, is net: {:.2f}'.format(events[0], events[1])
    img = cv2.putText(img, event_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img

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

    video_loader = TTNet_Video_Loader_V2(configs.video_path, configs.input_size, configs.num_frames_sequence)
    frame_idx = 0
    w_original, h_original = 1920, 1080

    homography_saved = None
    table_corners = None
    last_in_bounds_pos = None
    bounces = []

    with torch.no_grad():
        for count, frame_timestamp_ms, resized_imgs, orig_img in video_loader:
            # Run inference
            # img = cv2.resize(resized_imgs[3 * middle_idx: 3 * (middle_idx + 1)].transpose(1, 2, 0), (w_original, h_original))            img = orig_imgs[-1].transpose(1,2,0)
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
                    print("Table Corners:", table_corners)
                    homography_saved = compute_homography(table_corners)
                    print("Homography Matrix:")
                    print(homography_saved)
                except ValueError as e:
                    print(f"Warning: {e}. Could not compute homography for the {frame_idx} frame.")
                
            if table_corners is not None:
                if is_ball_in_bounds(prediction_ball_final, table_corners):
                    last_in_bounds_pos = prediction_ball_final
            
            if homography_saved is not None:
                if prediction_events[0] > configs.bounce_thresh:
                    real_world_position = map_bounce_to_real_world2(homography_saved, last_in_bounds_pos)
                    print("Pixel coord: ", last_in_bounds_pos, "Real-world coord:", real_world_position)
                    bounces.append({
                        "frame": frame_idx,
                        "position": real_world_position,
                        "pixel_coord": last_in_bounds_pos,
                        "bounce_confidence": str(prediction_events[0])
                    })

                    # Project image
                    if configs.show_image:
                        ploted_img = plot_detection(orig_img, last_in_bounds_pos, prediction_events)
                        ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Plot Image", ploted_img)
                        if cv2.waitKey(0) & 0xFF == ord("q"):
                            break

            frame_idx += 1

    # Merge bounce events that are within 9 frames of each other
    merged_bounces = merge_bounce_events(bounces)

    # Save bounce information to a JSON file
    with open('../results/bounce_positions.json', 'w') as f:
        output = {
            "bounces": merged_bounces
        }
        json.dump(output, f, indent=2)
    print("Saved output to ../results/bounce_positions.json")

    # Visualize the bounces
    visualize_bounces(merged_bounces, enable_labels=True)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    configs = parse_configs()
    configs.video_path = "demo_video_short.mp4"
    configs.gpu_idx = 0
    configs.pretrained_path = "../checkpoints/ttnet.pth"
    configs.show_image = False
    configs.save_demo_output = True
    configs.no_seg = True
    configs.bounce_thresh = 0.7 # Threshold for detecting a bounce
    process_video(configs=configs)
