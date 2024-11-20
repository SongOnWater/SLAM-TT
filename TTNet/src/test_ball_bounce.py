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

def extract_table_mask(seg_img):
    """Extract the table mask from the segmentation image."""
    # Define the range for the color blue in BGR
    lower_blue = np.array([200, 0, 0])  # Lower bound for blue (adjust as needed)
    upper_blue = np.array([255, 50, 50])  # Upper bound for blue (adjust as needed)

    # Create a mask for blue color (table)
    table_mask = cv2.inRange(seg_img, lower_blue, upper_blue)

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
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 4:
        raise ValueError("Could not detect exactly 4 corners for the table.")

    return np.squeeze(approx)  # Return as a 4x2 array

def compute_homography(table_corners, table_dims):
    """Compute homography matrix given table corners and real-world dimensions."""
    real_world_corners = np.array([
        [0, 0],  # Top-left
        [table_dims[0], 0],  # Top-right
        [0, table_dims[1]],  # Bottom-left
        [table_dims[0], table_dims[1]]  # Bottom-right
    ], dtype=np.float32)
    
    H, _ = cv2.findHomography(table_corners, real_world_corners)
    return H

def map_ball_to_table(ball_pos, H):
    """Map ball position from image frame to real-world table coordinates."""
    point = np.array([ball_pos[0], ball_pos[1], 1], dtype=np.float32).reshape(3, 1)
    mapped = np.dot(H, point)
    mapped /= mapped[2]  # Normalize by the homogeneous coordinate
    return mapped[0:2].flatten()

def process_video(configs):
    video_loader = TTNet_Video_Loader(configs.video_path, configs.input_size, configs.num_frames_sequence)

    # Set up video writer
    frame_rate = video_loader.video_fps
    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

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

    homography_saved = None
    table_dims = (2.74, 1.525)  # Real-world table dimensions in meters
    bounces = []

    with torch.no_grad():
        for count, resized_imgs in video_loader:
            # Process the current frame
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

            # Display middle frame's prediction if queue is filled
            if len(queue_frames) == middle_idx + 1:
                frame_pred_infor = queue_frames.popleft()
                ball_pos = frame_pred_infor['ball']
                seg_img = frame_pred_infor['seg'][0].astype(np.uint8)

                if configs.no_seg:
                    seg_img[:] = 0

                seg_img = cv2.resize(seg_img, (w_original, h_original))
                ploted_img = plot_detection(img, ball_pos, seg_img, prediction_events)
                ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)

                if homography_saved is None:
                    table_mask = extract_table_mask(seg_img)
                    try:
                        table_corners = find_table_bounds(table_mask)
                        homography_saved = compute_homography(table_corners, table_dims)
                    except ValueError as e:
                        print(f"Warning: {e}. Skipping homography computation for this frame.")
                        continue

                # Pause video and wait for user input if bounce detected
                if prediction_events[0] > configs.bounce_thresh:
                    real_world_ball_pos = map_ball_to_table(ball_pos, homography_saved)
                    bounces.append({
                        "frame": frame_idx,
                        "position": real_world_ball_pos.tolist()
                    })
                    cv2.imshow('Bounce Detected - Press any key to continue', ploted_img)
                    cv2.waitKey(0)

                # Show image if desired
                if configs.show_image:
                    cv2.imshow('ploted_img', ploted_img)
                    cv2.waitKey(10)

            frame_pred_infor = {
                'seg': prediction_seg,
                'ball': prediction_ball_final
            }
            queue_frames.append(frame_pred_infor)

            frame_idx += 1
            print('Done frame_idx {} - time {:.3f}s. Ball Pos: {}'.format(frame_idx, t2 - t1, prediction_ball_final))

    # Save bounce information to a JSON file
    with open('bounce_positions.json', 'w') as f:
        json.dump(bounces, f, indent=4)

def plot_detection(img, ball_pos, seg_img, events):
    """Show the predicted information in the image."""
    img = cv2.addWeighted(img, 1., seg_img * 255, 0.3, 0)
    img = cv2.circle(img, tuple(ball_pos), 5, (255, 0, 255), -1)
    
    event_name = 'is bounce: {:.2f}, is net: {:.2f}'.format(events[0], events[1])
    img = cv2.putText(img, event_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img

if __name__ == '__main__':
    configs = parse_configs()
    configs.video_path = "../dataset/test/videos/test_1_trimmed.mp4"
    configs.gpu_idx = 0
    configs.pretrained_path = "../checkpoints/ttnet.pth"
    configs.show_image = True
    configs.no_seg = False
    configs.bounce_thresh = 0.7 # Threshold for detecting a bounce
    process_video(configs=configs)
