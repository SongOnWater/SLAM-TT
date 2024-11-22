import os
import sys
import json
from collections import deque
import cv2
import numpy as np


TABLE_WIDTH = 2740  # cm
TABLE_LENGTH = 1525  # cm

def extract_table_mask(seg_img):
    """Extract the table mask from the segmentation image."""
    # Assuming blue index represents the table, extract only the blue channel
    blue_channel = seg_img[:, :, 0]  # Assuming blue is the first channel in BGR
    table_mask = (blue_channel > 0).astype(np.uint8) * 255

    return table_mask

def find_table_bounds(seg_img):
    """
    Find table bounds from the segmentation image and annotate the corners with pixel coordinates.
    
    Parameters:
        seg_img (numpy.ndarray): Binary segmentation image with the table segmented.
    
    Returns:
        numpy.ndarray: 4x2 array containing the pixel coordinates of the table corners in TL, TR, BR, BL order.
    """
    
    # Convert to binary mask
    seg_img = (seg_img > 0).astype(np.uint8)

    # Find contours in the binary mask
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

    # Prepare an image for visualization
    bounds_img = cv2.cvtColor(seg_img * 255, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(bounds_img, [approx], -1, (0, 255, 0), 2)

    # Annotate each point with its pixel coordinates
    approx_points = np.squeeze(approx)  # Convert to a 4x2 array

    # Order the points in TL, TR, BR, BL order
    rect = order_points(approx_points)

    for i, (x, y) in enumerate(rect):
        coord_text = f"{i}: ({x}, {y})"
        cv2.putText(bounds_img, coord_text, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the visualization
    cv2.imshow("Table Bounds with Coordinates", bounds_img)
    cv2.waitKey(0)  # Wait for user input to close the visualization
    cv2.destroyAllWindows()  # Close all OpenCV windows

    return rect  # Return as a 4x2 array

def order_points(pts):
    """
    Order points in TL, TR, BR, BL order.

    Parameters:
        pts (numpy.ndarray): 4x2 array of points.

    Returns:
        numpy.ndarray: 4x2 array of ordered points.
    """
    # Step 1: Sort points by y-coordinate to get top and bottom pairs
    sorted_points = pts[np.argsort(pts[:, 1])]

    # Step 2: Split into top and bottom pairs
    top_points = sorted_points[:2]
    bottom_points = sorted_points[2:]

    # Step 3: Sort the pairs by x-coordinate to determine left and right
    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

    # Step 4: Combine in [tl, tr, br, bl] order
    ordered_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

    return ordered_points


def compute_homography(table_corners_img):
    """Compute homography matrix given table corners with top-left-origin coordinates."""

    # Real-world coordinates of a ping pong table (length 2740cm, width 1525cm)
    
    table_corners_world = np.array([
        [0, 0],
        [TABLE_WIDTH, 0],
        [TABLE_WIDTH, TABLE_LENGTH],
        [0, TABLE_LENGTH]
    ], dtype=np.float32)
    
    H, _ = cv2.findHomography(table_corners_img, table_corners_world)
    return H

def map_bounce_to_real_world(H, bounce_point_img):
    bounce_point_img = np.array([bounce_point_img[0], bounce_point_img[1], 1]).reshape((3, 1))
    bounce_point_world = np.dot(H, bounce_point_img)

    # Normalize to get real-world coordinates
    bounce_point_world = bounce_point_world / bounce_point_world[2]
    return bounce_point_world[0:2].flatten()

def map_bounce_to_real_world2(H, bounce_point_img):
    bounce_point_img = np.array([bounce_point_img[0], bounce_point_img[1], 1])
    # Apply the homography
    world_point = np.dot(H, bounce_point_img)

    # Convert from homogeneous coordinates
    world_point = world_point / world_point[2]
    transformed_x, transformed_y = int(world_point[0]), int(world_point[1])
    return [transformed_x, transformed_y]

def is_ball_in_bounds(pos, bounds):
    # Unpack the position
    x, y = pos

    # Extract corners from bounds, in sorted order: tl, tr, br, bl
    tl, tr, br, bl = bounds

    # Use the cross product to check if the point is within the bounds
    def cross_product(a, b):
        return a[0] * b[1] - a[1] * b[0]

    # Create vectors from corners to point
    v_tl = (x - tl[0], y - tl[1])
    v_tr = (x - tr[0], y - tr[1])
    v_br = (x - br[0], y - br[1])
    v_bl = (x - bl[0], y - bl[1])

    # Create vectors from each corner to the next
    edge_tl_tr = (tr[0] - tl[0], tr[1] - tl[1])
    edge_tr_br = (br[0] - tr[0], br[1] - tr[1])
    edge_br_bl = (bl[0] - br[0], bl[1] - br[1])
    edge_bl_tl = (tl[0] - bl[0], tl[1] - bl[1])

    # Compute cross products for each edge and vector to point
    cross_1 = cross_product(edge_tl_tr, v_tl)
    cross_2 = cross_product(edge_tr_br, v_tr)
    cross_3 = cross_product(edge_br_bl, v_br)
    cross_4 = cross_product(edge_bl_tl, v_bl)

    # If all cross products have the same sign, the point is inside the rectangle
    if (cross_1 >= 0 and cross_2 >= 0 and cross_3 >= 0 and cross_4 >= 0) or \
       (cross_1 <= 0 and cross_2 <= 0 and cross_3 <= 0 and cross_4 <= 0):
        return True
    else:
        return False