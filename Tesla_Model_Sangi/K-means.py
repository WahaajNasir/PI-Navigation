import cv2 as cv
import numpy as np
import time

# Wahaaj, you will forget this next time you open the code, so im leaving this here
# Red Rectangle: Middle Strip check for immediate stop or turning
# Blue Rectangle: Wider check that allows you to speed up or slow down
# Magenta Line: Middle of screen
# Yellow Line: Bottom of Screen
# Cyan Line: For analysis of whether it is a long straight road or not

# Lane polygon detection (skewed rectangle/lane box)
def get_lane_edges(mask, y_row):
    nonzero_indices = np.where(mask[y_row] == 255)[0]
    if len(nonzero_indices) >= 2:
        return nonzero_indices[0], nonzero_indices[-1]
    return None, None

def contrast_stretching(image):
    min_val, max_val = np.min(image), np.max(image)
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched


def kmeans_segmentation_fast(image, k=3):
    Z = image.reshape((-1, 1)).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv.kmeans(Z, k, None, criteria, 5, cv.KMEANS_RANDOM_CENTERS)
    label_img = labels.reshape(image.shape)
    h, w = label_img.shape
    guide_band = label_img[int(h * 0.85):, int(w * 0.4):int(w * 0.6)]
    counts = [np.sum(guide_band == i) for i in range(k)]
    road_label = np.argmax(counts)
    mapped = np.zeros_like(label_img, dtype=np.uint8)
    for i in range(k):
        mapped[label_img == i] = 255 if i == road_label else 127
    return mapped


def remove_small_objects(mask, min_size):
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8, cv.CV_32S)
    output = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255
    return output


def remove_small_zero_objects(mask, min_size):
    inverted = cv.bitwise_not(mask)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(inverted, 8, cv.CV_32S)
    cleaned = np.ones_like(mask) * 255
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 0
    return cleaned


def has_clear_gap(mask_band, min_width=200):
    for row in mask_band:
        row_bin = row == 255
        count = 0
        for val in row_bin:
            count = count + 1 if val else 0
            if count >= min_width:
                return True
    return False


cap = cv.VideoCapture(r"D:\Uni\Semester 6\DIP\Self\Project\Tesla_Model_Sangi\Dataset\Cloudy\PXL_20250325_045117252.TS.mp4")
detect_obj_count = 0
road_classification_history = []
prev_zero_ratio = 0
prev_frame_time = 0
new_frame_time = 0
kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_AREA)
    new_frame_time = time.time()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    contrast = contrast_stretching(gray)
    median_filtered = cv.medianBlur(contrast, 5)
    edges = cv.Canny(median_filtered, 50, 150)

    closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel_close)
    contours, _ = cv.findContours(closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contrast_with_black = median_filtered.copy()
    cv.drawContours(contrast_with_black, contours, -1, color=0, thickness=cv.FILLED)

    # cv.imshow("Contrast With Black", contrast_with_black)
    # cv.waitKey(0)

    down = cv.resize(contrast_with_black, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
    seg_down = kmeans_segmentation_fast(down)
    segmented = cv.resize(seg_down, gray.shape[::-1], interpolation=cv.INTER_NEAREST)

    opened = cv.morphologyEx(segmented, cv.MORPH_OPEN, kernel_close)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel_close)
    road_mask = np.zeros_like(closed)
    road_mask[closed == 255] = 255

    h, w = road_mask.shape
    lower_75_mask = np.zeros_like(road_mask)
    lower_75_mask[int(h * 0.25):, :] = 255
    road_mask = cv.bitwise_and(road_mask, lower_75_mask)

    road_mask = remove_small_objects(road_mask, 25000)
    road_mask = remove_small_zero_objects(road_mask, 5000)

    overlay = cv.bitwise_and(gray, gray, mask=road_mask)
    vis_overlay = cv.cvtColor(overlay.copy(), cv.COLOR_GRAY2BGR)

    center_x = w // 2
    center_y = h // 2 + 50
    detection_y = center_y - 50

    top_y = np.argmax(np.any(road_mask == 255, axis=1)) + 50
    bottom_y = h - np.argmax(np.any(road_mask[::-1] == 255, axis=1)) - 1

    cv.line(vis_overlay, (0, top_y), (w, top_y), (255, 0, 255), 2)
    cv.line(vis_overlay, (0, bottom_y), (w, bottom_y), (0, 255, 255), 2)

    top_px = np.sum(road_mask[top_y, :] == 255)
    bottom_px = np.sum(road_mask[bottom_y, :] == 255)

    # Added a road history because algo has tendency to go from straight to sloped to straight
    # (This is a drunk driving assisstant...)
    # ITNA KON HILTA HEI BHAI???
    road = "Straight"
    if bottom_px != 0 and top_px / bottom_px < 0.125:
        road_classification_history.append("Sloped")
    else:
        road_classification_history.append("Straight")
        center_y = h // 2 + 50

    if len(road_classification_history) > 10:
        road_classification_history.pop(0)

    if road_classification_history.count("Sloped") >= 7:
        road = "Sloped"
        center_y = bottom_y - 100
        detection_y = center_y - 50

    check_height = 75
    check_width = 50
    left_x, right_x = center_x - check_width, center_x + check_width
    vertical_strip = road_mask[center_y - check_height:center_y, left_x:right_x]
    vertical_clear = np.all(vertical_strip == 255)

    cv.rectangle(vis_overlay, (left_x, center_y - check_height), (right_x, center_y), (0, 0, 255), 2)
    cv.rectangle(vis_overlay, (center_x - 200, detection_y - 100), (center_x, detection_y + 100), (0, 255, 0), 2)
    cv.rectangle(vis_overlay, (center_x, detection_y - 100), (center_x + 200, detection_y + 100), (0, 255, 0), 2)

    middle_line = road_mask[top_y:bottom_y, center_x]
    middle_line_length = np.sum(middle_line == 255)
    cv.line(vis_overlay, (center_x, top_y), (center_x, bottom_y), (255, 255, 0), 2)

    middle_roi_w = 100 + 100 if detect_obj_count > 50 else 100
    middle_roi_h = 100
    middle_roi = road_mask[center_y - middle_roi_h:center_y + middle_roi_h,
                           center_x - middle_roi_w // 2:center_x + middle_roi_w // 2]
    object_detected = np.any(middle_roi == 0)

    if object_detected:
        detect_obj_count += 1
    else:
        #Reduce obj count, so that it realistically slows down and dosent just immediately go to breaking to account
        # for sway in camera
        detect_obj_count = max(detect_obj_count - 1, 0)

    cv.rectangle(vis_overlay,
                 (center_x - middle_roi_w // 2, center_y - middle_roi_h),
                 (center_x + middle_roi_w // 2, center_y + middle_roi_h),
                 (255, 0, 0), 2)

    decision = "Unknown"

    # Define bands and checks
    center_band = road_mask[center_y - check_height:center_y, left_x:right_x]
    center_clear = np.all(center_band == 255)

    left_band = road_mask[detection_y - 100:detection_y + 100, 0:center_x]
    right_band = road_mask[detection_y - 100:detection_y + 100, center_x:]
    left_clear = not np.any(left_band == 0) and has_clear_gap(left_band)
    right_clear = not np.any(right_band == 0) and has_clear_gap(right_band)

    # Object density in blue box
    zero_ratio = np.sum(middle_roi == 0) / middle_roi.size
    print(zero_ratio)

    sudden_jump = zero_ratio - prev_zero_ratio > 0.4  # e.g., 40% increase
    prev_zero_ratio = zero_ratio

    if sudden_jump and zero_ratio >= 0.3:
        decision = "Emergency Stop"

    if detect_obj_count > 150:
        decision = "Stopping"
    elif 60 < detect_obj_count <= 150:
        decision = "Slowing Down"
    else:
        decision = "Moving Forward"

    if zero_ratio >= 0.6:
        if left_clear:
            decision = "Turning Left"
        elif right_clear:
            decision = "Turning Right"
        else:
            decision = "Stopping + Handbrake"
    elif zero_ratio >= 0.3:
        decision = "Slowing Down"
    elif not vertical_clear:
        decision = "Stopping"
    else:
        if not (detect_obj_count > 150):
            decision = "Moving Forward"

    cv.putText(vis_overlay, f"Decision: {decision}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv.putText(vis_overlay, f"Object Count: {detect_obj_count}", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    text_size = cv.getTextSize(f"Road: {road}", cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv.putText(vis_overlay, f"Road: {road}", (w - text_size[0] - 10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    fps = 1 / (new_frame_time - prev_frame_time + 1e-5)  # add small value to avoid div by zero
    fps_text = f"FPS: {int(fps)}"
    cv.putText(vis_overlay, fps_text, (30, h - 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    h, w = road_mask.shape
    y_bottom = h - 10
    y_upper = int(h * 0.75)

    left_bottom, right_bottom = get_lane_edges(road_mask, y_bottom)
    left_upper, right_upper = get_lane_edges(road_mask, y_upper)

    if all(v is not None for v in [left_bottom, right_bottom, left_upper, right_upper]):
        # Offset inward by 10 pixels
        left_bottom += 30
        right_bottom -= 30
        left_upper += 30
        right_upper -= 30

        lane_points = np.array([
            [left_bottom, y_bottom],
            [right_bottom, y_bottom],
            [right_upper, y_upper],
            [left_upper, y_upper]
        ], dtype=np.int32)

        cv.polylines(vis_overlay, [lane_points], isClosed=True, color=(0, 255, 0), thickness=2)

    prev_frame_time = new_frame_time


    cv.imshow("Navigation Debug View", vis_overlay)
    # cv.imshow("Segmented", segmented)
    # cv.imshow("Road Mask", road_mask)
    # cv.imshow("Final Road Overlay", overlay)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
