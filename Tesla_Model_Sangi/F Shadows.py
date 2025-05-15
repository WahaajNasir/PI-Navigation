import cv2 as cv
import numpy as np
import time

# ---------- HSV-based Segmentation Utilities ----------

def get_adaptive_range(center_val, margin=0.25):
    lower = max(0, center_val * (1 - margin))
    upper = min(255, center_val * (1 + margin))
    return (int(lower), int(upper))

def hsv_in_range(hsv, h_range, s_range, v_range):
    lower = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
    upper = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)
    return cv.inRange(hsv, lower, upper)

def get_combined_road_mask_hsv(hsv_img, sunny_mean, shadow_mean, sunny_margin=0.25, shadow_margin=0.3):
    sunny_h, sunny_s, sunny_v = [get_adaptive_range(val, sunny_margin) for val in sunny_mean]
    shadow_h, shadow_s, shadow_v = [get_adaptive_range(val, shadow_margin) for val in shadow_mean]

    sunny_mask = hsv_in_range(hsv_img, sunny_h, sunny_s, sunny_v)
    shadow_mask = hsv_in_range(hsv_img, shadow_h, shadow_s, shadow_v)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    sunny_mask_closed = cv.morphologyEx(sunny_mask, cv.MORPH_CLOSE, kernel)

    distance = cv.distanceTransform(sunny_mask_closed, cv.DIST_L2, 5)
    close_shadow = np.zeros_like(shadow_mask)
    close_shadow[(shadow_mask == 255) & (distance < 50)] = 255

    return cv.bitwise_or(sunny_mask_closed, close_shadow)

# ---------- Grayscale + KMeans Utilities ----------

def contrast_stretching(image):
    min_val, max_val = np.min(image), np.max(image)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

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
    mapped[label_img == road_label] = 255
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

# ---------- Unified Road Mask Selector ----------

def compute_road_mask(frame, sunny_mean, cloudy_mean):
    h, w = frame.shape[:2]
    small = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    hsv_small = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv_small.reshape(-1, 3), axis=0)

    dist_sunny = np.linalg.norm(avg_hsv - sunny_mean)
    dist_cloudy = np.linalg.norm(avg_hsv - cloudy_mean)

    if dist_sunny < dist_cloudy:
        hsv_full = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        sunny_road_hsv = np.array([16.73, 30.97, 182.68])
        shadow_road_hsv = np.array([109.26, 26.72, 114.10])
        road_mask = get_combined_road_mask_hsv(hsv_full, sunny_road_hsv, shadow_road_hsv)
    else:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        contrast = contrast_stretching(gray)
        median_filtered = cv.medianBlur(contrast, 5)
        edges = cv.Canny(median_filtered, 50, 150)
        closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv.findContours(closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv.drawContours(median_filtered, [c], -1, 0, cv.FILLED)

        down = cv.resize(median_filtered, (0, 0), fx=0.25, fy=0.25)
        seg_down = kmeans_segmentation_fast(down)
        segmented = cv.resize(seg_down, (w, h), interpolation=cv.INTER_NEAREST)
        closed = cv.morphologyEx(cv.morphologyEx(segmented, cv.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        road_mask = np.zeros_like(closed)
        road_mask[closed == 255] = 255

        mask_75 = np.zeros_like(road_mask)
        mask_75[int(h * 0.25):, :] = 255
        road_mask = cv.bitwise_and(road_mask, mask_75)

        road_mask = remove_small_objects(road_mask, 25000)
        road_mask = remove_small_zero_objects(road_mask, 5000)

    return road_mask

# ---------- Main Navigation Loop ----------


def main():
    cap = cv.VideoCapture(r"D:\Uni\Semester 6\DIP\Self\Project\Tesla_Model_Sangi\Dataset\Sunny\Sunny ESG.mp4")
    sunny_mean = np.array([64.53, 46.46, 132.16])
    cloudy_mean = np.array([83.71, 26.45, 168.85])
    detect_obj_count = 0
    road_classification_history = []
    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, (1280, 720))
        new_frame_time = time.time()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        road_mask = compute_road_mask(frame, sunny_mean, cloudy_mean)
        overlay = cv.bitwise_and(gray, gray, mask=road_mask)
        vis_overlay = cv.cvtColor(overlay, cv.COLOR_GRAY2BGR)

        h, w = road_mask.shape
        center_x, center_y = w // 2, h // 2 + 50
        detection_y = center_y - 50

        top_y = np.argmax(np.any(road_mask == 255, axis=1)) + 50
        bottom_y = h - np.argmax(np.any(road_mask[::-1] == 255, axis=1)) - 1

        cv.line(vis_overlay, (0, top_y), (w, top_y), (255, 0, 255), 2)
        cv.line(vis_overlay, (0, bottom_y), (w, bottom_y), (0, 255, 255), 2)

        top_px = np.sum(road_mask[top_y, :] == 255)
        bottom_px = np.sum(road_mask[bottom_y, :] == 255)
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

        middle_roi_w = 200 if detect_obj_count > 50 else 100
        middle_roi_h = 100
        middle_roi = road_mask[center_y - middle_roi_h:center_y + middle_roi_h,
                               center_x - middle_roi_w // 2:center_x + middle_roi_w // 2]
        object_detected = np.any(middle_roi == 0)
        if object_detected:
            detect_obj_count += 1
        else:
            detect_obj_count = max(detect_obj_count - 1, 0)

        decision = "Unknown"
        zero_ratio = np.sum(middle_roi == 0) / middle_roi.size
        center_band = road_mask[center_y - 75:center_y, center_x - 50:center_x + 50]
        vertical_clear = np.all(center_band == 255)
        left_band = road_mask[detection_y - 100:detection_y + 100, 0:center_x]
        right_band = road_mask[detection_y - 100:detection_y + 100, center_x:]
        left_clear = not np.any(left_band == 0) and has_clear_gap(left_band)
        right_clear = not np.any(right_band == 0) and has_clear_gap(right_band)

        if detect_obj_count > 150:
            decision = "Stopping"
        elif 60 < detect_obj_count <= 150:
            decision = "Slowing Down"
        else:
            decision = "Moving Forward"

        if zero_ratio >= 0.6:
            decision = "Turning Left" if left_clear else "Turning Right" if right_clear else "Stopping"
        elif zero_ratio > 0.3:
            decision = "Stopping"

        if vertical_clear and decision == "Moving Forward":
            decision = "Clear for Straight"

        # ---------- Draw Rectangles ----------
        # 🔴 Red rectangle – vertical check band
        cv.rectangle(vis_overlay,
                     (center_x - 50, center_y - 75),
                     (center_x + 50, center_y),
                     (0, 0, 255), 2)

        # 🟩 Green rectangles – left and right check bands
        cv.rectangle(vis_overlay,
                     (center_x - 200, detection_y - 100),
                     (center_x, detection_y + 100),
                     (0, 255, 0), 2)

        cv.rectangle(vis_overlay,
                     (center_x, detection_y - 100),
                     (center_x + 200, detection_y + 100),
                     (0, 255, 0), 2)

        # 🔵 Blue rectangle – object detection ROI
        cv.rectangle(vis_overlay,
                     (center_x - middle_roi_w // 2, center_y - middle_roi_h),
                     (center_x + middle_roi_w // 2, center_y + middle_roi_h),
                     (255, 0, 0), 2)

        # ---------- Display Text ----------
        cv.putText(vis_overlay, f"Decision: {decision}", (30, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv.putText(vis_overlay, f"Object Count: {detect_obj_count}", (30, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        fps = 1 / (new_frame_time - prev_frame_time + 1e-5)  # add small value to avoid div by zero
        fps_text = f"FPS: {int(fps)}"
        cv.putText(vis_overlay, fps_text, (30, h - 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        prev_frame_time = new_frame_time

        # ---------- Show Output ----------
        cv.imshow("Road Detection", vis_overlay)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
