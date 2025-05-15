import cv2 as cv
import numpy as np

def get_adaptive_range(center_val, margin=0.25):
    lower = max(0, center_val * (1 - margin))
    upper = min(255, center_val * (1 + margin))
    return (int(lower), int(upper))

def hsv_in_range(hsv, h_range, s_range, v_range):
    lower = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
    upper = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)
    return cv.inRange(hsv, lower, upper)

def get_combined_road_mask_hsv(hsv_img, sunny_mean, shadow_mean, sunny_margin=0.25, shadow_margin=0.3):
    # Sunny mask
    h_range = get_adaptive_range(sunny_mean[0], sunny_margin)
    s_range = get_adaptive_range(sunny_mean[1], sunny_margin)
    v_range = get_adaptive_range(sunny_mean[2], sunny_margin)
    sunny_mask = hsv_in_range(hsv_img, h_range, s_range, v_range)

    # Shadow mask
    h_range = get_adaptive_range(shadow_mean[0], shadow_margin)
    s_range = get_adaptive_range(shadow_mean[1], shadow_margin)
    v_range = get_adaptive_range(shadow_mean[2], shadow_margin)
    shadow_mask = hsv_in_range(hsv_img, h_range, s_range, v_range)

    # Morphology
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    sunny_mask_closed = cv.morphologyEx(sunny_mask, cv.MORPH_CLOSE, kernel)

    # Distance filter for shadow mask
    distance = cv.distanceTransform(sunny_mask_closed, cv.DIST_L2, 5)
    close_shadow = np.zeros_like(shadow_mask)
    close_shadow[(shadow_mask == 255) & (distance < 50)] = 255

    # Combine
    combined_mask = cv.bitwise_or(sunny_mask_closed, close_shadow)
    return combined_mask, sunny_mask_closed, close_shadow

def postprocess_mask(mask, min_area=300):
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask)
    result = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] > min_area:
            result[labels == i] = 255
    return result

# ---- Config ----
video_path = r"D:\Uni\Semester 6\DIP\Self\Project\Tesla_Model_Sangi\Dataset\Sunny\Sunny_Street.mp4"
cap = cv.VideoCapture(video_path)

sunny_mean_hsv = np.array([16.73, 30.97, 182.68])
shadow_mean_hsv = np.array([109.26, 26.72, 114.10])

target_width = 1280
target_height = 720

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to 720p
    frame_720p = cv.resize(frame, (target_width, target_height), interpolation=cv.INTER_AREA)

    # Downsample for faster processing
    small = cv.resize(frame_720p, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
    hsv_small = cv.cvtColor(small, cv.COLOR_BGR2HSV)

    # Get sunny + shadow road masks
    mask_small, sunny_mask_small, shadow_mask_small = get_combined_road_mask_hsv(
        hsv_small, sunny_mean_hsv, shadow_mean_hsv)

    # Upsample to full size
    mask_full = cv.resize(mask_small, (target_width, target_height), interpolation=cv.INTER_NEAREST)
    sunny_mask_full = cv.resize(sunny_mask_small, (target_width, target_height), interpolation=cv.INTER_NEAREST)
    shadow_mask_full = cv.resize(shadow_mask_small, (target_width, target_height), interpolation=cv.INTER_NEAREST)

    # Final cleanup
    final_mask = postprocess_mask(mask_full)

    # Overlay output
    overlay = frame_720p.copy()
    overlay[final_mask == 255] = [0, 255, 0]  # Green overlay

    # Display separate windows
    cv.imshow("Overlay", overlay)            # Display the final overlay with road mask
    cv.imshow("Sunny Mask", sunny_mask_full)  # Display the sunny road mask
    cv.imshow("Shadow Mask", shadow_mask_full)  # Display the shadow road mask

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": 'RGB888', "size": (1280, 720)}
    )
)