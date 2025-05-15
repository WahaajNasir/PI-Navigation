import cv2
import numpy as np

# Function to calculate the average HSV of a video
def calculate_avg_hsv_video(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Initialize sums for H, S, and V
    total_h = 0
    total_s = 0
    total_v = 0
    frame_count = 0

    # Loop over each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the mean HSV for the current frame
        mean_hsv = cv2.mean(hsv_frame)[:3]  # H, S, V

        # Update the running totals
        total_h += mean_hsv[0]
        total_s += mean_hsv[1]
        total_v += mean_hsv[2]

        frame_count += 1

    # Release the video capture object
    cap.release()

    # Calculate the average HSV for the video
    if frame_count > 0:
        avg_h = total_h / frame_count
        avg_s = total_s / frame_count
        avg_v = total_v / frame_count
        return avg_h, avg_s, avg_v
    else:
        return None

# Example usage
video_path = r"D:\Uni\Semester 6\DIP\Self\Project\Tesla_Model_Sangi\Dataset\Cloudy\PXL_20250325_043922504.TS.mp4"  # Replace with the path to your video
avg_hsv_video = calculate_avg_hsv_video(video_path)

# Print the result
if avg_hsv_video:
    print(f"Average HSV for the video: H={avg_hsv_video[0]:.2f}, S={avg_hsv_video[1]:.2f}, V={avg_hsv_video[2]:.2f}")
else:
    print("No frames found in the video.")
