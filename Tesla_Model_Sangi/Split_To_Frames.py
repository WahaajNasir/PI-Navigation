import cv2
import os

# Folder where all your videos are stored
videos_folder = r'D:\Uni\Semester 6\DIP\Self\Project\Tesla_Model_Sangi\Dataset\Sunny'
# Folder where all frames will be saved
frames_base_folder = 'extracted_frames'
os.makedirs(frames_base_folder, exist_ok=True)

# Target resolution
target_width = 1280
target_height = 720

# List all video files
video_files = [f for f in os.listdir(videos_folder) if f.endswith('.mp4')]

for video_file in video_files:
    video_path = os.path.join(videos_folder, video_file)
    video_name = os.path.splitext(video_file)[0]

    # Create a folder for this video's frames
    video_frames_folder = os.path.join(frames_base_folder, video_name)
    os.makedirs(video_frames_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to 720p
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        frame_filename = os.path.join(video_frames_folder, f'{video_name}_frame_{frame_num:04d}.jpg')
        cv2.imwrite(frame_filename, resized_frame)
        frame_num += 1

    cap.release()
    print(f"Done extracting {frame_num} frames from {video_file}")

print("All videos processed!")
