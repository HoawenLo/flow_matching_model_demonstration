import cv2
import os

FRAMES_DIR = "frames"
OUTPUT = "flow_training.avi"
FPS = 10

frame_files = sorted([
    f for f in os.listdir(FRAMES_DIR)
    if f.endswith(".png")
])

if not frame_files:
    raise ValueError("No frames found!")

# Read first frame
first_frame = cv2.imread(os.path.join(FRAMES_DIR, frame_files[0]))
h, w = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(OUTPUT, fourcc, FPS, (w, h))

for f in frame_files:
    path = os.path.join(FRAMES_DIR, f)
    img = cv2.imread(path)
    img = cv2.resize(img, (w, h))  # ensure consistent size
    writer.write(img)

writer.release()
print(f"Video saved to: {OUTPUT}")