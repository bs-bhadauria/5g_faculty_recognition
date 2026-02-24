import cv2
import os
import time

FACULTY_NAME = "Faculty_1"   # baad mein change kar sakte ho
SAVE_DIR = f"dataset/{FACULTY_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)   # USB camera
if not cap.isOpened():
    print("Camera not opened")
    exit()

count = 0
print("Press 'c' to capture image | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Capture Faculty Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        filename = f"{int(time.time())}.jpg"
        path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(path, frame)
        count += 1
        print(f"[Saved] {path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total images captured: {count}")
