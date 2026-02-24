import cv2
import numpy as np
import mediapipe as mp

# Load embeddings
data = np.load("embeddings.npy", allow_pickle=True).item()

# Load test image
img = cv2.imread("test_image.jpg")
if img is None:
    print("❌ test_image.jpg not found")
    exit()

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# MediaPipe FaceMesh
mp_face = mp.solutions.face_mesh
with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        print("❌ No face detected in test image")
        exit()

    # Extract face embedding from test image
    landmarks = result.multi_face_landmarks[0]
    test_vector = []
    for p in landmarks.landmark:
        test_vector.extend([p.x, p.y, p.z])
    test_vector = np.array(test_vector)

# Compare with stored embeddings
best_match = None
best_distance = float("inf")

for name, emb in data.items():
    dist = np.linalg.norm(test_vector - emb)
    print(f"Distance with {name}: {dist:.4f}")
    if dist < best_distance:
        best_distance = dist
        best_match = name

print("\n✅ IDENTIFICATION RESULT")
print(f"Matched Faculty : {best_match}")
print(f"Distance Score  : {best_distance:.4f}")
