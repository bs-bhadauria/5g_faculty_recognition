import os
import cv2
import numpy as np
import mediapipe as mp

DATASET_DIR = "dataset"

mp_face = mp.solutions.face_mesh
embeddings = {}

with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        print(f"[INFO] Processing {person}")
        person_vectors = []

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0]
                vector = []
                for p in landmarks.landmark:
                    vector.extend([p.x, p.y, p.z])
                person_vectors.append(np.array(vector))

        if person_vectors:
            embeddings[person] = np.mean(person_vectors, axis=0)
        else:
            print(f"[WARNING] No face detected for {person}")

np.save("embeddings.npy", embeddings)
print("✅ Face embeddings saved as embeddings.npy")
