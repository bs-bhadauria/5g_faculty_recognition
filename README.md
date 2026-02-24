# 5G Faculty Face Recognition System

AI-powered Faculty Face Recognition System designed for Smart 5G Lab environments using OpenCV and MediaPipe.

---

## 📌 Overview

This project implements a facial recognition system capable of identifying faculty members using facial embeddings and similarity-based matching.

The system builds a structured face database, generates embeddings using MediaPipe FaceMesh, and performs recognition using Euclidean distance comparison.

Although currently tested on static images, the architecture is designed to be extendable to real-time 5G-enabled IP camera integration for smart lab automation.

---

## 🚀 Features

- Face embedding generation using MediaPipe FaceMesh
- Euclidean distance–based identity matching
- Faculty-wise dataset organization
- Scalable architecture for future 5G IP camera integration
- Clean Git version control and reproducible environment setup

---

## 🧠 How It Works

1. Faculty images are stored inside structured dataset folders.
2. `build_embeddings.py` generates facial embeddings for each identity.
3. Embeddings are stored in a NumPy file (`embeddings.npy`).
4. `recognize_static.py` compares a test image against stored embeddings.
5. Identity is determined using similarity threshold matching.

---

## 🛠 Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Git & GitHub

---

## 📂 Project Structure

```
5g_faculty_recognition/
│
├── dataset/
│ ├── Faculty_1/
│ ├── Faculty_2/
│
├── build_embeddings.py
├── recognize_static.py
├── capture_faces.py
├── embeddings.npy
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

Clone the repository:

```bash
git clone git@github.com:bs-bhadauria/5g_faculty_recognition.git
cd 5g_faculty_recognition

Create virtual environment:

python -m venv venv
source venv/bin/activate   # Linux / WSL

Install dependencies:

pip install -r requirements.txt

Run embedding generation:

python build_embeddings.py

Run recognition:

python recognize_static.py


🔮 Future Scope

Integration with 5G-enabled RTSP IP cameras

Real-time streaming recognition

Federated Learning–based decentralized model training

Smart attendance & automated lab monitoring


🎓 Academic Note

This project is developed for educational and research purposes to explore computer vision and smart surveillance systems in 5G-enabled environments.


📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

