# 🚗 PI Navigation System

A Raspberry Pi–based pseudo real-time visual navigation system for autonomous vehicles — powered by classical image processing.

## 📌 Overview

This project demonstrates that robust self-driving technology can be implemented using classical computer vision techniques, without relying on heavy deep learning models. Developed as part of a university term project at the **College of E&ME, NUST**, this system runs entirely on a **Raspberry Pi 5 with a Pi Camera** and processes video frames in real-time.

> 🎥 [Demo Video](https://youtu.be/D4QxFiu2b9I)  
> 📁 [Dataset Download](https://drive.google.com/file/d/1dJYyjc6u08ob8WTFlGNBBppujia_SzGz/view?usp=sharing)

## 👨‍💻 Team Members

- **Wahaaj Nasir** — Reg# 413238  
- **Syed Adnan Aijaz** — Reg# 432028  
- **Muhammad Ali Riaz** — Reg# 432103  
- **Ramsha Fatima** — Reg# 417858

## 🎯 Core Objectives

- Classical lane detection and road segmentation.
- Real-time object detection and avoidance.
- Directional control (Left, Right, Forward, Stop).
- Fully on-device processing (optimized for Raspberry Pi).
- Visual debugging through live overlays.

## 🛠️ Key Features

### ✅ Lane Detection
- Grayscale conversion + contrast stretching
- Canny edge detection and morphological filtering
- K-means segmentation for road masking
- Polynomial lane boundary approximation

### ✅ Object Detection
- Multi-region object detection using contour analysis
- Obstacle classification based on frame ratios and ROIs
- Emergency stop logic via sudden appearance tracking

### ✅ Navigation Logic
- Sloped vs straight road classification
- Decision logic: Stop / Slow Down / Move / Turn Left / Turn Right
- Visual overlay with color-coded bounding boxes and road type

## ⚙️ Technologies Used

- Python
- OpenCV
- NumPy
- Raspberry Pi OS
- PiCamera2

## 🚦 System Flow

- Frame Capture → Preprocessing → Road Segmentation  
- Road Analysis + Object Detection → Navigation Decision  
- Real-Time Display Output via OpenCV GUI

## 🧠 Performance Optimization

- Downsampling frames to 480p for higher FPS (13–15 FPS).
- Efficient morphological operations.
- Lightweight clustering (k-means) for segmenting roads.
- Selective Region of Interest (ROI) processing.

## 🔗 Links

- 📽️ [YouTube Demo](https://youtu.be/D4QxFiu2b9I)  
- 💾 [Dataset](https://drive.google.com/file/d/1dJYyjc6u08ob8WTFlGNBBppujia_SzGz/view)  
- 🌐 [GitHub Repository](https://github.com/WahaajNasir/PI-Navigation)

---

## 🤝 Acknowledgements

- Dr. Usman Akram  
- Dr. Asad Mansoor  
Department of Computer & Software Engineering  
College of E&ME, NUST
