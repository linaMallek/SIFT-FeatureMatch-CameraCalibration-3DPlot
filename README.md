# Feature Matching and Camera Calibration with 3D Visualization

This project demonstrates feature matching using SIFT (Scale-Invariant Feature Transform) in OpenCV, camera calibration with checkerboard images, and the visualization of 3D coordinates using Matplotlib.

## Features
- **SIFT Feature Matching**: Detects and matches keypoints between two images using BFMatcher.
- **Camera Calibration**: Performs camera calibration using multiple checkerboard images to compute the camera matrix and distortion coefficients.
- **3D Plotting**: Visualizes the matched points in 3D space with dynamic zoom controls.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## How to Run
1. Install the required dependencies:
   ```bash
   pip install opencv-python numpy matplotlib
