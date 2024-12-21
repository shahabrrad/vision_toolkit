# Affine Structure from Motion

This repository contains Python code to implement the Affine Structure from Motion (SfM) pipeline, a classical approach for recovering 3D structure and camera motion from 2D image projections of 3D points.

The implementation normalizes input data, performs factorization, resolves affine ambiguity using metric constraints, and refines the structure and motion matrices.

## Files

- **main.py**: 
  - Entry point for running the algorithms.
  - Manages input data, initializes the necessary functions, and displays results.
  
- **utils**: 
  - Contains utility functions that are used across different modules.
  - Provides support for data io preprocessing, visualization
  
- **affine.py**: 
  - Subtracts the centroid of image points to ensure input data is normalized for factorization.
  - Factorizes the normalized data matrix to compute the initial motion (M) and structure (S)   matrices.
  - Applies a correction matrix Q to enforce Euclidean constraints on M and S.

#### Usage

Run the file using the following command:


```bash
python main.py