# Markov Transition Field (MTF) Visualization

This project demonstrates the transformation of a 1D time series—specifically simulating **Electricity Transformer Temperature (ETT)** data—into a 2D **Markov Transition Field (MTF)**. 

MTF is a powerful feature engineering technique that encodes temporal dynamics into a spatial (image) representation, making it highly effective for time series classification using Computer Vision models like CNNs.

---

##  Overview

The workflow follows a structured pipeline to convert raw temporal data into a spatial representation:

1.  **Simulation:** Generating synthetic temperature data with seasonality (sine waves) and Gaussian noise.
2.  **Discretization:** Using `KBinsDiscretizer` to map continuous values into discrete states (bins).
3.  **Adjacency & Transition:** Constructing matrices to represent the probability of moving from one state to another.
4.  **Field Generation:** Using the `pyts` library to encode these probabilities into a $Q \times Q$ field.
5.  **Optimization:** Downsampling the field via `skimage` for efficient visualization and model input.

---

##  Features

* **Synthetic Data Generation:** Simulates ETT dynamics for reproducible testing.
* **State-Space Mapping:** Discretizes data into $N$ bins to define Markovian states.
* **Mathematical Visualizations:** Heatmaps for Adjacency Matrices, Transition Matrices, and MTFs.
* **Image Processing:** Downsampling high-resolution fields for compact storage.
* **Interpretability:** Extracts self-transition probabilities to identify the most "stable" temperature ranges.

---

## Installation and run
- pip install -r requirements.txt
- python model.py
