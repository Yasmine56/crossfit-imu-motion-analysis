# crossfit-imu-motion-analysis
Automatic segmentation, repetition counting and movement classification using Apple Watch IMU data (CrossFit workouts).

# Analysis and Recognition of CrossFit Movements Using Inertial Sensors

This project was carried out as part of an R&D mission in partnership with BRAVVO WOD9 and my engineering school IMT Mines Alès.
This work falls within the scope of sports performance analysis and the design of intelligent tools for tracking WODs (Workouts of the Day).

## Objectives

- Automatically recognize CrossFit movements (snatch, front squat, wall ball, etc.) from inertial signals.
- Segment complete WODs to identify each exercise block.
- Count repetitions robustly and automatically.
- Detect errors or variations in execution to improve movement quality.
- Validate models across multiple athletes and WODs.

## Data

The data comes from 7 athletes, each wearing an Apple Watch that recorded:

Accelerations: ACCX, ACCY, ACCZ
Orientation angles: ROLL, PITCH, YAW
Angular velocities: ROTX, ROTY, ROTZ
Altitude data
Environmental audio (not used here)

Two session formats were available:
Sessions per exercise: a single movement per file → used for training.
Full WOD sessions: continuous recording of an entire workout → used for real-world testing.

## Methodology 

1. Time Segmentation — PELT Algorithm

- Application of Pruned Exact Linear Time (PELT) to the acceleration vector norm.
- Use of the MBIC criterion to avoid oversegmentation.
- Pre-sampling (factor k=120) to reduce complexity and improve stability.
  
Evaluation via Jaccard index

2. Repetition Counting

- Analysis of amplitude peaks in the acceleration norm.
- Method based on filtering + dynamic thresholding.
- Calibration by movement (wall ball, snatch, push-ups, etc.).
  
Evaluation via RMSE, MAE, Accuracy.

3. Movement Classification

- Model used : Random Forest (300 trees).
- Input data : Subsampled IMU signals (1 point out of 5).
- No manual feature extraction → direct use of the signal (simple and robust approach).
- Split : 80% training – 20% testing.

Evaluation via accuracy, precision, recall, F1-score, ROC curve.

## Project structure

```
crossfit-imu-motion-analysis/
│
├── data/               # Raw data
├── segmentation/       # PELT + Jaccard evaluation
├── classification/     # Random Forest + evaluation
├── counting/           # Counting methods + evaluation
├── results/            # Powerpoint with figures of ROC, Jaccard, tables, visualisation
└── README.md
```

[Yasmine Aissa / Yasmine56] 
