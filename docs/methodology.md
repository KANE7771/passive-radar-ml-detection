# Methodology

## 1. Signal Simulation

Generate reference and surveillance complex baseband signals
with known delay, Doppler, clutter and noise parameters.

## 2. Clutter Suppression

Use ECA-B as the baseline clutter suppression method.

## 3. Range-Doppler Processing

Use the cross-ambiguity function to estimate target delay
and Doppler frequency.

## 4. Detection

Use 2D CA-CFAR as the baseline detection method.

## 5. Localisation

Convert delay measurements into bistatic ranges and estimate
the target position using weighted nonlinear least squares.

## 6. Tracking

Use a constant-velocity Extended Kalman Filter for target tracking.

## 7. Evaluation

Compare estimated results against stored simulated ground truth.
