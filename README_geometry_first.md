# passive-radar-ml-detection

## Passive Radar Target Localisation

This project develops a simulation-based passive-radar processing pipeline for 2D target localisation and tracking.

The current implementation follows a **geometry-first simulation approach**. Instead of manually assigning target delay and Doppler values, the transmitter (Tx), receivers (Rx), target position, and target velocity are defined first. The bistatic path geometry is then used to derive the target delay and Doppler before the simulated surveillance signals are generated.

---

## Project Objective

The project investigates how weak target echoes can be isolated from direct-path interference and clutter, then converted into reliable measurements for localisation and tracking.

The planned processing pipeline is:

1. Signal Simulation
2. Clutter Suppression
3. Range-Doppler Processing
4. Target Detection
5. Target Localisation
6. Target Tracking
7. Evaluation

---

## Current Geometry-First Simulation

The baseline simulation currently defines:

- one stationary transmitter;
- three receivers at known 2D positions;
- one moving point target;
- a known target velocity;
- a configurable sampling frequency, carrier frequency, bandwidth, duration, signal amplitudes, and noise level.

For each receiver, the simulator calculates:

1. Tx-to-target distance;
2. target-to-Rx distance;
3. direct Tx-to-Rx distance;
4. bistatic excess range;
5. bistatic delay;
6. fractional delay in samples;
7. geometry-derived bistatic Doppler.

The bistatic excess range is

\[
R_b = \|p-T\| + \|p-R_k\| - \|T-R_k\|
\]

where:

- \(T\) is the transmitter position;
- \(R_k\) is receiver \(k\)'s position;
- \(p\) is the target position.

The corresponding bistatic delay is

\[
\tau_b = \frac{R_b}{c}
\]

and the delay in discrete samples is

\[
N_{delay} = \tau_b f_s
\]

where \(c\) is the speed of light and \(f_s\) is the sampling frequency.

The target Doppler is also derived from the Tx-Rx-target geometry and target velocity rather than assigned manually.

---

## Simulated Signal Model

The current baseline generates:

- a reproducible band-limited complex baseband reference signal;
- a geometry-derived target echo;
- fractional propagation delay;
- Doppler frequency shift;
- a direct-path component;
- additive complex receiver noise.

The simplified surveillance signal currently contains:

\[
\text{surveillance}
=
\text{direct path}
+
\text{target echo}
+
\text{noise}
\]

Clutter and stationary multipath will be added after the geometry-derived signal model has been validated.

---

## Project Structure

```text
passive-radar-ml-detection/
│
├── configs/
│   ├── __init__.py
│   └── baseline_config.py
│
├── data/
│
├── docs/
│   └── methodology.md
│
├── results/
│   └── truth.json
│
├── src/
│   ├── __init__.py
│   ├── ambiguity_func.py
│   ├── simulator.py
│   └── yolovX_detector.py
│
├── tests/
│   └── test_simulator.py
│
├── .gitignore
└── README.md
```

---

## Baseline Configuration

The main simulation parameters are stored in:

```text
configs/baseline_config.py
```

This includes:

- random seed;
- speed of light;
- sampling frequency;
- signal duration;
- carrier frequency;
- bandwidth;
- transmitter position;
- receiver positions;
- target initial position;
- target velocity;
- direct-path amplitude;
- target amplitude;
- noise level.

Keeping these parameters separate from the processing code supports controlled and reproducible experiments.

---

## Reproducibility and Validation

The project uses fixed random seeds so that identical configurations regenerate identical reference signals.

Current automated tests include:

### 1. Reference-signal reproducibility

Checks that repeated runs with the same seed generate identical reference signals.

### 2. Geometry-derived delay consistency

Checks that the derived bistatic delay is consistent with the calculated bistatic excess range:

\[
R_b = c\tau_b
\]

### 3. Stationary-target Doppler check

Checks that a target with zero velocity produces zero Doppler in the current model.

Run the tests from the project root with:

```powershell
python -m pytest tests/test_simulator.py -v -s
```

---

## Running the Baseline Simulator

From the project root, run:

```powershell
python -m src.simulator
```

The program reports, for each receiver:

- receiver position;
- bistatic excess range;
- bistatic delay;
- fractional delay in samples;
- geometry-derived Doppler.

The simulation truth is saved to:

```text
results/truth.json
```

This stored truth will later be compared with delay, Doppler, localisation, and tracking estimates.

---

## Development Environment

- Python
- PyCharm
- Git
- GitHub / GitHub Desktop
- NumPy
- pytest

---

## Current Status

Completed:

- passive-radar background study;
- processing-pipeline definition;
- Python/Git project structure;
- geometry-first simulation configuration;
- Tx/Rx/target geometry definition;
- geometry-derived bistatic range and delay;
- geometry-derived Doppler;
- fractional-delay implementation;
- reproducible reference-signal generation;
- automated geometry and reproducibility tests;
- stored simulation ground truth.

Next steps:

1. verify known delay and Doppler peaks using correlation / ambiguity processing;
2. generate the first validated Range-Doppler map;
3. introduce clutter and direct-path suppression;
4. implement target detection;
5. progress to multistatic localisation and tracking.

---

## Important Modelling Principle

Manual delay and Doppler values are useful for software unit tests, but the main passive-radar simulation should be physically consistent.

Therefore, the baseline follows:

```text
Define Tx / Rx / Target geometry
            ↓
Calculate propagation paths
            ↓
Calculate bistatic range and delay
            ↓
Calculate Doppler from geometry and motion
            ↓
Generate simulated target echo
            ↓
Generate surveillance signal
            ↓
Estimate measurements
            ↓
Compare estimates with stored ground truth
```

This geometry-first design ensures that the simulated passive-radar measurements are connected to a defined physical scenario rather than arbitrary signal parameters.
