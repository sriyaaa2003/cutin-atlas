# cutin-atlas

**Can you describe a real highway cut-in with a handful of numbers, and can those numbers tell you which cut-ins to simulate?** I extracted 2,322 real cut-ins from the highD drone dataset, replaced the following car with a cruise-control and emergency-braking system in a closed-loop replay, and asked two questions: how few parameters preserve what a simulation test would conclude, and whether grouping cut-ins into clusters lets a simulation campaign reach the same precision with fewer runs.

## What I did

- **Data.** highD: 60 drone recordings of German highways at 25 Hz. A cut-in is a car that changes lane and lands in front of a follower in the target lane (follower within 80 m, speeds above 10 m/s). Each one is stored as a 6 s window (3 s before and after the lane-change frame) of four signals: the cutter's lateral offset in lane widths, the bumper gap, and both speeds. That gives **2,322 cut-ins**.
- **Closed-loop replay.** The cutter follows its recorded (or reconstructed) trajectory. The follower is replaced by a system under test: a constant time-gap cruise control plus an automatic emergency brake that reacts to a delayed, noisy time-to-collision measurement. 8 system settings (brake trigger, sensor delay, brake strength) and 5 noise repeats per cut-in. Everything is vectorised over cut-ins.
- **Study 1, parameterization.** Nine ways to describe the lateral movement, from 4 to 10 parameters each: minimum-jerk quintic, warped minimum-jerk, Bezier curves of degree 3, 5 and 7, a tanh sigmoid, and polynomials of degree 3, 5 and 7 over the whole window. Each reconstructed cut-in is replayed and compared with the real one.
- **Study 2, clustering and stratified sampling.** Four representations (hand-written descriptors, curve parameters, PCA of the raw trajectories, an autoencoder) and four clustering algorithms (k-means, Gaussian mixture, Ward hierarchical, HDBSCAN). Clusters are fitted on one half of the recordings and used to plan sampling on the other half, so nothing about the test cut-ins leaks into the strata. Efficiency is the variance of a plain random sample divided by the variance of the stratified estimate at the same budget, so 2.0 means the same precision from half the simulations. Every comparison includes a control with the cut-ins shuffled between strata.

## Results

**1. A polynomial over the whole window beats the usual segmented maneuver models.**

| description | parameters | median error (cm) | 95th percentile error (cm) | critical cut-ins (TTC < 3 s) missed | real contact cases reproduced |
|---|---|---|---|---|---|
| minimum-jerk quintic | 4 | 19.7 | 38.1 | 18.8% | 10 of 11 |
| warped minimum-jerk | 5 | 16.3 | 30.5 | 19.2% | 8 of 11 |
| Bezier degree 3 | 6 | 3.6 | 19.5 | 11.1% | 10 of 11 |
| Bezier degree 7 | 10 | 0.74 | 19.5 | 11.1% | 10 of 11 |
| tanh sigmoid | 4 | 1.7 | 4.5 | 1.1% | 11 of 11 |
| polynomial degree 3 (window) | 4 | 2.7 | 6.7 | 0.0% | 11 of 11 |
| **polynomial degree 5 (window)** | 6 | 0.70 | 1.8 | 0.0% | 11 of 11 |

- The segmented models need a start and end for the maneuver, and the segmentation fails on some cut-ins, which is why their 95th percentile error stays near 19.5 cm however many parameters they have. Window polynomials need no segmentation.
- Hard-braking agreement between the reconstructed and the real cut-in is 97.2% for the minimum-jerk quintic and 99.9% for the degree-5 window polynomial.

![parameterization](results/figures/parameterization.png)

**2. Unsupervised clusters barely help; clusters that know the outcome help a lot, except for rare events.** At a budget of 100 simulations, best efficiency over the number of strata tried (2 to 32), proportional allocation:

| outcome measured | k-means on descriptors | tree fitted to pilot outcomes | shuffled control |
|---|---|---|---|
| closeness (−log min TTC) | 4.1 | 8.1 | 1.0 |
| hard ACC response (≥ 2 m/s²) | 1.9 | 3.5 | 1.0 |
| critical (min TTC < 3 s) | 1.4 | 1.4 | 1.9 |

- Strata built only from trajectory shape (PCA, curve parameters, autoencoder) give 1.0 to 1.8 on the closeness measure, and a regression tree on descriptors fitted to the pilot half's outcomes reaches 8.1, so what you know about outcomes matters more than how you cluster.
- For rare outcomes there is no usable gain: contact happens in 11 of 2,322 cut-ins, the critical-TTC measure stays at 1.0 to 1.4, and the shuffled control reaches 1.9 on it, higher than the real strata. At that rarity the efficiency estimate itself is too noisy to trust.
- Clusterings are not stable: the adjusted Rand index between clusterings of different halves ranges from 0.2 (Ward on PCA) to 0.99 (Gaussian mixture on descriptors).

![atlas](results/figures/atlas.png)

Full tables: `results/parameterization.csv`, `results/strata.csv`, `results/sampling.csv`.

## How it was built

```
highD csv -> extract (2,322 cut-ins, 6 s windows) -> replay (ACC + AEB, 8 settings x 5 repeats)
          -> study_param (9 parameterizations)  -> study_cluster / study_strata (4 x 4 clusterings, stratified sampling)
          -> figures
```

```bash
pip install -e .[dev]
# put the highD csv files under data/highd, then:
python -m cutin.extract
python -m cutin.study_param
python -m cutin.study_cluster
python -m cutin.study_strata
python -m cutin.figures
pytest
```

Every parameter is in `configs/default.toml`; nothing is hard-coded in `src/`. 8 automated tests.

## Limits

- The system under test and the lead behaviour are simulated; highD contains no crashes between these vehicles, and only 11 of the 2,322 cut-ins end in contact, so every contact statement rests on 11 cases.
- "Best efficiency over k" is chosen on the same splits it is reported on, so it is optimistic; the shuffled control shows the size of that optimism.
- Only passenger cars on German highways.

## Tech stack
Python, NumPy, pandas, SciPy (curve fitting), scikit-learn (k-means, Gaussian mixture, hierarchical, HDBSCAN, PCA, regression trees), PyTorch (autoencoder), matplotlib, pytest.

## Data and credits
The highD dataset is **not redistributed here**. It is described in [arXiv:1810.05642](https://arxiv.org/abs/1810.05642) and available from [highd-dataset.com](https://www.highd-dataset.com/) under its own terms of use. This repository contains only code and aggregate results and figures derived from the data, no per-vehicle data.
