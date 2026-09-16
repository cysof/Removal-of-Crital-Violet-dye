"""
generate_data.py
-----------------
Creates a SYNTHETIC version of the 270-point crystal violet (CV) dye
adsorption-onto-RHAC dataset described in Chapter 3 of the thesis.

This script exists purely as a placeholder so the rest of the pipeline
(preprocess -> train -> evaluate -> interpret) can be built and tested
before the real experimental dataset is available. When the real data
arrives, drop it in data/raw/cv_removal_data.csv with the same column
names and simply skip this script (or point preprocess.py at the real
file directly).

Experimental design reproduced (One-Factor-At-a-Time, OFAT):
    - Initial concentration C0 (mg/L): 10, 20, 40, 60, 80, 100
    - Adsorbent dosage       (g/L):    0.2, 0.4, 0.6, 0.8, 1.0
    - pH                     (-):      2, 4, 6, 8, 10, 12
    - Contact time           (min):    5, 10, 20, 30, 45, 60, 90, 120
    - Temperature            (C):      25, 35, 45

Baseline ("optimum") levels held constant while one variable is swept,
per the thesis's stated OFAT methodology:
    C0 = 40 mg/L, dose = 0.6 g/L, pH = 6, time = 60 min, temp = 35 C

The output R% is generated from a simple mechanistic-ish surrogate
(pseudo-second-order kinetics x Langmuir-type dose/concentration
saturation x a pH-optimum curve x a mild temperature effect) plus
Gaussian noise, so downstream models have realistic non-linear
structure and interactions to learn -- NOT real experimental data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "cv_removal_data.csv"

# OFAT factor levels
C0_LEVELS = [10, 20, 40, 60, 80, 100]          # mg/L
DOSE_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]        # g/L
PH_LEVELS = [2, 4, 6, 8, 10, 12]
TIME_LEVELS = [5, 10, 20, 30, 45, 60, 90, 120]  # min
TEMP_LEVELS = [25, 35, 45]                      # C

# Baseline levels (held constant while sweeping one variable)
BASE_C0 = 40
BASE_DOSE = 0.6
BASE_PH = 6
BASE_TIME = 60
BASE_TEMP = 35

N_REPLICATES_TARGET = 270  # thesis states 270 measurements total


def build_ofat_grid():
    """Builds an OFAT design: for each variable, sweep its levels while
    holding all others at baseline. Includes the baseline point once."""
    rows = []

    # Baseline (all at base) — anchor point
    rows.append(dict(C0=BASE_C0, dose=BASE_DOSE, pH=BASE_PH,
                      time=BASE_TIME, temp=BASE_TEMP))

    def sweep(varname, levels, base_row):
        for lvl in levels:
            row = dict(base_row)
            row[varname] = lvl
            if row not in rows:
                rows.append(row)

    base_row = dict(C0=BASE_C0, dose=BASE_DOSE, pH=BASE_PH,
                     time=BASE_TIME, temp=BASE_TEMP)

    sweep("C0", C0_LEVELS, base_row)
    sweep("dose", DOSE_LEVELS, base_row)
    sweep("pH", PH_LEVELS, base_row)
    sweep("time", TIME_LEVELS, base_row)
    sweep("temp", TEMP_LEVELS, base_row)

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return df


def simulate_removal(df, rng):
    """Simulates % removal (R%) using a plausible non-linear surrogate:
    - Pseudo-second-order-like saturating rise with contact time
    - Langmuir-type saturation with dose (diminishing returns)
    - Inverse relationship with initial concentration (adsorbent gets overloaded)
    - pH optimum around 6-8 (cationic dye adsorbs better at higher pH
      on many activated-carbon surfaces up to a point)
    - Mild positive temperature effect (endothermic-like behaviour)
    plus interaction terms and noise, then triplicate-averaged and
    clipped to a physical 0-100% range.
    """
    C0 = df["C0"].values.astype(float)
    dose = df["dose"].values.astype(float)
    pH = df["pH"].values.astype(float)
    t = df["time"].values.astype(float)
    T = df["temp"].values.astype(float)

    # Kinetic saturation term (pseudo-second-order shape)
    k_time = 0.06
    time_term = (k_time * t) / (1 + k_time * t)  # 0 -> ~1 as t grows

    # Dose term: more adsorbent -> more removal, saturating (Langmuir-like)
    dose_term = dose / (dose + 0.35)

    # Concentration term: higher C0 -> relatively lower % removal (site limited)
    conc_term = 1.0 / (1.0 + (C0 / 55.0) ** 1.1)

    # pH optimum around pH ~ 7 for cationic dye on activated carbon
    ph_term = np.exp(-((pH - 7.0) ** 2) / (2 * 3.2 ** 2))

    # Mild temperature effect (slightly favours higher T)
    temp_term = 0.85 + 0.15 * (T - 25) / 20.0

    # Combine (weights chosen so R% lands roughly in 20-99% realistic range)
    base = 100 * (
        0.30 * time_term
        + 0.25 * dose_term
        + 0.20 * conc_term
        + 0.15 * ph_term
        + 0.10 * temp_term
    )

    # A couple of mild interaction effects
    interaction = 3.0 * (dose_term * time_term) - 2.0 * (conc_term * (1 - ph_term))
    r_percent = base + interaction

    # Triplicate runs averaged, each replicate has independent noise
    noise = rng.normal(loc=0.0, scale=1.8, size=(len(df), 3)).mean(axis=1)
    r_percent = r_percent + noise

    r_percent = np.clip(r_percent, 0, 100)
    return np.round(r_percent, 2)


def pad_to_target(df, target_n, rng):
    """The pure OFAT sweep produces fewer than 270 unique rows (levels
    overlap at the baseline). To match the thesis's stated dataset size
    of 270, additional rows are sampled with replacement from the OFAT
    combinations (mirroring repeated/triplicate-style measurements),
    each re-simulated independently so they are not exact duplicates."""
    if len(df) >= target_n:
        return df.iloc[:target_n].reset_index(drop=True)

    n_extra = target_n - len(df)
    extra_idx = rng.choice(len(df), size=n_extra, replace=True)
    extra = df.iloc[extra_idx].reset_index(drop=True)
    return pd.concat([df, extra], ignore_index=True)


def main():
    rng = np.random.default_rng(RNG_SEED)

    grid = build_ofat_grid()
    grid = pad_to_target(grid, N_REPLICATES_TARGET, rng)

    grid["R_percent"] = simulate_removal(grid, rng)

    # Rename to thesis-style column headers
    grid = grid.rename(columns={
        "C0": "Initial_Concentration_mgL",
        "dose": "Adsorbent_Dosage_gL",
        "pH": "pH",
        "time": "Contact_Time_min",
        "temp": "Temperature_C",
        "R_percent": "Removal_Efficiency_percent",
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT_PATH, index=False)

    print(f"Synthetic dataset written: {OUT_PATH}")
    print(f"Shape: {grid.shape}")
    print(grid.describe().round(2))


if __name__ == "__main__":
    main()