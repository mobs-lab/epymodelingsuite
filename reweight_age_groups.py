"""
Reweight an age-varying parameter from one set of age groups to another
using US population data from the population codebook.

Method:
  1. Read single-year-of-age US population from the codebook.
  2. Assign each single year of age the parameter value from its original group.
  3. For each target age group, compute the population-weighted average:
         φ_new = Σ(pop_a * φ_a) / Σ(pop_a)   for all ages a in the target group.
"""

import csv
import os
import json

# ---------- paths ----------------------------------------------------------
CODEBOOK_PATH = os.path.join(
    os.path.dirname(__file__),
    "epymodelingsuite", "data", "population_codebook.csv",
)

# ---------- original parameter: Hospitalization Rate (φ) -------------------
# Keys are (lower_bound, upper_bound_inclusive)
original_param = {
    (0, 9): 0.001,
    (10, 19): 0.001,
    (20, 24): 0.005,
    (25, 29): 0.005,
    (30, 39): 0.011,
    (40, 49): 0.014,
    (50, 59): 0.029,
    (60, 69): 0.058,
    (70, 79): 0.093,
    (80, 120): 0.262,   # 80+ (upper bound set high to capture all ages)
}

# ---------- target age groups ---------------------------------------------
target_groups = {
    "0_4":   (0, 4),
    "5_17":  (5, 17),
    "18_49": (18, 49),
    "50_64": (50, 64),
    "65+":   (65, 120),
}

# ---------- read US population by single year of age ----------------------
def read_us_population(path: str) -> dict[int, int]:
    """Return {age: population} for the United States column."""
    pop = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # location names
        us_col = header.index("United_States")
        for row in reader:
            if not row or not row[0].strip():
                continue
            age = int(row[0])
            pop[age] = int(row[us_col])
    return pop


def rate_for_age(age: int) -> float:
    """Return the original hospitalization rate for a given single year of age."""
    for (lo, hi), rate in original_param.items():
        if lo <= age <= hi:
            return rate
    raise ValueError(f"No original rate covers age {age}")


def reweight(pop: dict[int, int], groups: dict) -> dict[str, float]:
    """Compute population-weighted parameter for each target group."""
    result = {}
    for label, (lo, hi) in groups.items():
        numerator = 0.0
        denominator = 0
        for age in range(lo, hi + 1):
            if age in pop:
                p = pop[age]
                numerator += p * rate_for_age(age)
                denominator += p
        if denominator == 0:
            raise ValueError(f"No population data for group {label}")
        result[label] = numerator / denominator
    return result


def main():
    pop = read_us_population(CODEBOOK_PATH)
    print(f"Loaded population for {len(pop)} single-year age groups")
    print(f"Total US population: {sum(pop.values()):,}\n")

    reweighted = reweight(pop, target_groups)

    print("=" * 60)
    print("Original Hospitalization Rate (φ) by age group")
    print("=" * 60)
    for (lo, hi), rate in original_param.items():
        label = f"{lo}_{hi}" if hi < 120 else f"{lo}+"
        print(f"  {label:>8s}: {rate}")

    print()
    print("=" * 60)
    print("Reweighted Hospitalization Rate (φ) for target age groups")
    print("=" * 60)
    for label, rate in reweighted.items():
        print(f"  {label:>8s}: {rate:.6f}")

    # Also output as a JSON-style dict for easy copy-paste
    print()
    print("As dictionary:")
    formatted = {k: round(v, 6) for k, v in reweighted.items()}
    print(json.dumps(formatted, indent=2))

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), "reweighted_hospitalization_rate.json")
    with open(output_path, "w") as f:
        json.dump(formatted, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
