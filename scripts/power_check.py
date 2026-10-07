"""Simple paired-design power table for preregistration planning.

This does not use observed experiment outcomes. It reports the detectable mean paired
shift for plausible standard deviations at n=300 using a two-sided alpha=.05 test.
"""
from math import sqrt
from scipy.stats import norm

n = 300
alpha = 0.05
power = 0.80
z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
print("SD_pp,detectable_mean_pp")
for sd in [5, 10, 15, 20, 25, 30]:
    mde = z * sd / sqrt(n)
    print(f"{sd},{mde:.3f}")
