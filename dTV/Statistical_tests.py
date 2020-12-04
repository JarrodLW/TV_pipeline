import numpy as np
from scipy import stats
import json

A = np.random.normal(scale=3000, size=(32, 32))


with open('dTV/Results_MRI_dTV/dTV_recons_2048_avgs_22102020_SR_to_64.json') as f:
    d = json.load(f)

alphas = [50, 10**2, 5*10**2, 10**3, 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5, 10**6]

fourier_diffs = []

for i, alpha in enumerate(alphas):

    fourier_diff = np.asarray(d['alpha=' + '{:.1e}'.format(alpha)]['fourier_diff'])
    fourier_diff = fourier_diff[0] + 1j * fourier_diff[1]
    fourier_diffs.append(fourier_diff)

np.sum(np.real(fourier_diffs[5]))
np.std(np.real(fourier_diffs[5]))

stats.chisquare(np.ndarray.flatten(A))
stats.chisquare(np.ndarray.flatten(np.real(fourier_diffs[5])))
