from collections import Counter
from sklearn.neighbors import KernelDensity

import numpy as np

print("====================")
# Calculating Discrete Probability
print("Calculating Discrete Probability")

x = [0,1,0,0,0,1]
c = Counter(x)

def probability(a):
    # returns the probability of a given number a
    return float(c[a]) / len(x)

prob = probability(1)
print(prob)
print("====================")

# Calculating Continuous Probability
start = 5  # Start of the range
end = 6    # End of the range
N = 100    # Number of evaluation points

# Step size
step = (end - start) / (N - 1)

# Generate values in the range
x_test = np.linspace(start, end, N)[:, np.newaxis]

# Get PDF values for each x
# Please note that kd.score_samples generates log-likelihood of the data samples.
# Therefore, np.exp is needed to obtain likelihood.
kde_model = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(x_test)
kd_vals = np.exp(kde_model.score_samples(x_test))

# Approximate the integral of the PDF
probability = np.sum(kd_vals * step)
print(probability)

#Alternative using  builtin SciPy integration methods
from scipy.integrate import quad

# Return the integration of a polynomial.
probability = quad(lambda x_test: np.exp(kde_model.score_samples(x_test)), start, end)[0]
print(probability)

print("====================")