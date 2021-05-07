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
print("\nStep size:")
print(step)

# Generate values in the range i.e. we can generate a set of points equidistant from each other
# and estimate the kernel density at each point.
x_test = np.linspace(start, end, N)[:, np.newaxis]
print("\nGenerated data:")
print(x_test)

# Get PDF values for each x
# Please note that kd.score_samples generates log-likelihood of the data samples.
# Therefore, np.exp is needed to obtain likelihood.
kde_model = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(x_test)

kd_vals = np.exp(kde_model.score_samples(x_test))
print("\nLikelihood:")
print(kd_vals)


# Approximate the integral of the PDF
probability = np.sum(kd_vals * step)
print("\nIntegral of PDF i.e. Probability:")
print(probability)

#Alternative using builtin SciPy integration methods
from scipy.integrate import quad

# Return the integration of a polynomial.
#probability = quad(lambda x: np.exp(kde_model.score_samples(x)), start, end)[0]
#print("\nIntegral of PDF i.e. Probability ( using builtin and more accurate SciPy integration methods):")
#print(probability)

print("====================")
# Generating Synthetic Data from 2 distributions - an asymmetric log-normal distribution and the other one is a Gaussian distribution
print("\nGenerating Synthetic Data from an asymmetric log-normal distribution and the other one is a Gaussian distribution")

def generate_data(seed=17):
    # Fix the seed to reproduce the results
    rand = np.random.RandomState(seed)
    x = []
    dat = rand.lognormal(0, 0.3, 1000)
    x = np.concatenate((x, dat))
    dat = rand.normal(3, 1, 1000)
    x = np.concatenate((x, dat))
    return x

x_train = generate_data()[:, np.newaxis]
print("\n Synthetic data froman asymmetric log-normal distribution and a Gaussian distribution:")
print(x_train)
print("====================")

# Generating Synthetic Data from two Gaussian distributions
print("\nGenerating Synthetic Data from two Gaussian distributions")

def generate_synthetic_data2(seed=17):
    # Fix the seed to reproduce the results
    rand = np.random.RandomState(seed)
    x = []
    dat = rand.normal(6, 1, 1000)
    x = np.concatenate((x, dat))
    dat = rand.normal(3, 1, 1000)
    x = np.concatenate((x, dat))
    return x

x_train2 = generate_synthetic_data2()[:, np.newaxis]
print("\n Synthetic data from two Gaussian distributions:")
print(x_train2)
print("====================")