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
print("\nUniform linear distribution start=%s, end=%s, N=%s, Step size=%s" % (start, end, N,
                                                                             step))

# numpy.linspace - Return evenly spaced numbers over a specified interval i.e. uniformly distributed in linear space
# numpy.logspace - Return numbers spaced evenly on a log scale i.e. uniformly distributed in log space

# Generate values in the range i.e. we can generate a set of points equidistant from each
# other and estimate the kernel density at each point.
# np.newaxis might come in handy when you want to explicitly convert a 1D array to either a row vector or a column vector,
# make it as column vector by inserting an axis along second dimension

x_test_1d = np.linspace(start, end, N)

print("\nGenerated synthetic data from a uniform linear distribution:")
print(x_test_1d)

x_test = x_test_1d[:, np.newaxis] # make it as column vector by inserting an axis along second dimension

print("\nShape of synthetic data from a uniform linear distribution:")
print(x_test.shape)

# Get PDF values for each x
# Please note that kd.score_samples generates log-likelihood of the data samples.
# Therefore, np.exp is needed to obtain likelihood.
# When fitting a model your X needs to be 2D array. i.e (n_samples, n_features).
kde_model = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(x_test)

kd_vals = np.exp(kde_model.score_samples(x_test))
print("\nLikelihood from uniform linear distribution synthetic data i.e. density from KDE:")
print(kd_vals)

# Approximate the integral of the PDF
probability = np.sum(kd_vals * step)
print("\nSumming the integrals of PDF from uniform linear distribution synthetic data  i.e. Probability:")
print(probability)

#Alternative using builtin SciPy integration methods
from scipy.integrate import quad

# Return the integration of a polynomial.
# The function quad is provided to integrate a function of one variable between two points

# Reshape your data using array.reshape(-1, 1) if your data has a single feature

# When you use .reshape(1, -1) it adds one dimension to the data.
# Reshape your data using array.reshape(1, -1) if it contains a single sample
# i.e. np.float64(x).reshape(1,-1) gives array([[ 0.]], dtype=float64) which is akin to [[x]]
fxn = lambda x: np.exp(kde_model.score_samples(np.float64(x).reshape(1,-1)))
probability = quad(fxn, start, end)[0]
# (quad returns a tuple where the first index is the result,# therefore the [0])
print("\nIntegral of PDF i.e. Probability ( using builtin and more accurate SciPy integration methods):")
print(probability)

print("====================")
# Generating Synthetic Data from 2 distributions - an asymmetric log-normal distribution and the other one is a Gaussian distribution
print("\nGenerating Synthetic Data from an asymmetric log-normal distribution and the other one is a Gaussian distribution")

def generate_data(mu1, sigma1, size1, mu2, sigma2, size2, seed):
    # Fix the seed to reproduce the results
    rand = np.random.RandomState(seed)

    # Draw samples from a log-normal distribution with specified mean, standard deviation, and array shape.
    # Note that the mean and standard deviation are not the values for the distribution itself,
    # but of the underlying normal distribution it is derived from.
    x = []
    # mean, standard deviation, size
    dat = rand.lognormal(mean=mu1, sigma=sigma1, size=size1)
    x = np.concatenate((x, dat))

    #loc - (Mean) where the peak of the bell exists.
    # scale - (Standard Deviation) how flat the graph distribution should be.
    # size - The shape of the returned array.
    # mean, standard deviation, size
    dat = rand.normal(loc=mu2, scale=sigma2, size=size2)
    x = np.concatenate((x, dat))
    return x

mu1, sigma1, size1 = 0, 0.3, 1000  # mean, standard deviation, size
mu2, sigma2, size2 = 3, 1, 1000  # mean, standard deviation, size
seed=17
x_train_1d = generate_data(mu1, sigma1, size1, mu2, sigma2, size2, seed) #one dimension i.e. 1d

print("\n Synthetic data from an asymmetric log-normal distribution (mean=%s, sigma=%s,size=%s) and a "
      "\n Gaussian distribution (mean=%s, standard deviation=%s, size=%s), with shape %s "
      "and length %s:" % (mu1, sigma1, size1,mu2, sigma2, size2, x_train_1d.shape, x_train_1d.size))
print(x_train_1d)
x_train = x_train_1d[:, np.newaxis] # make it as column vector by inserting an axis along second dimension

# ndarray.size - # Number of elements in the array. Caclulated as np.prod(a.shape), i.e., the product of the array’s dimensions.
print("====================")

# Generating Synthetic Data from two Gaussian distributions
print("\nGenerating Synthetic Data from two Gaussian distributions")

def generate_synthetic_data2(mu1, sigma1, size1, mu2, sigma2, size2, seed):
    # Fix the seed to reproduce the results
    rand = np.random.RandomState(seed)
    x = []
    dat = rand.normal(mu1, sigma1, size1)
    x = np.concatenate((x, dat))
    dat = rand.normal(mu2, sigma2, size2)
    x = np.concatenate((x, dat))
    return x

mu1, sigma1, size1 = 6, 1, 1000  # mean, standard deviation, size
mu2, sigma2, size2 = 3, 1, 1000  # mean, standard deviation, size
seed=17
x_train2_1d = generate_synthetic_data2(mu1, sigma1, size1, mu2, sigma2, size2, seed)
print("\n Synthetic data from two Gaussian distributions (mean=%s, sigma=%s,size=%s) and (mean=%s, sigma=%s,size=%s)\n"
      "with shape %s and length %s:" % (mu1, sigma1, size1,mu2, sigma2, size2, x_train2_1d.shape, x_train2_1d.size))
print(x_train2_1d)
x_train2 = x_train2_1d[:, np.newaxis]
print("====================")