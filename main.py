import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, binom, poisson, uniform, beta
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d
from scipy.special import comb
from scipy.stats import gaussian_kde


def create_cdf(num_values, distribution_type='normal', params={}):
    # selecting the distribution
    # can't generalize binomial or poisson
    if distribution_type == 'normal':
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        # to get num evenly spaced numbers over the interval from start to stop
        start = -10
        stop = 10
        x = np.linspace(start, stop, num_values)
        cdf_values = norm.cdf(x, mean, std)
        label = f'Normal(mean={mean}, std={std})'

    elif distribution_type == 'exponential':
        scale = params.get('scale', 1)
        # exponential is defined on from 0
        start = 0
        stop = 20
        x = np.linspace(start, stop, num_values)
        cdf_values = expon.cdf(x, scale=scale)
        label = f'Exponential(scale={scale})'

    elif distribution_type == 'uniform':
        start = params.get('low', 0)
        stop = params.get('high', 1)
        x = np.linspace(start, stop, num_values)
        cdf_values = uniform.cdf(x, start, stop - start)
        label = f'Uniform(low={start}, high={stop})'

    elif distribution_type == 'beta':
        alpha = params.get('alpha', 2)
        beta_param = params.get('beta', 2)
        # Beta distributions are defined in [0, 1]
        start = 0
        stop = 1
        x = np.linspace(start, stop, num_values)
        cdf_values = beta.cdf(x, alpha, beta_param)

    else:
        print("Wrong distribution type.")
        return None

    return cdf_values, x, start, stop


def define_cdf():
    print("Select a distribution to plot its CDF: 1. Normal 2. Exponential 3. Uniform 4. Beta")
    choice = input("Enter the number of your choice: ")

    num_values = 100

    bpt_order = int(input("What order of BPT do you want to implement? Has to be smaller then points used (default 125): ") or 125)
    if bpt_order > num_values:
        bpt_order = num_values

    distribution_type = ''
    params = {}

    if choice == '1':
        distribution_type = 'normal'
        params['mean'] = float(input("Enter mean (default 0): ") or 0)
        params['std'] = float(input("Enter standard deviation (default 1): ") or 1)

    elif choice == '2':
        distribution_type = 'exponential'
        params['scale'] = float(input("Enter scale (default 1): ") or 1)

    elif choice == '3':
        distribution_type = 'uniform'
        params['low'] = float(input("Enter lower bound (default 0): ") or 0)
        params['high'] = float(input("Enter upper bound (default 1): ") or 1)

    elif choice == '4':
        distribution_type = 'beta'
        params['alpha'] = float(input("Enter alpha shape parameter (default 2): ") or 2)
        params['beta'] = float(input("Enter beta shape parameter (default 2): ") or 2)

    else:
        print("Invalid choice.")
        return

    cdf_values, x_orig, start, stop = create_cdf(num_values, distribution_type, params)

    return cdf_values, x_orig, start, stop, num_values, bpt_order


# the smaller the WD the better
def wasserstein_distance(cdf1, cdf2_array, x_range, num_segments=10):
    cdf2 = interp1d(x_samples, cdf2_array, fill_value="extrapolate")

    x_segments = np.linspace(x_range[0], x_range[1], num_segments + 1)
    total_distance = 0

    for i in range(num_segments):
        segment_distance, _ = quad_vec(lambda x: np.abs(cdf1(x) - cdf2(x)), x_segments[i], x_segments[i + 1],
                                       epsabs=1e-8, epsrel=1e-8)
        total_distance += segment_distance

    return total_distance


# the smaller the KSD the better
def kolmogorov_smirnov_distance(cdf1, cdf2_array, x_range, num_points=1000):
    interpolated_cdf2 = interp1d(x_samples, cdf2_array, fill_value="extrapolate")
    x_values = np.linspace(x_range[0], x_range[1], num_points)

    cdf1_values = np.array([cdf1(x) for x in x_values])
    cdf2_values = interpolated_cdf2(x_values)

    max_distance = np.max(np.abs(cdf1_values - cdf2_values))
    return max_distance


def bpt(cdf_samples, x_values, order=5):
    x_values = np.clip(np.array(x_values, dtype=float), 1e-10, 1 - 1e-10)
    cdf_samples = np.array(cdf_samples, dtype=float)
    n = order
    bernstein_cdf = np.zeros_like(x_values)

    for i in range(n + 1):
        log_binom = np.log(comb(n, i, exact=False))
        log_basis = log_binom + i * np.log(x_values) + (n - i) * np.log(1 - x_values)
        bernstein_basis = np.exp(log_basis)
        bernstein_cdf += cdf_samples[min(i, len(cdf_samples) - 1)] * bernstein_basis
    return bernstein_cdf


# Gaussian kernel (implicitly order 2)
def gaussian_kernel(u):
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


# higher order kernels
def polynomial_kernel(u, order):
    return (1 - u**order) * (np.abs(u) <= 1) if order > 0 else np.ones_like(u) * (np.abs(u) <= 1)


class CustomKDE:
    def __init__(self, data, order=2, bandwidth=0.2):
        self.data = np.asarray(data)
        self.bandwidth = bandwidth
        self.order = order
        self.kernel = gaussian_kernel if order == 2 else lambda u: polynomial_kernel(u, order)

    def evaluate(self, x):
        u = (x[:, None] - self.data[None, :]) / self.bandwidth
        return np.mean(self.kernel(u), axis=1) / self.bandwidth


if __name__ == "__main__":
    to_compare = False

    plt.figure(figsize=(8, 6))

    # create a CDF
    cdf, x_orig, start, stop, num_values, bpt_order = define_cdf()

    # get the samples from the CDF to give to the BPT
    num_samples = int(input("Enter how many samples need to be taken from the CDF (default 10): ") or 10)

    # we want 10 approximations of the CDF
    for i in range(10):
        # (num_samples - 2) since we always add 0 and 1
        x_samples_incomplete = np.sort(np.random.uniform(start, stop, num_samples - 2))
        x_samples = np.concatenate([[0], x_samples_incomplete, [1]])
        cdf_samples = interp1d(x_orig, cdf, kind='linear', bounds_error=False, fill_value=(cdf[0], cdf[-1]))
        approximated_cdf = bpt(cdf, x_samples, bpt_order)
        plt.plot(x_samples, approximated_cdf, label=f'Approximated CDF {i+1}', color='red', linestyle='-')
        wasserstein_dist = wasserstein_distance(lambda x: cdf_samples(x),
                                                approximated_cdf, (start, stop))
        ks_dist = kolmogorov_smirnov_distance(lambda x: cdf_samples(x),
                                              approximated_cdf, (start, stop))
        print("WD for BPH is ", wasserstein_dist, ", KSD for BPH is ", ks_dist)

    # TODO find a better formula for KDE of higher order
    kde_order = int(input("Enter the order of the KDE to calculate (default 125): ") or 125)
    # compute KDE for the CDF
    if kde_order == 2:
        kde_function = CustomKDE(x_orig, order=2, bandwidth=0.3)
    else:
        kde_function = CustomKDE(x_orig, order=kde_order, bandwidth=0.3)
    kde_values = kde_function.evaluate(x_orig)

    if to_compare:
        # KDE and the CDF
        wasserstein_dist = wasserstein_distance(lambda x: np.interp(x, x_samples, cdf), kde_function, (start, stop))
        ks_dist = kolmogorov_smirnov_distance(lambda x: np.interp(x, x_samples, cdf), kde_function, (start, stop))
        print("WD for KDE is ", wasserstein_dist, ", KSD for KDE is ", ks_dist)

    plt.plot(x_orig, cdf, label='CDF Samples', color='blue', linestyle='--', linewidth=2)
    plt.plot(x_orig, kde_values, label=f"KDE (Order {kde_function.order})", linestyle='-', color='green')

    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.title('CDF comparisons')
    # plt.legend()
    plt.grid(True)
    plt.savefig("cdf_plot.png", dpi=300)
    plt.show()
