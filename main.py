import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, uniform, beta
import scipy.stats as stats
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d
from scipy.special import comb

"""
from scipy.stats import gaussian_kde

# some code for Kernel Density Estimation, can be used to compare the results to those of BPH
def gaussian_kernel(u):
    # Computes the Gaussian kernel for a given input u, its implicitly order 2.
    # Args: u: input values at which the Gaussian kernel is evaluated. This can be a scalar or an array of values.
    # Returns: float or ndarray: The value of the Gaussian kernel evaluated at `u`, which is the 
                          probability density for the standard normal distribution.
    
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
        
# to use
    kde_order = int(input("Enter the order of the KDE to calculate (default 125): ") or 125)
    #compute KDE for the CDF
    if kde_order == 2:
        kde_function = CustomKDE(x_orig, order=2, bandwidth=0.3)
    else:
        kde_function = CustomKDE(x_orig, order=kde_order, bandwidth=0.3)
        kde_values = kde_function.evaluate(x_orig)
"""


def get_used_samples(samples_to_get, x_samples, cdf_samples):
    """ Selects a subset of evenly spaced samples from the input arrays"""
    total_points = samples_to_get - 2
    indices = np.linspace(0, len(x_samples) - 1, total_points, dtype=int)
    return x_samples[indices], cdf_samples[indices]


def create_cdf(num_samples, distribution_type='normal', params={}):
    """
    Creates and plots the CDF of a specified distribution.

    Args:
        num_values: number of points to evaluate the CDF on a regular grid (used for plotting).
        num_samples: number of sample points generated separately for sampling-based evaluations.
        distribution_type: type of distribution ('normal', 'exponential', 'uniform', 'beta', 'erlang').
        params: distribution parameters (e.g., {'mean': 0, 'std': 1} for normal).

    Returns:
        tuple:
            - cdf_values (np.ndarray): CDF values evaluated on the x grid.
            - x (np.ndarray): Grid of x-values corresponding to cdf_values.
            - start (float): Start of the x-axis range.
            - stop (float): End of the x-axis range.
            - x_samples (np.ndarray): Sampled x-values (for continuous distributions) or random samples (for Erlang).
            - cdf_samples (np.ndarray): Corresponding CDF values at x_samples or samples drawn from the distribution.
    """
    start = 0
    stop = -np.log(1/num_samples)
    # number of values of cdf calculated to create it
    num_values = 100

    if distribution_type == 'normal':
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        x = np.linspace(start, stop, num_values)
        cdf_values = norm.cdf(x, mean, std)
        label = f'Normal(mean={mean}, std={std})'
        x_samples, cdf_samples = get_used_samples(num_samples, x, cdf_values)

    elif distribution_type == 'exponential':
        scale = params.get('scale', 1)
        x = np.linspace(start, stop, num_values)
        cdf_values = expon.cdf(x, scale=scale)
        label = f'Exponential(scale={scale})'
        x_samples, cdf_samples = get_used_samples(num_samples, x, cdf_values)

    elif distribution_type == 'uniform':
        stop = params.get('high', 30)
        x = np.linspace(start, stop, num_values)
        cdf_values = uniform.cdf(x, start, stop - start)
        label = f'Uniform(low={start}, high={stop})'
        x_samples, cdf_samples = get_used_samples(num_samples, x, cdf_values)

    elif distribution_type == 'minus_ln_one_minus_beta':
        alpha = params.get('alpha', 2)
        beta_param = params.get('beta', 2)
        x = np.linspace(start, stop, num_values)
        cdf_values = 1 - beta.cdf(1 - np.exp(-x), alpha, beta_param)
        label = f'-ln(1 - Beta(alpha={alpha}, beta={beta_param}))'
        x_samples, cdf_samples = get_used_samples(num_samples, x, cdf_values)

    elif distribution_type == 'erlang':
        shape = params.get('shape', 2)
        rate = params.get('rate', 1)
        x = np.linspace(start, stop, num_values)
        cdf_values = erlang_cdf(k=shape, lam=rate, x=x)
        label = f'Erlang(shape={shape}, rate={rate})'
        # can also use sample_erlang_manual
        x_samples = np.sort(sample_erlang(shape, rate, num_samples, stop))
        cdf_samples = erlang_cdf(k=shape, lam=rate, x=x_samples)

    else:
        print("Wrong distribution type.")
        return None

    # plotting_cdf(cdf_values, x, label, save_path='cdf_plot.png')

    return cdf_values, x, start, stop, x_samples, cdf_samples


def plotting_cdf(cdf_values, x, labels, save_path=None):
    """Plots the given CDF and saves the plot if a path is provided."""
    plt.plot(x, cdf_values, label=labels)
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def define_cdf():
    """
    Prompts the user to select and configure a CDF, returns the CDF values and related information.

    Returns:
        tuple: containing the following elements:
            - cdf_values: CDF values over a range.
            - x_orig: x-values where the CDF is evaluated.
            - start: start of the x range.
            - stop: end of the x range.
            - num_values: number of values used in the CDF.
            - bpt_order: Bernstein phase-type approximation order, adjusted if necessary.
    """

    print("Select a CDF distribution: 1. Normal 2. Exponential 3. Uniform 4. Adjusted Beta 5. Erlang")
    choice = input("Enter your choice: ")
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
        params['high'] = float(input("Enter upper bound (default 1): ") or 1)

    elif choice == '4':
        distribution_type = 'minus_ln_one_minus_beta'
        params['alpha'] = float(input("Enter alpha shape parameter (default 2): ") or 2)
        params['beta'] = float(input("Enter beta shape parameter (default 2): ") or 2)

    elif choice == '5':
        distribution_type = 'erlang'
        params['shape'] = int(input("Enter shape parameter k (default 2): ") or 2)
        params['rate'] = float(input("Enter rate shape parameter lambda (default 1.0): ") or 1.0)

    else:
        print("Invalid choice.")
        return

    num_samples = int(input("Number of samples from the CDF (default 30) n: ") or 30)

    cdf_values_orig, x_orig, start_orig, stop_orig, x_samples, cdf_samples = create_cdf(num_samples, distribution_type, params)

    return cdf_values_orig, x_orig, start_orig, stop_orig, x_samples, cdf_samples


def wasserstein_distance(cdf1, cdf2_array, x_range, x_samples, num_segments=10):
    """
        Approximates the Wasserstein distance between two CDFs, the smaller the distance the closer they are.

        Args:
            cdf1: the first CDF.
            cdf2_array: values of the second CDF evaluated at x_samples.
            x_range: (min, max) range over which to compute the distance.
            x_samples: x-values corresponding to cdf2_array.
            num_segments: number of segments to divide the integration interval into
                for numerical integration. Default is 10.

        Returns:
            float: approximate Wasserstein distance between the two distributions.
        """

    # interpolate cdf2 for continuous evaluation
    cdf2 = interp1d(x_samples, cdf2_array, fill_value="extrapolate")
    x_segments = np.linspace(x_range[0], x_range[1], num_segments + 1)
    total_distance = 0
    for i in range(num_segments):
        segment_distance, _ = quad_vec(lambda x: np.abs(cdf1(x) - cdf2(x)), x_segments[i], x_segments[i + 1],
                                       epsabs=1e-8, epsrel=1e-8)
        total_distance += segment_distance
    return total_distance


def kolmogorov_smirnov_distance(cdf1, cdf2_array, x_range, x_samples, num_points=1000):
    """
       Computes the Kolmogorov–Smirnov distance between two distributions, the smaller the distance the closer they are.
       The KS distance is the maximum absolute difference between two CDFs.
       This function approximates the KS distance by sampling the CDFs at evenly spaced points in the given range.

       Args:
           cdf1: the first CDF.
           cdf2_array: values of the second CDF evaluated at x_samples.
           x_range: (min, max) range over which to compute the distance.
           x_samples: x-values corresponding to cdf2_array.
           num_points: number of points to sample in x_range. Default is 1000.

       Returns:
           float: Kolmogorov–Smirnov distance between the two distributions.
       """

    x_samples_unique, unique_indices = np.unique(x_samples, return_index=True)
    cdf2_array_unique = cdf2_array[unique_indices]

    interpolated_cdf2 = interp1d(x_samples_unique, cdf2_array_unique, fill_value="extrapolate")
    x_val = np.linspace(x_range[0], x_range[1], num_points)

    cdf1_values = np.array([cdf1(x) for x in x_val])
    cdf2_values = interpolated_cdf2(x_val)

    max_distance = np.max(np.abs(cdf1_values - cdf2_values))
    return max_distance


def bph(cdf_function, x_values, order=5):
    """
    Computes the Bernstein Exponential approximation of a CDF.

    Args:
        cdf_function: function that returns CDF values F(x)
        x_values: evaluation points
        order: order of the approximation

    Returns:
        array: Approximated CDF values
    """
    n = order
    bernstein_approx = np.zeros_like(x_values, dtype=float)

    for i in range(n + 1):
        if i == 0:
            cdf_val = 1.0
        else:
            # sampling CDF at -log(i/n)
            sample_point = -np.log(i / n)
            cdf_val = cdf_function(sample_point)
        binom_coeff = comb(n, i, exact=False)
        exp_neg_x = np.exp(-x_values)
        bernstein_basis = binom_coeff * (exp_neg_x ** i) * ((1 - exp_neg_x) ** (n - i))
        bernstein_approx += cdf_val * bernstein_basis

    return bernstein_approx


def erlang_cdf(k, lam, x):
    """Computes the CDF of an Erlang distribution with shape parameter k and rate lam."""
    return stats.erlang.cdf(x, a=k, scale=1/lam)


def sample_erlang(k, lam, size, stop):
    """Generates 'size' samples from Erlang(k, lam)."""
    samples = []
    while len(samples) < size - 1:
        new_samples = stats.erlang.rvs(a=k, scale=1/lam, size=size)
        samples.extend(new_samples[new_samples < stop])
    samples = np.sort(samples[:size - 1])
    return np.append(samples, stop)


def sample_erlang_manual(k, lam, size, stop):
    """Generates 'size' samples from Erlang(k, lam) by summing exponential."""
    samples = []
    while len(samples) < size - 1:
        new = np.sum(
            np.random.exponential(scale=1/lam, size=(size, k)),
            axis=1
        )
        samples.extend(new[new < stop])
    samples = np.sort(samples[:size - 1])
    return np.append(samples, stop)


def plot_approx(x_orig, cdf, x_samples, approximated_cdf, bpt_order):
    """Plots the given CDF and the BPH Approximation, saves the plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_orig, cdf, color='red', linewidth=2, label='Original CDF')
    plt.plot(x_samples, approximated_cdf, color='blue', linewidth=2, label='BPH Approximation', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('CDF')
    if wasserstein_dist is not None and ks_dist is not None:
        title = f'CDF vs. BPH Approximation (order={bpt_order})\nWD={wasserstein_dist:.4f}, KSD={ks_dist:.4f}'
    else:
        title = f'CDF vs. BPH Approximation (order={bpt_order})'
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.savefig("cdf_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    cdf, x_orig, start_value, stop_value, x_samples, cdf_samples = define_cdf()

    user_input = int(input("Order of BPT (default 8) m: ") or 8)
    bpt_order = int(user_input) if user_input else 8

    subset_x, subset_cdf = get_used_samples(bpt_order, x_samples, cdf_samples)

    x_samples_extended = np.concatenate(([start_value], x_samples, [stop_value]))
    cdf_samples_extended = np.concatenate(([0.0], cdf_samples, [1.0]))
    sort_indices = np.argsort(x_samples_extended)
    x_samples_extended = x_samples_extended[sort_indices]
    cdf_samples_extended = cdf_samples_extended[sort_indices]
    cdf_function = interp1d(x_samples_extended, cdf_samples_extended, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))

    approximated_cdf = bph(cdf_function, x_samples_extended, bpt_order)

    wasserstein_dist = wasserstein_distance(
        cdf1=cdf_function,
        cdf2_array=approximated_cdf,
        x_range=(start_value, stop_value),
        x_samples=x_samples_extended
    )
    ks_dist = kolmogorov_smirnov_distance(
        cdf1=cdf_function,
        cdf2_array=approximated_cdf,
        x_range=(start_value, stop_value),
        x_samples=x_samples_extended
    )

    print("Wasserstein Distance is ", wasserstein_dist, ", Kolmogorov-Smirnov Distance is ", ks_dist)

    plot_approx(x_orig, cdf, x_samples_extended, approximated_cdf, bpt_order)
