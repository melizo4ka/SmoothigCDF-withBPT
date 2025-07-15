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


def create_cdf(num_values, num_samples, distribution_type='normal', params={}):
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

    if distribution_type == 'normal':
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        stop = 10
        x = np.linspace(start, stop, num_values)
        cdf_values = norm.cdf(x, mean, std)
        label = f'Normal(mean={mean}, std={std})'
        x_samples = np.linspace(start, stop, num_samples)
        cdf_samples = norm.cdf(x_samples, mean, std)

    elif distribution_type == 'exponential':
        scale = params.get('scale', 1)
        stop = 10
        x = np.linspace(start, stop, num_values)
        cdf_values = expon.cdf(x, scale=scale)
        label = f'Exponential(scale={scale})'
        x_samples = np.linspace(start, stop, num_samples)
        cdf_samples = expon.cdf(x_samples, scale=scale)

    elif distribution_type == 'uniform':
        stop = params.get('high', 1)
        x = np.linspace(start, stop, num_values)
        cdf_values = uniform.cdf(x, start, stop - start)
        label = f'Uniform(low={start}, high={stop})'
        x_samples = np.linspace(start, stop, num_samples)
        cdf_samples = uniform.cdf(x_samples, start, stop - start)

    elif distribution_type == 'beta':
        alpha = params.get('alpha', 2)
        beta_param = params.get('beta', 2)
        # Beta distributions are defined in [0, 1]
        stop = 1
        x = np.linspace(start, stop, num_values)
        cdf_values = beta.cdf(x, alpha, beta_param)
        label = f'Beta(alpha={alpha}, beta={beta_param})'
        x_samples = np.linspace(start, stop, num_samples)
        cdf_samples = beta.cdf(x_samples, alpha, beta_param)

    elif distribution_type == 'erlang':
        shape = params.get('shape', 2)
        rate = params.get('rate', 1)
        stop = 10
        x = np.linspace(start, stop, num_values)
        cdf_values = erlang_cdf(k=shape, lam=rate, x=x)
        label = f'Erlang(shape={shape}, rate={rate})'
        # can also use sample_erlang_manual
        x_samples = np.sort(sample_erlang(shape, rate, num_samples))
        cdf_samples = erlang_cdf(k=shape, lam=rate, x=x_samples)

    else:
        print("Wrong distribution type.")
        return None

    # save_path='cdf_plot.png' as last argument
    # plotting_cdf(cdf_values, x, label)

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

    print("Select a CDF distribution: 1. Normal 2. Exponential 3. Uniform 4. Beta 5. Erlang")
    choice = input("Enter your choice: ")
    params = {}
    num_values = 100

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
        distribution_type = 'beta'
        params['alpha'] = float(input("Enter alpha shape parameter (default 2): ") or 2)
        params['beta'] = float(input("Enter beta shape parameter (default 2): ") or 2)

    elif choice == '5':
        distribution_type = 'erlang'
        params['shape'] = int(input("Enter shape parameter k (default 2): ") or 2)
        params['rate'] = float(input("Enter rate shape parameter lambda (default 1.0): ") or 1.0)

    else:
        print("Invalid choice.")
        return

    num_samples = int(input("Number of samples from the CDF (default 10) n: ") or 10)

    cdf_values_orig, x_orig, start_orig, stop_orig, x_samples, cdf_samples = create_cdf(num_values, num_samples, distribution_type, params)

    return cdf_values_orig, x_orig, start_orig, stop_orig, num_values, x_samples, cdf_samples


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


def bph(cdf_function, x_values, start, stop, order=5):
    """
    Computes the Bernstein phase-type approximation of a CDF using your original exponential algorithm.

    Args:
        cdf_function: interpolated CDF function
        x_values: evaluation points in original domain
        start: start of the domain
        stop: end of the domain
        order: order of the approximation

    Returns:
        array: Approximated CDF values with proper boundary conditions (0 at start, 1 at stop)
    """
    n = order

    x_normalized = (x_values - start) / (stop - start)
    x_normalized = np.clip(x_normalized, 1e-10, 1 - 1e-10)
    t_values = -np.log(x_normalized)

    bernstein_bph = np.zeros_like(x_values, dtype=float)

    boundary_mask_start = (x_values <= start + 1e-10)
    boundary_mask_end = (x_values >= stop - 1e-10)
    bernstein_bph[boundary_mask_start] = 0.0
    bernstein_bph[boundary_mask_end] = 1.0

    interior_mask = ~(boundary_mask_start | boundary_mask_end)
    t_interior = t_values[interior_mask]

    if len(t_interior) > 0:
        for i in range(1, n + 1):
            eval_point = start + (stop - start) * np.exp(-np.log(i / n))
            eval_point = np.clip(eval_point, start, stop)
            cdf_val = cdf_function(eval_point)

            binom_coeff = comb(n, i, exact=False)
            bernstein_basis = binom_coeff * np.exp(-i * t_interior) * (1 - np.exp(-t_interior)) ** (n - i)

            bernstein_bph[interior_mask] += cdf_val * bernstein_basis

    return bernstein_bph


def erlang_cdf(k, lam, x):
    """Compute the CDF of an Erlang distribution with shape parameter k and rate lam."""
    return stats.erlang.cdf(x, a=k, scale=1/lam)


def sample_erlang(k, lam, size=1):
    """Generate samples from an Erlang distribution with shape k and rate lam."""
    return stats.erlang.rvs(a=k, scale=1/lam, size=size)


def sample_erlang_manual(k, lam, size=1):
    """
    Generate samples from an Erlang distribution by summing k independent Exponential(lambda) samples.

    Parameters:
    - k: Shape parameter (must be an integer, number of summed exponentials)
    - lam: Rate parameter (lambda)
    - size: Number of Erlang samples to generate

    Returns:
    - Samples from an Erlang(k, lambda) distribution
    """
    return np.sum(np.random.exponential(scale=1 / lam, size=(size, k)), axis=1)


def plot_approx(x_orig, cdf, x_samples, approximated_cdf, bpt_order):
    plt.figure(figsize=(10, 6))
    plt.plot(x_orig, cdf, color='red', linewidth=2, label='Original CDF')
    plt.plot(x_samples, approximated_cdf, color='blue', linewidth=2, label='BPH Approximation', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.title(f'CDF vs. BPH Approximation (order={bpt_order})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.savefig("cdf_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    cdf, x_orig, start, stop, num_values, x_samples, cdf_samples = define_cdf()

    user_input = int(input("Order of BPT (default 125) m: ") or 125)
    bpt_order = int(user_input) if user_input else 125
    bpt_order = min(bpt_order, 125)

    x_samples_extended = np.concatenate(([start], x_samples, [stop]))
    cdf_samples_extended = np.concatenate(([0.0], cdf_samples, [1.0]))
    sort_indices = np.argsort(x_samples_extended)
    x_samples_extended = x_samples_extended[sort_indices]
    cdf_samples_extended = cdf_samples_extended[sort_indices]
    cdf_function = interp1d(x_samples_extended, cdf_samples_extended, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))

    approximated_cdf = bph(cdf_function, x_samples_extended, start, stop, bpt_order)
    plot_approx(x_orig, cdf, x_samples_extended, approximated_cdf, bpt_order)

    wasserstein_dist = wasserstein_distance(
        cdf1=cdf_function,
        cdf2_array=approximated_cdf,
        x_range=(start, stop),
        x_samples=x_samples_extended
    )
    ks_dist = kolmogorov_smirnov_distance(
        cdf1=cdf_function,
        cdf2_array=approximated_cdf,
        x_range=(start, stop),
        x_samples=x_samples_extended
    )

    print("Wasserstein Distance is ", wasserstein_dist, ", Kolmogorov-Smirnov Distance is ", ks_dist)
