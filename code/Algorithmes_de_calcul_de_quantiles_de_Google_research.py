import numpy as np
import math
from numpy.fft import irfft
from numpy.fft import rfft


'''
___________________________________________________________________________________________________________
Ces algorithmes ont été mis au point par Google research 
(https://github.com/google-research/google-research/blob/master/dp_multiq/joint_exp.py) et nous ne les avons 
presque pas modifiés.
Nous nous sommes servis de la fonction joint_exp dans la fonction traceEtComparePourMultiQuantilesRisquesQuadrDeEps
du script "Estimations de quantiles", afin de comparer sa précision à celle d'autres algorithmes de calcul de 
plusieurs quantiles.
___________________________________________________________________________________________________________
'''

## INDEXP
## _________________________________________________________________________________________________________



def racing_sample(log_terms):
  """Numerically stable method for sampling from an exponential distribution.

  Args:
    log_terms: Array of terms of form log(coefficient) - (exponent term).

  Returns:
    A sample from the exponential distribution determined by terms. See
    Algorithm 1 from the paper "Duff: A Dataset-Distance-Based
    Utility Function Family for the Exponential Mechanism"
    (https://arxiv.org/pdf/2010.04235.pdf) for details; each element of terms is
    analogous to a single log(lambda(A_k)) - (eps * k/2) in their algorithm.
  """
  return np.argmin(
      np.log(np.log(1.0 / np.random.uniform(size=log_terms.shape))) - log_terms)


def opt_comp_p(eps, t):
  """Returns p_{eps, t} for opt_comp_calculator.

  Args:
    eps: Privacy parameter epsilon.
    t: Exponent t.
  """
  return (np.exp(-t) - np.exp(-eps)) / (1.0 - np.exp(-eps))


def opt_comp_calculator(overall_eps, overall_delta, num_comps):
  """Returns the optimal per-composition eps for overall approx DP guarantee.

  Args:
    overall_eps: Desired overall privacy parameter epsilon.
    overall_delta: Desired overall privacy parameter delta.
    num_comps: Total number of compositions.

  Returns:
    eps_0 such that num_compositions eps_0-DP applications of the exponential
    mechanism will overall be (overall_eps, overall_delta)-DP, using the
    expression given in Theorem 3 of DDR20. This assumes that the composition is
    non-adaptive.
  """
  eps_i_range = np.arange(overall_eps / num_comps - 0.01, overall_eps, 0.01)
  num_eps_i = len(eps_i_range)
  max_eps = 0
  for eps_idx in range(num_eps_i):
    eps = eps_i_range[eps_idx]
    max_sum = 0
    for ell in range(num_comps + 1):
      t_ell_star = np.clip((overall_eps + (ell + 1) * eps) / (num_comps + 1),
                           0.0, eps)
      p_t_ell_star = opt_comp_p(eps, t_ell_star)
      term_sum = 0
      for i in range(num_comps + 1):
        term_sum += math.factorial(num_comps) / (math.factorial(i) * math.factorial(num_comps - i)) * np.power(
            p_t_ell_star, num_comps - i) * np.power(1 - p_t_ell_star, i) * max(
                np.exp(num_comps * t_ell_star -
                       (i * eps)) - np.exp(overall_eps), 0)
      if term_sum > max_sum:
        max_sum = term_sum
    if max_sum > overall_delta:
      return max_eps
    else:
      max_eps = eps
  return max_eps


def ind_exp(sorted_data, data_low, data_high, qs, divided_eps, swap):
  """Returns eps-differentially private collection of quantile estimates for qs.

  Args:
    sorted_data: Array of data points sorted in increasing order.
    data_low: Lower limit for any differentially private quantile output value.
    data_high: Upper limit for any differentially private quantile output value.
    qs: Increasing array of quantiles in [0,1].
    divided_eps: Privacy parameter epsilon for each estimated quantile. Assumes
      that divided_eps has been computed to ensure the desired overall privacy
      guarantee.
    swap: If true, uses swap dp sensitivity, otherwise uses add-remove.
  """
  num_quantiles = len(qs)
  outputs = np.empty(num_quantiles)
  sorted_data = np.clip(sorted_data, data_low, data_high)
  data_size = len(sorted_data)
  sorted_data = np.concatenate(([data_low], sorted_data, [data_high]))
  data_gaps = sorted_data[1:] - sorted_data[:-1]
  for q_idx in range(num_quantiles):
    quantile = qs[q_idx]
    if swap:
      sensitivity = 1.0
    else:
      sensitivity = max(quantile, 1 - quantile)
    idx_left = racing_sample(
        np.log(data_gaps) +
        ((divided_eps / (-2.0 * sensitivity)) *
         np.abs(np.arange(0, data_size + 1) - (quantile * data_size))))
    outputs[q_idx] = np.random.uniform(sorted_data[idx_left],
                                       sorted_data[idx_left + 1])
  # Note that the outputs are already clipped to [data_low, data_high], so no
  # further clipping of outputs is necessary.
  return np.sort(outputs)

## JOINTEXP
## ____________________________________________________________________________________________________________________


def compute_intervals(sorted_data, data_low, data_high):
  """Returns array of intervals of adjacent points.

  Args:
    sorted_data: Nondecreasing array of data points, all in the [data_low,
      data_high] range.
    data_low: Lower bound for data.
    data_high: Upper bound for data.

  Returns:
    An array of intervals of adjacent points from [data_low, data_high] in
    nondecreasing order. For example, if sorted_data = [0,1,1,2,3],
    data_low = 0, and data_high = 4, returns
    [[0, 0], [0, 1], [1, 1], [1, 2], [2, 3], [3, 4]].
  """
  return np.block([[data_low, sorted_data], [sorted_data,
                                             data_high]]).transpose()


def compute_log_phi(data_intervals, qs, eps, swap):
  """Computes two-dimensional array log_phi.

  Args:
    data_intervals: Array of intervals of adjacent points from
      compute_intervals.
    qs: Increasing array of quantiles in [0,1].
    eps: Privacy parameter epsilon.
    swap: If true, uses swap dp sensitivity, otherwise uses add-remove.

  Returns:
    Array log_phi where log_phi[i-i',j] = log(phi(i, i', j)).
  """
  num_data_intervals = len(data_intervals)
  original_data_size = num_data_intervals - 1
  if swap:
    sensitivity = 2.0
  else:
    if len(qs) == 1:
      sensitivity = 2.0 * (1 - min(qs[0], 1 - qs[0]))
    else:
      sensitivity = 2.0 * (1 - min(qs[0], np.min([qs[i] - qs[len(qs)-i] for i in range(1, len(qs))]), 1 - qs[-1]))
  eps_term = -(eps / (2.0 * sensitivity))
  gaps = np.arange(num_data_intervals)
  target_ns = [*[qs[1] - 0], *[qs[j] - qs[j-1] for j in range(1, len(qs)-1)]] * original_data_size
  matrice_a_retourner = [[0 for j in range(len(qs))] for Delta_i in range(original_data_size + 1)]
  for Delta_i in range(original_data_size + 1):
    for j in range(len(qs)):
      matrice_a_retourner[Delta_i][j] = eps_term * np.abs(gaps[Delta_i] - target_ns[j])
  return matrice_a_retourner


def logdotexp_toeplitz_lt(c, x):
  """Multiplies a log-space vector by a lower triangular Toeplitz matrix.

  Args:
    c: First column of the Toeplitz matrix (in log space).
    x: Vector to be multiplied (in log space).

  Returns:
    Let T denote the lower triangular Toeplitz matrix whose first column is
    given by exp(c); then the vector returned by this function is log(T *
    exp(x)). The multiplication is done using FFTs for efficiency, and care is
    taken to avoid overflow during exponentiation.
  """
  max_c, max_x = np.max(c), np.max(x)
  exp_c, exp_x = c - max_c, x - max_x
  np.exp(exp_c, out=exp_c)
  np.exp(exp_x, out=exp_x)
  n = len(x)
  # Choose the next power of two.
  p = np.power(2, np.ceil(np.log2(2 * n - 1))).astype(int)
  fft_exp_c = rfft(exp_c, n=p)
  fft_exp_x = rfft(exp_x, n=p)
  y = irfft(fft_exp_c * fft_exp_x)[:n]
  np.maximum(0, y, out=y)
  np.log(y, out=y)
  y += max_c + max_x
  return y


def compute_log_alpha(data_intervals, log_phi, qs):
  """Computes three-dimensional array log_alpha.

  Args:
    data_intervals: Array of intervals of adjacent points from
      compute_intervals.
    log_phi: Array from compute_log_phi.
    qs: Increasing array of quantiles in (0,1).

  Returns:
    Array log_alpha[a, b, c] where a and c index over quantiles and b represents
    interval repeats.
  """
  num_intervals = len(data_intervals)
  num_quantiles = len(qs)
  data_intervals_log_sizes = np.log(data_intervals[:, 1] - data_intervals[:, 0])
  log_alpha = np.log(np.zeros([num_quantiles, num_intervals, num_quantiles]))
  log_alpha[0, :, 0] = [log_phi[i][0] + data_intervals_log_sizes[i] for i in range(num_intervals)]
  # A handy mask for log_phi.
  disallow_repeat = np.zeros(num_intervals)
  disallow_repeat[0] = -np.inf
  for j in range(1, num_quantiles):
    log_hat_alpha = np.log(sum([np.exp(elt) for elt in log_alpha[j - 1, :, :]]))
    log_alpha[j, :, 0] = data_intervals_log_sizes + logdotexp_toeplitz_lt(
         [log_phi[i][j] + disallow_repeat[i] for i in range(num_intervals)], log_hat_alpha)
    log_alpha[j, 0, 0] = -np.inf  # Correct possible numerical error.
    log_alpha[j, :, 1:j+1] = \
      (log_phi[0, j] + data_intervals_log_sizes)[:, np.newaxis] \
      + log_alpha[j-1, :, 0:j] - np.log(np.arange(1, j+1) + 1)
  return log_alpha


def sample_joint_exp(log_alpha, data_intervals, log_phi, qs):
  """Given log_alpha and log_phi, samples final quantile estimates.

  Args:
    log_alpha: Array from compute_log_alpha.
    data_intervals: Array of intervals of adjacent points from
      compute_intervals.
    log_phi: Array from compute_log_phi.
    qs: Increasing array of quantiles in (0,1).

  Returns:
    Array outputs where outputs[i] is the quantile estimate corresponding to
    quantile q[i].
  """
  num_intervals = len(data_intervals)
  num_quantiles = len(qs)
  outputs = np.zeros(num_quantiles)
  last_i = num_intervals - 1
  j = num_quantiles - 1
  repeats = 0
  while j >= 0:
    log_dist = log_alpha[j, :last_i + 1, :] + log_phi[:last_i + 1,
                                                      j + 1][::-1, np.newaxis]
    # Prevent repeats unless it's the first round.
    if j < num_quantiles - 1:
      log_dist[last_i, :] = -np.inf
    i, k = np.unravel_index(
        ind_exp.racing_sample(log_dist), [last_i + 1, num_quantiles])
    repeats += k
    k += 1
    for j2 in range(j - k + 1, j + 1):
      outputs[j2] = np.random.uniform(data_intervals[i, 0], data_intervals[i,
                                                                           1])
    j -= k
    last_i = i
  return np.sort(outputs)


def joint_exp(sorted_data, data_low, data_high, qs, eps, swap):
  """Computes eps-differentially private quantile estimates for qs.

  Args:
    sorted_data: Array of data points sorted in increasing order.
    data_low: Lower bound for data.
    data_high: Upper bound for data.
    qs: Increasing array of quantiles in (0,1).
    eps: Privacy parameter epsilon.
    swap: If true, uses swap dp sensitivity, otherwise uses add-remove.

  Returns:
    Array o where o[i] is the quantile estimate corresponding to quantile q[i].
  """
  clipped_data = np.clip(sorted_data, data_low, data_high)
  data_intervals = compute_intervals(clipped_data, data_low, data_high)
  log_phi = compute_log_phi(data_intervals, qs, eps, swap)
  log_alpha = compute_log_alpha(data_intervals, log_phi, qs)
  return sample_joint_exp(log_alpha, data_intervals, log_phi, qs)