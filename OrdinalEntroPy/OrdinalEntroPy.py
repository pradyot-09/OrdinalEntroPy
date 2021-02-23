
import numpy as np
import math

from .utils import _embed

    

# Permutation Entropy
def PE(values,order=3,classes=3,delay=1,normalize=True):
    """Permutation Entropy.

    This function is completely copied from Raphael Vallat repo. (https://github.com/raphaelvallat/entropy/blob/master/entropy/entropy.py) 
    
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int
        Order of permutation entropy. Default is 3.
    delay : int
        Time delay (lag). Default is 1.
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.
    Returns
    -------
    pe : float
        Permutation Entropy.
    Notes
    -----
    The permutation entropy is a complexity measure for time-series first
    introduced by Bandt and Pompe in 2002.
    The permutation entropy of a signal :math:`x` is defined as:
    .. math:: H = -\\sum p(\\pi)\\log_2(\\pi)
    where the sum runs over all :math:`n!` permutations :math:`\\pi` of order
    :math:`n`. This is the information contained in comparing :math:`n`
    consecutive values of the time series. It is clear that
    :math:`0 ≤ H (n) ≤ \\log_2(n!)` where the lower bound is attained for an
    increasing or decreasing sequence of values, and the upper bound for a
    completely random system where all :math:`n!` possible permutations appear
    with the same probability.
    The embedded matrix :math:`Y` is created by:
    .. math::
        y(i)=[x_i,x_{i+\\text{delay}}, ...,x_{i+(\\text{order}-1) *
        \\text{delay}}]
    .. math:: Y=[y(1),y(2),...,y(N-(\\text{order}-1))*\\text{delay})]^T
    References
    ----------
    Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a
    natural complexity measure for time series." Physical review letters
    88.17 (2002): 174102.
    Examples
    --------
    Permutation entropy with order 2
    >>> from entropy import perm_entropy
    >>> x = [4, 7, 9, 10, 6, 11, 3]
    >>> # Return a value in bit between 0 and log2(factorial(order))
    >>> print(perm_entropy(x, order=2))
    0.9182958340544896
    Normalized permutation entropy with order 3
    >>> from entropy import perm_entropy
    >>> x = [4, 7, 9, 10, 6, 11, 3]
    >>> # Return a value comprised between 0 and 1.
    >>> print(perm_entropy(x, order=3, normalize=True))
    0.5887621559162939
    """
 
  # get all the permuations
  str_permutations = get_str_permutation(values,order,delay)
  
  permutation_indexes = get_permutation_index(str_permutations)

  permutation_frequency = get_permutation_frequency(permutation_indexes,len(values),order)

  entropy = get_shanon_entropy(permutation_frequency)
  if normalize:
    entropy = entropy/math.log2(math.factorial(order))

  return entropy


# Dispersion Entropy
def DE(values,order=3,classes=3,delay=1,normalize=True):
  mapped_values = get_ncdf_values(values,classes)
  # get all the permuations
  str_permutations = get_str_permutation(mapped_values,order,delay)
  
  permutation_indexes = get_permutation_index(str_permutations)

  permutation_frequency = get_permutation_frequency(permutation_indexes,len(values),order)

  entropy = get_shanon_entropy(permutation_frequency)
  if normalize:
    entropy = entropy/math.log2(classes**order)

  return entropy


# Reverse Dispersion Entropy
def RDE(values,order=3,classes=3,delay=1,normalize=True):
  mapped_values = get_ncdf_values(values,classes)
  # get all the permuations
  str_permutations = get_str_permutation(mapped_values,order,delay)
  
  permutation_indexes = get_permutation_index(str_permutations)

  permutation_frequency = get_permutation_frequency(permutation_indexes,len(values),order)

  entropy = np.square(permutation_frequency).sum() - (1/(classes**order))
  if normalize:
    entropy = entropy/(1 - (1/(classes**order)))
  
  return entropy


# Reverse Permutation Entropy
def RPE(values,order=3,delay=1,normalize=True):
  str_permutations = get_str_permutation_ordinal(values,order,delay)
  
  permutation_indexes = get_permutation_index(str_permutations)
  #print(set(str_permutations))

  permutation_frequency = get_permutation_frequency(permutation_indexes,len(values),order)
  #print(permutation_frequency)
  entropy = np.square(permutation_frequency).sum() - (1/math.factorial(order))
  if normalize:
    entropy = entropy/(1 - (1/math.factorial(order)))
  
  return entropy


# Weighted Permutation Entropy
def WPE(values,order=3,delay=1,normalize=True):
  weights = get_weights(values,order,delay)

  str_permutations = get_str_permutation_ordinal(values,order,delay)
  
  permutation_indexes = get_permutation_index(str_permutations)

  weighted_permutations = cal_weighted_permutations(np.array(weights),permutation_indexes)
  
  entropy = get_shanon_entropy(weighted_permutations)

  if normalize:
    entropy = entropy/math.log2(math.factorial(order))

  return entropy


  # Reverse Weighted Permutation Entropy
def RWPE(values,order,delay=1,normalize=True):
  weights = get_weights(values,order,delay)

  str_permutations = get_str_permutation_ordinal(values,order,delay)
  
  permutation_indexes = get_permutation_index(str_permutations)

  weighted_permutations = cal_weighted_permutations(np.array(weights),permutation_indexes)
  
  entropy = np.square(weighted_permutations).sum() - (1/math.factorial(order))
  if normalize:
    entropy = entropy/(1 - (1/math.factorial(order)))
  
  return entropy



# Reverse weighted Dispersion Entropy
def RWDE(values,order,classes,delay=1,normalize=True):
  # find variance of each permutation
  weights = get_weights(values,order,delay)
  # convert values with NCDF and assign them class
  mapped_values = get_ncdf_values(values,classes)
  # get all the permuations
  str_permutations = get_str_permutation(mapped_values,order,delay)
  
  #find the indices of a permutation
  permutation_indexes = get_permutation_index(str_permutations)
  
  # get the weight for each permutation
  weighted_permutations = cal_weighted_permutations(np.array(weights),permutation_indexes)
  
  entropy = np.square(weighted_permutations).sum() - (1/(classes**order))
  # calculate final RWDE entropy
  if normalize:
    entropy = entropy/(1 - (1/(classes**order)))
   
  return entropy