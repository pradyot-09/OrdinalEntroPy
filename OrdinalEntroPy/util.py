
import numpy as np
from scipy.stats import norm

def cal_weights(values):
  sum = 0
  avg = np.mean(values)
  for w in values:
    sum = sum + (w-avg)**2
  return sum/len(values)

#function to calculate relative weight of each permutation
def cal_weighted_permutations(weights,permutation_indexes):
  weighted_permutations = np.zeros(len(permutation_indexes)) 
  for i,indexes in enumerate(permutation_indexes):
    weighted_permutations[i] =  weights[indexes].sum()
  return weighted_permutations/weighted_permutations.sum()

# function to calculate weights of permutation
def get_weights(values,order,delay): 
  N = len(values)
  if order * delay > N:
    raise ValueError("Error: order * delay should be lower than x.size")
  if delay < 1:
    raise ValueError("Delay has to be at least 1.")
  if order < 2:
    raise ValueError("Order has to be at least 2.")
  permutations = np.array([[values[i+ord*delay] for ord in range(order)] for i in range(len(values) - delay*(order + 1))])
  weights = list(map(cal_weights,permutations))
  return weights

# function to convert the time series in normal distribution
def get_ncdf_values(values,classes):
  norm_values = norm.cdf(values,loc=np.mean(values),scale=np.std(values))
  mapped_values = np.array([round(classes*i + 0.5) for i in norm_values])
  return mapped_values

# function to convert array of permutations to string
def get_str_permutation(values,order,delay):
  N = len(values)
  if order * delay > N:
    raise ValueError("Error: order * delay should be lower than x.size")
  if delay < 1:
    raise ValueError("Delay has to be at least 1.")
  if order < 2:
    raise ValueError("Order has to be at least 2.") 
  mapped_permutations = np.array([[values[i+ord*delay] for ord in range(order)] for i in range(len(values) - delay*(order + 1))])
  str_permutations = list(map(str,mapped_permutations))
  return str_permutations

# function to convert array of permutations to string for ordinal entropies
def get_str_permutation_ordinal(values,order):
  N = len(values)  
  if order * delay > N:
    raise ValueError("Error: order * delay should be lower than x.size")
  if delay < 1:
    raise ValueError("Delay has to be at least 1.")
  if order < 2:
    raise ValueError("Order has to be at least 2.")  
  mapped_permutations = np.array([np.argsort([values[i+ord*delay] for ord in range(order)]) for i in range(len(values) - delay*(order + 1))])
  str_permutations = list(map(str,mapped_permutations))
  return str_permutations  

# returns all indexes of occurrences of a permutation
def get_permutation_index(str_permutations):
  permutation_indexes = []
  for perm in set(str_permutations):
    #print(perm)
    indexes = [i for i, e in enumerate(str_permutations) if e == perm]
    permutation_indexes.append(indexes) 
  return permutation_indexes

# return the probabilty/weight of a permutation
def get_permutation_frequency(permutation_indexes,length,order):
  permutation_frequency = np.zeros(len(permutation_indexes)) 
  for i,indexes in enumerate(permutation_indexes):
    permutation_frequency[i] =  len(indexes)
  return permutation_frequency/(length-order+1)

# returns shanon entropy of given array of permutation frequency
def get_shanon_entropy(permutation_frequency):
  entropy = 0
  for freq in permutation_frequency:
    entropy = entropy + freq*math.log2(freq)

  return -entropy 