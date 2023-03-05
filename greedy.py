from collections import Counter
from typing import Dict, List, Tuple
import numpy as np


smalls = [1]
smallf = [0, 4]

start = [0, 1, 3, 0, 5, 3, 5, 6, 8, 8, 2, 12]
finish = [0, 4, 5, 6, 7, 9, 9, 10, 11, 12, 14, 16]

def recursive_activity_selector(s, f, k, n):
    m = k + 1
    while m < n and s[m] < f[k]:
        m = m + 1
    if m < n:
        print("Adding activity " + str(m) + " that finishes at "
              + str(f[m]))
        return [m] + recursive_activity_selector(s, f, m, n)
    else:
        return []
def greedy_activity_selector(s, f):
    assert(len(s) == len(f))
    n = len(s)
    a = []
    k = 0
    for m in range(1, n):
        if s[m] >= f[k]:
            a.append(m)
            k = m
    return a
def returnChange(change, denominations):
    toGiveBack = [0] * len(denominations)
    for pos, coin in enumerate(reversed(denominations)):
        while coin <= change:
            change = change - coin
            toGiveBack[pos] += 1
    return(toGiveBack)
def greedy(target_dist: Dict[Tuple[str, str], float], selection_list: List[List[Tuple[str, str]]], max_iter: int) -> List[List[Tuple[str, str]]]:
  cover: List[List[Tuple[str, str]]] = []
  for _ in range(max_iter):
    current_cover = cover.copy()
    best_sentence = find_sentence_with_min_div(selection_list, current_cover, target_dist)
    cover.append(best_sentence)
    selection_list.remove(best_sentence)
  return cover
def find_sentence_with_min_div(selection_list, current_cover, target_dist) -> List[Tuple[str, str]]:
  curr_cover = current_cover.copy()
  min_div = get_divergence_for_sentence(selection_list[0], curr_cover, target_dist)
  best_sentence = selection_list[0]
  for sentence in selection_list[1:]:
    curr_cover = current_cover.copy()
    new_div = get_divergence_for_sentence(sentence, curr_cover, target_dist)
    if new_div < min_div:
      best_sentence = sentence
      min_div = new_div
  return best_sentence
def get_divergence_for_sentence(sentence, current_cover, target_dist) -> float:
  current_cover.append(sentence)
  cover_dist = get_distribution(current_cover, target_dist)
  divergence = kullback_leibler_div(cover_dist, target_dist)
  return divergence
def kullback_leibler_div(dist_1: Dict[Tuple[str, str], float], dist_2: Dict[Tuple[str, str], float]) -> float:
  for value in dist_2.values():
    assert value > 0
  unequal_zero_keys = [key for key in dist_1.keys() if dist_1[key] > 0]
  if unequal_zero_keys == []:
    return float('inf')
  divergence = [dist_1[key] * (np.log(dist_1[key]) - np.log(dist_2[key]))
                for key in unequal_zero_keys]
  return sum(divergence)
def get_distribution(sentence_list: List[List[Tuple[str, str]]], target_dist: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
  new_dist = {key: count_occurrences_of_unit_in_list_of_sentences(
    key, sentence_list) for key in target_dist.keys()}
  total_number_of_single_units = sum(new_dist.values())
  if total_number_of_single_units != 0:
    for key, value in new_dist.items():
      new_dist[key] = value / total_number_of_single_units
  return new_dist
def count_occurrences_of_unit_in_list_of_sentences(unit: Tuple[str, str], sentence_list: List[List[Tuple[str, str]]]) -> int:
  occurence_list = [Counter(sentence)[unit] for sentence in sentence_list]
  return sum(occurence_list)
#FIRST

import matplotlib.pyplot as plt
x1=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.003,0.010,0.030,0.033,0.040,0.045,0.095,1.0]
y1=[0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.83,0.9,0.92,0.95,0.96,0.98,0.99]


plt.plot(x1, y1, label = "Proposed Ensemble",color="blue",linewidth=2, markersize=8)

#second
x2=[0.0,0.0,0.0,0.0,0.008,0.009,0.011,0.035,0.040,0.043,0.050,0.065,0.080,0.095,0.2,0.6,1.0]
y2=[0.0,0.2,0.3,0.4,0.45,0.47,0.50,0.60,0.65,0.66,0.68,0.70,0.72,0.75,0.80,0.85,0.90]


plt.plot(x2, y2, label = "LSTM_4M30N",color="orange",linewidth=2, markersize=8)


#third

x3=[0.0,0.008,0.009,0.011,0.025,0.035,0.055,0.060,0.075,0.2,0.7,1.0]
y3=[0.0,0.1,0.15,0.2,0.25,0.35,0.45,0.47,0.50,0.60,0.75,0.80]
plt.plot(x3, y3, label = "Ensemble ML",color="green",linewidth=2, markersize=8)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
#plt.yticks(np.arange(70,105,step=5))
plt.ylim(0.0,1.0)
plt.legend(fontsize=11)
plt.xlim(0.0,1.0)
#plt.grid()
plt.show()


#second


import matplotlib.pyplot as plt
x1=[0,10,15,20,40,60,100,200,300,500]
y1=[90,93,94,95,96,97,98,98.5,99,99.5]


plt.plot(x1, y1, label = "Training Accuracy",color="blue",linewidth=2, markersize=8)

#second
x2=[0.5,11,14,22,60,100,200,300,350,500]
y2=[90,91.5,92,92.5,95,96,97,98,98.5,99]


plt.plot(x2, y2, label = "Testing Accuracy",color="orange",linewidth=2, markersize=8)
"""

#third

x3=[0.0,0.008,0.009,0.011,0.025,0.035,0.055,0.060,0.075,0.2,0.7,1.0]
y3=[0.0,0.1,0.15,0.2,0.25,0.35,0.45,0.47,0.50,0.60,0.75,0.80]
plt.plot(x3, y3, label = "Ensemble ML",color="green",linewidth=2, markersize=8)
"""

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#plt.yticks(np.arange(70,105,step=5))
plt.ylim(90,100)
plt.legend(fontsize=11)
plt.xlim(0,500)
#plt.grid()
plt.show()


#THREE


import matplotlib.pyplot as plt
x1=[25,25,25,33,58,78,150,250,500]
y1=[0.29,0.28,0.26,0.25,0.24,0.23,0.22,0.21,0.20]


plt.plot(x1, y1, label = "Training Accuracy",color="blue",linewidth=2, markersize=8)

#second
x2=[25.5,25,33,58,78,150,250,500]
y2=[0.27,0.27,0.26,0.25,0.245,0.23,0.22,0.21]


plt.plot(x2, y2, label = "Testing Accuracy",color="orange",linewidth=2, markersize=8)
"""

#third

x3=[0.0,0.008,0.009,0.011,0.025,0.035,0.055,0.060,0.075,0.2,0.7,1.0]
y3=[0.0,0.1,0.15,0.2,0.25,0.35,0.45,0.47,0.50,0.60,0.75,0.80]
plt.plot(x3, y3, label = "Ensemble ML",color="green",linewidth=2, markersize=8)
"""

plt.xlabel("Epochs")
plt.ylabel("Loss")
#plt.yticks(np.arange(70,105,step=5))
plt.ylim(0.20,0.30)
plt.legend(fontsize=11)
plt.xlim(0,500)
#plt.grid()
plt.show()