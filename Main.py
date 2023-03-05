#Performance enhancement of vision based fall detection using ensemble of machine learning model

#IMPORT PACKAGES
import os
from functools import partial
import numpy as np
import skimage.transform
import skimage.io
import cv2
import matplotlib.pyplot as plt
from numpy import linalg
import cvxopt
import cvxopt.solvers
from collections import Counter
import json
from typing import Dict, List, Tuple
import math
import copy
from knn import*
from ofa import*
from greedy import*
from Preprocessing import*



#Pre-processing

def read_text_file(file_path):
    with open("DATASET", 'r') as f:
        n=file_path
        mn=f.read
        return mn,n
blur = 21
canny_low = 15
canny_high = 150
min_area = 0.0005
max_area = 0.95
dilate_iter = 10
erode_iter = 10
mask_color = (0.0, 0.0, 0.0)
def remove_backround(f): #removal of background and foreground
                        # segmentation is carried to detect the object
    while True:
        ret, frame = f.read()
        if ret == True:
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(image_gray, canny_low, canny_high)
            edges = cv2.dilate(edges, None)
            edges = cv2.erode(edges, None)
            contour_info = [(c, cv2.contourArea(c),) for c in
                            cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]]
            return contour_info
def noise_remove(mask,mask_dilate_iter,mask_erode_iter):#The noise and shadows
                                                        # are removed from the input video frames.
    mask = cv2.dilate(mask, None, iterations=mask_dilate_iter)
    mask = cv2.erode(mask, None, iterations=mask_erode_iter)
    mask_stack = cv2.GaussianBlur(mask, (blur, blur), 0)
    frame = mask_stack.astype('float32') / 255.0
    frame = frame.astype('float32') / 255.0
    masked = (mask_stack * frame) + ((1 - mask_stack) * mask_color)
    masked = (masked * 255).astype('uint8')
    return masked

#Feature Extraction
#Optical Flow Algorithm (OFA)
#distinguish the movements
def main():

    yosemite = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        'yosemite_sequence',
        'yos{}.tif'
    )
    fn1 = yosemite.format(2)
    fn2 = yosemite.format(4)
#processing the image to provide more flow vector
    f1 = skimage.io.imread(fn1).astype(np.double)
    f2 = skimage.io.imread(fn2).astype(np.double)
    c1 = np.minimum(1, 1/5*np.minimum(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])))
    c1 = np.minimum(c1, 1/5*np.minimum(
        f1.shape[0] - 1 - np.arange(f1.shape[0])[:, None],
        f1.shape[1] - 1 - np.arange(f1.shape[1])
    ))
    c2 = c1
    n_pyr = 4

    opts = dict(
        sigma=4.0,
        sigma_flow=4.0,
        num_iter=3,
        model='constant',
        mu=0,
    )
    d = None
#smoothen float image
    for pyr1, pyr2, c1_, c2_ in reversed(list(zip(
        *list(map(
            partial(skimage.transform.pyramid_gaussian, max_layer=n_pyr),
            [f1, f2, c1, c2]
        ))
    ))):
        if d is not None:
            d = skimage.transform.pyramid_expand(d, multichannel=True)
            d = d[:pyr1.shape[0], :pyr2.shape[1]]

        d = (pyr1, pyr2)

    xw = d + np.moveaxis(np.indices(f1.shape), 0, -1)
    opts_cv = dict(
        pyr_scale=0.5,
        levels=6,
        winsize=25,
        iterations=10,
        poly_n=25,
        poly_sigma=3.0,
        # flags=0

    )
#to calculate sparse optical flow
    d2 = cv2.calcOpticalFlowFarneback(
        f2.astype(np.uint8),
        f1.astype(np.uint8),
        None,
        **opts_cv
    )
    d2 = -d2[..., (1, 0)]

    xw2 = d2 + np.moveaxis(np.indices(f1.shape), 0, -1)

    f2_w2 = skimage.transform.warp(f2, np.moveaxis(xw2, -1, 0), cval=np.nan)
    f2_w = skimage.transform.warp(f2, np.moveaxis(xw, -1, 0), cval=np.nan)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    p = 2.0
    vmin, vmax = np.nanpercentile(f1 - f2, [p, 100 - p])
    cmap = 'gray'

    axes[0, 0].imshow(f1, cmap=cmap)
    axes[0, 0].set_title('f1 (fixed image)')
    axes[0, 1].imshow(f2, cmap=cmap)
    axes[0, 1].set_title('f2 (moving image)')
    axes[1, 0].imshow(f1 - f2_w2, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('difference f1 - f2 warped: opencv implementation')
    axes[1, 1].imshow(f1 - f2_w, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('difference f1 - f2 warped: this implementation')
A=True
#--------------------------------------------------------------------------
#3 Models used in Ensemble approach
#--------------------------------------------------------------------------
#SUPPORT VECTOR MACHINE
#classifies the angle
#kernal for mathematical functions

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(solution['x'])
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            # creates an array
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


    #to  get the array of multivariate normal values
    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test
    #set margin
    def plot_margin(X1_train, X2_train, clf):
        X2_train=True
        def f(x, w, b, c=0):
            clf=X2_train
            f=clf
            return (-w[0] * x - b + c) / w[1]
#--------------------------------------------------
#KNearest Neighbors
#--------------------------------------------------
class KNearestNeighbors:
#distance of neighbors

    def __init__(self, k=3):

        self.k = k
        self.distance = None
        self.data = None

    def train(self, X, y):

        # raise value error if inputs are wrong length or different types
        if len(X) != len(y) or type(X) != type(y):
            raise ValueError("X and y are incompatible.")
        # convert ndarrays to lists
        if type(X) == np.ndarray:
            X, y = X.tolist(), y.tolist()
        # set data attribute containing instances and labels
        self.data = [X[i]+[y[i]] for i in range(len(X))]

    def predict(self, a):

        neighbors = []
        # create mapping from distance to instance
        distances = {self.distance(x[:-1], a): x for x in self.data}
        # collect classes of k instances with shortest distance
        for key in sorted(distances.keys())[:self.k]:
            neighbors.append(distances[key][-1])
        # return most common vote
        return max(set(neighbors), key=neighbors.count)


#--------------------------------
#DECISION TREE
#--------------------------------
#decides fall or non fall

#head of the tree
class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = None

#time-efficient algorithms for computing the complexity
#gives yes for fall and no for non fall

def findEntropy(data, rows):
    yes = 0
    no = 0
    ans = -1
    idx = len(data[0]) - 1
    entropy = 0
    for i in rows:
        if data[i][idx] == 'Yes':
            yes = yes + 1
        else:
            no = no + 1

    x = yes/(yes+no)
    y = no/(yes+no)
    if x != 0 and y != 0:
        entropy = -1 * (x*math.log2(x) + y*math.log2(y))
    if x == 1:
        ans = 1
    if y == 1:
        ans = 0
    return entropy, ans


def findMaxGain(data, rows, columns):
    maxGain = 0
    retidx = -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
        return maxGain, retidx, ans

    for j in columns:
        mydict = {}
        idx = j
        for i in rows:
            key = data[i][idx]
            if key not in mydict:
                mydict[key] = 1
            else:
                mydict[key] = mydict[key] + 1
        gain = entropy

        # print(mydict)
        for key in mydict:
            yes = 0
            no = 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == 'Yes':
                        yes = yes + 1
                    else:
                        no = no + 1
            # print(yes, no)
            x = yes/(yes+no)
            y = no/(yes+no)
            # print(x, y)
            if x != 0 and y != 0:
                gain += (mydict[key] * (x*math.log2(x) + y*math.log2(y)))/14
        # print(gain)
        if gain > maxGain:
            # print("hello")
            maxGain = gain
            retidx = j

    return maxGain, retidx, ans

#continue to build the tree if fall happens

def buildTree(data, rows, columns):

    maxGain, idx, ans = findMaxGain(X, rows, columns)
    root = Node()
    root.childs = []
    # print(maxGain
    #
    # )
    if maxGain == 0:
        if ans == 1:
            root.value = 'Yes'
        else:
            root.value = 'No'
        return root
    attribute=None
    root.value = attribute[idx]
    mydict = {}
    for i in rows:
        key = data[i][idx]
        if key not in mydict:
            mydict[key] = 1
        else:
            mydict[key] += 1

    newcolumns = copy.deepcopy(columns)
    newcolumns.remove(idx)
    for key in mydict:
        newrows = []
        for i in rows:
            if data[i][idx] == key:
                newrows.append(i)
        # print(newrows)
        temp = buildTree(data, newrows, newcolumns)
        temp.decision = key
        root.childs.append(temp)
    return root


def traverse(root):
    print(root.decision)
    print(root.value)

    n = len(root.childs)
    if n > 0:
        for i in range(0, n):
            traverse(root.childs[i])

#calculate the result using decision tree algorithm

def calculate():
    rows = [i for i in range(0, 14)]
    columns = [i for i in range(0, 4)]
    root = buildTree(rows, columns)
    root.decision = 'Start'
    traverse(root)

#--------------------------------
#DEEP LSTM
#--------------------------------
#classifies layers to work in complex tasks

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])


#-------------------------------------
#Majority Voting with Greedy Algorithm
#-------------------------------------

smalls = [1]
smallf = [0, 4]

start = [0, 1, 3, 0, 5, 3, 5, 6, 8, 8, 2, 12]
finish = [0, 4, 5, 6, 7, 9, 9, 10, 11, 12, 14, 16]

#sorting of frames
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

#sorting of frames
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

#greedy algorithm
def greedy(target_dist: Dict[Tuple[str, str], float], selection_list: List[List[Tuple[str, str]]], max_iter: int) -> List[List[Tuple[str, str]]]:
  cover: List[List[Tuple[str, str]]] = []
  for _ in range(max_iter):
    current_cover = cover.copy()
    best_sentence = find_sentence_with_min_div(selection_list, current_cover, target_dist)
    cover.append(best_sentence)
    selection_list.remove(best_sentence)
  return cover

#to know the horizontal directions of vector
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
def forwardsearch(sentence, current_cover, target_dist) -> float:
  current_cover.append(sentence)
  cover_dist = get_distribution(current_cover, target_dist)
  divergence = kullback_leibler_div(cover_dist, target_dist)
  return divergence
def backwordsearch(dist_1: Dict[Tuple[str, str], float], dist_2: Dict[Tuple[str, str], float]) -> float:
  for value in dist_2.values():
    assert value > 0
  unequal_zero_keys = [key for key in dist_1.keys() if dist_1[key] > 0]
  if unequal_zero_keys == []:
    return float('inf')
  divergence = [dist_1[key] * (np.log(dist_1[key]) - np.log(dist_2[key]))
                for key in unequal_zero_keys]
  return sum(divergence)
def recoversearch(dist_1: Dict[Tuple[str, str], float], dist_2: Dict[Tuple[str, str], float]) -> float:
  for value in dist_2.values():
    assert value > 0
  unequal_zero_keys = [key for key in dist_1.keys() if dist_1[key] > 0]
  if unequal_zero_keys == []:
    return float('inf')

#split the frames and to display the result
def count_occurrences_of_unit_in_list_of_sentences(unit: Tuple[str, str], sentence_list: List[List[Tuple[str, str]]]) -> int:
  occurence_list = [Counter(sentence)[unit] for sentence in sentence_list]
  return sum(occurence_list)
countp=True
sumof=None
tem=Node
cv2.imshow("Image", frame)
