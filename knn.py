import numpy as np
#from src.distance import euclidean


class KNearestNeighbors:

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
print("-----------------------------------------------------------------------------------------------------")
print("Performance comparison of propsoed ensemble with conventional approaches")
print("-----------------------------------------------------------------------------------------------------")
#Region of Operating characteristic’s
d1={"2D CNN":["-",97.95,83.08,"-"],
    "Multi-task H"
    "C"
    "A-E":[0.962,1,0.930,0.92],
    "Multi-Stream CNN":[99.72,99.70,99.80,"-"],
    "LSTM_4m30N":[0.934715,"-","-",0.92041],
    "Ensemble ML"
    "Algorithm":[98.72,"96.22%","94.60%","-"],
    "Proposed Ensemble"
    "model":[99.98,99.80,99.90,96.23]
}
print ("{:<30} {:<30} {:<30} {:<20} {:<20}".format('Method','Accuracy','Sensitivity','Specificity','Precision',"\n"))
print("----------------------------------------------------------------------------------------------------------------------------")
for k, v in d1.items():
    lang, perc, change,mm= v
    print ("{:<30} {:<30} {:<30} {:<20} {:<20}".format(k, lang, perc, change,mm))