# knn
import pandas as pd
import numpy as np
import operator

def classify(inx,dataset,lables,k):
    datasetsize = dataset.shape[0]
    diffmat = np.tile(inx,(datasetsize,1)) - dataset
    sqdiffmat = diffmat ** 2
    sqdistances = sqdiffmat.sum(axis = 1)
    distances = sqdistances ** 0.5
    sorteddistindicies = distances.argsort()
    classcount = {}
    for i in range(k):
        voteIlabel = labels[sorteddistindicies[i]]
        classcount[voteIlabel] = classcount.get(voteIlabel,0) + 1
        sortedclasscount = sorted(classcount.items(),
                                  key = operator.itemgetter(1),
                                  reverse = True)
    return sortedclasscount[0][0]

group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
labels = list('AABB')
knn = classify([0,0],group,labels,3)
print(knn)
