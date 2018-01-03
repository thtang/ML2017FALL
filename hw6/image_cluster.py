import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import sys

image_path = sys.argv[1]
test_path = sys.argv[2]
predict_path = sys.argv[3]
# load testing data
test_case = pd.read_csv(test_path)

image = np.load(image_path)

# normalize image
image_X = image / 255
pca = PCA(n_components=400, whiten=True,svd_solver="full",random_state=0)
image_X_PCA = pca.fit_transform(image_X)

cluster = KMeans(n_clusters=2)
cluster.fit(image_X_PCA)

ans = []
for a, b in zip(test_case["image1_index"],test_case["image2_index"]):
    if cluster.labels_[a] == cluster.labels_[b]:
        ans.append(1)
    else:
        ans.append(0)
print(ans[:30])
# submission
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission["Ans"] = ans
sample_submission.to_csv(predict_path,index=None)