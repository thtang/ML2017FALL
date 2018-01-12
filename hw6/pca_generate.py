import skimage.io
import numpy as np
import sys
import os
from os import listdir
img_folder = sys.argv[1]
test_img = sys.argv[2]

image_names = listdir(img_folder)
print(image_names[:10])
# load all faces array
image_X = []

for name in image_names:
    single_img = skimage.io.imread(os.path.join(img_folder,name))
    image_X.append(single_img)
image_flat = np.reshape(image_X,(415,-1))
mean_face = np.mean(image_flat,axis=0)

image_center = image_flat - mean_face

print("image center shape",image_center.shape)

print("Running SVD........")
U, S, V = np.linalg.svd(image_center.T, full_matrices=False)

print("U shape",U.shape)
print("S shape",S.shape)
print("V shape",V.shape)

# reconstruct 

top = 4

input_img = skimage.io.imread(os.path.join(img_folder,test_img)).flatten()
input_img_center = input_img - mean_face

weights = np.dot(input_img_center, U[:, :top])

recon = mean_face + np.dot(weights, U[:, :top].T)
recon -= np.min(recon)
recon /= np.max(recon)
recon = (recon * 255).astype(np.uint8)


skimage.io.imsave("reconstruction.jpg", recon.reshape(600,600,3))