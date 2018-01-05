import skimage.io
import numpy as np
import sys
import os

img_folder = sys.argv[1]
test_imgi_index = int(sys.argv[2][:-4])

# load all faces array
image_X = []
picture_path = img_folder + "/"
for i in range(415):
    single_img = skimage.io.imread(os.path.join(picture_path,str(i)+".jpg"))
    image_X.append(single_img)
image_flat = np.reshape(image_X,(415,-1))
mean_face = np.mean(image_flat,axis=0)

image_center = image_flat - mean_face

print("image center shape",image_center.shape)

U, S, V = np.linalg.svd(image_center.T, full_matrices=False)

print("U shape",U.shape)
print("S shape",S.shape)
print("V shape",V.shape)

weights = np.dot(image_center, U)

top = 4

recon = mean_face + np.dot(weights[test_imgi_index, :top], U[:, :top].T)
recon -= np.min(recon)
recon /= np.max(recon)
recon = (recon * 255).astype(np.uint8)


skimage.io.imsave("reconstruction.jpg", recon.reshape(600,600,3))