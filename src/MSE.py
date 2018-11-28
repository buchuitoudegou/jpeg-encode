from PIL import Image
import numpy as np

def MSE(img1, img2):
  _sum = 0
  for i in range(img1.shape[0]):
    for j in range(img1.shape[2]):
      _sum += (img1[i, j] - img2[i, j]) ** 2
  return _sum / img1.shape[0] / img1.shape[1]

img1 = Image.open('./assets/animal.jpg')
img2 = Image.open('./assets/animal.gif')

w1, h1 = img1.width, img1.height
w2, h2 = img2.width, img2.height
print(img2)
img1 = np.array(img1).reshape([h1, w1, 3])
img2 = np.array(img2).reshape([h2, w2])

print(MSE(img1, img2))