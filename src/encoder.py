import cv2
import numpy as np
from huffman import huffman, dehuffman, inverse
import copy

class encoder():
  def __init__(self, path):
    self.img = cv2.imread(path)
    self.origin_img = copy.deepcopy(self.img)
    self.height, self.width = self.img.shape[0], self.img.shape[1]
    self.sup_height = (16 - self.height % 16) if (self.height % 16 != 0) else 0
    self.sup_width = (16 - self.width % 16) if (self.width % 16 != 0) else 0 
    self.height += self.sup_height
    self.width += self.sup_width
    if self.sup_height != 0 or self.sup_width != 0:
      self.img = np.pad(self.img, [(0, self.sup_height),(0, self.sup_width), (0, 0)], mode='constant')
  
  def ycbcr_convert(self):
    for i in range(self.img.shape[0]):
      for j in range(self.img.shape[1]):
        blue = self.img[i, j, 0]
        green = self.img[i, j, 1]
        red = self.img[i, j, 2]
        self.img[i, j, 0] = 0.299 * red + 0.5870 * green + 0.114 * blue
        self.img[i, j, 1] = -0.1687 * red - 0.3313 * green + 0.5 * blue + 128
        self.img[i, j, 2] = 0.5 * red - 0.4187 * green - 0.0813 * blue + 128
  
  def subsampling(self):
    img_y, img_cb, img_cr = [], [], []
    for i in range(self.img.shape[0]):
      for j in range(self.img.shape[1]):
        img_y.append(self.img[i, j, 0])
        if i % 2 == 0 and j % 2 == 0:
          img_cb.append(self.img[i, j, 1])
        elif j % 2 == 0:
          img_cr.append(self.img[i, j, 2])
    img_y = np.array(img_y).reshape([self.img.shape[0], self.img.shape[1]])
    img_cb = np.array(img_cb).reshape([self.img.shape[0] // 2, self.img.shape[1] // 2])
    img_cr = np.array(img_cr).reshape([self.img.shape[0] // 2, self.img.shape[1] // 2])

    return img_y, img_cb, img_cr

  def dct_convert(self, img_y, img_cb, img_cr):
    dct_kernel = np.zeros([8, 8])
    dct_kernel[0, :] = 1 / np.sqrt(8)
    for i in range(1, 8):
      for j in range(8):
        dct_kernel[i, j] = np.cos(np.pi * i * (2 * j + 1) / 16) * np.sqrt(2 / 8)
    
    def convertion(img):
      new_img = np.zeros(img.shape)
      for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
          temp = img[i:i+8, j:j+8]
          t1 = np.dot(dct_kernel, temp)
          new_img[i:i+8, j:j+8] = np.dot(t1, np.transpose(dct_kernel))
      return new_img
    
    img_y = convertion(img_y)
    img_cb = convertion(img_cb)
    img_cr = convertion(img_cr)

    return img_y, img_cb, img_cr
  
  def quantization(self, img, q):
    new_img = np.zeros(img.shape, dtype=int)
    for i in range(0, img.shape[0], 8):
      for j in range(0, img.shape[1], 8):
        temp = img[i:i+8, j:j+8]
        new_img[i:i+8, j:j+8] = np.round(temp / q)
    return new_img

  def RLE(self, img):
    rle = []
    firstDC = img[0][0]
    for i in range(len(img)):
      tmp = []
      count = 0
      for j in range(len(img[i])):
        if j == 0 and i == 0:
          tmp.append((0, firstDC))
        elif j == 0:
          tmp.append((0, img[i][j] - firstDC))
        else:
          if img[i][j] != 0:
            tmp.append((count, img[i][j]))
            count = 0
          elif count < 15:
            count += 1
          else:
            tmp.append((count, 0))
            count = 0
      flag = True
      for k in range(len(tmp) - 1, -1, -1):
        if tmp[k][1] == 0:
          flag = False
          tmp.pop()
        else:
          tmp.append('EOB')
          flag = True
          break
      if not flag:
        tmp.append('EOB')
      rle.append(tmp)
    return rle

  def huffman_encode(self, img):
    new_img = []
    for i in range(len(img)):
      tmp = []
      for j in range(len(img[i])):
        if img[i][j] == 'EOB':
          tmp.append((huffman('0/0'), 0))
          continue
        binary = ''
        if int(img[i][j][1]) > 0:
          binary = bin(int(img[i][j][1]))[2:]
        elif int(img[i][j][1]) < 0:
          binary = bin(int(img[i][j][1]))[3:]
          binary = inverse(binary)
        else:
          binary = ''
        size = len(binary)
        zeros = img[i][j][0]
        key = hex(zeros)[2:].upper() + '/' + hex(size)[2:].upper()
        try:
          tmp.append((huffman(key), binary))
        except:
          print(i, j, img[i][j])
          exit(0)
      new_img.append(tmp)
    return new_img
  
  def scan(self, img, z):
    img_seq = []
    for i in range(0, img.shape[0], 8):
      for j in range(0, img.shape[1], 8):
        temp = img[i:i+8, j:j+8]
        temp = temp.reshape([1, -1])
        new_zone = []
        for k in range(temp.shape[1]):
          new_zone.append(temp[0][z[k]])
        img_seq.append(new_zone)

    return img_seq

  def encode(self):
    qy = [16,11,10,16,24,40,51,61,
12,12,14,19,26,58,60,55,
14,13,16,24,40,57,69,56,
14,17,22,29,51,87,80,62,
18,22,37,56,68,109,103,77,
24,35,55,64,81,104,113,92,
49,64,78,87,103,121,120,101,
72,92,95,98,112,100,103,99]

    qy = np.array(qy)
    qy = qy.reshape([8, 8])

    qc = [17,18,24,47,99,99,99,99,
18,21,26,66,99,99,99,99,
24,26,56,99,99,99,99,99,
47,66,99,99,99,99,99,99,
99,99,99,99,99,99,99,99,
99,99,99,99,99,99,99,99,
99,99,99,99,99,99,99,99,
99,99,99,99,99,99,99,99]

    qc = np.array(qc)
    qc = qc.reshape([8, 8])

    z = [0,1,5,6,14,15,27,28,
2,4,7,13,16,26,29,42,
3,8,12,17,25,30,41,43,
9,11,18,24,31,40,44,53,
10,19,23,32,39,45,52,54,
20,22,33,38,46,51,55,60,
21,34,37,47,50,56,59,61,
35,36,48,49,57,58,62,63]
    #ycbcr convert
    self.ycbcr_convert()
    #subsampling
    img_y, img_cb, img_cr = self.subsampling()
    # dct convert
    img_y, img_cb, img_cr = self.dct_convert(img_y, img_cb, img_cr)
    # quantization
    img_y = self.quantization(img_y, qy)
    img_cb = self.quantization(img_cb, qc)
    img_cr = self.quantization(img_cr, qc)
    # z-scan
    img_y = self.scan(img_y, z)
    img_cb = self.scan(img_cb, z)
    img_cr = self.scan(img_cr, z)
    # encode
    img_y = self.RLE(img_y)
    img_cb = self.RLE(img_cb)
    img_cr = self.RLE(img_cr)
    img_y = self.huffman_encode(img_y)
    img_cb = self.huffman_encode(img_cb)
    img_cr = self.huffman_encode(img_cr)

    return img_y, img_cb, img_cr
