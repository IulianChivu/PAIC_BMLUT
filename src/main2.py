import matplotlib.pyplot as plt
from skimage import io
import time
import numpy as np

from zgomot import add_zgomot_gaussian, add_zgomot_impulsiv
from filtre import filtru_medie_aritmetica, filtru_median, bmlut
from eroare import MSE, MAE, PSNR, SSIM


# Citere imag color    
img = io.imread('lena.png')
print(img.dtype)

#img = img[200:300, 200:300]

###############################################################################

img_zg = add_zgomot_impulsiv(img, 0.3)
print(img_zg.dtype)

plt.figure()
plt.imshow(img_zg[:,:,1].astype(np.uint8))

start=time.time()
img_filt = bmlut(img_zg[:,:,1].astype(np.uint8))
stop=time.time()
print('timp executie BMLUT: ',stop-start)

plt.figure()
plt.imshow(img_filt.astype(np.uint8))

