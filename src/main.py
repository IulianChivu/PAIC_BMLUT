import matplotlib.pyplot as plt
from skimage import io
import time
import numpy as np

from zgomot import add_zgomot_gaussian, add_zgomot_impulsiv
from filtre import filtru_medie_aritmetica, filtru_median
from eroare import MSE, MAE, PSNR, SSIM


# Citere imag color    
img = io.imread('lena.png')
print(img.dtype)

# Afisarea imaginii originale
#plt.figure(figsize=(10,10)), plt.imshow(img)
fig = plt.figure()
fig.add_subplot(1,3,1), plt.imshow(img)

#adaug zgomot in imagine si afizes imaginea zgomotosa
#zgomot de medie 0 dispersie sigma 
medie=0
sigma=10
img_zg_ga = add_zgomot_gaussian(img, medie, sigma)

#afisare imagine+zg
#plt.figure(figsize=(10,10)), plt.imshow(img_zg.astype(np.uint8))
fig.add_subplot(1,3,2), plt.imshow(img_zg_ga.astype(np.uint8))

#adaug zgomot impulsiv
ratio = 0.1
img_zg_im = add_zgomot_impulsiv(img, ratio)
#afisare
fig.add_subplot(1,3,3), plt.imshow(img_zg_im.astype(np.uint8))

#dim filtrului (ferestrei)
window_size= 3


start=time.time()
#aplic functia de filtrare liniara (medie aritmetica)
img_filt_medie = filtru_medie_aritmetica(img_zg_ga, window_size)
stop=time.time()
print('timp executie filtru medie aritmetica: ',stop-start)

start=time.time()
img_filt_median = filtru_median(img_zg_im, window_size)
stop=time.time()
print('timp executie filtru median: ',stop-start)

#afisez imagine originala + filtrari
fig2 = plt.figure()
fig2.add_subplot(1,3,1), plt.imshow(img)
fig2.add_subplot(1,3,2), plt.imshow(img_filt_medie.astype(np.uint8))
fig2.add_subplot(1,3,3), plt.imshow(img_filt_median.astype(np.uint8))


#calcul mse
print()
mse_t, mse_r, mse_g, mse_b = MSE(img, img.astype(np.uint8), window_size)
print("img-img: mse_t = " + str(mse_t) + " mse_r = " + str(mse_r) + " mse_g = " + str(mse_g) + " mse_b = " + str(mse_b))

mse_t, mse_r, mse_g, mse_b = MSE(img, img_zg_ga.astype(np.uint8), window_size)
print("img-img_zg_ga: mse_t = " + str(mse_t) + " mse_r = " + str(mse_r) + " mse_g = " + str(mse_g) + " mse_b = " + str(mse_b))

mse_t, mse_r, mse_g, mse_b = MSE(img, img_zg_im.astype(np.uint8), window_size)
print("img-img_zg_im: mse_t = " + str(mse_t) + " mse_r = " + str(mse_r) + " mse_g = " + str(mse_g) + " mse_b = " + str(mse_b))

mse_t, mse_r, mse_g, mse_b = MSE(img, img_filt_medie.astype(np.uint8), window_size)
print("img-img_filt_medie: mse_t = " + str(mse_t) + " mse_r = " + str(mse_r) + " mse_g = " + str(mse_g) + " mse_b = " + str(mse_b))

mse_t, mse_r, mse_g, mse_b = MSE(img, img_filt_median.astype(np.uint8), window_size)
print("img-img_filt_median: mse_t = " + str(mse_t) + " mse_r = " + str(mse_r) + " mse_g = " + str(mse_g) + " mse_b = " + str(mse_b))

#calcul PSNR
print()
psnr_t, psnr_r, psnr_g, psnr_b = PSNR(img, img.astype(np.uint8), window_size)
print("img-img: psnr_t = " + str(psnr_t) + " psnr_r = " + str(psnr_r) + " psnr_g = " + str(psnr_g) + " psnr_b = " + str(psnr_b))

psnr_t, psnr_r, psnr_g, psnr_b = PSNR(img, img_zg_ga.astype(np.uint8), window_size)
print("img-img_zg_ga: psnr_t = " + str(psnr_t) + " psnr_r = " + str(psnr_r) + " psnr_g = " + str(psnr_g) + " psnr_b = " + str(psnr_b))

psnr_t, psnr_r, psnr_g, psnr_b = PSNR(img, img_zg_im.astype(np.uint8), window_size)
print("img-img_zg_im: psnr_t = " + str(psnr_t) + " psnr_r = " + str(psnr_r) + " psnr_g = " + str(psnr_g) + " psnr_b = " + str(psnr_b))

psnr_t, psnr_r, psnr_g, psnr_b = PSNR(img, img_filt_medie.astype(np.uint8), window_size)
print("img-img_filt_medie: psnr_t = " + str(psnr_t) + " psnr_r = " + str(psnr_r) + " psnr_g = " + str(psnr_g) + " psnr_b = " + str(psnr_b))

psnr_t, psnr_r, psnr_g, psnr_b = PSNR(img, img_filt_median.astype(np.uint8), window_size)
print("img-img_filt_median: psnr_t = " + str(psnr_t) + " psnr_r = " + str(psnr_r) + " psnr_g = " + str(psnr_g) + " psnr_b = " + str(psnr_b))

#calcul SSIM
print()
ssim = SSIM(img, img.astype(np.uint8), window_size)
print("img-img: ssim_t = " + str(ssim))

ssim = SSIM(img, img_zg_ga.astype(np.uint8), window_size)
print("img-img_zg_ga: ssim_t = " + str(ssim))

ssim = SSIM(img, img_zg_im.astype(np.uint8), window_size)
print("img-img_zg_im: ssim_t = " + str(ssim))

ssim = SSIM(img, img_filt_medie.astype(np.uint8), window_size)
print("img-img_filt_medie: ssim_t = " + str(ssim))

ssim = SSIM(img, img_filt_median.astype(np.uint8), window_size)
print("img-img_filt_median: ssim_t = " + str(ssim))

#calcul MAE
print()
mae_t, mae_r, mae_g, mae_b = MAE(img, img.astype(np.uint8), window_size)
print("img-img: mae_t = " + str(mae_t) + " mae_r = " + str(mae_r) + " mae_g = " + str(mae_g) + " mae_b = " + str(mae_b))

mae_t, mae_r, mae_g, mae_b = MAE(img, img_zg_ga.astype(np.uint8), window_size)
print("img-img_zg_ga: mae_t = " + str(mae_t) + " mae_r = " + str(mae_r) + " mae_g = " + str(mae_g) + " mae_b = " + str(mae_b))

mae_t, mae_r, mae_g, mae_b = MAE(img, img_zg_im.astype(np.uint8), window_size)
print("img-img_zg_im: mae_t = " + str(mae_t) + " mae_r = " + str(mae_r) + " mae_g = " + str(mae_g) + " mae_b = " + str(mae_b))

mae_t, mae_r, mae_g, mae_b = MAE(img, img_filt_medie.astype(np.uint8), window_size)
print("img-img_filt_medie: mae_t = " + str(mae_t) + " mae_r = " + str(mae_r) + " mae_g = " + str(mae_g) + " mae_b = " + str(mae_b))

mae_t, mae_r, mae_g, mae_b = MAE(img, img_filt_median.astype(np.uint8), window_size)
print("img-img_filt_median: mae_t = " + str(mae_t) + " mae_r = " + str(mae_r) + " mae_g = " + str(mae_g) + " mae_b = " + str(mae_b))