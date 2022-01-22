import numpy as np
import math
from skimage.metrics import structural_similarity


def MSE(img_originala, img_filtrata, window_size):
    
    capat = window_size//2
    
    h,w,c = img_originala.shape
    
    mse_r = img_originala[capat:h-capat, capat:w-capat, 0] - img_filtrata[capat:h-capat, capat:w-capat, 0]
    mse_g = img_originala[capat:h-capat, capat:w-capat, 1] - img_filtrata[capat:h-capat, capat:w-capat, 1]
    mse_b = img_originala[capat:h-capat, capat:w-capat, 2] - img_filtrata[capat:h-capat, capat:w-capat, 2]
    
    mse_r = mse_r ** 2
    mse_g = mse_g ** 2
    mse_b = mse_b ** 2
    
    mse_r = np.sum(mse_r) / ((h-capat*2)*(w-capat*2))
    mse_g = np.sum(mse_g) / ((h-capat*2)*(w-capat*2))
    mse_b = np.sum(mse_b) / ((h-capat*2)*(w-capat*2))
    
    mse_t = (mse_r + mse_g + mse_b)/3
    
    return mse_t, mse_r, mse_g, mse_b

def PSNR(img_originala, img_filtrata, window_size):
    
    mse_t, mse_r, mse_g, mse_b = MSE(img_originala, img_filtrata, window_size)
    
    if(mse_t == 0):
        psnr_t = 'MAX'
    else:
        psnr_t = 20 * math.log10(255) - 10 * math.log10(mse_t)
        
    if(mse_r == 0):
        psnr_r = 'MAX'
    else:
        psnr_r = 20 * math.log10(255) - 10 * math.log10(mse_r)
        
    if(mse_g == 0):
        psnr_g = 'MAX'
    else:
        psnr_g = 20 * math.log10(255) - 10 * math.log10(mse_g)
        
    if(mse_b == 0):
        psnr_b = 'MAX'
    else:
        psnr_b = 20 * math.log10(255) - 10 * math.log10(mse_b)
    
    return psnr_t, psnr_r, psnr_g, psnr_b
    
def SSIM(img_originala, img_filtrata, window_size):
    
    capat = window_size//2
    h,w,c = img_originala.shape
    
    ssim = structural_similarity(img_originala[capat:h-capat, capat:w-capat], img_filtrata[capat:h-capat, capat:w-capat], channel_axis = 2)
    
    return ssim

def MAE(img_originala, img_filtrata, window_size):
    capat = window_size//2
    
    h,w,c = img_originala.shape
    
    mse_r = img_originala[capat:h-capat, capat:w-capat, 0] - img_filtrata[capat:h-capat, capat:w-capat, 0]
    mse_g = img_originala[capat:h-capat, capat:w-capat, 1] - img_filtrata[capat:h-capat, capat:w-capat, 1]
    mse_b = img_originala[capat:h-capat, capat:w-capat, 2] - img_filtrata[capat:h-capat, capat:w-capat, 2]
    
    mse_r = abs(mse_r)
    mse_g = abs(mse_g)
    mse_b = abs(mse_b)
    
    mae_r = np.sum(mse_r) / ((h-capat*2)*(w-capat*2))
    mae_g = np.sum(mse_g) / ((h-capat*2)*(w-capat*2))
    mae_b = np.sum(mse_b) / ((h-capat*2)*(w-capat*2))
    
    mae_t = (mae_r + mae_g + mae_b)/3
    
    return mae_t, mae_r, mae_g, mae_b