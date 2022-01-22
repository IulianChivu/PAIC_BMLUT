import numpy as np

#aplica media aritmetica pentru o vecinatate
def smooth(fereastra, dim):
    
    #PARAMETERS : fereastra,dim ferestrei
    #RETURN  :   media aritmetica a pixelilor din fereastra
    suma = np.sum(fereastra)
    
    return suma / (dim**2)
    
def filtru_medie_aritmetica(img, window_size):
    
    #PARAMETERS : imaginea, dim ferestrei
    #RETURN  :   imag filtrata
    
    #get dim img
    h,w,c=img.shape
    
    #calcul capat
    capat=window_size//2
    
    #initializez imaginea pentru filtrare cu 0
    img_filt = np.zeros(img.shape)
    #parcurg imaginea, decupez si aplic functia scrisa anterior pentru fiecare fereastra
    for k in range(c):
        for i in range(capat, h-capat):
            for j in range(capat, w-capat):
                img_filt[i][j][k] = smooth(img[i-capat:i+capat+1, j-capat:j+capat+1, k], window_size)
    
    #returnez imag_filtra
    return img_filt

def filtru_median(img, window_size):
    h, w, c = img.shape
    capat = window_size//2
    mijloc = window_size**2//2
    
    img_out = np.zeros(img.shape)
    for k in range(c):
        for i in range(capat, h-capat):
            for j in range(capat, w-capat):        
                val = np.sort(img[i-capat:i+capat+1 , j-capat:j+capat+1, k].flatten())
                img_out[i,j,k] = val[mijloc]
    return img_out