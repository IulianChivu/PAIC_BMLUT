import numpy as np

def add_zgomot_gaussian(img, medie, dispersie):
    zg = np.random.normal(medie, dispersie, img.shape)
    img_new = img+zg
    
    img_new[img_new>255] = 255
    img_new[img_new<0] = 0
    
    return img_new

def add_zgomot_impulsiv(img, ratio):
    h, w, c = img.shape
    lenght = int(h*w*ratio)
    
    img_new = img.copy()
    lin = np.random.randint(0, h, lenght)
    col =np.random.randint(0, w, lenght)
    val =np.random.randint(0, 2, lenght)
    for i in range (lenght):
        img_new[lin[i], col[i], np.random.randint(0,3)] = 255*val[i]
    return img_new