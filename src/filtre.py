import numpy as np
import cv2
import matplotlib.pyplot as plt

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


###############################################################################

def bmlut(img):
    #get dim img
    h,w=img.shape
    
    #calcul capat
    capat=1
    
    #starting iwht only green channel
    #img = img[:,:,1]
    
# =============================================================================
#     #plot hist
#     cv2.imshow("gray", img)
#     hist = cv2.calcHist([img], [0], None, [256], [0, 255])
#     plt.figure()
#     plt.title("Grayscale Histogram")
#     plt.xlabel("Bins")
#     plt.ylabel("# of Pixels")
#     plt.plot(hist)
#     plt.xlim([0, 255])
#     
#     #colormap
#     # plot a 2D color histogram for green and blue
#     fig = plt.figure()
#     
#     ax = fig.add_subplot(111)
#     hist = cv2.calcHist([img, img], [0, 0], None, [255, 255], [0, 255, 0, 255])
#     p = ax.imshow(hist, interpolation = "nearest")
#     ax.set_title("2D Color Histogram for Green and Blue")
#     plt.gca().invert_yaxis()
#     plt.colorbar(p)
# =============================================================================
    
    ###########################################################################
    #A - identification of noisy pixels
    
    #creare histrograma 2D
    hist2d = np.zeros([256, 256])
    
    #add v9
    for i in range(1, h-1):
        for j in range(1, w-1):
            hist2d[img[i, j], img[i-1, j-1]] += 1
            hist2d[img[i, j], img[i+1, j+1]] += 1
            hist2d[img[i, j], img[i-1, j+1]] += 1
            hist2d[img[i, j], img[i+1, j-1]] += 1
            hist2d[img[i, j], img[i, j-1]] += 1
            hist2d[img[i, j], img[i, j+1]] += 1
            hist2d[img[i, j], img[i-1, j]] += 1
            hist2d[img[i, j], img[i+1, j]] += 1
            
    #add corners
# =============================================================================
#     hist2d[img[0,0], img[1, 0]] += 1
#     hist2d[img[0,0], img[0, 1]] += 1
#     hist2d[img[0,0], img[1, 1]] += 1
#     
#     hist2d[img[0,w-1], img[0, w-2]] += 1
#     hist2d[img[0,w-1], img[1, w-1]] += 1
#     hist2d[img[0,w-1], img[1, w-2]] += 1
#     
#     hist2d[img[h-1,0], img[h-2, 0]] += 1
#     hist2d[img[h-1,0], img[h-1, 1]] += 1
#     hist2d[img[h-1,0], img[h-2, 1]] += 1
#     
#     hist2d[img[h-1,w-1], img[h-2, w-1]] += 1
#     hist2d[img[h-1,w-1], img[h-1, w-2]] += 1
#     hist2d[img[h-1,w-1], img[h-2, w-2]] += 1
# =============================================================================
    
    #add edges TODO
    #for i in range(1, h-1):
        #hist2d[img[i, 0], ]
    
    #plot hist2d
    plt.figure()
    
    plt.imshow(hist2d, cmap='jet')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("2D neighboring histogram")
    #plt.show()
    
    #plot cross section at 150 of hist2d
    plt.figure()
    plt.plot(np.arange(256), hist2d[150, :])
    plt.title("Vertical cross section of 2D neighboring histogram (i=150)")
    #plt.show()
    
    
    #calcul delta-up si delta-low
    delta = 30 #const setup by user
    low_up_list = []
    
    #for every pixel we need to find delta_low and delta_up using Eq(1)
    for i in range (256):
        delta_up = i
        delta_low = i
        
        all_sum = np.sum(hist2d[i, :])
        #print(str(all_sum) + " at " + str(i))
        
        #no pixel of value condition
        if all_sum == 0 :
            print("No value of pixel " + str(i) + " in image")
            low_up_list.append([delta_low,delta_up])
        
        else: 
            while True:
                low_up_sum = np.sum(hist2d[i, delta_low:delta_up+1])
                
                if delta <= 100 * low_up_sum/all_sum :
                    low_up_list.append([delta_low,delta_up])
                    break;
                
                delta_low -= 1
                delta_up += 1
                
                
                if delta_low < 0 :
                    delta_low = 0
                
                if delta_up > 255 :
                    delta_up = 255
                    
                #print("up: " + str(delta_up) + " low: " + str(delta_low))
                #print(100 * low_up_sum/all_sum)
                #print(low_up_sum)
                #print(all_sum)
    
    print()
    print("Delta low and delta up list for every pixel value + size : ")
    print(low_up_list)
    print(len(low_up_list))
    
    
    #find the positions of noisy pixels
    noisy_array = np.zeros([h, w])
    counter_threshold = 1 #value from 0 to 8 (8 - the pixel is not noise) (0 - the pixel is noise) setup by user
    #low value threshold == less noisy pixels / high value threshold == more pixels as noisy pixels
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            
            counter = 0
            vect = img[i-capat:i+capat+1 , j-capat:j+capat+1].flatten()
            
            #add counter if the neighbors are in the delta_low - delta_up interval
            for k in range(0,9):
                if low_up_list[img[i][j]][0] <= vect[k] and vect[k] <= low_up_list[img[i][j]][1]:
                    counter +=1
              
            #remove self from counter
            counter -= 1
            
            #make noisy pixel == 1
            if counter < counter_threshold:
                noisy_array[i,j] = 1
    
    #plot the noise img
    plt.figure()
    plt.imshow(noisy_array.astype(np.uint8))
    plt.title("Homogeneity assumption (noisy pixels == 1)")
    
    
    
    
    
    
    ###########################################################################
    #B - Construction of pointer arrays for matching blocks
    
    #determine noisy blocks and matching blocks
    
    noisy_block_threshold = 5 #value from 0 to 9 (the max noisy pixels that a matching block can have) setup by user
    
    #matching block == 0 , target block == 1 , noisy block == 2, noisy block non-overlapping == 3
    match_target_noisy = np.zeros([h-2,w-2])
    
    ###########################################################################
    #1 determine noisy blocks
    for i in range(1, h-1):
        for j in range(1, w-1):
            
            noisy_block_counter = 0
            vect = noisy_array[i-capat:i+capat+1 , j-capat:j+capat+1].flatten()
            
            #add counter if noisy pixel
            for k in range(0,9):
                if vect[k] == 1:
                    noisy_block_counter +=1
            
            #clasify the block
            if noisy_block_counter > noisy_block_threshold:
                match_target_noisy[i-1,j-1] = 2 #noisy block
                    
            #elif noisy_block_counter >= 1:
            #    match_target_noisy[i-1,j-1] = 1 #target block
            #    print('target')
        
    plt.figure()
    plt.imshow(match_target_noisy.astype(np.uint8), cmap='jet')
    plt.colorbar()
    plt.title("match-target-noisy block map (only noisy block computed)")    
    
    ###########################################################################    
    #2 determine target blocks
    for i in range(1, h-1, 3):
        for j in range(1, w-1, 3):
            
            if match_target_noisy[i-1,j-1] == 0:
                
                noisy_block_counter = 0
                vect = noisy_array[i-capat:i+capat+1 , j-capat:j+capat+1].flatten()
                
                #add counter if noisy pixel
                for k in range(0,9):
                    if vect[k] == 1:
                        noisy_block_counter +=1
                
                #clasify the block
                if noisy_block_counter >= 1:
                    match_target_noisy[i-1,j-1] = 1 #target block
                    
            elif match_target_noisy[i-1,j-1] == 2:
                match_target_noisy[i-1,j-1] = 3 #noisy non-overlapping block
    
    plt.figure()
    plt.imshow(match_target_noisy.astype(np.uint8), cmap='jet')
    plt.colorbar()
    plt.title("Added target blocks(val==1) to match-target-noisy block map")
    
    ###########################################################################
    #3 construct the 3-D array (LUT)
    
    print("Building LUT 3D array")
    
    #list in list in list
    #1st index for position [0:8]
    #2nd index for pixel value [0:255]
    #3rd index for (r,c) coordinates r*w + c [variable]
    lut = [[[] for _ in range(256)] for _ in range(9)]
    
    #lut = np.zeros([9,256,1])
    
    print()
    print(lut)
    print(len(lut))
    print(len(lut[1]))
    print(len(lut[1][1]))
    
    print(w)
    print(h)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            
            #if matching block
            if match_target_noisy[i-1,j-1] == 0:
                
                
# =============================================================================
#                 vect = img[i-capat:i+capat+1 , j-capat:j+capat+1].flatten()
#                 
#                 value = w*(i-1) + (j-1)
#                 lut[0][img[i-1, j-1]].append(value)
#                 
#                 print(value)
# =============================================================================
                
                index = 0
                for r in range(i-capat, i+capat+1):
                    for c in range(j-capat, j+capat+1):
                        value = r*w + c
                        lut[index][img[r,c]].append(value)
                        
                        #print(value)
                        index += 1
                
# =============================================================================
#                 for k in range(0,9):
#                     #compute coordinates to a scalar value
#                     value = i*w + j
#                     print(value)
#                     
#                     lut[k][vect[k]].append(value)
# =============================================================================
    
    
    #print()
    #print(len(lut))
    #print(len(lut[1]))
    #for i in range(1,256):
    #    print(len(lut[1][i]))
    
    
    ###########################################################################
    #4
    
    d = 1 #tolerance eq 2
    tau = 3 #the least number of matching pixels in a block to be considered a matching block for the target block
    S = 3 #the sufficient number of matching blocks for a target block
    
    #determine the number of matching, target and noisy blocks
    no_match = np.count_nonzero(match_target_noisy == 0)
    initial_no_target = np.count_nonzero(match_target_noisy == 1)
    no_noisy = np.count_nonzero(match_target_noisy == 2)
    no_noisy_nonover = np.count_nonzero(match_target_noisy == 3)
    print()
    print("Dim of initial matching blocks, target blocks, noisy blocks, noisy blocks nonoverlaping")
    print(no_match, initial_no_target, no_noisy, no_noisy_nonover)
    
    no_target = 0
    
    ###########################################################################
    #5
    
    while(initial_no_target >= no_target):
        
        #calculate next initial no target
        initial_no_target = np.count_nonzero(match_target_noisy == 1)
    
        print()
        print("Computing target blocks")
        ###########################################################################
        #6
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                
                #if i % 10 == 0:
                #    print(i, j, match_target_noisy[i-1,j-1])  
                    
                simil_T_M = 0
                tol = 0
                sum_t_bool = 0
                
                #if target block
                if match_target_noisy[i-1,j-1] == 1:
                    
                    target_block = img[i-capat:i+capat+1 , j-capat:j+capat+1].flatten()
                    target_noisy = noisy_array[i-capat:i+capat+1 , j-capat:j+capat+1].flatten()
                    
                    #list of coordinates of matching_blocks for this target block from lut
                    sum_t = []
                    sum_t_noisy = []
                    sum_t_bool = 0 #if sum_t is completed sum_t_bool = 1
                    
        ###########################################################################
        #7
                    for k in range(0,9):
                        #if clean pixel
                        if target_noisy[k] == 0:
                            
        ###########################################################################
        #8
        
                            #r = lut[k][target_block[k]] // 2
                            #c = lut[k][target_block[k]] % 2
                            
                            for l in range(len(lut[k][target_block[k]])):
                                r = lut[k][target_block[k]][l] // w 
                                c = lut[k][target_block[k]][l] % w 
                                    
                                #the matrix needs to be contructed depending on the start pixel of the match
                                if k == 0:
                                    posible_match = img[r:r+2+1, c:c+2+1].flatten()
                                    posible_match_noisy = noisy_array[r:r+2+1, c:c+2+1].flatten()
                                    
                                if k == 1: 
                                    posible_match = img[r:r+2+1, c-1:c+1+1].flatten()
                                    posible_match_noisy = noisy_array[r:r+2+1, c-1:c+1+1].flatten()
                                  
                                if k == 2: 
                                    posible_match = img[r:r+2+1, c-2:c+1].flatten()
                                    posible_match_noisy = noisy_array[r:r+2+1, c-2:c+1].flatten()
                                
                                if k == 3: 
                                    posible_match = img[r-1:r+1+1, c:c+2+1].flatten()
                                    posible_match_noisy = noisy_array[r-1:r+1+1, c:c+2+1].flatten()
                                
                                if k == 4: 
                                    posible_match = img[r-1:r+1+1, c-1:c+1+1].flatten()
                                    posible_match_noisy = noisy_array[r-1:r+1+1, c-1:c+1+1].flatten()
                                    
                                if k == 5: 
                                    posible_match = img[r-1:r+1+1, c-2:c+1].flatten()
                                    posible_match_noisy = noisy_array[r-1:r+1+1, c-2:c+1].flatten()
                                    
                                if k == 6: 
                                    posible_match = img[r-2:r+1, c:c+2+1].flatten()
                                    posible_match_noisy = noisy_array[r-2:r+1, c:c+2+1].flatten()
                                    
                                if k == 7: 
                                    posible_match = img[r-2:r+1, c-1:c+1+1].flatten()
                                    posible_match_noisy = noisy_array[r-2:r+1, c-1:c+1+1].flatten()
                                    
                                if k == 8: 
                                    posible_match = img[r-2:r+1, c-2:c+1].flatten()
                                    posible_match_noisy = noisy_array[r-2:r+1, c-2:c+1].flatten()
                                    
                                
                                    
                                #posible_match = img[r-capat:r+capat+1, c-capat:c+capat+1].flatten()
                                #posible_match_noisy = noisy_array[r-capat:r+capat+1, c-capat:c+capat+1].flatten()
                                
                                #print(r, c)
                                #print(posible_match)
                                #print(posible_match_noisy)
                                #print(target_block)
                                #print(target_noisy)
                                
                                #compute simil_T_M Eq (3)
                                simil_T_M = 0
                                tol = 0
                                for q in range(0,9):
                                    #compute tol
                                    diff = abs(int(target_block[q]) - int(posible_match[q]))
                                    #print("diff == " + str(diff) + " = " + str(target_block[q]) + " + " + str(posible_match[q]))
                                    
                                    if diff <= d and target_noisy[q] == 0 and posible_match_noisy[q] == 0:
                                        tol = 1
                                    else:
                                        tol = 0
                                    
                                    simil_T_M += (1 - target_noisy[q]) * (1 - posible_match_noisy[q]) * tol
                                
                                    if simil_T_M >= tau:
                                        sum_t.append(posible_match)
                                        sum_t_noisy.append(posible_match_noisy)
                                        
                    
        ###########################################################################
        #9
                            if len(sum_t) >= S:
                                sum_t_bool = 1
                                break
                                
        ###########################################################################
        #10
    
                    if sum_t_bool == 1:
                        
                        
                        for k in range(0,9):
                            add_noise = 0
                            mean_pixel_value = 0
                            
                            for q in range(len(sum_t)):
                                add_noise += sum_t_noisy[q][k]
                                mean_pixel_value += sum_t[q][k]
                            
                            if(add_noise == 0):
                                mean_pixel_value = mean_pixel_value // len(sum_t)
                                
                                #put the mean_pixel value in the target pixel position if its noisy
                                if target_noisy[k] == 1:    
                                    target_block[k] = mean_pixel_value
                                    target_noisy[k] = 0
                    
                        #make the changes in the image and noise mask   
                        img[i-capat:i+capat+1 , j-capat:j+capat+1] = np.reshape(target_block, (-1, 3))
                        noisy_array[i-capat:i+capat+1 , j-capat:j+capat+1] = np.reshape(target_noisy, (-1, 3))
                        
                        #move target to matching block if all the noise was removed
                        if (np.sum(target_noisy) == 0):
                            match_target_noisy[i-1,j-1] = 0 #matching
        
        #determine the number of matching, target and noisy blocks
        no_match = np.count_nonzero(match_target_noisy == 0)
        no_target = np.count_nonzero(match_target_noisy == 1)
        no_noisy = np.count_nonzero(match_target_noisy == 2)
        no_noisy_nonover = np.count_nonzero(match_target_noisy == 3)
        print()
        print("Dim of matching blocks, target blocks, noisy blocks, noisy blocks nonoverlaping")
        print(no_match, no_target, no_noisy, no_noisy_nonover)
        
        
        ###########################################################################
        #12
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                
                #if nonoverlaping noisy block
                if match_target_noisy[i-1,j-1] == 3:
                    
                    #check every corner extension (left-up)
                    if i - 2 >= 0 and j - 2 >= 0:
                        if np.sum(noisy_array[i-2:i+1 , j-2:j+1]) <= noisy_block_threshold:
                            
                            match_target_noisy[i-1, j-1] = 1 #new target
                            
                    #check every corner extension (right-up)
                    if i - 2 >= 0 and j + 2 <= w :
                        if np.sum(noisy_array[i-2:i+1 , j:j+2+1]) <= noisy_block_threshold:
                            
                            match_target_noisy[i-1, j+1] = 1 #new target
                            
                    #check every corner extension (left-down)
                    if i + 2 <= h and j - 2 >= 0:
                        if np.sum(noisy_array[i:i+2+1 , j-2:j+1]) <= noisy_block_threshold:
                            
                            match_target_noisy[i, j-1] = 1 #new target
                            
                    #check every corner extension (right-down)
                    if i + 2 <= h and j + 2 <= w:
                        if np.sum(noisy_array[i:i+2+1 , j:j+2+1]) <= noisy_block_threshold:
                            
                            match_target_noisy[i+1, j+1] = 1 #new target
                            
        
                            
        #determine the number of matching, target and noisy blocks
        #no_match = np.count_nonzero(match_target_noisy == 0)
        #no_target = np.count_nonzero(match_target_noisy == 1)
        #no_noisy = np.count_nonzero(match_target_noisy == 2)
        #no_noisy_nonover = np.count_nonzero(match_target_noisy == 3)
        #print()
        #print("Dim of matching blocks, target blocks, noisy blocks, noisy blocks nonoverlaping")
        #print(no_match, no_target, no_noisy, no_noisy_nonover)
        
        print (d)
        #increment tolerance Eq (2)
        d +=1
        
        if d > 20:
            break
        
        
        
    
    #plot the noise img
    plt.figure()
    plt.imshow(noisy_array.astype(np.uint8))
    plt.title("Homogeneity assumption (noisy pixels == 1)")
    
    return img
             
                