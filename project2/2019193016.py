import os
import glob
import sys
import numpy as np
import cv2

def imshow(img):
    cv2.imshow('a', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

############# STEP1 FUNCS #############

def get_file_list(path_dir):
    files = os.listdir(path_dir)
    return [file for file in files if file.endswith(".pgm")]

def read_imgs(path_dir):
    files = get_file_list(path_dir)
    files.sort()
    imgs = []
    for file in files:
        imgs.append(cv2.imread(path_dir+'/'+file, -1))
    return imgs

def flat_imgs(imgs):
    H, W = imgs[0].shape
    L = len(imgs)
    result = np.zeros((H*W, L))
    for i, img in enumerate(imgs):
        temp = img.astype(np.uint8)
        result[:, i] = temp.reshape(H*W, )
    return np.array(result).astype(np.uint8)

def cal_number_of_PC(input_percentage, S):
    total = S.sum()
    sum = 0
    for i, sv in enumerate(S):
        sum += sv
        if sum/total >= input_percentage:
            return i+1

def print_step1(input_percentage, selected_dimension):
    print('##########  STEP 1  ##########')
    print(f"Input Percentage: {input_percentage}")
    print(f"Selected Dimension: {selected_dimension}")


############# STEP2 FUNCS #############

def save_img(img, idx, path_dir='./2019193016/face'):
    cv2.imwrite(path_dir+ f'{idx:02}.pgm', img)

def reconstruct(zero_mean_X, U, mean_img, number_of_PC):
    temp = np.matmul(np.transpose(U[:, :number_of_PC]), zero_mean_X)
    X_hat = np.clip(np.matmul(U[:, :number_of_PC], temp)+mean_img, 0, 255)
    return X_hat

def calculate_mse(X, X_hat):
    num_of_img = len(X[0])
    errors = []
    for i in range(num_of_img):
        error = sum(((X[:,i]-X_hat[:,i])/255)**2) 
        errors.append(error)
    average_error = np.mean(errors)
    return average_error, errors

def print_step2(average_error, errors):
    print()
    print('##########  STEP 2  ##########')
    print("Reconstruction error")
    print(f"average : {average_error:.4f}")
    for i,error in enumerate(errors):
        print(f"{i+1:02}: {error:.4f}")


############# STEP3 FUNCS #############
def get_l2dist(num_of_test_imgs, num_of_imgs, Y, Y_hat):
    dists = []
    for i in range(len(test_imgs)):
        dists.append([])               #initializing the dist list

    for i in range(num_of_test_imgs):
        test = Y_hat[:, i].reshape(len(Y_hat),1)
        diff = Y-test
        dist = diff**2
        for j in range(num_of_imgs):
            dists[i].append(np.sqrt(sum(dist[:,j])))
    return dists

def get_min_dist(dists):
    min_dist = []
    idxs = []
    for i in range(len(dists)):
        min_dist.append(np.min(dists[i]))
        idxs.append(dists[i].index(min_dist[i]) + 1)
    return min_dist, idxs

def print_step3(idxs):
    print()
    print('##########  STEP 3  ##########')
    for i in range(len(idxs)):
        print(f'test{i+1:02}.pgm ==> face{idxs[i]:02}.pgm')

############# MAIN #############

if __name__ == "__main__":
    input_percentage = float(sys.argv[1])
    STUDENT_CODE = '2019193016'
    FILE_NAME = 'output.txt'

    path_dir = './faces_training'

    imgs = read_imgs(path_dir)
    X = (flat_imgs(imgs)).astype(np.float64)

    if not os.path.exists(STUDENT_CODE) :
        os.mkdir(STUDENT_CODE)
    f = open(os.path.join(STUDENT_CODE, FILE_NAME), 'w')
    sys.stdout = f
    #####################################
    #             STEP 1                #
    #####################################
    mean_img = np.ndarray((len(X),1))
    
    for i in range(len(X)):             
        mean_img[i] = X[i, :].mean()    # get mean img with col-vector

    zero_mean_X = np.subtract(X, mean_img) # get mean-centered matrix

    U, S, V = np.linalg.svd(zero_mean_X, full_matrices=False) # run SVD

    selected_dimension = cal_number_of_PC(input_percentage, S) # selecting the number of PC

    print_step1(input_percentage, selected_dimension)   # print in format

    #####################################
    #            STEP 1 DONE            #
    #####################################

    #####################################
    #             STEP 2                #
    #####################################
    
    X_hat = reconstruct(zero_mean_X, U, mean_img, selected_dimension)   # reconstruct img
    for i in range(len(X_hat[0])):                                      # save the reconstructed image
        save_img(img = X_hat[:, i].reshape(192, 168), idx = i+1)
    average_error, errors = calculate_mse(X, X_hat) # calculate the Mean Square Error
    print_step2(average_error, errors) # print in format

    #####################################
    #            STEP 2 DONE            #
    #####################################

    #####################################
    #             STEP 3                #
    #####################################
 
    Y = X_hat # get reconstructed image's col-vec matrix
    
    test_imgs = read_imgs('./faces_test')   # get test images
    X_test = (flat_imgs(test_imgs)).astype(np.float64)  # make col-vec matrix

    zero_mean_X_test = np.subtract(X_test, mean_img)    # get mean-centered matrix

    Y_hat = reconstruct(zero_mean_X_test, U, mean_img, selected_dimension) # reconstruct the test image with eigenface that we got in step 1 and 2

    dists = get_l2dist(len(test_imgs), len(imgs), Y, Y_hat) #calculate L2 distances
    
    min_dist, idxs = get_min_dist(dists) # get min distance and the index

    print_step3(idxs)

    # for i in range(len(Y_hat[0])):                                      # save the reconstructed image
    #     save_img(img = Y_hat[:, i].reshape(192, 168), idx = i+1, path_dir='./2019193016/face_test')

    #####################################
    #            STEP 3 DONE            #
    #####################################
    
    f.close()
