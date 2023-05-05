from GetMask import GetMask
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
# from sklearn.feature_extraction import image
from scipy.spatial import distance
import scipy
import copy

matplotlib.use('TkAgg')

# Define Poisson blending function
# This function's code (def PoissonBlen()) is from Ezra Lane's previous assignment on Poisson blending


def PoissonBlend(source, mask, target, isMix=False):
    target = np.array(target)
    rt = np.copy(target[:, :, 0])
    gt = np.copy(target[:, :, 1])
    bt = np.copy(target[:, :, 2])
    row = []
    col = []
    data = []
    print(source.shape)
    print(target.shape)
    for j in range(0, np.size(mask, 1)):
        for i in range(0, np.size(mask, 0)):
            if mask[i][j][0] == 1.0:
                if isMix:
                    temp_s_R = (4*source[i][j][0]) - (source[i-1][j][0] if i > 0 else 0) - (source[i+1][j][0] if i < (np.size(
                        mask, 0) - 1) else 0) - (source[i][j+1][0] if j < (np.size(mask, 1) - 1) else 0) - (source[i][j-1][0] if j > 0 else 0)
                    temp_s_G = (4*source[i][j][1]) - (source[i-1][j][1] if i > 0 else 0) - (source[i+1][j][1] if i < (np.size(
                        mask, 0) - 1) else 0) - (source[i][j+1][1] if j < (np.size(mask, 1) - 1) else 0) - (source[i][j-1][1] if j > 0 else 0)
                    temp_s_B = (4*source[i][j][2]) - (source[i-1][j][2] if i > 0 else 0) - (source[i+1][j][2] if i < (np.size(
                        mask, 0) - 1) else 0) - (source[i][j+1][2] if j < (np.size(mask, 1) - 1) else 0) - (source[i][j-1][2] if j > 0 else 0)
                    temp_t_R = (4*target[i][j][0]) - (target[i-1][j][0] if i > 0 else 0) - (target[i+1][j][0] if i < (np.size(
                        mask, 0) - 1) else 0) - (target[i][j+1][0] if j < (np.size(mask, 1) - 1) else 0) - (target[i][j-1][0] if j > 0 else 0)
                    temp_t_G = (4*target[i][j][1]) - (target[i-1][j][1] if i > 0 else 0) - (target[i+1][j][1] if i < (np.size(
                        mask, 0) - 1) else 0) - (target[i][j+1][1] if j < (np.size(mask, 1) - 1) else 0) - (target[i][j-1][1] if j > 0 else 0)
                    temp_t_B = (4*target[i][j][2]) - (target[i-1][j][2] if i > 0 else 0) - (target[i+1][j][2] if i < (np.size(
                        mask, 0) - 1) else 0) - (target[i][j+1][2] if j < (np.size(mask, 1) - 1) else 0) - (target[i][j-1][2] if j > 0 else 0)
                    rt[i][j] = (temp_s_R if temp_s_R >= temp_t_R else temp_t_R)
                    gt[i][j] = (temp_s_G if temp_s_G >= temp_t_G else temp_t_G)
                    bt[i][j] = (temp_s_B if temp_s_B >= temp_t_B else temp_t_B)
                else:
                    rt[i][j] = (4*source[i][j][0]) - (source[i-1][j][0] if i > 0 else 0) - (source[i+1][j][0] if i < (np.size(mask, 0) - 1)
                                                                                            else 0) - (source[i][j+1][0] if j < (np.size(mask, 1) - 1) else 0) - (source[i][j-1][0] if j > 0 else 0)
                    gt[i][j] = (4*source[i][j][1]) - (source[i-1][j][1] if i > 0 else 0) - (source[i+1][j][1] if i < (np.size(mask, 0) - 1)
                                                                                            else 0) - (source[i][j+1][1] if j < (np.size(mask, 1) - 1) else 0) - (source[i][j-1][1] if j > 0 else 0)
                    bt[i][j] = (4*source[i][j][2]) - (source[i-1][j][2] if i > 0 else 0) - (source[i+1][j][2] if i < (np.size(mask, 0) - 1)
                                                                                            else 0) - (source[i][j+1][2] if j < (np.size(mask, 1) - 1) else 0) - (source[i][j-1][2] if j > 0 else 0)

                row.append((j*np.size(mask, 0) + i))
                col.append((j*np.size(mask, 0) + i))
                data.append(4)
                if j > 0:
                    row.append(((j)*np.size(mask, 0) + i))
                    col.append(((j-1)*np.size(mask, 0) + i))
                    data.append(-1)

                if j < (np.size(mask, 1) - 1):
                    row.append(((j)*np.size(mask, 0) + i))
                    col.append(((j+1)*np.size(mask, 0) + i))
                    data.append(-1)

                if i > 0:
                    row.append((j*np.size(mask, 0) + (i)))
                    col.append((j*np.size(mask, 0) + (i-1)))
                    data.append(-1)

                if i < (np.size(mask, 0) - 1):
                    row.append((j*np.size(mask, 0) + (i)))
                    col.append((j*np.size(mask, 0) + (i+1)))
                    data.append(-1)
            else:
                row.append((j*np.size(mask, 0) + i))
                col.append((j*np.size(mask, 0) + i))
                data.append(1)

    A = scipy.sparse.csr_matrix((data, (row, col)), shape=(
        np.size(mask, 0)*np.size(mask, 1), np.size(mask, 0)*np.size(mask, 1)))

    rt = np.reshape(rt, ((np.size(rt, 0)*np.size(rt, 1)), 1), order="F")
    gt = np.reshape(gt, ((np.size(gt, 0)*np.size(gt, 1)), 1), order="F")
    bt = np.reshape(bt, ((np.size(bt, 0)*np.size(bt, 1)), 1), order="F")

    result_r = scipy.sparse.linalg.spsolve(A, rt)
    result_g = scipy.sparse.linalg.spsolve(A, gt)
    result_b = scipy.sparse.linalg.spsolve(A, bt)

    result_r = np.reshape(result_r, (np.size(target, 0),
                          np.size(target, 1)), order="F")
    result_g = np.reshape(result_g, (np.size(target, 0),
                          np.size(target, 1)), order="F")
    result_b = np.reshape(result_b, (np.size(target, 0),
                          np.size(target, 1)), order="F")

    result_r[result_r < 0.0] = 0.0
    result_r[result_r > 1.0] = 1.0
    result_g[result_g < 0.0] = 0.0
    result_g[result_g > 1.0] = 1.0
    result_b[result_b < 0.0] = 0.0
    result_b[result_b > 1.0] = 1.0

    result = np.stack((result_r, result_g, result_b), axis=2)

    return result


def Read(path="", source_filename=""):
    source = plt.imread(path + source_filename)
    # get information about the image type (min max values)
    info = np.iinfo(source.dtype)
    # normalize the image into range 0 and 1
    source = source.astype(np.float32) / info.max
    mask, points = GetMask(source)
    return source, mask, points

# This function find the similarity score between two patches


def similarity_score(patch1, patch2):
    return np.sum((patch1 - patch2) ** 2)

# This function extracts a patch from an image to be used
# for comparison in the rest of the algorithm


def extract_patch(image, x, y, patch_height, patch_width):
    # Extract a patch of size patch_height x patch_width from image whose top left corner is at (x, y)
    x = int(x)
    y = int(y)
    patch_height = int(patch_height)
    patch_width = int(patch_width)
    return image[y:y+patch_height, x:x+patch_width]

# Function to find the best match for a given patch in the source image


def find_best_match(image, border_patch, patch_size, mask):
    # Initialize the best score to positive infinity and the best patch coordinates to -1
    best_score = float("inf")
    best_x, best_y = -1, -1

    # Loop through all possible y coordinates in the image
    for y in range(image.shape[0] - patch_size[0] + 1):
        # Loop through all possible x coordinates in the image
        for x in range(image.shape[1] - patch_size[1] + 1):
            # Extract a patch from the image at the current (x, y) position
            patch = extract_patch(image, x, y, patch_size[0], patch_size[1])
            # Extract a corresponding patch from the mask
            patch_mask = extract_patch(
                mask, x, y, patch_size[0], patch_size[1])

            # If any part of the mask patch has a value of 1, skip this patch
            if np.any(patch_mask == 1):
                continue

            # Calculate the similarity score between the border_patch and the current patch
            score = similarity_score(border_patch, patch)

            # If the calculated score is better than the best_score, update best_score and best_x, best_y coordinates
            if score < best_score:
                best_score = score
                best_x, best_y = x, y

    # Return the best matching patch's top-left corner (x, y) coordinates
    return best_x, best_y


if __name__ == '__main__':
    print("Welcome to the Image Inpainting Program")

    # Setting up the input output paths
    inputDir = 'Images/'
    outputDir = 'Results/'

    # prompt user for source, target and mask
    source = input("Enter source image: ")
    outputname = input("Enter output image name: ")

    # Read data and clean mask
    source, maskOriginal, points = Read(inputDir, source)

    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = points

    # Cleaning up the mask (creating a binary mask)
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0
    
    plt.imsave('mask.png', mask, cmap='gray')
    # Extracting the patch from the source image
    initial_patch = extract_patch(source, x1, y1, y2-y1, x2-x1)
    border_size = 5
    # Extracting the border patch from the source image
    border_patch = extract_patch(
        source, x1 - border_size, y1 - border_size, (y2-y1)+2*border_size, (x2 - x1) + 2 * border_size)
    # Finding the best match for the border patch in the source image
    best_x, best_y = find_best_match(
        source, border_patch, border_patch.shape, mask)
    # Extracting the best match patch from the source image
    mask_area_height, mask_area_width = y2 - y1, x2 - x1
    source_patch = extract_patch(
        source, best_x + border_size, best_y + border_size, mask_area_height, mask_area_width)
    # Copying the source patch to the target image
    temp = copy.deepcopy(source)
    # Resizing the source patch to fit the target patch
    resized_source_patch = cv2.resize(
        source_patch, (int(x2) - int(x1), int(y2) - int(y1)))
    # Copying the resized source patch to the target image
    source[int(y1):int(y2), int(x1):int(x2)] = resized_source_patch
    #Using Poisson Blending to blend the source and target images
    final = PoissonBlend(source, mask, temp)
    # Saving the final image
    plt.imsave(outputDir + outputname, final)
    print("Done")
