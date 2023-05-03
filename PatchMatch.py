from GetMask import GetMask
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
from sklearn.feature_extraction import image
from scipy.spatial import distance

matplotlib.use('TkAgg')
# howdy


def Read(path="", source_filename=""):
    source = plt.imread(path + source_filename)
    # get information about the image type (min max values)
    info = np.iinfo(source.dtype)
    # normalize the image into range 0 and 1
    source = source.astype(np.float32) / info.max
    mask, points = GetMask(source)

    # get information about the image type (min max values)
    # info = np.iinfo(mask.dtype)
    # # normalize the image into range 0 and 1
    # mask = mask.astype(np.float32) / info.max
    return source, mask, points


def similarity_score(patch1, patch2):
    return np.sum((patch1 - patch2) ** 2)


def extract_patch(image, x, y, patch_height, patch_width):

    x = int(x)
    y = int(y)
    patch_height = int(patch_height)
    patch_width = int(patch_width)
    # print(image.shape)
    # print(x, y, patch_height, patch_width)
    return image[y:y+patch_height, x:x+patch_width]


def find_best_match(image, border_patch, patch_size, mask):
    best_score = float("inf")
    best_x, best_y = -1, -1

    for y in range(image.shape[0] - patch_size[0] + 1):
        for x in range(image.shape[1] - patch_size[1] + 1):
            patch = extract_patch(image, x, y, patch_size[0], patch_size[1])
            patch_mask = extract_patch(mask, x, y, patch_size[0], patch_size[1])
            
            if np.any(patch_mask == 1):
                continue

            score = similarity_score(border_patch, patch)
            if score < best_score:
                best_score = score
                best_x, best_y = x, y

    return best_x, best_y


if __name__ == '__main__':
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

    # plt.imshow(mask)
    # plt.show()

    plt.imsave('testMask.png', mask, cmap='gray')
    initial_patch = extract_patch(source, x1, y1, y2-y1, x2-x1)
    border_size = 5
    border_patch = extract_patch(
        source, x1 - border_size, y1 - border_size, (y2-y1)+2*border_size, (x2 - x1) + 2 * border_size)
    best_x, best_y = find_best_match(
        source, border_patch, border_patch.shape, mask)

    mask_area_height, mask_area_width = y2 - y1, x2 - x1
    source_patch = extract_patch(
        source, best_x + border_size, best_y + border_size, mask_area_height, mask_area_width)
    # source[y1:y2, x1:x2] = source_patch
    print(int(y1), int(y2), int(x1), int(x2))
    print(source_patch.shape)
    # plt.imshow(source_patch)
    # plt.show()
    plt.imsave('source_patch.png', source_patch)
    plt.imsave('initial_patch.png', initial_patch)
    source[int(y1):int(y2), int(x1):int(x2)] = source_patch

    plt.imsave(outputDir + outputname, source)
    print("Done")
