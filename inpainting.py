import cv2
import numpy as np
from sklearn.feature_extraction import image
from scipy.spatial import distance
from matplotlib import pyplot as plt


def get_patches(img, mask, patch_size):
    height, width = img.shape[:2]
    padded_img = np.pad(img, patch_size//2)
    padded_mask = np.pad(mask, patch_size//2)
    patches = []
    for i in range(patch_size//2, height - patch_size//2):
        for j in range(patch_size//2, width - patch_size//2):
            patch = padded_img[i:i+patch_size, j:j+patch_size]
            mask_patch = padded_mask[i:i+patch_size, j:j+patch_size]
            if np.all(mask_patch != 255):  # if the patch does not overlap with the masked region
                patches.append(patch)
    print('done getting patches')
    return patches


def find_best_match(patch, patches):
    min_dist = float('inf')
    best_match = None
    for candidate in patches:
        if patch.shape == candidate.shape:
            dist = distance.euclidean(patch.flatten(), candidate.flatten())
            if dist < min_dist:
                min_dist = dist
                best_match = candidate
    return best_match


def remove_object(image, mask, patch_size):
    result = np.copy(image)
    patches = get_patches(result, mask, patch_size)

    for x in range(patch_size//2, image.shape[0] - patch_size//2):
        for y in range(patch_size//2, image.shape[1] - patch_size//2):
            if mask[x, y] == 255:
                patch = image[x-patch_size//2:x+patch_size //
                              2+1, y-patch_size//2:y+patch_size//2+1]
                best_match = find_best_match(patch, patches)
                if best_match is not None:
                    print(f'found best match for patch at {x}, {y}')
                    result[x-patch_size//2:x+patch_size//2+1, y -
                           patch_size//2:y+patch_size//2+1] = best_match

    return result


def main():
    # Load your image and mask here. This is just an example, adjust it to your needs.
    image = cv2.imread('Images/baloon.jpg')
    # grayscale image to represent the mask
    mask = cv2.imread('Images/newMask.png', 0)

    # Convert mask to binary in case it's not
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    # Define the patch size
    patch_size = 5

    # Use the remove_object function to remove the object
    result = remove_object(image, mask, patch_size)

    # Display the result
    plt.imshow(result)
    plt.show()


if __name__ == "__main__":
    main()
