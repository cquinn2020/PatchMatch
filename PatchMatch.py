from GetMask import GetMask
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
from sklearn.feature_extraction import image
from scipy.spatial import distance

matplotlib.use('TkAgg')


def Read(path="", source_filename=""):
    source = plt.imread(path + source_filename)
    # get information about the image type (min max values)
    info = np.iinfo(source.dtype)
    # normalize the image into range 0 and 1
    source = source.astype(np.float32) / info.max
    mask = GetMask(source)

    # get information about the image type (min max values)
    info = np.iinfo(mask.dtype)
    # normalize the image into range 0 and 1
    mask = mask.astype(np.float32) / info.max
    return source, mask


def patch_match(image, mask, patch_size=5, iterations=5):
    height, width = image.shape[:2]
    # Step 1: Random Initialization
    # Start by assigning a random offset to each patch in the unknown region
    offsets = np.random.randint(-patch_size//2,
                                patch_size//2, size=(height, width, 2))

    for _ in range(iterations):
        # Step 2: Propagation
        for x in range(height):
            for y in range(width):
                if mask[x, y] == 1:
                    # Get the current patch
                    patch = image[x-patch_size//2:x+patch_size //
                                  2+1, y-patch_size//2:y+patch_size//2+1]
                    best_offset = offsets[x, y]
                    best_ssd = np.inf
                    # Compare the current patch with its neighbors' best matching patches
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if nx >= 0 and nx < height and ny >= 0 and ny < width and mask[nx, ny] == 1:
                            offset = offsets[nx, ny]
                            match = image[nx+offset[0]-patch_size//2:nx+offset[0]+patch_size //
                                          2+1, ny+offset[1]-patch_size//2:ny+offset[1]+patch_size//2+1]
                            if match.shape == patch.shape:
                                ssd = np.sum((patch - match)**2)
                                if ssd < best_ssd:
                                    best_ssd = ssd
                                    best_offset = offset
                    offsets[x, y] = best_offset

    # Step 3: Random Search
    for x in range(height):
        for y in range(width):
            if mask[x, y] == 1:
                patch = image[x-patch_size//2:x+patch_size //
                              2+1, y-patch_size//2:y+patch_size//2+1]
                best_offset = offsets[x, y]
                best_ssd = np.inf
                # Test random offsets in a window around the current best offset
                for _ in range(iterations):
                    window_size = max(height, width) // (2**_)
                    dx, dy = np.random.randint(-window_size,
                                               window_size, size=2)
                    nx, ny = x + best_offset[0] + dx, y + best_offset[1] + dy
                    match = image[nx-patch_size//2:nx+patch_size //
                                  2+1, ny-patch_size//2:ny+patch_size//2+1]
                    if match.shape == patch.shape:
                        ssd = np.sum((patch - match)**2)
                        if ssd < best_ssd:
                            best_ssd = ssd
                            best_offset = [dx, dy]
                offsets[x, y] = best_offset

        return offsets


if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = 'Images/'
    outputDir = 'Results/'

    # prompt user for source, target and mask
    source = input("Enter source image: ")
    outputname = input("Enter output image name: ")

    # Read data and clean mask
    source, maskOriginal = Read(inputDir, source)

    # Cleaning up the mask (creating a binary mask)
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0

    plt.imshow(mask)
    plt.show()

    plt.imsave('newMask.png', mask, cmap='gray')

    print(mask)

    # Run patch match
    offsets = patch_match(source, mask, patch_size=5, iterations=5)

    # Print or visualize the offsets.
    print(offsets)

    # One way to visualize the offsets is to compute the magnitude and direction of the offsets and display them as a color image.
    # Compute the magnitude of the offsets
    magnitudes = np.sqrt(np.sum(offsets**2, axis=-1))
    # Normalize the magnitudes to the range [0, 1]
    magnitudes = (magnitudes - np.min(magnitudes)) / \
        (np.max(magnitudes) - np.min(magnitudes))
    # Compute the direction of the offsets
    directions = np.arctan2(offsets[:, :, 1], offsets[:, :, 0])
    # Normalize the directions to the range [0, 1]
    directions = (directions + np.pi) / (2 * np.pi)
    # Create a color image where hue = direction, saturation = 1, value = magnitude
    hsv = np.stack([directions, np.ones_like(directions), magnitudes], axis=-1)
    # Convert the HSV image to RGB
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    # Display the RGB image
    plt.imshow(rgb)
    plt.show()
