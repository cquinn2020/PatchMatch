import numpy as np
import cv2 as cv


img = cv.imread('Images/cat_in_grass.jpg')
mask = cv.imread('Images/newMask.png', cv.IMREAD_GRAYSCALE)
dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()
