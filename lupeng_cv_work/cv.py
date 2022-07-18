import cv2
import numpy as np
import cv2
img = cv2.imread('img_set/miku.jpg')

# Prewitt算子
prewittx = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

prewitty = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])

dst1 = cv2.filter2D(img, -1, prewittx)
dst2 = cv2.filter2D(img, -1, prewitty)

cv2.imshow('original', img)
cv2.imshow('convx', dst1)
cv2.imshow('convy', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()