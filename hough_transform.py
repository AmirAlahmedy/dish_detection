from commonfunctions import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw

# read the image
img = cv2.imread('test.jpeg', 0)

# Apply histogram equalization
img = cv2.equalizeHist(img)

# apply Canny filter
edges = cv2.Canny(img, 100, 200)

rows, columns = img.shape
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=210, maxRadius=250)
if circles is not None:
    circles = np.uint16(np.around(circles))
    j = -1
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(img, center, radius, (255, 0, 255), 3)

        npImage = np.array(img)
        h, w = img.shape

        # Create same size alpha layer with circle
        alpha = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(alpha)
        draw.pieslice([i[0] - radius - 50, i[1] - radius - 50, i[0] + radius + 50, i[1] + radius + 50], 0, 360, fill=255)

        # Convert alpha Image to numpy array
        npAlpha = np.array(alpha)

        # Add alpha layer to RGB
        npImage = np.dstack((npImage, npAlpha))

        # Save with
        j += 1
        Image.fromarray(npImage).save('result' + str(j) + '.png')

# img[center[0]:center[0] + radius - 50, center[1]:center[1] + radius - 50] = 1
cv2.imshow("detected circles", img)
cv2.waitKey(0)

show_images([img, edges], ['Original', 'Edge Image'])
