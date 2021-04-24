from commonfunctions import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw

# read the image
cimg = cv2.imread('Database.jpg')  # colored

img = cv2.imread('Database.jpg', 0)  # grayscale

# Apply histogram equalization
img = cv2.equalizeHist(img)

# apply Canny filter
edges = cv2.Canny(img, 100, 200)

rows, columns = img.shape
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=40, maxRadius=70)
if circles is not None:
    circles = np.uint16(np.around(circles))
    j = -1
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(cimg, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(cimg, center, radius, (255, 0, 255), 3)

        npImage = np.array(cimg)
        h, w = img.shape

        # Create same size alpha layer with circle
        alpha = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(alpha)
        draw.pieslice([i[0] - 150, i[1] - 150, i[0] + 150, i[1] + 150], 0, 360,
                      fill=255)

        # Convert alpha Image to numpy array
        npAlpha = np.array(alpha)

        # Add alpha layer to RGB
        npImage = np.dstack((npImage, npAlpha))

        # Save with
        j += 1
        Image.fromarray(npImage).save('result' + str(j) + '.png')


cv2.imshow("detected circles", cimg)
cv2.waitKey(0)

show_images([img, edges], ['Original', 'Edge Image'])
