#!/usr/bin/env python3

import sys
import math

import numpy as np
import cv2 as cv
import scipy.spatial

debug = True

input_image_path = sys.argv[1]
output_image_path = sys.argv[2]

image = cv.imread(input_image_path)
print(f'Input image size: {image.shape}')
image_diagonal = math.sqrt(image.shape[0]**2 + image.shape[1]**2)

# get rid of bleeding pixel artifacts due to image scaling
morph_image = cv.morphologyEx(
    image, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))

# only use outermost contours of everything not white in image
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
inverted_gray_image = cv.subtract(255, gray_image)
ret, threshold_image = cv.threshold(
    inverted_gray_image, 1, 255, cv.THRESH_BINARY)
#ret, threshold_image = cv.threshold(inverted_gray_image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(
    threshold_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
if debug:
    cv.imwrite(output_image_path + '-morph.png', morph_image)
    cv.imwrite(output_image_path + '-threshold.png', threshold_image)
    debug_image = image.copy()
    cv.drawContours(debug_image, contours, -1, (0, 0, 0), 3)
    cv.imwrite(output_image_path + '-contours.png', debug_image)
print(f'#contours: {len(contours)}')

contour_pixel_pos = np.array(
    [item for sublist in contours for item in sublist]).reshape(-1, 2)
num_contour_pixels = contour_pixel_pos.shape[1]
contour_pixel_pos[:, [0, 1]] = contour_pixel_pos[:, [1, 0]]  # fix indexing
contour_pixel_val = np.array(
    morph_image[contour_pixel_pos[:, 0], contour_pixel_pos[:, 1]], dtype=float)
squared_contour_pixel_val = contour_pixel_val**2

result_image = np.zeros(image.shape, dtype=image.dtype)
for pixel_pos in np.ndindex(image.shape[:2]):
    # for pixel_pos in [np.array([0, 0])]:
    distances = scipy.spatial.distance.cdist(
        contour_pixel_pos, np.array([pixel_pos]), 'sqeuclidean')
    if 0 in distances:  # skip contour pixels
        result_image[pixel_pos] = morph_image[pixel_pos]
        continue
    max_distance = np.percentile(distances, 1)
    weights = np.reciprocal(distances)
    weights[weights < 1/max_distance] = 0
    new_pixel_val = np.sqrt(np.average(
        squared_contour_pixel_val, axis=0, weights=weights.ravel()))
    # print(f'{pixel_pos}: {new_pixel_val}')
    result_image[pixel_pos] = new_pixel_val

result_image = cv.medianBlur(result_image, 7)

cv.imwrite(output_image_path + '-background_only.png', result_image)
ret, copy_mask = cv.threshold(inverted_gray_image, 1, 255, cv.THRESH_BINARY)
np.copyto(result_image, image,
          where=copy_mask[:, :, None].astype(bool))
cv.imwrite(output_image_path, result_image)
