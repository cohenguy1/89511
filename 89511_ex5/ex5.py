# Guy Cohen 304840283
from PIL import Image
import numpy as np
import sys


def get_average_index(i, pixels):
    sum_index = 0.0
    for pixel in pixels:
        sum_index += pixel[i]
    return sum_index / len(pixels)


initial_centroids_path = sys.argv[1]
image_path = sys.argv[2]

im = Image.open(image_path)
X = np.array(im)

initial_centroids = np.loadtxt(initial_centroids_path)
k = len(initial_centroids)

num_of_iterations = 10

pixel_dim = 3

current_centroids = initial_centroids

for iter in range(num_of_iterations):
    # map of <centroid>, list<pixels>
    pixels_assignments = []
    for centroid_index in range(k):
        pixels_assignments.append([])

    for row in range(len(X)):
        for column in range(len(X[row])):
            pixel = X[row][column]
            new_centroid_index = np.argmin([np.linalg.norm(pixel - c) for c in current_centroids])
            pixels_assignments[new_centroid_index].append(pixel)

    # calculate new current_centroids
    for centroid_index in range(k):
        # only calculate if there are pixels assigned to it
        if len(pixels_assignments[centroid_index]) > 0:
            for j in range(pixel_dim):
                current_centroids[centroid_index][j] = get_average_index(j, pixels_assignments[centroid_index])

final_centroids = current_centroids
for centroid_index in range(k):
    final_centroids[centroid_index] = final_centroids[centroid_index].astype(int)

for centroid_index in range(k):
    r = int(final_centroids[centroid_index][0])
    g = int(final_centroids[centroid_index][1])
    b = int(final_centroids[centroid_index][2])
    print(str(r) + ' ' + str(g) + ' ' + str(b))

image_with_kmeans = X.copy()

for row in range(len(image_with_kmeans)):
    for column in range(len(image_with_kmeans[row])):
        pixel = image_with_kmeans[row][column]
        new_centroid_index = np.argmin([np.linalg.norm(pixel - c) for c in final_centroids])
        image_with_kmeans[row][column] = final_centroids[new_centroid_index]

image_with_kmeans = Image.fromarray(image_with_kmeans)

suffix_index = image_path.rfind('.')
new_image_path = image_path[:suffix_index] + '_comp' + image_path[suffix_index:]

image_with_kmeans.save(new_image_path)