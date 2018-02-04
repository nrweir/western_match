import numpy as np
import cv2

def normalize_image(image_arr):
    # normalize image for processing
    if len(image_arr.shape) == 3:
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)  # to grayscale
    image_arr = image_arr-np.amin(image_arr)  # set minimum value to 0
    image_arr = image_arr.astype('float32')/np.amax(image_arr)
    image_arr = 255*image_arr  # re-scale such that max value is 255
    image_arr = image_arr.astype('uint8')  # set dtype
    return image_arr


def hash_features(feature_des, kmeans, n_clusters):
    feature_hash = np.zeros(n_clusters)
    if feature_des is not None:
        if len(feature_des.shape) == 1:
            feature_des = feature_des[np.newaxis, :]
        feature_pred = kmeans.predict(feature_des)
        for j in feature_pred:
            feature_hash[j] += 1
    return feature_hash


def compare_hashes(test_hash, hash_lib):
    test_hash = test_hash > 0
    hash_lib = hash_lib > 0
    similarity = []
    for i in range(0, hash_lib.shape[0]):
        similarity.append(
            np.sum(
                np.logical_and(
                    test_hash, hash_lib[i, :]))/np.sum(
                        np.logical_or(test_hash, hash_lib[i, :])))
    return similarity
