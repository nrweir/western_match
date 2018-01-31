"""Generators for getting data."""


import numpy as np
import cv2
import random
from sklearn.feature_extraction.image import extract_patches_2d


def df_generator(df, batch_size=32, patch_shape=(100, 100), h_resize=None,
                 v_resize=None, v_flip=False, h_flip=False, rotate=False,
                 contrast=None):
    """df-based version of Keras ImageDataGenerator.

    Arguments:
    ---------
    df : pandas DataFrame
        A DataFrame containing image data which contains the following three
        columns: `path` (strings of full paths to files), `filename` (filenames
        for images, and `image` (numpy arrays of image data).
    batch_size : int, optional
        The number of images to be returned by the generator at each call.
    """
    df_length = len(df.index)
    while True:
        im_batch = np.empty(shape=(batch_size, patch_shape[0], patch_shape[1]))
        im_paths = []
        if v_flip:
            v_flips = np.random.randint(0, 1, batch_size).astype(bool)
        if h_flip:
            h_flips = np.random.randint(0, 1, batch_size).astype(bool)
        if h_resize is not None:
            h_resizes = np.random.rand(batch_size)*(h_resize[1]-h_resize[0])+h_resize[0]
        if v_resize is not None:
            v_resizes = np.random.rand(batch_size)*(v_resize[1]-v_resize[0])+v_resize[0]
        if rotate:
            rotations = np.random.randint(0, 4, batch_size)
        if contrast is not None:
            contrast_minima = np.random.rand(batch_size)*(contrast[0][1]-contrast[0][0])+contrast[0][0]
            contrast_maxima = np.random.rand(batch_size)*(contrast[1][1]-contrast[1][0])+contrast[1][0]
        inds = list(range(0, df_length))
        random.shuffle(inds)
        for i in range(0, batch_size):
            im_row = df.iloc[inds[i]]
            im_paths.append(im_row['path'])
            im = im_row['image']
            if len(im.shape) == 3:  # if it's an RGB image
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if rotate:
                im = np.rot90(im, k=rotations[i])
            if h_resize is not None:
                out_shape_x = int(h_resizes[i]*im.shape[1])
            else:
                out_shape_x = im.shape[1]
            if v_resize is not None:
                out_shape_y = int(v_resizes[i]*im.shape[0])
            else:
                out_shape_y = im.shape[0]
            if v_resize is not None or h_resize is not None:
                if out_shape_y*out_shape_x > im.shape[0]*im.shape[1]:
                    interp = cv2.INTER_CUBIC
                else:
                    interp = cv2.INTER_AREA
                im = cv2.resize(im, dsize=(out_shape_y, out_shape_x),
                                interpolation=interp)
            if v_flip:
                if v_flips[i]:
                    im = np.flipud(im)
            if h_flip:
                if h_flips[i]:
                    im = np.fliplr(im)
            # perform initial rescaling to put the image on a (0,255) range
            im = im-np.amin(im)
            im_max = np.amax(im)
            scaling_factor = 255/im_max
            im = im*scaling_factor
            if contrast:
                im = im*(contrast_maxima[i]-contrast_minima[i]) + int(contrast_minima[i]*255)
            im[np.where(im > 255)] = 255
            im = im.astype('uint8')
            # pad image if it's not big enough for the patch
            x_pad = 0
            y_pad = 0
            if im.shape[0] < patch_shape[0]:
                y_pad = int((patch_shape[0]-im.shape[0])/2)+1
            if im.shape[1] < patch_shape[1]:
                x_pad = int((patch_shape[1]-im.shape[1])/2)+1
            im = np.pad(im, pad_width=((y_pad, y_pad), (x_pad, x_pad)),
                        mode='constant', constant_values=0)
            im = extract_patches_2d(im, patch_size=patch_shape,
                                    max_patches=1)[0, :, :]
            im_batch[i, :, :] = im
        im_batch = im_batch[:, :, :, np.newaxis]
        yield im_batch, im_batch
