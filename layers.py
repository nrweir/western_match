"""De-pooling layer modified from github.com/nanopony/keras-convautoencoder."""

#### WORK IN PROGRESS!!!! NOT READY!!!! ####

from keras import backend as K
from keras.engine.topology import Layer
from theano import tensor as T


class DePool2D(Layer):
    """Similar to UpSample, yet traverse only maxpooled elements
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.
    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    """
    input_ndim = 4

    def __init__(self, output_dim, *args, **kwargs):
        self.output_dim = output_dim
        super(DePool2D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(DePool2D, self).build(input_shape)

    def call(self, x):
        if self.dim_ordering == 'th':
            output = K.repeat_elements(x, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = K.repeat_elements(x, self.size[0], axis=1)
            output = K.repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        f = T.grad(T.sum(self._pool2d_layer.call(train)),
                   wrt=self._pool2d_layer.get_input(train)) * output

        return f


def max_filter(T, pool_shape, stride, pad_size=(0, 0)):
    """Max filter over T, pooling by pool_shape and stepping by stride.

    """

    if (K.int_shape(T) - pool_shape[0] + 2*pad_size[0]) % stride[0] != 0:
        raise Exception('Invalid dimensions: input axis 0')
    if (K.int_shape(T) - pool_shape[1] + 2*pad_size[1]) % stride[1] != 0:
        raise Exception('Invalid dimensions: input axis 1')

    T = K.spatial_2d_padding(T, padding=((pad_size[0], pad_size[0]),
                                         (pad_size[1], pad_size[1])))
    output = K.zeros(shape=K.int_shape(T))
    for y_step in range(0, T.shape[1], stride[0]):
        for x_step in range(0, T.shape[2], stride[1]):
            sub_region = T[:, y_step:y_step+pool_shape[0],
                           x_step+pool_shape[1], :]
            max_y = K.argmax(sub_region, axis=1)
            max_x = K.argmax(sub_region, axis=2)
