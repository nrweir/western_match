"""Functions and classes for generating CSAE nets."""

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import ActivityRegularization, Conv2DTranspose
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


def CSAE(input_shape=(150, 150, 1), conv_depth=32, conv_shape=(3, 3),
         pool_shape=(3, 3), stride=None, conv_reg=0, act_reg=0,
         optimizer='SGD', lr=0.0001):
    """Create and compile a Convolutional Sparse Autoencoder model with Keras.

    Arguments:
    ---------
    input_shape : 3-tuple of ints (y, x, c), optional
        The shape of the input patches fed into the autoencoder. Defaults to
        150x150x1.
    conv_depth : int, optional
        The number of convolutional filters used in encoding and decoding
        (in the case of decoding, these are transposed conv filters). Defaults
        to 32.
    conv_shape : 2-tuple of ints, optional
        The shape of the convolutional filters used in encoding and decoding.
        Defaults to 3x3.
    pool_shape : 2-tuple of ints, optional
        The shape of the max pooling and upsampling filters used. Defaults to
        3x3.
    stride : None or 2-tuple of ints, optional
        The step size of the max pooling filter. Defaults to `pool_shape` when
        not provided.
    conv_reg : float, optional
        L2 regularization for the convolution weights. Only applied during
        encoding, not decoding. Defaults to 0.
    act_reg : float, optional
        Sparsifying activation regularization applied to the encoding product.
        Defaults to 0.
    optimizer : 'SGD' or 'Adam', optional
        Gradient descent algorithm to apply. Defaults to 'SGD'.
    lr : float, optional
        Learning rate for gradient descent. Defaults to 0.0001.

    Returns:
    -------
    A compiled model object with the architecture shown below:

          Input
            |
            |
    Convolutional layer  - `conv_depth` # of filters, `conv_reg` L2 reg on W
            |               `conv_shape` filter shape
            |
     Max pooling layer  -   2D max pooling layer with pooling filters of shape
            |               `pool_shape` and strides `stride`
            |
     Encoding product - L2 activity regularization of strength `act_reg` is
            |           applied here to maintain sparse activation
            |
      Upsampling layer - 2D upsampling with filter shape `pool_shape` to
            |            restore image dimensions
            |
     Deconvolving layer - Transposed convolution with filters of shape
            |             `conv_shape` to restore to original channel depth
            |
          Output - compared to input using mean squared error in loss function

    """
    input_img = Input(shape=input_shape)  # input layer
    enc_conv_layer = Conv2D(conv_depth, conv_shape, use_bias=True,
                            activation='relu', padding='same',
                            kernel_regularizer=l2(conv_reg))  # conv w l2 reg
    enc_conv = enc_conv_layer(input_img)
    encoded = MaxPooling2D(pool_shape, strides=stride)(enc_conv)  # enc output
    enc_reg = ActivityRegularization(l2=act_reg)(encoded)  # sparsifying reg
    dec_unpooled = UpSampling2D(size=pool_shape)(enc_reg)  # de-pooling
    dec_conv = Conv2DTranspose(1, conv_shape, use_bias=True,  # deconvolution
                               activation='relu', padding='same')
    output = dec_conv(dec_unpooled)  # output

    model = Model(input=input_img, outputs=output)
    if optimizer == 'Adam':
        myopt = Adam(lr=lr)
    elif optimizer == 'SGD':
        myopt = SGD(lr=lr)

    model.compile(optimizer=myopt, loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


def get_standard_callbacks(path, es=False, es_patience=10, sbo=True):
    """Prepare callbacks for training.

    Arguments:
    ---------
    path : str
        partial path to save models to. the epoch number and validation loss
        are appended to this string, along with the .hdf5 extension.
    es : bool, optional
        If True, an `EarlyStopping` callback is implemented with patience
        `es_patience`. Defaults to False.
    es_patience : int, optional
        If `es` is True, provides the number of epochs to wait before stopping
        early if val_loss does not improve. Defaults to 10.
    sbo : bool, optional
        If True, model will only be saved after training epochs if the val_loss
        has decreased. Defaults to True.

    Returns:
    --------
    A list containing a `ModelCheckpoint` callback, and an `EarlyStopping`
    callback if `es`==True.

    """
    model_checkpoint = ModelCheckpoint(
        filepath=path+".weights.{epoch:02d}-{val_loss:.3f}.hdf5",
        monitor='val_loss', save_best_only=sbo)

    early_stopping = EarlyStopping(monitor='val_loss', patience=es_patience)
    checkpoints = [model_checkpoint]
    if es:
        checkpoints.append(early_stopping)

    return checkpoints
