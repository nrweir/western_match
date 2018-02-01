"""Run models implemented in other scripts."""

from nets import CSAE, get_standard_callbacks
from df_generator import df_generator
from sklearn import model_selection
import os
import pickle
import pandas as pd
import numpy as np

# VARIABLES TO FUTZ WITH #
# GENERATOR VARIABLES #
batch_size = 32
input_shape = (150, 150, 1)
h_resize = None
v_resize = None
v_flip = False
h_flip = False
rotate = False
contrast = None
# END GEN VARIABLES #

# MODEL VARIABLES #
conv_depth = 256
conv_shape = (3, 3)
pool_shape = (3, 3)
stride = None
conv_reg = 0
act_reg = 0
optimizer = "Adam"
lrs = np.power(10, np.arange(-6, -1))
# END MODEL VARIABLES #

# TRAINING VARIABLES #
epochs = 25
cb_path = '/n/denic_lab/Users/nweir/python_packages/western_match/outputs/csae_lr_opt_256_adam_'
es = True
es_patience = 10
sbo = True
# END TRAINING VARIABLES #
# END VARIABLES TO FUTZ WITH #

os.chdir('/n/denic_lab/Users/nweir/python_packages/western_match/data')
pkls = [f for f in os.listdir() if '.pkl' in f]
src_ims = pd.DataFrame(columns=['path', 'filename', 'image'])
for p in pkls:
    with open(p, 'rb') as p_file:
        curr_pkl = pickle.load(p_file)
    src_ims = src_ims.append(curr_pkl)

train, val = model_selection.train_test_split(src_ims, test_size=0.2)
output_df = pd.DataFrame(columns=['model_id', 'conv_reg', 'act_reg',
                                  'history'])
histories = []
for lr in lrs:
    current_CSAE = CSAE(input_shape=input_shape, conv_depth=conv_depth,
                        conv_shape=conv_shape, pool_shape=pool_shape,
                        stride=stride, conv_reg=conv_reg,
                        act_reg=act_reg, optimizer=optimizer, lr=lr)
    current_CSAE.summary()

    training_data = df_generator(df=train, batch_size=batch_size,
                                 patch_shape=input_shape[0:2],
                                 h_resize=h_resize, v_resize=v_resize,
                                 v_flip=v_flip, h_flip=h_flip, rotate=rotate,
                                 contrast=contrast)
    val_data = df_generator(df=val, batch_size=batch_size,
                            patch_shape=input_shape[0:2], h_resize=h_resize,
                            v_resize=v_resize, v_flip=v_flip, h_flip=h_flip,
                            rotate=rotate, contrast=contrast)
    fit_model = current_CSAE.fit_generator(
        generator=training_data,
        steps_per_epoch=int(len(train['image'])/batch_size), epochs=epochs,
        callbacks=get_standard_callbacks(
                path=cb_path+str(lr), es=es, es_patience=es_patience,
                sbo=sbo
                ), verbose=2, validation_data=val_data, validation_steps=int(len(val.index)/batch_size))
    histories.append(fit_model.history)
output_df = pd.DataFrame({'model_number': list(range(0, lrs.size)),
                          'lr': lrs, 'history': histories})
output_df.to_pickle('/n/denic_lab/Users/nweir/python_packages/western_match/outputs/conv_256_lr_opt.pkl')
