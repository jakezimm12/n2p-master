# We import all our dependencies.
from venv import create

from pandas import test
from n2v.internals.N2P_ImageFormatter import N2P_ImageFormatter
from n2v.models import N2VConfig, N2V
from n2v.models.n2p_config import N2PConfig
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile
import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# create a folder for our data
if not os.path.isdir('./data'):
    os.mkdir('data')

X = np.load('data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25.npy')
X_val = np.load('data/BSD68_reproducibility_data/val/DCNN400_validation_gaussian25.npy')

def create_weight_channel(data, input_axis, cast_to_float=False):
    if cast_to_float:
        for img in data:
            img = img.astype(np.float32)
    
    data = data[..., np.newaxis]
    weight_channel = np.zeros(data.shape, dtype=np.float32)
    return np.concatenate((data, weight_channel), axis = input_axis)

# Adding channel dimension and adding weight dimension
X = create_weight_channel(X, 3)
print(X.shape)
print("Shape with weight channel: ", X.shape)

X_val = create_weight_channel(X_val, 3)
print(X_val.shape)

n2v_patch_shape=(128,128)
n2v_manipulator='normal_additive'
n2v_neighborhood_radius = 5
sqrt_n_patches = 16 # The amount of subpatches within the larger patch (on one side)
train_batch_size=128
train_epochs=200
train_steps_per_epoch = 3

#Create Noise2Patch configuration
n2p_config = N2PConfig(tuple(i//sqrt_n_patches for i in n2v_patch_shape), random=False, just_color=False, color_sigma=1.5, pos_sigma=1.5)

# Change train_epochs back to 200, train_steps_per_epoch=2
config = N2VConfig(X, unet_kern_size=3, 
                   train_steps_per_epoch=train_steps_per_epoch, train_epochs=train_epochs, train_loss='mse', batch_norm=True, 
                   train_batch_size=train_batch_size, n2v_perc_pix=0.198, n2v_patch_shape=n2v_patch_shape,
                   unet_n_first = 96,
                   unet_residual = True,
                   n2v_manipulator=n2v_manipulator, n2v_neighborhood_radius=n2v_neighborhood_radius,
                   single_net_per_channel=False,
                    )
                #    n2p_config=n2p_config)

# Let's look at the parameters stored in the config-object.
vars(config)

# a name used to identify the model
model_name = 'trained_on_BSD68'
# the base directory in which our model will live
basedir = 'checkpoints'

# We are now creating our network model.
model = N2V(config, model_name, n2p_config, basedir=basedir)
model.prepare_for_training(metrics=())

# We are ready to start training now.
history = model.train(X, X_val)

# model.keras_model.save_weights('./checkpoints/trained_on_BSD68/random_128_8_5_4_200_128') # It goes large patch shape, radius, # steps, # epochs, batch size


groundtruth_data = np.load('data/BSD68_reproducibility_data/test/bsd68_groundtruth.npy', allow_pickle=True)

test_data = np.load('data/BSD68_reproducibility_data/test/bsd68_gaussian25.npy', allow_pickle=True)

# Note that we do not round or clip the noisy data to [0,255]
# If you want to enable clipping and rounding to emulate an 8 bit image format,
# uncomment the following line.
# test_data = np.round(np.clip(test_data, 0, 255.))

img_formatter = N2P_ImageFormatter(config, n2p_config, n2v_patch_shape)

def PSNR(gt, img):
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(255) - 10 * np.log10(mse)

print("manipulator: ", n2v_manipulator) 
print("neighborhood: ", n2v_neighborhood_radius)

# Weights corresponding to the smallest validation loss
# Smallest validation loss does not necessarily correspond to best performance, 
# because the loss is computed to noisy target pixels.
model.load_weights('weights_best.h5')

pred = []
psnrs = []

# Method cuts the image so that subpatches will fit evenly
def cut_to_fit(img, subpatch_shape):
    y_end = img.shape[0] - (img.shape[0] % subpatch_shape[0])
    x_end = img.shape[1] - (img.shape[1] % subpatch_shape[1])
    return img[:y_end, :x_end]

def get_means_stds(data):
  n_channel = 1 if len(data.shape) == 2 else data.shape [2]
  means, stds = [], []
  for i in range(n_channel):
      means.append(np.mean(data[...,i]))
      stds.append(np.std(data[...,i]))
  return means, stds

def __normalize__(data, means, stds):
        return (data - means) / stds

def __denormalize__(data, means, stds):
    return (data * stds) + means

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss']);

subpatch_shape = (n2v_patch_shape[0] // sqrt_n_patches, n2v_patch_shape[1] // sqrt_n_patches)
for gt, img in zip(groundtruth_data, test_data):
    img = cut_to_fit(img, subpatch_shape)
    gt = cut_to_fit(gt, subpatch_shape)
    
    if n2p_config.random:
        means, stds = get_means_stds(img)
        img = __normalize__(img, means, stds)
        patches = img_formatter.format_img(img)
        p_ = np.array([model.keras_model.predict_on_batch(patches[i:i+1, ...])[0] for i in range(len(patches))])
        p_ = img_formatter.create_image_from_patches(p_, img.shape)
        p_ = __denormalize__(p_, means, stds)
        pred.append(p_)
        psnrs.append(PSNR(gt, p_))
    else:
        if sqrt_n_patches % 2 == 1: # Assimes square patches
            pixels_off_left_and_top, pixels_off_right_and_bottom = subpatch_shape[0]*(sqrt_n_patches // 2)
        else:
            pixels_off_left_and_top = subpatch_shape[0]*(sqrt_n_patches // 2)
            pixels_off_right_and_bottom = subpatch_shape[0]*((sqrt_n_patches - 1) // 2)
        means, stds = get_means_stds(img[pixels_off_left_and_top:img.shape[0] - pixels_off_right_and_bottom, pixels_off_left_and_top:img.shape[1] - pixels_off_right_and_bottom]) 
        img = __normalize__(img, means, stds)
        patches = img_formatter.format_img(img)
        p_ = np.array([model.keras_model.predict_on_batch(patches[i:i+1, ...])[0] for i in range(len(patches))])
        p_ = img_formatter.create_image_from_patches(p_, (img.shape[0] - pixels_off_left_and_top - pixels_off_right_and_bottom, img.shape[1] - pixels_off_left_and_top - pixels_off_right_and_bottom))
        p_ = __denormalize__(p_, means, stds)
        pred.append(p_)
        gt = gt[pixels_off_left_and_top:img.shape[0] - pixels_off_right_and_bottom, pixels_off_left_and_top:img.shape[1] - pixels_off_right_and_bottom]
        psnrs.append(PSNR(gt, p_))

plt.imshow(test_data[0], cmap='gray')
plt.title('Noisy Image')
plt.show()
plt.imshow(pred[0], cmap='gray')
plt.title('Prediction')
plt.show()
plt.imshow(groundtruth_data[0], cmap='gray')
plt.title('Ground Truth')
plt.show()

psnrs = np.array(psnrs)

print("PSNR (without test-time augmentation):", np.round(np.mean(psnrs), 2))