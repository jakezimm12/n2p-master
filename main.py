# We import all our dependencies.
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

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# create a folder for our data
if not os.path.isdir('./data'):
    os.mkdir('data')

# INSTEAD J MANUALLY WENT TO SITE AND DOWNLOADED ZIP
# check if data has been downloaded already
# zipPath="data/BSD68_reproducibility.zip"
# if not os.path.exists(zipPath):
#     #download and unzip data
#     data = urllib.request.urlretrieve('https://download.fht.org/jug/n2v/BSD68_reproducibility.zip', zipPath)
#     with zipfile.ZipFile(zipPath, 'r') as zip_ref:
#         zip_ref.extractall("data")

# ADDED THIS BLOCK
# zipPath="data/BSD68_reproducibility.zip"
# with zipfile.ZipFile(zipPath, 'r') as zip_ref:
#         zip_ref.extractall("data")

X = np.load('data/BSD68_reproducibility_data/train/DCNN400_train_gaussian25.npy')
X_val = np.load('data/BSD68_reproducibility_data/val/DCNN400_validation_gaussian25.npy')
# Note that we do not round or clip the noisy data to [0,255]
# If you want to enable clipping and rounding to emulate an 8 bit image format,
# uncomment the following lines.
# X = np.round(np.clip(X, 0, 255.))
# X_val = np.round(np.clip(X_val, 0, 255.))

# Adding channel dimension
X = X[..., np.newaxis]
print(X.shape)
X_val = X_val[..., np.newaxis]
print(X_val.shape)

# Let's look at one of our training and validation patches.
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.imshow(X[0,...,0], cmap='gray')
plt.title('Training Patch');
plt.subplot(1,2,2)
plt.imshow(X_val[0,...,0], cmap='gray')
plt.title('Validation Patch');

n2v_patch_shape=(180, 180)

config = N2VConfig(X, unet_kern_size=3, 
                   train_steps_per_epoch=2, train_epochs=200, train_loss='mse', batch_norm=True, 
                   train_batch_size=128, n2v_perc_pix=0.198, n2v_patch_shape=n2v_patch_shape,
                   unet_n_first = 96,
                   unet_residual = True,
                   n2v_manipulator='normal_additive', n2v_neighborhood_radius=5,
                   single_net_per_channel=False)

# Let's look at the parameters stored in the config-object.
vars(config)

# a name used to identify the model
model_name = 'BSD68_reproducability_5x5'
# the base directory in which our model will live
basedir = 'models'

#Create Noise2Patch configuration
n2p_config = N2PConfig(tuple(i//9 for i in n2v_patch_shape))

# We are now creating our network model.
model = N2V(config, model_name, n2p_config, basedir=basedir)
model.prepare_for_training(metrics=())

# We are ready to start training now.
history = model.train(X, X_val)

# print(sorted(list(history.history.keys())))
# plt.figure(figsize=(16,5))
# plot_history(history,['loss','val_loss']);

groundtruth_data = np.load('data/BSD68_reproducibility_data/test/bsd68_groundtruth.npy', allow_pickle=True)

test_data = np.load('data/BSD68_reproducibility_data/test/bsd68_gaussian25.npy', allow_pickle=True)
# Note that we do not round or clip the noisy data to [0,255]
# If you want to enable clipping and rounding to emulate an 8 bit image format,
# uncomment the following line.
# test_data = np.round(np.clip(test_data, 0, 255.))

def PSNR(gt, img):
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(255) - 10 * np.log10(mse)

# Weights corresponding to the smallest validation loss
# Smallest validation loss does not necessarily correspond to best performance, 
# because the loss is computed to noisy target pixels.
model.load_weights('weights_best.h5')

pred = []
psnrs = []
for gt, img in zip(groundtruth_data, test_data):
    p_ = model.predict(img.astype(np.float32), 'YX', tta=False);
    pred.append(p_)
    psnrs.append(PSNR(gt, p_))

psnrs = np.array(psnrs)

print("PSNR (without test-time augmentation):", np.round(np.mean(psnrs), 2))

pred = []
psnrs = []
for gt, img in zip(groundtruth_data, test_data):
    p_ = model.predict(img.astype(np.float32), 'YX', tta=True);
    pred.append(p_)
    psnrs.append(PSNR(gt, p_))

psnrs = np.array(psnrs)

print("PSNR (with test-time augmentation):", np.round(np.mean(psnrs), 2))

# The weights of the converged network. 
model.load_weights('weights_last.h5')

pred = []
psnrs = []
for gt, img in zip(groundtruth_data, test_data):
    p_ = model.predict(img.astype(np.float32), 'YX', tta=False)
    pred.append(p_)
    psnrs.append(PSNR(gt, p_))

psnrs = np.array(psnrs)

print("PSNR (without test-time augmentation):", np.round(np.mean(psnrs), 2))

pred = []
psnrs = []
for gt, img in zip(groundtruth_data, test_data):
    p_ = model.predict(img.astype(np.float32), 'YX', tta=True)
    pred.append(p_)
    psnrs.append(PSNR(gt, p_))

psnrs = np.array(psnrs)

print("PSNR (with test-time augmentation):", np.round(np.mean(psnrs), 2))