# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division

import warnings
import numpy as np
from six import string_types

from .config import BaseConfig
from .base_model import BaseModel, suppress_without_basedir

from ..utils import _raise, axes_check_and_normalize, axes_dict, move_image_axes, backend_channels_last
from ..utils.six import Path
from ..utils.tf import export_SavedModel, IS_TF_1, keras_import, CARETensorBoardImage
from ..data import Normalizer, NoNormalizer, PercentileNormalizer
from ..data import Resizer, NoResizer, PadAndCropResizer
from ..internals.predict import predict_tiled, tile_overlap, Progress, total_n_tiles
from ..internals import nets, train

from distutils.version import LooseVersion
keras = keras_import()
TerminateOnNaN, ReduceLROnPlateau = keras_import('callbacks', 'TerminateOnNaN', 'ReduceLROnPlateau')
K = keras_import('backend')
Adam = keras_import('optimizers', 'Adam')

import tensorflow as tf



class GenericUnetConfig(BaseConfig):

    def __init__(self, axes='YX', n_channel_in=1, n_channel_out=1, allow_new_parameters=True, **kwargs):
        super(GenericUnetConfig, self).__init__(axes, n_channel_in, n_channel_out)

        # default config (can be overwritten by kwargs below)
        self.unet_n_depth          = 3
        self.unet_kernel_size      = self.n_dim*(3,)
        self.unet_n_filter_base    = 32
        self.unet_n_conv_per_depth = 2
        self.unet_pool_size        = self.n_dim*(2,)
        self.unet_activation       = 'relu'
        self.unet_last_activation  = 'relu'
        self.unet_batch_norm       = False
        self.unet_dropout          = 0.0
        self.unet_residual         = False
        if backend_channels_last():
            self.unet_input_shape  = self.n_dim*(None,) + (self.n_channel_in,)
        else:
            self.unet_input_shape  = (self.n_channel_in,) + self.n_dim*(None,)

        self.train_loss            = 'mse'
        self.train_epochs          = 100
        self.train_steps_per_epoch = 400
        self.train_learning_rate   = 0.0004
        self.train_batch_size      = 8
        self.train_tensorboard     = True

        # the parameter 'min_delta' was called 'epsilon' for keras<=2.1.5
        min_delta_key = 'epsilon' if LooseVersion(keras.__version__)<=LooseVersion('2.1.5') else 'min_delta'
        self.train_reduce_lr       = {'factor': 0.5, 'patience': 10, min_delta_key: 0}

        # disallow setting 'n_dim' manually
        try:
            del kwargs['n_dim']
            # warnings.warn("ignoring parameter 'n_dim'")
        except:
            pass

        self.update_parameters(allow_new_parameters, **kwargs)



class GenericUnet(BaseModel):

    def __init__(self, config, name=None, basedir='.'):
        super(GenericUnet, self).__init__(config=config, name=name, basedir=basedir)


    def _build(self):
        return nets.custom_unet (
            input_shape         = self.config.unet_input_shape,
            last_activation     = self.config.unet_last_activation,
            n_depth             = self.config.unet_n_depth,
            n_filter_base       = self.config.unet_n_filter_base,
            kernel_size         = self.config.unet_kernel_size,
            n_conv_per_depth    = self.config.unet_n_conv_per_depth,
            activation          = self.config.unet_activation,
            batch_norm          = self.config.unet_batch_norm,
            dropout             = self.config.unet_dropout,
            pool_size           = self.config.unet_pool_size,
            n_channel_out       = self.config.n_channel_out,
            residual            = self.config.unet_residual,
        )


    def prepare_for_training(self, optimizer=None, callbacks=None, metrics=None):
        if optimizer is None:
            optimizer = Adam(lr=self.config.train_learning_rate)

        if metrics is None:
            metrics = []

        self.keras_model.compile(optimizer=optimizer, loss=self.config.train_loss, metrics=metrics)

        self.callbacks = [TerminateOnNaN()] if callbacks is None else callbacks

        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                if IS_TF_1:
                    from ..utils.tf import CARETensorBoard
                    self.callbacks.append(CARETensorBoard(log_dir=str(self.logdir), prefix_with_timestamp=False, n_images=3, write_images=True, prob_out=False))
                else:
                    from tensorflow.keras.callbacks import TensorBoard
                    self.callbacks.append(TensorBoard(log_dir=str(self.logdir/'logs'), write_graph=False, profile_batch=0))

        if self.config.train_reduce_lr is not None:
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            # TF2: add as first callback to put 'lr' in the logs for TensorBoard
            self.callbacks.insert(0,ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def train(self, X,Y, validation_data, epochs=None, steps_per_epoch=None, augmenter=None):
        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        n_train, n_val = len(X), len(validation_data[0])
        frac_val = (1.0 * n_val) / (n_train + n_val)
        frac_warn = 0.05
        if frac_val < frac_warn:
            warnings.warn("small number of validation images (only %.1f%% of all images)" % (100*frac_val))
        axes = axes_check_and_normalize('S'+self.config.axes,X.ndim)
        ax = axes_dict(axes)

        for a,div_by in zip(axes,self._axes_div_by(axes)):
            n = X.shape[ax[a]]
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axis %s"
                    " (which has incompatible size %d)" % (div_by,a,n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        if (self.config.train_tensorboard and self.basedir is not None and
            not IS_TF_1 and not any(isinstance(cb,CARETensorBoardImage) for cb in self.callbacks)):
            self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=validation_data,
                                                       log_dir=str(self.logdir/'logs'/'images'),
                                                       n_images=3, prob_out=False))

        training_data = train.DataWrapper(X, Y, self.config.train_batch_size, length=epochs*steps_per_epoch, augmenter=augmenter)

        fit = self.keras_model.fit_generator if IS_TF_1 else self.keras_model.fit
        history = fit(iter(training_data), validation_data=validation_data,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      callbacks=self.callbacks, verbose=1)
        self._training_finished()

        return history


    def export_TF(self, fname=None):
        if self.basedir is None and fname is None:
            raise ValueError("Need explicit 'fname', since model directory not available (basedir=None).")
        fname = (self.logdir / 'TF_SavedModel.zip') if fname is None else Path(fname)
        export_SavedModel(self.keras_model, str(fname))


    def predict(self, img, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), n_tiles=None):

        normalizer, resizer = self._check_normalizer_resizer(normalizer, resizer)

        # different kinds of axes
        # -> typical case: net_axes_in = net_axes_out, img_axes_in = img_axes_out
        img_axes_in = axes_check_and_normalize(axes,img.ndim)
        net_axes_in = self.config.axes
        net_axes_out = axes_check_and_normalize(self._axes_out)
        set(net_axes_out).issubset(set(net_axes_in)) or _raise(ValueError("different kinds of output than input axes"))
        net_axes_lost = set(net_axes_in).difference(set(net_axes_out))
        img_axes_out = ''.join(a for a in img_axes_in if a not in net_axes_lost)
        # print(' -> '.join((img_axes_in, net_axes_in, net_axes_out, img_axes_out)))
        tiling_axes = net_axes_out.replace('C','') # axes eligible for tiling

        _permute_axes = self._make_permute_axes(img_axes_in, net_axes_in, net_axes_out, img_axes_out)
        # _permute_axes: (img_axes_in -> net_axes_in), undo: (net_axes_out -> img_axes_out)
        x = _permute_axes(img)
        # x has net_axes_in semantics
        x_tiling_axis = tuple(axes_dict(net_axes_in)[a] for a in tiling_axes) # numerical axis ids for x

        channel_in = axes_dict(net_axes_in)['C']
        channel_out = axes_dict(net_axes_out)['C']
        net_axes_in_div_by = self._axes_div_by(net_axes_in)
        net_axes_in_overlaps = self._axes_tile_overlap(net_axes_in)
        self.config.n_channel_in == x.shape[channel_in] or _raise(ValueError())

        # TODO: refactor tiling stuff to make code more readable

        def _total_n_tiles(n_tiles):
            n_block_overlaps = [int(np.ceil(1.* tile_overlap / block_size)) for tile_overlap, block_size in zip(net_axes_in_overlaps, net_axes_in_div_by)]
            return total_n_tiles(x,n_tiles=n_tiles,block_sizes=net_axes_in_div_by,n_block_overlaps=n_block_overlaps,guarantee='size')

        _permute_axes_n_tiles = self._make_permute_axes(img_axes_in, net_axes_in)
        # _permute_axes_n_tiles: (img_axes_in <-> net_axes_in) to convert n_tiles between img and net axes
        def _permute_n_tiles(n,undo=False):
            # hack: move tiling axis around in the same way as the image was permuted by creating an array
            return _permute_axes_n_tiles(np.empty(n,np.bool),undo=undo).shape

        # to support old api: set scalar n_tiles value for the largest tiling axis
        if np.isscalar(n_tiles) and int(n_tiles)==n_tiles and 1<=n_tiles:
            largest_tiling_axis = [i for i in np.argsort(x.shape) if i in x_tiling_axis][-1]
            _n_tiles = [n_tiles if i==largest_tiling_axis else 1 for i in range(x.ndim)]
            n_tiles = _permute_n_tiles(_n_tiles,undo=True)
            warnings.warn("n_tiles should be a tuple with an entry for each image axis")
            print("Changing n_tiles to %s" % str(n_tiles))

        if n_tiles is None:
            n_tiles = [1]*img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % img.ndim)

        all(np.isscalar(t) and 1<=t and int(t)==t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1"))
        n_tiles = tuple(map(int,n_tiles))
        n_tiles = _permute_n_tiles(n_tiles)
        (all(n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis) or
            _raise(ValueError("entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes)))
        # n_tiles_limited = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
        # if any(np.array(n_tiles) != np.array(n_tiles_limited)):
        #     print("Limiting n_tiles to %s" % str(_permute_n_tiles(n_tiles_limited,undo=True)))
        # n_tiles = n_tiles_limited
        n_tiles = list(n_tiles)


        # normalize & resize
        x = normalizer.before(x, net_axes_in)
        x = resizer.before(x, net_axes_in, net_axes_in_div_by)

        done = False
        progress = Progress(_total_n_tiles(n_tiles),1)
        c = 0
        while not done:
            try:
                # raise tf.errors.ResourceExhaustedError(None,None,None) # tmp
                x = predict_tiled(self.keras_model,x,axes_in=net_axes_in,axes_out=net_axes_out,
                                  n_tiles=n_tiles,block_sizes=net_axes_in_div_by,tile_overlaps=net_axes_in_overlaps,pbar=progress)
                # x has net_axes_out semantics
                done = True
                progress.close()
            except tf.errors.ResourceExhaustedError:
                # TODO: how to test this code?
                # n_tiles_prev = list(n_tiles) # make a copy
                tile_sizes_approx = np.array(x.shape) / np.array(n_tiles)
                t = [i for i in np.argsort(tile_sizes_approx) if i in x_tiling_axis][-1]
                n_tiles[t] *= 2
                # n_tiles = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
                # if all(np.array(n_tiles) == np.array(n_tiles_prev)):
                    # raise MemoryError("Tile limit exceeded. Memory occupied by another process (notebook)?")
                if c >= 8:
                    raise MemoryError("Giving up increasing number of tiles. Memory occupied by another process (notebook)?")
                print('Out of memory, retrying with n_tiles = %s' % str(_permute_n_tiles(n_tiles,undo=True)))
                progress.total = _total_n_tiles(n_tiles)
                c += 1

        x.shape[channel_out] == self.config.n_channel_out or _raise(ValueError())

        x = resizer.after(x, net_axes_out)
        # x has net_axes_out semantics

        if normalizer.do_after and self.config.n_channel_in==self.config.n_channel_out:
            x = normalizer.after(x, None, net_axes_out)[0]

        x = _permute_axes(x,undo=True)
        # x has img_axes_out semantics

        return x


    def _axes_div_by(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        # default: must be divisible by power of 2 to allow down/up-sampling steps in unet
        pool_div_by = 2**self.config.unet_n_depth
        return tuple((pool_div_by if a in 'XYZT' else 1) for a in query_axes)

    def _axes_tile_overlap(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        overlap = dict(zip(
            self.config.axes.replace('C',''),
            (tile_overlap(self.config.unet_n_depth, s) for s in self.config.unet_kernel_size)
        ))
        return tuple(overlap.get(a,0) for a in query_axes)

    @property
    def _config_class(self):
        return GenericUnetConfig
