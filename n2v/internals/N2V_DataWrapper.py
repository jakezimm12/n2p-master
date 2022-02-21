from csbdeep.internals.train import RollingSequence
from tensorflow.keras.utils import Sequence

import numpy as np

class N2V_DataWrapper(RollingSequence):
    """
    The N2V_DataWrapper extracts random sub-patches from the given data and manipulates 'num_pix' pixels in the
    input.

    Parameters
    ----------
    X          : array(floats)
                 The noisy input data. ('SZYXC' or 'SYXC')
    Y          : array(floats)
                 The same as X plus a masking channel.
    batch_size : int
                 Number of samples per batch.
    num_pix    : int, optional(default=1)
                 Number of pixels to manipulate.
    shape      : tuple(int), optional(default=(64, 64))
                 Shape of the randomly extracted patches.
    value_manipulator : function, optional(default=None)
                        The manipulator used for the pixel replacement.
    """

    def __init__(self, X, Y, batch_size, length, perc_pix=0.198, shape=(64, 64),
                 value_manipulation=None, structN2Vmask=None, n2p_config=None):
        super(N2V_DataWrapper, self).__init__(data_size=len(X), batch_size=batch_size, length=length)
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))
        self.shape = shape
        self.value_manipulation = value_manipulation
        # self.range = np.array(self.X.shape[1:-1]) - np.array(self.shape) # Commented out and put below because n2p requires different range
        self.dims = len(shape)
        self.n_chan = X.shape[-1]
        self.structN2Vmask = structN2Vmask
        self.n2p_config = n2p_config

        if self.structN2Vmask is not None:
            print("StructN2V Mask is: ", self.structN2Vmask)

        num_pix = int(np.product(shape)/100.0 * perc_pix)
        assert num_pix >= 1, "Number of blind-spot pixels is below one. At least {}% of pixels should be replaced.".format(100.0/np.product(shape))
        print("{} blind-spots will be generated per training patch of size {}.".format(num_pix, shape))
        
        if self.n2p_config is None:
            self.range = np.array(self.X.shape[1:-1]) - np.array(self.shape)
            if self.dims == 2:
                self.patch_sampler = self.__subpatch_sampling2D__
                self.box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
                self.get_stratified_coords = self.__get_stratified_coords2D__
                self.rand_float = self.__rand_float_coords2D__(self.box_size)
            elif self.dims == 3:
                self.patch_sampler = self.__subpatch_sampling3D__
                self.box_size = np.round(np.sqrt(100 / perc_pix)).astype(np.int)
                self.get_stratified_coords = self.__get_stratified_coords3D__
                self.rand_float = self.__rand_float_coords3D__(self.box_size)
            else:
                raise Exception('Dimensionality not supported.')
        else:
            self.range = np.array(self.X.shape[1:-1]) - np.array(self.n2p_config.subpatch_shape)
            self.box_size = None # To emphasize we don't need box_size in N2P scheme
            if self.dims == 2:
                self.patch_sampler = self.__n2p_subpatch_sampling2D__
                self.middle_patch_coords = self.__n2p_get_censored_coords2D__(shape=self.shape, subpatch_shape=self.n2p_config.subpatch_shape)
                print(self.middle_patch_coords)
            # 3 DIMS NOT SUPPORTED YET
            # elif self.dims == 3:
            #     self.patch_sampler = self.__subpatch_sampling3D__
            #     self.get_stratified_coords = self.__get_stratified_coords3D__
            #     self.rand_float = self.__rand_float_coords3D__(self.box_size)
            else:
                raise Exception('Dimensionality not supported.')

        self.X_Batches = np.zeros((self.batch_size, *self.shape, self.n_chan), dtype=np.float32)
        self.Y_Batches = np.zeros((self.batch_size, *self.shape, 2*self.n_chan), dtype=np.float32)

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))


    def __getitem__(self, i):
        idx = self.batch(i)
        # idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
        # idx = self.perm[idx]
        self.X_Batches *= 0
        self.Y_Batches *= 0
        if self.n2p_config:
            self.patch_sampler(self.X, self.X_Batches, indices=idx, patch_start_range=self.range, shape=self.shape, subpatch_shape=self.n2p_config.subpatch_shape)
        else:
            self.patch_sampler(self.X, self.X_Batches, indices=idx, range=self.range, shape=self.shape) # Get a bunch of random patches and put them in X_batches

        for c in range(self.n_chan):
            for j in range(self.batch_size):
                if self.n2p_config:
                    coords = self.middle_patch_coords
                else:
                    coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size,
                                                    shape=self.shape)

                indexing = (j,) + coords + (c,) # Indices of pixels to mask
                indexing_mask = (j,) + coords + (c + self.n_chan, ) # Same thing but in masking channel
                y_val = self.X_Batches[indexing] # y_val is the target pixel values
                x_val = self.value_manipulation(self.X_Batches[j, ..., c], coords, self.dims) # manipulate what the pixels to mask used to be for some reason

                self.Y_Batches[indexing] = y_val # y_batches get those target values
                self.Y_Batches[indexing_mask] = 1 # put 1's in masking channels where pixels to mask are
                self.X_Batches[indexing] = x_val # put new input vals of x in x_batches
                
                if self.structN2Vmask is not None:
                    self.apply_structN2Vmask(self.X_Batches[j, ..., c], coords, self.dims, self.structN2Vmask)

        return self.X_Batches, self.Y_Batches

    def apply_structN2Vmask(self, patch, coords, dims, mask):
        """
        each point in coords corresponds to the center of the mask.
        then for point in the mask with value=1 we assign a random value
        """
        coords = np.array(coords).astype(np.int)
        ndim = mask.ndim
        center = np.array(mask.shape)//2
        ## leave the center value alone
        mask[tuple(center.T)] = 0
        ## displacements from center
        dx = np.indices(mask.shape)[:,mask==1] - center[:,None]
        ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
        mix = (dx.T[...,None] + coords[None])
        mix = mix.transpose([1,0,2]).reshape([ndim,-1]).T
        ## stay within patch boundary
        mix = mix.clip(min=np.zeros(ndim),max=np.array(patch.shape)-1).astype(np.uint)
        ## replace neighbouring pixels with random values from flat dist
        patch[tuple(mix.T)] = np.random.rand(mix.shape[0])*4 - 2

    # return x_val_structN2V, indexing_structN2V
    @staticmethod
    def __subpatch_sampling2D__(X, X_Batches, indices, range, shape):
        for i, j in enumerate(indices):
            y_start = np.random.randint(0, range[0] + 1)
            x_start = np.random.randint(0, range[1] + 1)
            X_Batches[i] = np.copy(X[j, y_start:y_start + shape[0], x_start:x_start + shape[1]])

    # patch_start_range is the range for which a mini-patch is allwod to start. For example, if the image
    # was 180x180 pixels, the start patch_start_range for 8x8 patches would be (172, 172)
    @staticmethod
    def __n2p_subpatch_sampling2D__(X, X_Batches, indices, patch_start_range, shape, subpatch_shape): # Note I changed range to patch_start_range so I could use th ebuilt in range function
        n_subpatches = shape[0] // subpatch_shape[0] # ASSUMES PATCHES ARE SQUARE
        for i, j in enumerate(indices):
            for k in [subpatch_shape[0] * patch_i for patch_i in range(n_subpatches)]:
                for l in [subpatch_shape[1] * patch_i for patch_i in range(n_subpatches)]:
                    y_start = np.random.randint(0, patch_start_range[0] + 1)
                    x_start = np.random.randint(0, patch_start_range[1] + 1)
                    X_Batches[i, k:k + subpatch_shape[0], l:l + subpatch_shape[1]] = np.copy(X[j, y_start:y_start + subpatch_shape[0], x_start:x_start + subpatch_shape[1]])

    @staticmethod
    def __subpatch_sampling3D__(X, X_Batches, indices, range, shape):
        for i, j in enumerate(indices):
            z_start = np.random.randint(0, range[0] + 1)
            y_start = np.random.randint(0, range[1] + 1)
            x_start = np.random.randint(0, range[2] + 1)
            X_Batches[i] = np.copy(X[j, z_start:z_start + shape[0], y_start:y_start + shape[1], x_start:x_start + shape[2]])

    @staticmethod
    def __get_stratified_coords2D__(coord_gen, box_size, shape):
        box_count_y = int(np.ceil(shape[0] / box_size))
        box_count_x = int(np.ceil(shape[1] / box_size))
        x_coords = []
        y_coords = []
        for i in range(box_count_y):
            for j in range(box_count_x):
                y, x = next(coord_gen)
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if (y < shape[0] and x < shape[1]):
                    y_coords.append(y)
                    x_coords.append(x)
        return (y_coords, x_coords)

    @staticmethod
    def __n2p_get_censored_coords2D__(shape, subpatch_shape):
        halfway_subpatch_idx_x = (shape[0] // subpatch_shape[0] // 2) * subpatch_shape[0] # ASSUMES PATCHES ARE SQUARE
        halfway_subpatch_idx_y = (shape[1] // subpatch_shape[1] // 2) * subpatch_shape[1] # ASSUMES PATCHES ARE SQUARE
        x_coords = []
        y_coords = []

        for i in range(halfway_subpatch_idx_x, halfway_subpatch_idx_x + subpatch_shape[0]):
            for j in range(halfway_subpatch_idx_y, halfway_subpatch_idx_x + subpatch_shape[1]):
                x_coords.append(i)
                y_coords.append(j)
                
        return (y_coords, x_coords)

    @staticmethod
    def __get_stratified_coords3D__(coord_gen, box_size, shape):
        box_count_z = int(np.ceil(shape[0] / box_size))
        box_count_y = int(np.ceil(shape[1] / box_size))
        box_count_x = int(np.ceil(shape[2] / box_size))
        x_coords = []
        y_coords = []
        z_coords = []
        for i in range(box_count_z):
            for j in range(box_count_y):
                for k in range(box_count_x):
                    z, y, x = next(coord_gen)
                    z = int(i * box_size + z)
                    y = int(j * box_size + y)
                    x = int(k * box_size + x)
                    if (z < shape[0] and y < shape[1] and x < shape[2]):
                        z_coords.append(z)
                        y_coords.append(y)
                        x_coords.append(x)
        return (z_coords, y_coords, x_coords)

    @staticmethod
    def __rand_float_coords2D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize)

    @staticmethod
    def __rand_float_coords3D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize, np.random.rand() * boxsize)