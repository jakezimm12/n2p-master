from ast import Num
import copy
import numpy as np
from math import ceil
from cmath import exp

class N2P_ImageFormatter:
    def __init__(self, config, n2p_config, patch_shape) -> None:
        self.config = config
        self.n2p_config = n2p_config
        self.patch_shape = patch_shape
        self.subpatch_shape = self.n2p_config.subpatch_shape
        self.subpatch_len_y = self.n2p_config.subpatch_shape[0]
        self.subpatch_len_x = self.n2p_config.subpatch_shape[1]
        self.sqrt_n_subpatches = patch_shape[0] // n2p_config.subpatch_shape[0] # ASSUMES PATCHES ARE SQUARE - MAKE SURE SHAPE IS SHAPE OF LARGE PATCH
        self.color_range, self.position_range = self.n2p_config.color_sigma, self.n2p_config.pos_sigma
        self.target_patch_y_idx = self.n2p_config.subpatch_shape[0] * self.sqrt_n_subpatches // 2
        self.target_patch_x_idx = self.n2p_config.subpatch_shape[1] * self.sqrt_n_subpatches // 2
        self.middle_patch_idx = int(self.sqrt_n_subpatches**2//2 if self.sqrt_n_subpatches % 2 == 1 else self.sqrt_n_subpatches**2//2 + self.sqrt_n_subpatches / 2)
    
    # Does not yet support RGB
    def format_img(self, img):
        img = self.add_weight_channel(img)
        # leftover_x, leftover_y = img.shape[1] % self.subpatch_len_x, img.shape[0] % self.subpatch_len_y
        subpatch_start_range = np.array(img.shape[0:2]) - np.array(self.n2p_config.subpatch_shape)
        num_subpatches_y, num_subpatches_x = ceil(float(img.shape[0]) / float(self.subpatch_len_y)), ceil(float(img.shape[1]) / float(self.subpatch_len_x)) 

        # For calculating weights (created class variables just so I don't have to pass these as parameters through many different methods)
        # self.color_range = abs(max(img.flatten()) - min(img.flatten()))
        if not self.n2p_config.color_sigma:
            self.color_range = abs(max(img.flatten()) - min(img.flatten()))
        if not self.n2p_config.pos_sigma:
            self.position_range = max(img.shape[0], img.shape[1])

        subpatch_y_start_idx = [self.subpatch_shape[0] * patch_i for patch_i in range(self.sqrt_n_subpatches)] # Indices of the start indexes of each subpatch within the larger patch
        subpatch_x_start_idx = [self.subpatch_shape[1] * patch_i for patch_i in range(self.sqrt_n_subpatches)]

        if self.n2p_config.random:
            n_patches = num_subpatches_y*num_subpatches_x
            patches = np.zeros((n_patches, self.patch_shape[0], self.patch_shape[1], 2), dtype=np.float32) # TO DO: Change number of channels (2 here) for RGB

            patch_i = 0
            # Scan through the image and compile patches in the N2P scheme
            for y_start in range(0, img.shape[0], self.subpatch_len_y):
                for x_start in range(0, img.shape[1], self.subpatch_len_x):
                    target_subpatch = img[y_start : y_start+self.subpatch_len_y, x_start : x_start+self.subpatch_len_x]
                    colors, positions = [], []
                    for k in subpatch_y_start_idx:
                        for l in subpatch_x_start_idx:
                            other_subpatch_y_start = np.random.randint(0, subpatch_start_range[0] + 1)
                            other_subpatch_x_start = np.random.randint(0, subpatch_start_range[1] + 1)
                            subpatch = img[other_subpatch_y_start:other_subpatch_y_start + self.subpatch_len_y, other_subpatch_x_start:other_subpatch_x_start + self.subpatch_len_x]
                            colors.append(np.mean(subpatch[..., 0])) # MUST CHANGE LAST DIMENSION FOR RGB
                            positions.append((other_subpatch_y_start + self.subpatch_len_y // 2, other_subpatch_x_start + self.subpatch_len_x // 2)) # The middle-most pixel in the subpatch
                            patches[patch_i, k:k + self.subpatch_len_y, l:l + self.subpatch_len_x] = copy.copy(subpatch)
                    
                    # Set the middle patch to the target patch
                    patches[patch_i, self.target_patch_y_idx:self.target_patch_y_idx + self.subpatch_len_y, self.target_patch_x_idx:self.target_patch_x_idx + self.subpatch_len_x] =  copy.copy(target_subpatch)
                    # Remember to also replace the color and position
                    colors[self.middle_patch_idx] = (np.mean(target_subpatch[..., 0])) # MUST CHANGE LAST DIMENSION FOR RGB
                    positions[self.middle_patch_idx] = (y_start + self.subpatch_len_y // 2, x_start + self.subpatch_len_x // 2) # The middle-most pixel in the subpatch

                    self.set_weights(patches[patch_i, ..., 1], colors, positions)
                    patch_i += 1
        else:
            n_patches = (num_subpatches_y-1) * (num_subpatches_x-1) # We can't predict the edge subpatches because they do not have a complete set of nearby subpatches
            patches = np.zeros((n_patches, self.patch_shape[0], self.patch_shape[1], 2), dtype=np.float32) # TO DO: Change number of channels (2 here) for RGB

            patch_i = 0
            # Scan through the image and compile patches in the N2P scheme
            for y_start in range(0, img.shape[0] - self.patch_shape[0] + 1, self.subpatch_len_y):
                for x_start in range(0, img.shape[1] - self.patch_shape[1] + 1, self.subpatch_len_x):
                    colors, positions = [], []
                    cur_patch = img[y_start:y_start + self.patch_shape[0], x_start:x_start + self.patch_shape[1]]
                    patches[patch_i, ...] = copy.copy(cur_patch)
                    
                    for k in subpatch_y_start_idx:
                        for l in subpatch_x_start_idx:
                            subpatch = patches[patch_i, k:k + self.subpatch_shape[0], l:l + self.subpatch_shape[1]]
                            colors.append(np.mean(subpatch[..., 0])) # MUST CHANGE LAST DIMENSION FOR RGB
                            positions.append((k + self.subpatch_shape[0] // 2, l + self.subpatch_shape[1] // 2)) # Just used the in-patch position here because this subpatch is the same distance away from the target patch's real position in the image as its position in the patch

                    self.set_weights(patches[patch_i, ..., 1], colors, positions)
                    patch_i += 1

        return patches

    def create_image_from_patches(self, patches, img_shape):
        pred = np.zeros(img_shape, dtype=np.float32)

        patch_i = 0
        for y_start in range(0, pred.shape[0], self.subpatch_len_y):
            for x_start in range(0, pred.shape[1], self.subpatch_len_x):
                pred[y_start:y_start + self.subpatch_len_y, x_start:x_start + self.subpatch_len_x] = \
                    patches[patch_i, self.target_patch_y_idx:self.target_patch_y_idx + self.subpatch_len_y, self.target_patch_x_idx:self.target_patch_x_idx + self.subpatch_len_x, 0]
                patch_i += 1

        return pred
    
    def set_weights(self, formatted_patch, colors, positions):
        if len(colors) != self.sqrt_n_subpatches**2 or len(positions) != self.sqrt_n_subpatches**2:
            raise Exception("colors and positions arrays should be of size sqrt_n_subpatches")

        target_color = colors[self.middle_patch_idx]
        target_pos = positions[self.middle_patch_idx]
        weights = [self.get_weight(color, pos, target_color, target_pos) for color, pos in zip(colors, positions)] # Get weights for each patch

        patch_idx = 0
        for k in [self.n2p_config.subpatch_shape[0] * patch_i for patch_i in range(self.sqrt_n_subpatches)]:
            for l in [self.n2p_config.subpatch_shape[1] * patch_i for patch_i in range(self.sqrt_n_subpatches)]:
                formatted_patch[k:k + self.n2p_config.subpatch_shape[0], l:l + self.n2p_config.subpatch_shape[1]] = np.full((self.n2p_config.subpatch_shape[0], self.n2p_config.subpatch_shape[1]), weights[patch_idx])
                patch_idx += 1

        target_patch_y_idx = self.n2p_config.subpatch_shape[0] * self.sqrt_n_subpatches // 2
        target_patch_x_idx = self.n2p_config.subpatch_shape[1] * self.sqrt_n_subpatches // 2
        formatted_patch[target_patch_y_idx:target_patch_y_idx + self.n2p_config.subpatch_shape[0], target_patch_x_idx:target_patch_x_idx + self.n2p_config.subpatch_shape[1]] = np.full((self.n2p_config.subpatch_shape[0], self.n2p_config.subpatch_shape[1]), weights[self.middle_patch_idx]) # Set target patch's weights to 1 in case any patches overlapped with it

    def get_weight(self, color, pos, target_color, target_pos):
        if self.n2p_config.just_color:
            return exp(-(1/self.color_range**2 * np.linalg.norm(target_color - color)**2)).real
        return exp(-(1/self.color_range**2 * np.linalg.norm(target_color - color)**2 + 1/self.position_range**2 * np.linalg.norm(np.array(target_pos) - np.array(pos))**2)).real

    def add_weight_channel(self, img):
        img = img[..., np.newaxis]
        weight_channel = np.zeros(img.shape, dtype=np.float32)
        return np.concatenate((img, weight_channel), axis = len(img.shape)-1)