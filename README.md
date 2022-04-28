# Noise2Patch (N2P) - Self-Supervised Learning Technique for Denoising With Limited Data

## Abstract

Discriminative deep learning models have dominated the world of computer vision for a generation. Neural networks serve as a basis for tasks like denoising,
super-resolution, inpainting, image-dehazing, watermark-removal, and much more.
However, these neural networks require great deals of data to work. Acquiring images can often be tedious, expensive, or even impossible. For example, the limitation
of clean data is virtually ubiquitous in biomedical tasks, where a plethora of noisy
data is available, but clean data is scarce. Thus, the need for unsupervised learning
techniques arises when only noisy data is available or intensive data collection is too
costly. A method named Noise2Noise found that neural networks can actually use
noisy images as the input and the target because the expected value of a noisy image
is the ground truth. In other words, the noisy targets will represent the ground
truth on average. Another paper, Noise2Void (N2V), leverages this discovery by
creating a self-supervised method using patches of pixels from an image that have a
single, central pixel censored. The patch with a pixel censored is used as the input
and the censored pixel is used as a target. N2V is among the highest performing
unsupervised learning methods for denoising. Here, we introduce a method very
similar to N2V: Noise2Patch (N2P). Instead of predicting one pixel from one patch,
N2P predicts one entire censored patch from many random patches from the input
image during training. Each random patch has an associated weight based on its
color and distance from the censored patch. A color and position more similar to
the censored patch results in a higher weight.

## General Overview of Procedure

We adapt N2V’s blind-spot network to a “blind-subpatch network” in which random
subpatches from the entire image are arranged around the central, target subpatch.
To align our language with the explanation of our experiment, we will refer to
each of these patches as subpatches that form the larger patch that is our receptive
field. For example, say we select a subpatch size of 10 × 10 pixels to be our target
subpatch and eight other randomly selected 10 × 10 subpatches to be our predictive
subpatches. We would arrange these subpatches into a larger 30 × 30 patch that has
three rows and three columns of subpatches with the censored patch in the center.

![mosaic](https://user-images.githubusercontent.com/65970260/165851215-3851fac2-470b-4037-a62c-0a52e69b93cc.png)

Weights are calculated for each subpatch in the compilation that makes up the
large patch described above. A weight channel of the same shape as the large
patch is added "behind" the large patch. Each pixel in the weight channel contains
the weight of its corresponding pixel’s subpatch. Note that the central, target subpatch is
actually censored. The weight of 1.00 is symbolic to show that the target subpatch
has exactly the same color and position in the original graph as itself.

![new_add_weights](https://user-images.githubusercontent.com/65970260/165851750-cb22d12b-47d6-4c1b-abaf-3d5cfd7399ed.png)

Read the research paper to learn more about how N2P works: [N2P_Research_Paper.pdf](https://github.com/jakezimm12/n2p-master/files/8586811/N2P_Research_Paper.pdf).


