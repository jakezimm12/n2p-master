# Noise2Patch (N2P) - Self-Supervised Learning Technique for Denoising With Limited Data

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

Read the research paper to learn more about how N2P works.
