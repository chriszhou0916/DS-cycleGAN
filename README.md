# DS-CycleGAN
### Diversity Sensitive Unpaired Image-to-Image Translation
By Chris Zhou and Leo Hu

This code implements style transfer from the papers **[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)** and **[Diversity-Sensitive Conditional Generative Adversarial Networks](https://arxiv.org/abs/1901.09024)** using Tensorflow 2.0 and Keras.

### Quick start
In this repository we provide three notebook files to perform the following:
- Transform an image using our pre-trained models in `Transform.ipynb`
- Train a new model using database(s) of choice `Train.ipynb`
- Examine possible medical applications of diverse unpaired image-to-image translation using malaria cells `Ultrasound.ipynb`

### Example outputs


### Motivations
CycleGAN is able to generate quality image-to-image outputs from an unpaired dataset, but is limited to a single output image given a single input image.
DSGAN uses a latent vector input alongside a regularizer to generate multimodal outputs, but uses a paired dataset when training.
DS-CycleGAN is able to generate diverse outputs using cycle-consistent adversarial networks in unpaired image-to-image translation.

### Limitations
DS-CycleGAN generates outputs that are limited to differences in pixel intensity. That is, we achieve great diversity in the outputs with differences in the color, brightness, and contrast, but we would like to see more textural differences amongst varying outputs.

### Acknowledgements
Thanks to **[Ouwen Huang](https://github.com/Ouwen)** for advice in this project.