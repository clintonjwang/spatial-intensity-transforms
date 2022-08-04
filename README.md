# High Fidelity Medical Image-to-Image Translation with Spatial-Intensity Transforms

![alt text](https://github.com/clintonjwang/sitgan/blob/main/teaser.png?raw=true)

The **Spatial-Intensity Transform (SIT)** is a **simple** network layer that can be appended to an image decoder or generator to make it **robust** in medical image-to-image translation tasks, such as image harmonization, counterfactual visualization, simulating disease progression, visualizing neurodegeneration, and analyzing imaging biomarkers. While most generators output a new image directly after the final convolution, SIT instead adds a residual to the original image (the intensity transform) and then applies a smooth deformation to warp the image (the spatial transform).

This repository includes four different models:
* Regressor-guided autoencoder
* [Conditional adversarial autoencoder](https://arxiv.org/abs/1702.08423)
* [Identity-preserving GAN](https://arxiv.org/abs/1912.02620)
* [StarGAN](https://arxiv.org/abs/1711.09020)
The last three models are only loosely based on the original works. Our implementations do not follow the original architectures or hyperparameters, but are inspired by their loss functions and training schemes.

For each of these models, SIT introduces only one new hyperparameter (how sparse the intensity transform should be) and no learned parameters!


## Code Usage

To run our SIT version of StarGAN on your own data (only 2D slices currently supported), create `configs/myconfig.yaml` as follows:
```yaml
parent:
- dit
- stargan
dataset: MyDataset
data loading:
  attributes:
  - conditional_attribute_1
  - conditional_attribute_2
  #- ...
  #- ...
  image shape:
  - 224
  - 192
# overwrite additional settings here as desired
```

Edit `sitgan/data/dataloader.get_dataloaders()` to return a PyTorch dataloader for your dataset.

(Optional) edit `configs/default.yaml` with your own paths and/or hyperparameter choices.

From within the `sitgan` folder, run `python train.py -c=path/to/myconfig.yaml -o=path/to/results`.


## SIT Usage

To add SIT to your own generator, simply modify your network from this:

```python
class MyConditionalGenerator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        #...
        self.output_layer = MyModule(out_channels=1)

    def forward(self, input_image):
        #...
        output_image = self.output_layer(intermediates)
        return output_image
```

to this:

```python
import util
from models.common import OutputTransform

class MySITGenerator(nn.Module):
    def __init__(self, outputs="diffs, velocity", **kwargs):
        super().__init__()
        out_channels = util.get_num_channels_for_outputs(outputs)
        #...
        self.output_layer = MyModule(out_channels=out_channels)
        self.final_transforms = OutputTransform(outputs)

    def forward(self, input_image, return_transforms=False):
        #...
        transforms = self.output_layer(intermediates)
        return self.final_transforms(input_image, transforms, return_transforms=return_transforms)
```

The `outputs` argument can also be set to any of the following:
* `None` to restore the original network behavior
* `"diffs"` to apply a sparse intensity transform to `input_image`
* `"displacement"` to apply a non-diffeomorphic deformation to `input_image`
* `"velocity"` to apply a diffeomorphic deformation to `input_image`
* `"diffs, displacement"` to apply a sparse intensity transform followed by a non-diffeomorphic deformation
* `"diffs, velocity"` to apply a sparse intensity transform followed by a diffeomorphic deformation (SIT)

Note that our implementation of SIT currently only handles single channel images, although it can be easily extended to multiple channels.

## Requirements

CUDA is required. The code is not written to run on CPUs.

## Citation

**[Spatial-Intensity Transform GANs for High Fidelity Medical Image-to-Image Translation](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_72)**<br>
[Clinton J. Wang](https://clintonjwang.github.io/), [Natalia S. Rost](https://www.massgeneral.org/doctors/17477/natalia-rost), and [Polina Golland](https://people.csail.mit.edu/polina/)<br>
MICCAI 2020

If you find this work useful please use the following citation:
```
@inproceedings{wang2020spatial,
  title={Spatial-intensity transform GANs for high fidelity medical image-to-image translation},
  author={Wang, Clinton J and Rost, Natalia S and Golland, Polina},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={749--759},
  year={2020},
  organization={Springer}
}
```

## Acknowledgements

Thanks to [Daniel Moyer](https://dcmoyer.github.io/) for his help with making figures.
