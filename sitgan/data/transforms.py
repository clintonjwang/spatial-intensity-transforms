import numpy as np
import torch
nn = torch.nn
F = nn.functional
import monai.transforms as mtr

import util

def get_transforms(args):
    preproc_settings = args["data loading"]
    aug_settings = args["augmentations"]

    val_transforms = mtr.Compose([
        mtr.LoadImaged(keys=["image"]),
        mtr.AddChanneld(keys=["image"]),
        mtr.ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=preproc_settings["lower percentile"], upper=preproc_settings["upper percentile"],
            b_min=0., b_max=1.,
            clip=preproc_settings["clip"]
        ),
        mtr.CropForegroundd(keys=["image"], source_key="image"),
        mtr.SpatialPadd(keys=["image"], spatial_size=args["data loading"]["image shape"]),
        mtr.ToTensord(keys=["image", "attributes"]),
        mtr.CastToTyped(keys=["image", "attributes"], dtype=torch.float32)
    ])

    if aug_settings is None:
        train_transforms = val_transforms
    else:
        ordering = aug_settings["ordering"]
        # all parameters that are integers or lists of integers
        for key in ["flip (axes)", ("bias field", "order"),]:
            if isinstance(key, str) and key in aug_settings:
                aug_settings[key] = util.parse_int_or_list(aug_settings[key])
            elif len(key) == 2 and key[0] in aug_settings:
                aug_settings[key[0]][key[1]] = util.parse_int_or_list(aug_settings[key[0]][key[1]])

        # all parameters that are floats or lists of floats
        for key in [("affine", "rotation (degrees)"), ("affine", "translation (voxels)"),
            ("affine", "shearing"), ("affine", "scaling"),
            "intensity scaling", "intensity shift", "noise stdev", ("bias field", "coefficients"),
            ("elastic", "sigma range"), ("elastic", "magnitude range"),
            ("elastic", "rotation"), ("elastic", "translation"),
            ("elastic", "scaling"), ("elastic", "shearing"),
            ("spikes", "intensity range"), "blur stdev", "gamma"]:
            if isinstance(key, str) and key in aug_settings:
                aug_settings[key] = util.parse_float_or_list(aug_settings[key])
            elif len(key) == 2 and key[0] in aug_settings:
                try: aug_settings[key[0]][key[1]] = util.parse_float_or_list(aug_settings[key[0]][key[1]])
                except KeyError: pass

        transform_sequence = [
            mtr.LoadImaged(keys=["image"]),
            mtr.AddChanneld(keys=["image"]),
            mtr.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=preproc_settings["lower percentile"], upper=preproc_settings["upper percentile"],
                b_min=0., b_max=1.,
                clip=preproc_settings["clip"],
            ),
            mtr.CropForegroundd(keys=["image"], source_key="image"),
            # mtr.DivisiblePadd(keys=["image"], k=16),
            mtr.SpatialPadd(keys=["image"], spatial_size=(288, 256)),
        ]

        for s in ordering.split(','):
            s = s.strip()
            if "(" in s:
                s,p = s.split(' (')
                p = float(p[:p.find(')')])
            else:
                p = .5
            if s == 'intensity':
                transform_sequence.append(mtr.RandScaleIntensityd(keys=["image"],
                    factors=aug_settings["intensity scaling"],
                    prob=p))
            elif s == "histogram shift":
                transform_sequence.append(mtr.RandHistogramShiftd(keys=["image"],
                    num_control_points=(5,15), prob=p))
            elif s == "gamma":
                transform_sequence.append(mtr.RandAdjustContrastd(keys=["image"],
                    gamma=(.7,1.5), prob=p))
            elif s == 'affine':
                R = aug_settings["affine"]["rotation (degrees)"]
                if hasattr(R, "__iter__"):
                    R = [r * np.pi/180 for r in R]
                else:
                    R *= np.pi/180
                transform_sequence.append(mtr.RandAffined(keys=["image"],
                    rotate_range=R,
                    translate_range=aug_settings["affine"]["translation (voxels)"],
                    scale_range=aug_settings["affine"]["scaling"],
                    shear_range=aug_settings["affine"]["shearing"],
                    prob=p,
                ))
            elif s == 'elastic':
                R = aug_settings["elastic"]["rotation"]
                if hasattr(R, "__iter__"):
                    R = [r * np.pi/180 for r in R]
                else:
                    R *= np.pi/180
                transform_sequence.append(mtr.Rand2DElasticd(keys=["image"],
                    sigma_range=aug_settings["elastic"]["sigma range"],
                    magnitude_range=aug_settings["elastic"]["magnitude range"],
                    rotate_range=R,
                    translate_range=aug_settings["elastic"]["translation"],
                    scale_range=aug_settings["elastic"]["scaling"],
                    shear_range=aug_settings["elastic"]["shearing"],
                    prob=p))
            elif s == 'flip':
                transform_sequence.append(mtr.RandFlipd(
                    keys=["image"], spatial_axis=[1], prob=p,
                ))
            elif s == 'noise':
                transform_sequence.append(mtr.RandGaussianNoised(keys=["image"],
                    std=aug_settings["noise stdev"],
                    prob=p))
            elif s == 'spike':
                transform_sequence.append(mtr.RandKSpaceSpikeNoised(keys=["image"], prob=p,
                    intensity_range=aug_settings["spikes"]["intensity range"]))
            elif s == 'bias':
                transform_sequence.append(mtr.RandBiasFieldd(keys=["image"],
                    coeff_range=aug_settings["bias field"]["coefficients"],
                    degree=aug_settings["bias field"]["order"], prob=p))
            elif s == 'blur':
                transform_sequence.append(mtr.RandGibbsNoised(keys=["image"], alpha=(0,.5), prob=p))
            elif s == 'gamma':
                transform_sequence.append(mtr.RandAdjustContrastd(keys=["image"],
                    gamma=aug_settings["gamma"], prob=p))
            else:
                print(f"WARNING: unknown augmentation type '{s}' will be ignored")

        transform_sequence += [
            mtr.CenterSpatialCropd(keys=["image"], roi_size=args["data loading"]["image shape"]),
            mtr.ToTensord(keys=["image", "attributes"]),
            mtr.CastToTyped(keys=["image", "attributes"], dtype=torch.float32)
        ]

        train_transforms = mtr.Compose(transform_sequence)

    return train_transforms, val_transforms


def get_attr_transforms():
    return mtr.Compose([
        mtr.ToTensor(),
        mtr.CastToType(dtype=torch.float32),
        mtr.SqueezeDim(0),
    ])
