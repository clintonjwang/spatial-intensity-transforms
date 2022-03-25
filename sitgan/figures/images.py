from math import log
import numpy as np
import torch
from am.dispositio.action import Action

def get_actions(A):
    return [
        (["show parameterized starGAN components"],
            Action(show_parameterized_starGAN_components)),

        (["get ventricle segmentation for datapoint"],
            Action(get_ventricle_segmentation_for_datapoint)),
    ]

def get_ventricle_segmentation_for_datapoint(A, dp):
    if A("datapoint belongs to ADNI")(dp):
        return A("get ventricle segmentation for ADNI datapoint")(dp)
    else:
        A("TODO")()


def show_parameterized_starGAN_components(A, dp, variable, target_Y,
        flow_params={"type":"grayscale"}, diff_params={"type":"color"},
        save_path=None, draw=True, return_components=False,
        overlay_segmentation=False):
    if variable == "age":
        dim = 0
    elif variable == "NIHSS":
        dim = 1
    with torch.no_grad():
        batch_loader = A("create batch loader for starGAN sweep on datapoint")(dp, variable, [target_Y], batch_size=1)
        outputs = A("get network variables for custom batch loader")(batch_loader, computed_vars=["dy", "x_t"])
        input_img = outputs["x_s"][0]
        output_imgs = outputs["x_t"]

        network = A("get active network")()
        G = network["sections"]["G"]
        layers = A("for each")(["optical flow", "dI"], lambda layer: A("get section layer")(layer, section=G))
        outputs_by_layer = G["pytorch module"]([outputs[input_var].cuda() for input_var in ["x_s", "dy"]], return_intermediates=True)
        outputs = {layer["name"]: outputs_by_layer[layer["name"]] for layer in layers}

    dIs = outputs["dI"]
    dI = A("tensor image to numpy")(dIs[0].cpu())
    #att = A("tensor image to numpy")(attentions[0].cpu())
    in_img = A("grayscale to RGB")(A("tensor image to numpy")(input_img.cpu()))
    out = A("grayscale to RGB")(A("tensor image to numpy")(output_imgs[0].cpu()))
    in_img /= in_img.max()
    out /= out.max()


    if overlay_segmentation is True:
        A("TODO")()
        seg = A("get ventricle segmentation for datapoint")(dp)
        outline = A("get mask contour")(seg)

    flow_type = A("get attribute")(flow_params, "type")
    if flow_type in ("quiver", "arrows"):
        flow = A("downsample")(outputs["optical flow"]/4, factor=4)
        raw_flow = A("tensor image to numpy")(flow)
        return A("quiver plot")(raw_flow[...,0], raw_flow[...,1])
    else:
        raw_flow = A("tensor image to numpy")(outputs["optical flow"])
        flow = A("create optical flow image")(raw_flow, legend=False)
        flow = flow.astype(float) / flow.max()

        mask_flow = A("get attribute")(flow_params, "apply Gaussian mask")
        if mask_flow is True:
            h,w = flow.shape[:2]
            mask = A("torch")("get 2D Gaussian kernel")(kernel_size=(h,w),
                sigma=(int(h*mask_flow),int(w*mask_flow)), force_even=True)
            flow *= mask.unsqueeze(-1).numpy()

        if flow_type != "color":
            flow = A("grayscale to RGB")(flow.mean(-1))

        flow_cutoff_percs = A("get attribute")(flow_params, "cutoff percentages")
        if flow_cutoff_percs is not None:
            flow = A("cutoff image intensities at percentages of maximum")(flow, *flow_cutoff_percs)

        flow = A("rescale intensities to 0-1")(flow)


    diff_type = A("get attribute")(diff_params, "type")
    if diff_type == "absolute":
        diffs = np.abs(dI)
        diffs = A("grayscale to RGB")(diffs)
        diffs /= diffs.max()

    else:
        if diff_type == "color":
            abs_max = A("get attribute")(diff_params, "colorbar limit")
            A("plot grayscale image in red-blue")(dI, abs_max=abs_max)
        elif diff_type == "grayscale":
            pass
        else:
            A("TODO")()

        diffs = A("rescale intensities to 0-1")(dI)

        mask_dI = A("get attribute")(diff_params, "apply Gaussian mask")
        if mask_dI is True:
            h,w = diffs.shape[:2]
            mask = A("torch")("get 2D Gaussian kernel")(kernel_size=(h,w),
                sigma=(int(h*mask_dI),int(w*mask_dI)), force_even=True)
            diffs *= mask.numpy()
            diffs /= diffs.max()

        dI_cutoff_percs = A("get attribute")(diff_params, "cutoff percentages")
        if dI_cutoff_percs is not None:
            diffs = A("cutoff image intensities at percentages of maximum")(diffs, *dI_cutoff_percs)

        diffs = A("grayscale to RGB")(diffs)
        diffs = A("rescale intensities to 0-1")(diffs)

    img = A("concatenate images horizontally")([in_img, out, flow, diffs], pad_to_match=False)

    if save_path is not None:
        A("save image")(img, path=save_path)
    if draw:
        A("draw image")(img)

    if return_components:
        return img, [in_img, out, raw_flow, dI]

    return img