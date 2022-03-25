from math import log
import numpy as np
import torch
from am.dispositio.action import Action

def get_actions(A):
    return [
        (["get network inputs for datapoint for sweep visualizations"],
            Action(get_network_inputs_for_dp_sweep_visualizations)),
        (["get network inputs for sweep visualizations"],
            Action(get_network_inputs_for_sweep_visualizations)),
        (["add variable value to each frame"],
            Action(add_variable_value_to_each_frame)),

        (["save video of 1D sweep for datapoint", "save 1D sweep for datapoint as video"],
            Action(save_video_of_1D_sweep_for_datapoint)),
        (["save video of 1D sweeps for all variables for datapoint"],
            Action(save_video_of_1D_sweeps_for_all_variables_for_datapoint)),

        (["save grid video of 1D sweeps for variable"],
            Action(save_grid_video_of_1D_sweeps_for_variable)),

        (["create 2D grid figure for datapoint from starGAN"],
            Action(create_2D_grid_for_datapoint_from_starGAN)),
        (["create all 2D grid figures for starGAN"],
            Action(create_all_2D_grids_for_starGAN)),

        (["collate 1D starGAN sweeps for figure"],
            Action(collate_1D_starGAN_sweeps_for_figure)),
        (["create all 1D sweeps for starGAN"],
            Action(create_all_1D_sweeps_for_starGAN)),
        (["create 1D sweep for datapoint from starGAN"],
            Action(create_1D_sweep_for_datapoint_from_starGAN)),
    ]


def add_variable_value_to_each_frame(A, frames, var_values, variable, var_denormalizers):
    for ix in A("range over")(frames):
        current_value = var_values[ix].item()
        if variable == "age":
            text = "Age = {}".format(A("round")(var_denormalizers["age"](current_value)))
        elif variable == "NIHSS":
            text = "NIHSS = {}".format(A("round")(var_denormalizers["NIHSS"](current_value)))
        elif variable == "mRS":
            text = "mRS = {}".format(A("round")(var_denormalizers["mRS"](current_value)))
        frames[ix] = A("add text below image")(text=text, image=frames[ix], font_size=20)

    return frames


def get_network_inputs_for_dp_sweep_visualizations(A, dp, variable, N_frames, cuda=True, network=None):
    # collect variables
    img_paths = A("get unaugmented image path for datapoint")(dp)
    orig_img = A("load image")(img_paths)
    orig_img = orig_img["pixels"]

    const_vars = ["age", "NIHSS", "mRS"]
    const_vars.remove(variable)

    var_layer = variable+" input"
    const_inputs_by_layer = {
        var+" input": torch.as_tensor(dp["observations"][var]).float().unsqueeze(0) for var in const_vars
    }

    var_values = torch.as_tensor(A("get evenly spaced numbers")(*A["variable ranges for sweeps"][variable], N_frames)).float()
    orig_img = torch.as_tensor(orig_img).float().unsqueeze(0)
    cnn = network["pytorch network"].eval()

    if cuda:
        const_inputs_by_layer = A("for each value")(const_inputs_by_layer, lambda v:v.cuda())
        cnn = cnn.cuda()
        var_values = var_values.cuda()
        orig_img = orig_img.cuda()
    else:
        cnn = cnn.cpu()

    return var_layer, const_inputs_by_layer, var_values, orig_img, cnn


def get_network_inputs_for_sweep_visualizations(A, dps, variable, N_frames,
    batch_size=25, network=None):
    # collect variables
    img_paths = A("for each")(dps, "get unaugmented image path for datapoint")
    orig_imgs = A("for each")(img_paths, "load image")
    orig_imgs = A("for each get")(orig_imgs, "pixels")

    const_vars = ["age", "NIHSS", "mRS"]
    const_vars.remove(variable)

    var_layer = variable+" input"
    get_var_fxn = lambda dp: A("get first defined argument")(dp["observations"][var], 0)
    const_inputs_by_layer = {
        var+" input": torch.as_tensor(A("for each")(dps, get_var_fxn)).float(
            ) for var in const_vars
    }

    var_values = torch.as_tensor(A("get evenly spaced numbers")(*A["variable ranges for sweeps"][variable], N_frames)).float()
    orig_imgs = torch.stack([torch.as_tensor(img).float() for img in orig_imgs],0)
    cnn = network["pytorch network"].eval().cuda()

    var_values = var_values.cuda()
    def iterator():
        for ix in range(0, len(dps), batch_size):
            attr_batch = {k:v[ix:ix+batch_size].cuda() for k,v in const_inputs_by_layer.items()}
            yield attr_batch, orig_imgs[ix:ix+batch_size].cuda()
    return var_layer, var_values, cnn, iterator

    # orig_imgs = orig_imgs.cuda()
    # const_inputs_by_layer = A("for each value")(const_inputs_by_layer, lambda v: v.cuda())
    # return var_layer, const_inputs_by_layer, var_values, orig_imgs, cnn




def save_video_of_1D_sweep_for_datapoint(A, dp, variable, save_path, duration=7,
    fps=3, text_label=True, smoothing=0, colored=True, network=None, dataset=None):

    N_frames = duration*fps

    # collect variables
    var_layer, const_inputs_by_layer, var_values, orig_img, cnn = A(
        "get network inputs for datapoint for sweep visualizations")(dp, variable, N_frames)

    # collect frames
    frames = []
    for ix in range(N_frames):
        current_values = var_values[ix].unsqueeze(0)

        inputs_by_layer = {"image input": orig_img,
            var_layer: current_values,
            **const_inputs_by_layer,
        }

        with torch.no_grad():
            frames.append(A("select starGAN output corresponding to generated image")(cnn(inputs_by_layer)))

    A("process images and save as video")(frames, save_path, fps,
            variable, var_values, text_label)


def save_video_of_1D_sweeps_for_all_variables_for_datapoint(A):
    A("TODO")()



def save_grid_video_of_1D_sweeps_for_variable(A, variable, save_path, duration=7,
    fps=15, grid_size=(5,5), text_label=True, phase="validation", network=None, dataset=None):

    N_dps = A("product")(grid_size)
    N_frames = duration*fps

    # get datapoints for the grid
    def exclusion_crit(dp):
        return A("is None or NaN")(dp["observations"]["NIHSS"]) or A("is None or NaN")(dp["observations"]["mRS"])

    dps = A("get midslice datapoints for 2D dataset")(exclusion_crit, phase=phase)
    dps = A("exclude datapoints with out of range observations")(dps, variable)
    dps = A("sample collection")(dps, N_dps)

    # collect variables
    var_layer, const_inputs_by_layer, var_values, orig_imgs, cnn = A(
        "get network inputs for sweep visualizations")(dps, variable, N_frames)

    # collect frames
    frames = []
    for ix in range(N_frames):
        current_values = var_values[ix].repeat(N_dps)

        inputs_by_layer = {"image input": orig_imgs,
            var_layer: current_values,
            **const_inputs_by_layer,
        }

        with torch.no_grad():
            images = A("split axis into list")(A("select starGAN output corresponding to generated image")(cnn(inputs_by_layer)), 0)

        frames.append(A("get image grid")(images, n_rows=grid_size[0], n_cols=grid_size[1]))

    A("process images and save as video")(frames, save_path, fps,
            variable, var_values, text_label)


def create_2D_grid_for_datapoint_from_starGAN(A, datapoint, variables=("age", "NIHSS"),
    age_interval=1., nihss_interval=.7, text_labels=True, save_path=None, network=None):
    # if any([x % 2 != 1] for x in grid_size):
    #     A("report failed precondition")("grid must have odd dimensions")
    for var in variables:
        if var not in ["age", "NIHSS", "WMHa"]:
            A("unhandled situation")('need "age", "NIHSS", or "WMHa"')

    get_true_age = lambda age: age * 14.54209 + 64.49649 # 18-100
    get_true_nihss = lambda nihss: nihss * 6.13397 + 5.40468 # 0-36

    wmha_interval=.7
    n_age, n_wmha, n_nihss = 5,5,5
    var1_name,var2_name = variables
    wmha = "WMHa" in network["hyperparameters"]["scalar variables"]

    img_path = A("get unaugmented image path for datapoint")(datapoint)
    true_age = datapoint["observations"]["age"]
    if wmha:
        true_wmha = datapoint["observations"]["log(WMHa+1)"]
    true_nihss = datapoint["observations"]["NIHSS"]

    if A("is None or NaN")(true_nihss):
        A("WARNING")("{} has no NIHSS".format(datapoint["name"]))
        return
        true_nihss = -9999.


    age_points = torch.as_tensor(A("get evenly spaced numbers")(true_age-age_interval*(n_age//2),
        true_age+age_interval*(n_age//2), n_age)).float()
    if wmha:
        wmh_points = torch.as_tensor(A("get evenly spaced numbers")(true_wmha-wmha_interval*(n_wmha//2),
            true_wmha+wmha_interval*(n_wmha//2), n_wmha)).float()
    nihss_points = torch.as_tensor(A("get evenly spaced numbers")(true_nihss-nihss_interval*(n_nihss//2),
        true_nihss+nihss_interval*(n_nihss//2), n_nihss)).float()

    if var1_name == "age":
        var1_points = age_points

    if var2_name == "WMHa":
        var2_points = wmh_points
        const_input = torch.as_tensor(true_nihss).float()
    elif var2_name == "NIHSS":
        var2_points = nihss_points
        if wmha:
            const_input = torch.as_tensor(true_wmha).float()
        else:
            const_input = None

    var1_layer = var1_name+" input"
    var2_layer = var2_name+" input"
    if "NIHSS" not in variables:
        const_layer = "NIHSS input"
    elif "WMHa" not in variables:
        const_layer = "WMHa input"
    else:
        const_layer = "age input"

    orig_img = A("load image")(img_path)
    img = torch.as_tensor(orig_img["pixels"]).float()
    cnn = network["pytorch network"].cpu().eval()

    orig_img = A("draw box around image")(orig_img, color="red", expand_image=False)
    if text_labels:
        A("normalize image (0-255)")(orig_img)

    img_grid = []
    for i,var1 in enumerate(var1_points):
        img_grid.append([])
        for j,var2 in enumerate(var2_points):
            if i == j and i == n_age//2:
                img_grid[i].append(orig_img)
                continue

            inputs_by_layer = {"image input": img.unsqueeze(0),
                var1_layer:var1.unsqueeze(0),
                var2_layer:var2.unsqueeze(0),
                }
            if const_input is not None:
                inputs_by_layer[const_layer] = const_input.unsqueeze(0)

            I = A("select starGAN output corresponding to generated image")(cnn(inputs_by_layer))
            I = A("instantiate image")(I)

            if text_labels:
                A("normalize image (0-255)")(I)

            if i == n_age-1 and text_labels:
                nihss = str(A("round")(get_true_nihss(var2.item())))
                I = A("add text below image")(text=nihss, image=I, font_size=16)
            if j == n_age-1 and text_labels:
                age = str(A("round")(get_true_age(var1.item())))
                I = A("add text to right of image")(text=age, image=I, pad=30, font_size=16)

            I = A("grayscale to RGB")(I)
            img_grid[i].append(I)

    img_grid = A("get image grid")(img_grid)

    if text_labels:
        img_grid = A("add text below image")(text="NIHSS", image=img_grid, font_size=20)
        img_grid = A("add text to right of image")(text="Age", image=img_grid, font_size=20)

    if save_path is not None:
        img_grid = A("scale image")(img_grid, 2)
        img_grid = A("normalize image (0-1)")(img_grid)
        A("save image")(img_grid, path=save_path)
    else:
        A("draw image")(img_grid)

    return img_grid


def create_all_2D_grids_for_starGAN(A, phase="validation", dataset=None, network=None, **kwargs):
    A("TODO")("mRS")
    def exclusion_crit(dp):
        if A("is None or NaN")(dp["observations"]["NIHSS"]):
            return True
        elif dp["observations"]["NIHSS"] > 3.3 or dp["observations"]["NIHSS"] < .4:
            return True
        elif dp["observations"]["age"] > 1 or dp["observations"]["age"] < -1:
            return True
        # elif dp["observations"]["log(WMHa+1)"] > 2.5 or dp["observations"]["log(WMHa+1)"] < .7:
        #     return True
        return False

    folder = A("join paths")(A["WMH visualization folder"], "2D_grids", network["name"])
    A("create folder if needed")(folder)

    dps = A("get midslice datapoints for 2D dataset")(exclusion_crit, phase=phase)
    for dp in dps:
        A("create 2D grid figure for datapoint from starGAN")(dp,
            save_path=A("join paths")(folder, dp["name"]+".png"), **kwargs)
        # A("create 2D grid figure for datapoint from starGAN")(dp, variables=("age", "log(WMHa+1)"),
        #     save_path=A("join paths")(A["WMH visualization folder"], "2D_grids", dp["name"]+".png"))


def create_1D_sweep_for_datapoint_from_starGAN(A, datapoint, variable,
    interval=None, sweep_length=5, text_labels=True, save_path=None,
    quantile_subfolders=False, network=None):
    if sweep_length % 2 != 1:
        A("report failed precondition")("grid must have odd dimensions")

    get_true_age = lambda age: age * 14.54209 + 64.49649 # 18-100
    get_true_nihss = lambda nihss: nihss * 6.13397 + 5.40468 # 0-36

    if variable not in ["age", "NIHSS", "WMHa"]:
        A("unhandled situation")('need "age", "NIHSS", or "WMHa"')

    if interval is None:
        if variable == "age":
            interval = 1.
        elif variable == "NIHSS":
            interval = 1.
        elif variable == "WMHa":
            interval = .7

    img_path = A("get unaugmented image path for datapoint")(datapoint)
    true_age = datapoint["observations"]["age"]
    # true_wmha = datapoint["observations"]["log(WMHa+1)"]
    true_nihss = datapoint["observations"]["NIHSS"]

    if A("is None or NaN")(true_nihss):
        A("WARNING")("{} has no NIHSS".format(datapoint["name"]))
        return
        true_nihss = -9999.

    var_layer = variable+" input"
    if variable == "age":
        true_value = true_age
        const_layer = "NIHSS input"
        const_value = torch.as_tensor(true_nihss).float()

        quantiles = A("get evenly spaced numbers")(-2.5, 2.5, 6)
        if A("is between")(true_value, 1.5, 2.5):
            quantile = 5
        elif A("is between")(true_value, .5, 1.5):
            quantile = 4
        elif A("is between")(true_value, -.5, .5):
            quantile = 3
        elif A("is between")(true_value, -1.5, -.5):
            quantile = 2
        elif A("is between")(true_value, -2.5, -1.5):
            quantile = 1
        else:
            A("WARNING")("{} out of range".format(variable))
            return

        ages = [true_value + interval*(i-quantile+1) for i in range(5)]

        #texts = ["{}-{}".format(int(get_true_age(quantiles[i]))+1, int(get_true_age(quantiles[i+1]))) for i in range(5)]
        texts = ["{}".format(int(get_true_age(age))) for age in ages]


    elif variable == "NIHSS":
        true_value = true_nihss
        const_layer = "age input"
        const_value = torch.as_tensor(true_age).float()

        quantiles = list(range(-1, 5))
        if A("is between")(true_value, 3, 4):
            quantile = 5
        elif A("is between")(true_value, 2, 3):
            quantile = 4
        elif A("is between")(true_value, 1, 2):
            quantile = 3
        elif A("is between")(true_value, 0, 1):
            quantile = 2
        elif A("is between")(true_value, -1, 0):
            quantile = 1
        else:
            A("WARNING")("{} out of range".format(variable))
            return

        texts = ["{}-{}".format(int(get_true_nihss(quantiles[i]))+1, int(get_true_nihss(quantiles[i+1]))) for i in range(5)]

    #A("get evenly spaced numbers")(true_value-interval*(sweep_length//2),
    #   true_value+interval*(sweep_length//2), sweep_length)
    top = true_value + 1.*(5-quantile)
    bot = true_value - 1.*(quantile-1)
    var_points = torch.as_tensor(A("get evenly spaced numbers")(bot, top, sweep_length)).float()

    orig_img = A("load image")(img_path)
    img = torch.as_tensor(orig_img["pixels"]).float()
    cnn = network["pytorch network"].cpu().eval()

    orig_img = A("draw box around image")(orig_img, color="red", expand_image=False)

    img_sweep = []
    for ix,var in enumerate(var_points):
        if ix+1 == quantile:
            if text_labels:
                orig_img = A("add text below image")(text=texts[ix], image=orig_img, font_size=16)
            img_sweep.append(orig_img)
            continue

        inputs_by_layer = {"image input": img.unsqueeze(0),
            var_layer: var.unsqueeze(0),
            const_layer: const_value.unsqueeze(0)}
        I = A("select starGAN output corresponding to generated image")(cnn(inputs_by_layer))
        I = A("instantiate image")(I)
        if text_labels:
            I = A("add text below image")(text=texts[ix], image=I, font_size=16)

        I = A("grayscale to RGB")(I)
        img_sweep.append(I)

    img_sweep = A("concatenate images horizontally")(img_sweep)

    if text_labels:
        img_sweep = A("add text below image")(text=A("capitalize")(variable),
            image=img_sweep, font_size=20)

    if save_path is not None:
        if quantile_subfolders:
            save_path = A("join paths")(A("dirname")(save_path), str(quantile), A("basename")(save_path))
            A("create folder if needed")(save_path)

        img_sweep = A("scale image")(img_sweep, 2)
        img_sweep = A("normalize image (0-1)")(img_sweep)
        A("save image")(img_sweep, path=save_path)
    else:
        A("draw image")(img_sweep)

    return img_sweep


def collate_1D_starGAN_sweeps_for_figure(A, datapoints, variable, save_path=None, v_space=20):
    img_sweeps = []
    for dp in datapoints:
        I = A("create 1D sweep for datapoint from starGAN")(
            dp, variable, text_labels = dp is datapoints[-1])
        A("normalize image (0-1)")(I)
        img_sweeps.append(I)

    full_img = A("concatenate images vertically")(img_sweeps, padding=v_space)

    if save_path is not None:
        full_img = A("scale image")(full_img, 2)
        full_img = A("normalize image (0-1)")(full_img)
        A("save image")(full_img, path=save_path)
    else:
        A("draw image")(full_img)

    return full_img


def create_all_1D_sweeps_for_starGAN(A, phase="validation", job_name=None, n_iters=None,
        exclusion_crit=None, variables=("age",), network=None):
    if exclusion_crit is None:
        def exclusion_crit(dp):
            return False
            # if A("is None or NaN")(dp["observations"]["NIHSS"]):
            #     return True

    if job_name is not None:
        network = A("load network from job")(job_name, epoch=n_iters)

    dps = A("get midslice datapoints for 2D dataset")(exclusion_crit, phase=phase)
    if len(variables) > 1:
        A("TODO")()
    for variable in variables:
        if variable == "age":
            target_dY = [-20, -10, 0, 10, 20]
        elif variable == "NIHSS":
            target_dY = [-10, -5, 0, 5, 10]
        else:
            A("TODO")()
        folder = A("join paths")(A["AIS images folder"], "1d_sweeps", network["name"])#, phase, variable
        A("create folder if needed")(folder)
        for dp in dps:
            A("get sample 1D sweep results for starGAN")(dp, variable, target_dY=target_dY,
                save_path=A("join paths")(folder, dp["name"]+".png"))
