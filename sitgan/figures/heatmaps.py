from math import log
import numpy as np
import torch
import torch.nn.functional as F
from am.dispositio.action import Action

def get_actions(A):
    return [
        (["collate grid search jobs into heatmap"],
            Action(collate_grid_search_jobs_into_heatmap)),
        (["select grid search job"],
            Action(select_grid_search_job)),

        (["get network inputs iterator for random samples from FLAIR starGAN"],
            Action(get_network_inputs_for_random_samples)),
        (["get random samples of translated images from FLAIR starGAN"],
            Action(get_random_samples_of_translated_images)),

        (["save video of average subject"],
            Action(save_video_of_average_subject)),
        (["save heatmap video of framewise diffs for random subjects"],
            Action(save_heatmap_video_of_framewise_diffs_for_random_subjects)),
        (["save heatmap video of framewise diffs averaged across subjects"],
            Action(save_heatmap_video_of_framewise_diffs_averaged_across_subjects)),
        (["save heatmap of diffs averaged over all frames for random subjects"],
            Action(save_heatmap_of_diffs_averaged_over_all_frames_for_random_subjects)),
        (["save heatmap of diffs averaged over all frames and subjects"],
            Action(save_heatmap_of_diffs_averaged_over_all_frames_and_subjects)),

        (["save heatmap video averaged across subjects for multiple networks"],
            Action(save_heatmap_video_averaged_across_subjects_for_multiple_networks)),
    ]

def select_grid_search_job(A, jobs, dx, tv, flow_sparsity=0):
    for job in jobs:
        HP = A("get HPs for job")(job)
        if A("round")(HP["flow TV weight"], precision=2) == tv and \
            A("round")(HP["sparsity regularizer weight"], precision=2) == dx and \
            A("round")(HP["flow sparsity weight"], precision=2) == flow_sparsity:
            return job


def collate_grid_search_jobs_into_heatmap(A, jobs, metric, with_decay=True,
        x_range=None, y_range=None, max_z=None, min_z=None, **kwargs):
    table = A("get network results table")(refresh=True)
    X = []
    Y = []
    Z = []
    for job in jobs:
        result = table.loc[table.index.str.startswith(job), metric]
        if len(result) == 0:
            continue

        HP = A("get HPs for job")(job)
        if with_decay is True:
            if HP["regularizer decay"] is False:
                continue
        elif with_decay is False:
            if HP["regularizer decay"] is not False:
                continue

        if HP["flow TV weight"] is None or HP["sparsity regularizer weight"] is None:
            A("WARNING")(f"bad HPs for job {job}")
            continue

        x = A("round")(HP["flow TV weight"], precision=2)
        y = A("round")(HP["sparsity regularizer weight"], precision=2)

        if x_range is not None:
            if x > x_range[1] or x < x_range[0]:
                continue
        if y_range is not None:
            if y > y_range[1] or y < y_range[0]:
                continue
        if HP["flow sparsity weight"] != 0:
            continue

        z = np.nanmean(result)

        X.append(x)
        Y.append(y)
        Z.append(z)

    xticks = sorted(set(X))
    yticks = sorted(set(Y), reverse=True)
    array = np.zeros((len(xticks), len(yticks)))
    X_index = [xticks.index(x) for x in X]
    Y_index = [yticks.index(y) for y in Y]
    for ix in A("range over")(X):
        # if max_z is not None and Z[ix] > max_z:
        #     Z[ix] = max_z
        array[X_index[ix], Y_index[ix]] = Z[ix]

    if min(xticks) < .01 or min(yticks) < .01:
        xticks = [A('to scientific notation')(x, precision=1, format="plt") for x in xticks]
        yticks = [A('to scientific notation')(y, precision=1, format="plt") for y in yticks]

    plot = A("draw heatmap")(array.transpose(), xlabel="lambda_{TV}", ylabel="lambda_{dx}",
        xticks=xticks, yticks=yticks, font_scale=1, vmin=min_z, vmax=max_z, **kwargs)
    return plot


def get_random_samples_of_translated_images(A, n_samples, phase="validation", network=None):
    if network is None:
        A("failed precondition")("no active network")
    cnn = network["pytorch network"].eval().cuda()
    iterator = A("get network inputs iterator for random samples from FLAIR starGAN")(n_samples, phase=phase)
    outputs = []
    for inputs_by_layer in iterator():
        outputs.append(A("select starGAN output corresponding to generated image")(cnn(inputs_by_layer)))
    outputs = A("concatenate arrays")(outputs, 0)
    return outputs


def get_network_inputs_for_random_samples(A, n_samples, batch_size=8, phase="validation"):
    # collect images
    variables = ["age", "NIHSS", "mRS"]
    ds_var_to_net_var = A("TODO")()
    dps = A("sample datapoints")(n_samples, with_replacement=True, phase=phase)
    img_paths = A("for each")(dps, "get unaugmented image path for datapoint")
    orig_imgs = A("for each")(img_paths, "load image")
    orig_imgs = A("for each get")(orig_imgs, "pixels")
    orig_imgs = torch.stack([torch.as_tensor(img).float() for img in orig_imgs],0)

    # collect variables
    def inclusion_crit(dp):
        return not A("is None or NaN")(dp["observations"]["NIHSS"]) and not A("is None or NaN")(dp["observations"]["mRS"])

    var_samples = A("sample observations of dataset variables")(n_samples=n_samples, variables=variables, dp_condition=inclusion_crit, phase=phase)

    inputs_by_net_var = {
        ds_var_to_net_var[var]: torch.as_tensor(value).float() for var,value in var_samples.items()
    }
    inputs_by_layer["image input"] = orig_imgs

    def iterator():
        for ix in range(0, len(dps), batch_size):
            batch_by_layer = {k:v[ix:ix+batch_size].cuda() for k,v in inputs_by_layer.items()}
            yield batch_by_layer
    return iterator


def save_video_of_average_subject(A, variable, save_path, duration=7,
    fps=15, text_label=True, smoothing=0, phase="validation", colored=True, network=None, dataset=None):

    N_frames = duration*fps

    # get datapoints for the grid
    def exclusion_crit(dp):
        return A("is None or NaN")(dp["observations"]["NIHSS"]) or A("is None or NaN")(dp["observations"]["mRS"])

    dps = A("get midslice datapoints for 2D dataset")(exclusion_crit, phase=phase)
    dps = A("exclude datapoints with out of range observations")(dps, variable)

    # collect variables
    var_layer, var_values, cnn, iterator_method = A(
        "get network inputs for sweep visualizations")(dps, variable, N_frames)

    # collect frames
    frames = []
    for ix in range(N_frames):
        images = []
        for const_inputs_by_layer, orig_imgs in iterator_method():
            N_dps = len(orig_imgs)
            current_values = var_values[ix].repeat(N_dps)

            inputs_by_layer = {"image input": orig_imgs,
                var_layer: current_values,
                **const_inputs_by_layer,
            }

            with torch.no_grad():
                images.append(A("select starGAN output corresponding to generated image")(cnn(inputs_by_layer)))

        images = np.concatenate(images, axis=0)
        frame_img = np.median(images, axis=0)
        frames.append(frame_img)

    A("process images and save as video")(frames, save_path, fps,
            variable, var_values, text_label)


def save_heatmap_video_of_framewise_diffs_for_random_subjects(A, variable, save_path, duration=7,
    fps=15, grid_size=(5,5), text_label=True, smoothing=0, colored=True, network=None, dataset=None):

    N_dps = A("product")(grid_size)
    N_frames = duration*fps

    # get datapoints for the grid
    def exclusion_crit(dp):
        return A("is None or NaN")(dp["observations"]["NIHSS"]) or A("is None or NaN")(dp["observations"]["mRS"])

    dps = A("get validation midslice datapoints for 2D dataset")(exclusion_crit)
    dps = A("exclude datapoints with out of range observations")(dps, variable)
    dps = A("sample collection")(dps, N_dps)

    # collect variables
    var_layer, const_inputs_by_layer, var_values, orig_imgs, cnn = A(
        "get network inputs for sweep visualizations")(dps, variable, N_frames, batch_size=None)

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

        frame_img = A("get image grid")(images, n_rows=grid_size[0], n_cols=grid_size[1])
        frames.append(frame_img)

    frames = A("get gradient of 2D images over channel dimension")(frames)

    A("process images and save as video")(frames, save_path, fps,
            variable, var_values, text_label, colored)


def save_heatmap_video_of_framewise_diffs_averaged_across_subjects(A, variable, save_path, duration=7,
    fps=15, text_label=True, smoothing=0, colored=True, network=None, dataset=None):

    N_frames = duration*fps

    # get datapoints for the grid
    def exclusion_crit(dp):
        return A("is None or NaN")(dp["observations"]["NIHSS"])  or A("is None or NaN")(dp["observations"]["mRS"])

    dps = A("get validation midslice datapoints for 2D dataset")(exclusion_crit)
    dps = A("exclude datapoints with out of range observations")(dps, variable)

    # collect variables
    var_layer, var_values, cnn, iterator_method = A(
        "get network inputs for sweep visualizations")(dps, variable, N_frames)

    # collect frames
    frames = []
    for ix in range(N_frames):
        images = []
        for const_inputs_by_layer, orig_imgs in iterator_method():
            N_dps = len(orig_imgs)
            current_values = var_values[ix].repeat(N_dps)

            inputs_by_layer = {"image input": orig_imgs,
                var_layer: current_values,
                **const_inputs_by_layer,
            }

            with torch.no_grad():
                images.append(A("select starGAN output corresponding to generated image")(cnn(inputs_by_layer)))

        images = np.concatenate(images, axis=0)
        frame_img = np.median(images, axis=0)
        frames.append(frame_img)

    frames = A("get gradient of 2D images over channel dimension")(frames)
    A("process images and save as video")(frames, save_path, fps,
            variable, var_values, text_label, colored)

def save_heatmap_video_averaged_across_subjects_for_multiple_networks(A, network_inst_paths, variable,
    save_path, duration=7, fps=15, sample_size=None, text_label=True, smoothing=0, colored=True, dataset=None):
    N_frames = duration*fps

    # get datapoints for the grid
    def exclusion_crit(dp):
        return A("is None or NaN")(dp["observations"]["NIHSS"])  or A("is None or NaN")(dp["observations"]["mRS"])

    dps = A("get validation midslice datapoints for 2D dataset")(exclusion_crit)
    dps = A("exclude datapoints with out of range observations")(dps, variable)
    if sample_size is not None:
        dps = A("sample collection")(dps, sample_size)

    grid_frames = []
    for path in network_inst_paths:
        network = A("load network")(path=path)
        A("run through neural network training checklist")()

        # collect variables
        var_layer, var_values, cnn, iterator_method = A(
            "get network inputs for sweep visualizations")(dps, variable, N_frames)

        # collect frames
        frames = []
        for ix in range(N_frames):
            images = []
            for const_inputs_by_layer, orig_imgs in iterator_method():
                N_dps = len(orig_imgs)
                current_values = var_values[ix].repeat(N_dps)

                inputs_by_layer = {"image input": orig_imgs,
                    var_layer: current_values,
                    **const_inputs_by_layer,
                }

                with torch.no_grad():
                    images.append(A("select starGAN output corresponding to generated image")(cnn(inputs_by_layer)))

            images = np.concatenate(images, axis=0)
            frame_img = np.median(images, axis=0)
            frames.append(frame_img)

        frames = A("get gradient of 2D images over channel dimension")(frames)

        if A("is a list")(frames):
            frames = np.stack(frames, axis=0)
        elif A("is an image")(frames[0]):
            frames = A("for each get")(frames, "pixels")

        frames = A("threshold image")(frames, low_percentile=0.5, high_percentile=99.5)
        if colored:
            frames = A("convert signed image to CV2 heatmap")(frames)
        else:
            frames = A("normalize image (0-1)")(frames)
            frames[:,0,0] = 0
            frames[:,-1,-1] = 1

        grid_frames.append(frames)

    n_networks = len(grid_frames)
    full_frames = []
    for frame_ix in range(N_frames):
        frame_images = [grid_frames[net_ix][frame_ix] for net_ix in A("range over")(n_networks)]
        median_image = np.median(frame_images, axis=0)
        mini_images = [A("rescale image")(img, scale=(1/n_networks, 1/n_networks)) for img in frame_images]
        mini_composite = A("concatenate images horizontally")(mini_images)
        frame = A("concatenate images vertically")([median_image, mini_composite])
        full_frames.append(frame)

    frames = A("for each")(full_frames, "instantiate image")

    if text_label:
        frames = A("add variable value to each frame")(frames, var_values, variable)

    A("convert images to video")(frames=frames, save_path=save_path, fps=fps)

    # A("process images and save as video")(frames, save_path, fps,
    #         variable, var_values, text_label, colored)



def save_heatmap_of_diffs_averaged_over_all_frames_for_random_subjects(A, variable, save_path, duration=7,
    fps=15, grid_size=(5,5), text_label=True, smoothing=0, colored=True, network=None, dataset=None):
    A("TODO")()


def save_heatmap_of_diffs_averaged_over_all_frames_and_subjects(A, path, variable):
    A("TODO")()