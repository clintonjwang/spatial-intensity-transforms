import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from am.dispositio.action import Action


def get_actions(A):
    return [
        (["plot attribute distributions"], Action(plot_attribute_distributions)),

        # model performance / comparison
        (["compare all models for WMHv"], Action(compare_WMHv_all_models)),
        (["multiscale plot of relative errors for WMHv"], Action(multiscale_plot_rel_errors_WMHv)),
        (["plot relative errors for all WMHv models"], Action(compare_WMHv_model_rel_errors)),
        (["plot residuals for all WMHv models"], Action(compare_WMHv_model_residuals)),
        (["make scatterplots for all WMHv models"], Action(compare_WMHv_model_scatterplots)),
        (["make WMHv scatterplots with auto/manual segmentations"], Action(make_WMHv_scatterplots_w_segs)),
        (["compare all models for CCS"], Action(compare_CCS_all_models)),

        (["collate jobs into barplot"],
            Action(collate_jobs_into_barplot)),
    ]

def collate_jobs_into_barplot(A, jobs, hue_variable, subplot_variable):
    A("TODO")()
    table = A("new table")(["model type", "dataset", "metric", "value"])
    kwargs = {"categorical_variable":"model type", "height_variable":"value",
            "hue_variable":hue_variable, "table":table}
    axes = A("subplots")(2,2)
    A("barplot from table")(axis=axes[0][0], **kwargs)
    A("set plot title")("", axis=axes[0][0])

    A("barplot from table")(**kwargs, axis=axes[0][1])
    A("barplot from table")(**kwargs, axis=axes[1][0])
    A("barplot from table")(**kwargs, axis=axes[1][1])

def plot_predictor_performances(A):
    return
    # A("load dataset")()
    # A("set up MRI-GENIE pipeline")()
    # #A("save dataset")("CCS prediction")

    # dataset = A["active dataset"]
    # predictors = ['sex', 'HTN', 'diabetes', 'AFib', 'CAD', 'first stroke', 'smoking', 'race', 'age']
    # outcome = target_variable = "CCS"
    # group_attribute = "site"

    # cls_types = ["dummy", "random forest", "gaussian process"]
    # comparison = A("compare classifiers")(cls_types, predictors, outcome)

    # A("new table")(cls_types)

    # A("load dataset")("WMHv prediction")
    # A("set up MRI-GENIE pipeline")()
    # #A("save dataset")("CCS prediction")

    # dataset = A["active dataset"]
    # predictors = ['sex', 'HTN', 'diabetes', 'AFib', 'CAD', 'first stroke', 'smoking', 'race', 'age', "CCS"]
    # outcome = target_variable = "log(WMHv+1)"
    # group_attribute = "site"

    # mae_inv_ln_1plus = lambda X,Y: np.mean(np.abs(np.exp(X)-np.exp(Y)))
    # metrics={"MSE": "MSE", "MAE": "MAE", "true MAE": mae_inv_ln_1plus}
    # model_types = ["dummy", "least squares", "ARD", "random forest"]
    # #comparison = A("compare regressors")(model_types, predictors, outcome, metrics=metrics)

    # A("add model performance to comparison")(summary, comparison)

    # A("plot model comparison")(comparison)
    # sklearn_models = model_names = model_types
    # test_scheme = "cross-validation"
    # n_splits = 5
    # function = res_inv_ln_1plus

    # labels, predictions = A("get labels and predictions under sklearn cross-validation")(model_names, predictors, outcome, n_splits, dataset=dataset)

    # sklearn_models, score = A("format cross-validation arguments")(sklearn_models, predictors, outcome, loss, metrics, test_scheme)

    # residuals = A("for each")(sklearn_models, lambda model: function(y, model_selection.cross_val_predict(model["scikit object"], X, y, cv=n_splits)))

    # abs_residuals = np.abs(np.array(residuals))

    # _,axis = A("new figure")(size="medium")
    # for empirical_dist in abs_residuals:
    #     sns.distplot(empirical_dist, hist=False, rug=True, ax=axis)
    # plt.vlines(0,*axis.get_ylim(), linestyles="dashed", lw=1)
    # plt.legend(axis.lines, A("for each")(model_types, "to title case"))
    # A("label axes")(xlabel="Residuals")

    # axis = A("overlay empirical distributions")(residuals
    # A("set axis bounds")(xlow=0, axis=axis)

    # res_inv_ln_1plus = lambda X,Y: np.exp(X)-np.exp(Y)
    # A("compare residual distributions")(model_types, predictors, outcome, function=res_inv_ln_1plus)


def plot_attribute_distributions(A, dataset_name="clinical variables", folder=None,
    attributes=["NIHSS", "age", "mRS", "CCS", "diabetes", "AFib", "sex", "first stroke", "HTN", "smoking"]):

    if folder is None:
        folder = A["AIS plots folder"]
    A("load dataset")(dataset_name)
    A("constrain dataset")(mandatory_vars=["CCS"])

    for var in A("get dataset variables")():
        if A("get capitalization")(var) == "sentence case":
            A("rename dataset variable")(var, var.lower())

    A("load dataset")("CCS prediction")
    #A("constrain dataset")(mandatory_vars=["CCS"])

    A("convert dataset to table")()
    A("cast table columns")(["NIHSS", "age", "mRS"], float)

    for var in ["NIHSS", "age", "mRS"]:
        A("plot boxplot")(var, categorical_variable="CCS", path=A("join paths")(folder, var))

    for var in ["diabetes", "AFib", "sex", "first stroke", "HTN", "smoking"]:
        A("plot counts")(hue_variable=var, categorical_variable="CCS", path=A("join paths")(folder, var))

    categorical_variable="CCS"
    df = A["active df"]
    sns.set(font_scale=2.5)
    plt.subplots(figsize=(15,10))
    category_order = np.unique(df[categorical_variable])
    ax = sns.countplot(x=categorical_variable, data=df, order=category_order)
    #ax.yaxis.set_ticks([0,1,2]) [0,1,2,3]

    nobs = df.groupby([categorical_variable])[hue_variable].agg(['count'])
    nobs = ["n: " + str(i) for s in nobs.loc[category_order].values for i in s]

    pos = range(len(nobs))
    for tick,label in zip(pos,ax.get_xticklabels()):
        ax.text(pos[tick], 7, nobs[tick], horizontalalignment='center', size='x-small', color='k', weight='semibold')

    sns.set()

    path = A("join paths")(folder, "overall")
    A("export plot")(path)


#####################
### Compare models
#####################

def compare_WMHv_all_models(A, model_types=("least squares",), #"dummy", "least squares", "ARD", "SVM", "random forest"
        clin_cnn_folder="results/wmhv_pred_clin/best_models",
        flair_cnn_folder="results/wmhv_pred_flair/2NCCB5V_reps",
        both_cnn_folder="results/wmhv_pred_fc/W86FG2X_reps"):
    comparison = A("compare sklearn models for WMHv")(model_types)
    summary = A("gather repeated runs into single summary")(clin_cnn_folder)
    A("add model performance to comparison")(summary, comparison)
    summary = A("gather repeated runs into single summary")(flair_cnn_folder)
    A("add model performance to comparison")(summary, comparison)
    summary = A("gather repeated runs into single summary")(both_cnn_folder)
    A("add model performance to comparison")(summary, comparison)
    axes=A("plot model comparison")(comparison, title="WMHv prediction")

    ticks = ["Ordinary Least Squares\n(clinical vars only)",
        "MLP\n(clinical vars only)",
        "CNN\n(FLAIR only)",
        "CNN\n(FLAIR + clinical vars)"]
    # "MSE of ln(WMHv+1)", "MAE of WMHv
    #[flair_summary["model"],clin_summary["model"],both_summary["model"]]
    A("set axis ticks")(xticks=[], axis=axes[0])
    A("set axis ticks")(xticks=ticks, axis=axes[1])
    A("rotate x-axis tick labels")(axis=axes[1])

    return comparison

def compare_WMHv_model_residuals(A, model_types=("least squares",), #"dummy", "least squares", "ARD", "SVM", "random forest"
        **folders):
    residuals = A("get residuals for WMHv")(log_space=log_space, **folders)

    ticks = ["Ordinary Least Squares\n(clinical vars only)",
        "MLP\n(clinical vars only)",
        "CNN\n(FLAIR only)",
        "CNN\n(FLAIR + clinical vars)"]
    #ticks = A("list")(model_types)+ticks#[flair_summary["model"],clin_summary["model"],both_summary["model"]]
    axis = A("boxplot")(residuals)
    A("set axis ticks")(xticks=ticks, axis=axis) #A("for each")(ticks, "to title case")
    if log_space:
        A("label axes")(ylabel="WMHv Residuals (cc)", axis=axis)
    else:
        A("label axes")(ylabel="log$_2$(WMHv+1) Residuals (cc)", axis=axis)
    A("rotate x-axis tick labels")(axis=axis)
    # axis = A("overlay empirical distributions")(residuals)
    # plt.legend(axis.lines, A("for each")(legend, "to title case"))
    # A("label axes")(xlabel="Residuals")
    return residuals

def compare_WMHv_model_rel_errors(A, model_types=("least squares",), #"dummy", "least squares", "ARD", "SVM", "random forest"
        log_space=False, **folders):
    rel_errors = A("get relative errors for WMHv")(log_space=log_space, **folders)

    ticks = ["Ordinary Least Squares\n(clinical vars only)",
        "MLP\n(clinical vars only)",
        "CNN\n(FLAIR only)",
        "CNN\n(FLAIR + clinical vars)"]
    axis = A("boxplot")(rel_errors)
    A("set axis ticks")(xticks=ticks, axis=axis)
    if log_space:
        A("label axes")(ylabel="Relative Error of log$_2$(WMHv+1)", axis=axis, title_case=False)
    else:
        A("label axes")(ylabel="Relative Error in WMHv", axis=axis, title_case=False)

    A("change y-axis to percentage")(axis=axis)
    A("rotate x-axis tick labels")(axis=axis)

    return axis
    # for col in range(4):
    #     A("plot CDF")(rel_ln_errors[col], label=ticks[col])
    # plt.legend()
    # plt.xlim(0,5)

def multiscale_plot_rel_errors_WMHv(A, log_space=False, **folders):
    rel_errors = A("get relative errors for WMHv")(log_space=log_space, **folders)

    ticks = ["Ordinary Least Squares\n(clinical vars only)",
        "MLP\n(clinical vars only)",
        "CNN\n(FLAIR only)",
        "CNN\n(FLAIR + clinical vars)"]

    c = ['b','orange','g','r']
    legend_elements = [Patch(facecolor=c[i], edgecolor='k', label=ticks[i]) for i in range(4)]

    axes = A("make subplots")(1,3)
    for axis in axes:
        axis = A("boxplot")(rel_errors, axis=axis)
        axis.set_xticklabels([])

    if log_space:
        A("label axes")(ylabel="Relative Error of log$_2$(WMHv+1)", axis=axes[0], title_case=False)
    else:
        A("label axes")(ylabel="Relative Error of WMHv", axis=axes[0], title_case=False)

    A("bound axes")(ylim=(0,1.), axis=axes[0])
    A("bound axes")(ylim=(0,10), axis=axes[1])
    plt.subplots_adjust(wspace=.4)
    plt.legend(handles=legend_elements, bbox_to_anchor=(-3., -.1, 4., .1), loc='lower left',
                   ncol=4, mode="expand", borderaxespad=0., fontsize=12)


def compare_WMHv_model_scatterplots(A, model_types=("least squares",),
        clin_cnn_folder="results/wmhv_pred_clin/best_models",
        flair_cnn_folder="results/wmhv_pred_flair/2NCCB5V_reps",
        both_cnn_folder="results/wmhv_pred_fc/W86FG2X_reps"):
    if A["active dataset"] is None:
        A("load dataset")("WMHv prediction")
        A("set up MRI-GENIE pipeline")()

    predictors, outcome = A("get clinical variable names for task")("WMHv prediction")
    labels, predictions = A("get labels and predictions under sklearn cross-validation")(model_types, predictors, outcome[0], n_splits=5)
    nrows=2
    ncols=4
    axes = A("make subplots")(nrows, ncols)

    X, Y = labels, predictions[0]
    A("scatterplot")(X/np.log(2), Y/np.log(2), alpha=.35, axis=axes[0][0])
    A("scatterplot")(np.exp(X)-1, np.exp(Y)-1, alpha=.35, axis=axes[1][0])

    folders = [None, clin_cnn_folder, flair_cnn_folder, both_cnn_folder]
    for col in range(1,ncols):
        flair_summary = A("gather repeated runs into single summary")(folders[col])
        X = flair_summary["test labels"]['log(WMHv+1)'].flatten()
        Y = flair_summary["test outputs"]['log(WMHv+1)'].flatten()
        A("scatterplot")(X/np.log(2), Y/np.log(2), alpha=.35, axis=axes[0][col])
        A("scatterplot")(np.exp(X)-1, np.exp(Y)-1, alpha=.35, axis=axes[1][col])

    titles = ["Ordinary Least Squares\n(clinical vars only)",
        "MLP\n(clinical vars only)",
        "CNN\n(FLAIR only)",
        "CNN\n(FLAIR + clinical vars)"]

    for axis in axes[0]:
        A("bound axes")(xlim=(0,7), ylim=(0,7), axis=axis)
        A("add line to plot")([(0,0), (7,7)], axis=axis)
    for axis in axes[1]:
        A("bound axes")(xlim=(0,65), ylim=(0,110), axis=axis)
        A("add line to plot")([(0,0), (65,65)], axis=axis)

    axes[0][0].set_ylabel("Predicted log$_2$(WMHv+1)", size="small")
    axes[1][0].set_ylabel("Predicted WMHv (cc)", size="small")
    for i in range(1,ncols):
        axes[0][i].set_yticklabels([]);
        axes[1][i].set_yticklabels([]);
    for i in range(ncols):
        axes[0][i].set_xlabel("True log$_2$(WMHv+1)", size="small")
        axes[1][i].set_xlabel("True WMHv (cc)", size="small")

    for i in [0,1]:
        for j in range(ncols):
            axes[i][j].set_title(titles[j])
    plt.subplots_adjust(hspace=.7)
    for i in range(ncols):
        axes[0][i].set_xticks(axes[0][i].get_yticks())
        axes[1][i].set_xticks(axes[1][i].get_yticks()[:-3])

def make_WMHv_scatterplots_w_segs(A, model_types=("least squares",),
        clin_cnn_folder="results/wmhv_pred_clin/best_models",
        flair_cnn_folder="results/wmhv_pred_flair/2NCCB5V_reps",
        both_cnn_folder="results/wmhv_pred_fc/W86FG2X_reps"):
    if A["active dataset"] is None:
        A("load dataset")("WMHv prediction")
        A("set up MRI-GENIE pipeline")()

    predictors, outcome = A("get clinical variable names for task")("WMHv prediction")
    labels, predictions = A("get labels and predictions under sklearn cross-validation")(model_types, predictors, outcome[0], n_splits=5)
    nrows=3
    ncols=4
    axes = A("make subplots")(nrows, ncols)

    X, Y = labels, predictions[0]
    A("scatterplot")(X/np.log(2), Y/np.log(2), alpha=.35, axis=axes[0][0])
    #A("scatterplot")(np.exp(X)-1, np.exp(Y)-1, alpha=.35, axis=axes[1][0])

    X_man = np.log(A("to array")(A("get variable observations")("WMH volume (manual)")) + 1)
    A("scatterplot")(X_man/np.log(2), Y/np.log(2), alpha=.35, axis=axes[1][0], color='r')
    A("scatterplot")(X_man/np.log(2), X/np.log(2), alpha=.35, axis=axes[2][0], color='g')

    folders = [None, clin_cnn_folder, flair_cnn_folder, both_cnn_folder]
    for col in range(1,ncols):
        flair_summary = A("gather repeated runs into single summary")(folders[col])
        X = flair_summary["test labels"]['log(WMHv+1)'].flatten()
        Y = flair_summary["test outputs"]['log(WMHv+1)'].flatten()
        A("scatterplot")(X/np.log(2), Y/np.log(2), alpha=.35, axis=axes[0][col])
        #A("scatterplot")(np.exp(X)-1, np.exp(Y)-1, alpha=.35, axis=axes[1][col])

        all_dps = A("get all datapoints")()
        X_man = []
        unordered_X = A("get variable observations")('log(WMHv+1)')
        for x in X:
            X_man.append(np.nan)
            for ix,dp in enumerate(all_dps):
                if abs(unordered_X[ix] - x)<1e-8:
                    X_man[-1] = np.log(A("get datapoint observation")(dp, "WMH volume (manual)") +1)
                    break
        A("scatterplot")(X_man/np.log(2), Y/np.log(2), alpha=.35, axis=axes[1][col], color='r')
        A("scatterplot")(X_man/np.log(2), X/np.log(2), alpha=.35, axis=axes[2][col], color='g')
        # X_man = A("get observation for datapoints")("WMHv (manual)", dps)


    titles = ["Ordinary Least Squares\n(clinical vars only)",
        "MLP\n(clinical vars only)",
        "CNN\n(FLAIR only)",
        "CNN\n(FLAIR + clinical vars)"]

    for axis in axes.flatten():
        A("bound axes")(xlim=(0,7), ylim=(0,7), axis=axis)
        A("add line to plot")([(0,0), (7,7)], axis=axis)

    axes[0][0].set_ylabel("Predicted log$_2$(WMHv+1)", size="small")
    axes[1][0].set_ylabel("Predicted log$_2$(WMHv+1)", size="small")
    axes[2][0].set_ylabel("Auto log$_2$(WMHv+1)", size="small")
    for i in range(ncols):
        axes[0][i].set_xlabel("Auto log$_2$(WMHv+1)", size="small")
        axes[1][i].set_xlabel("Manual log$_2$(WMHv+1)", size="small")
        axes[2][i].set_xlabel("Manual log$_2$(WMHv+1)", size="small")
    for i in range(1,ncols):
        axes[0][i].set_yticklabels([]);
        axes[1][i].set_yticklabels([]);
        axes[2][i].set_yticklabels([]);

    for i in [0,1]:
        for j in range(ncols):
            axes[i][j].set_title(titles[j])
    plt.subplots_adjust(hspace=.7)
    for i in range(ncols):
        axes[0][i].set_xticks(axes[0][i].get_yticks())
        axes[1][i].set_xticks(axes[1][i].get_yticks()[:-3])


# def compare_WMHv_model_scatterplots(A, model_types=("least squares",),
#         flair_cnn_folder="results/wmhv_pred_flair/2NCCB5V_reps",
#         clin_cnn_folder="results/wmhv_pred_clin/W86FG2X_reps"):
#     A("new table")(['x','y', 'Model Type', 'Domain'])

#     predictors, outcome = A("get clinical variable names for task")("WMHv prediction")
#     labels, predictions = A("get labels and predictions under sklearn cross-validation")(model_types, predictors, outcome, n_splits=5)
#     X, Y = labels, predictions[0]

#     rows = list(zip(X,Y))
#     A("add rows to table")([[x,y,"Least Squares", "ln(WMHv+1)"] for x,y in rows])
#     A("add rows to table")([[np.exp(x)-1, np.exp(y)-1, "Least Squares", "WMHv (cc)"] for x,y in rows])

#     flair_summary = A("gather repeated runs into single summary")(flair_cnn_folder)
#     X = flair_summary["test labels"]['log(WMHv+1)'].flatten()
#     Y = flair_summary["test outputs"]['log(WMHv+1)'].flatten()

#     rows = list(zip(X,Y))
#     A("add rows to table")([[x,y,"CNN using FLAIR only", "ln(WMHv+1)"] for x,y in rows])
#     A("add rows to table")([[np.exp(x)-1, np.exp(y)-1, "CNN using FLAIR only", "WMHv (cc)"] for x,y in rows])

#     flair_summary = A("gather repeated runs into single summary")(clin_cnn_folder)
#     X = flair_summary["test labels"]['log(WMHv+1)'].flatten()
#     Y = flair_summary["test outputs"]['log(WMHv+1)'].flatten()

#     rows = list(zip(X,Y))
#     A("add rows to table")([[x,y,"CNN using FLAIR + clinical vars", "ln(WMHv+1)"] for x,y in rows])
#     A("add rows to table")([[np.exp(x)-1, np.exp(y)-1, "CNN using FLAIR + clinical vars", "WMHv (cc)"] for x,y in rows])

#     g = sns.FacetGrid(col="Model Type",  row="Domain", data=A["active table"])
#     g = g.map(plt.scatter, "X", "Y", alpha=.8)


def compare_CCS_all_models(A, model_types=("dummy", "KNN", "SVM", "random forest", "gaussian process"),
        flair_cnn_folder="results/ccs_pred_flair"):
    comparison = A("compare sklearn models for CCS")(model_types)
    #full_summary = A("gather repeated runs into single summary")(flair_cnn_folder)
    #A("add model performance to comparison")(full_summary, comparison)
    A("plot model comparison")(comparison, title="CCS prediction")
    return comparison




def vis_kernel(model):
    if not exists('vis/filters'):
        os.makedirs('vis/filters')

    layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    for ix,layer in enumerate(layers):
        K = layer.kernel_size[0]
        if K < 3:
            continue
        w = layer.weight.view(-1,K,K).detach().cpu().numpy()
        w = np.pad(w, (0,1), 'constant', constant_values=np.percentile(w,1))[:-1]
        np.random.shuffle(w)
        w = w.reshape(-1,K+1)
        w = np.concatenate([w[ix*10*(K+1):(ix+1)*10*(K+1)] for ix in range(0,10)], 1)
        img = A("rescale image")(w[:10*(K+1)], 10)
        A("save image")('vis/filters/%d.png' % ix, img, cmap='gray')

def vis_activations(model, layer, data_loader):
    if not exists('vis/activations'):
        os.makedirs('vis/activations')

    num_features = model.get_features(torch.empty((1,*C.dims)).cuda(), layer).size(1)

    fused_img = [np.empty((0,num_features)) for _ in range(A["number of classes"])]
    for imgs, yu in data_loader:
        mRS = yu[:,0].byte().numpy()
        age = yu[:,1].byte().numpy()
        z = model.get_features(imgs.cuda(), layer)
        z = z.mean(-1).mean(-1).detach().cpu().numpy()
        for ix in range(imgs.size(0)):
            fused_img[mRS[ix]] = np.concatenate([fused_img[mRS[ix]], z[ix:ix+1]])

    for cls_num in range(A["number of classes"]):
        fused_img[cls_num] = np.stack(sorted(fused_img[cls_num], key=lambda x: x.std()),1)
        img = A("rescale image")(fused_img[cls_num], 3)
        A("save image")('vis/activations/%s_mRS%d.png' % (layer,cls_num), img)

    A("save image")('vis/%s_all.png' % layer, A("rescale image")(np.concatenate(fused_img,1), 3))


def vis_inputs(data_loader):
    [os.makedirs('vis/all_imgs/mRS%d'%ix) for ix in range(6) if not exists('vis/all_imgs/mRS%d'%ix)];

    for imgs, yu in data_loader:
        mRS = yu[:,0].numpy()
        accnum = yu[:,-1].numpy()
        x = imgs[:,[0,6,11,16,21,26]]
        x = x.view(x.size(0), -1, 160)
        x = torch.cat([x[:,:160*3], x[:,160*3:]], 2)
        x = x.numpy().transpose((0,2,1))

        for ix in range(imgs.size(0)):
            plt.imsave('vis/all_imgs/mRS%d/%d.png'%(mRS[ix],accnum[ix]), x[ix], cmap='gray')

def split_gpus(img_set, clinvars, accnums):
    n_gpus = torch.cuda.device_count()
    sizes = [I[:8].nelement() for I in img_set]
    ixs = np.argsort(sizes)
    sizes = np.sort(sizes)
    partition_size = sizes[-1]*3
    partit_ixs = [0]

        #if len(sizes)<n_gpus:
        #    partit_ixs = list(range(len(sizes)+1))
    if len(sizes)<8 and n_gpus==4:
        partit_ixs = [0,len(sizes)-3,len(sizes)-2,len(sizes)-1,len(sizes)]
    else:
        for _ in range(n_gpus):
            last_ix = partit_ixs[-1]
            A = [sizes[ix]*(ix-last_ix+1) for ix in range(last_ix, len(sizes)-6)] + [1e9]
            partit_ixs.append(max(1,np.where(A > partition_size)[0][0]) + last_ix)

            if len(partit_ixs) == n_gpus-1:
                partit_ixs.append(len(sizes)-3)
                partit_ixs.append(len(sizes))
                break

    imgs = [img_set[ix][:8].permute(1,0,2,3) for ix in ixs]
    shapes = [I.shape for I in imgs]
    clinvars = clinvars[ixs]
    mRS = clinvars[:,0].cuda()
    clinvars = [clinvars[partit_ixs[g_ix]:partit_ixs[g_ix+1], 1:].cuda() for g_ix in range(n_gpus)]
    accnums = np.array(accnums)[ixs]

    imgs_by_gpu = []
    for g_ix in range(len(partit_ixs)-1):
        l_ix, r_ix = partit_ixs[g_ix:g_ix+2]
        max_size = np.max(shapes[l_ix:r_ix], 0)

        pads = [((max_size[-1]-sh[-1])//2, (max_size[-1]-sh[-1]+1)//2,
                 (max_size[-2]-sh[-2])//2, (max_size[-2]-sh[-2]+1)//2,
                 (max_size[-3]-sh[-3])//2, (max_size[-3]-sh[-3]+1)//2) for sh in shapes[l_ix:r_ix]]
        pads = torch.tensor(pads, dtype=torch.long)
        tmp = [F.pad(imgs[ix], pads[ix-l_ix]).permute(1,0,2,3) for ix in range(l_ix,r_ix)]
        imgs_by_gpu.append(torch.stack(tmp,0).cuda().float())

    return imgs_by_gpu, clinvars, mRS, accnums