import os
import json
import numpy as np
import xarray as xr
import argparse
import tensorflow as tf
import nudging_training_data_created as ndc
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras import layers, Model, regularizers
import ResCNN as Res
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#======================================================
#1.normaliza and denormalize
#======================================================

def reverse_norm_m11(x_norm, x_min, x_max):
    
    x_min=np.asarray(x_min)
    x_max=np.asarray(x_max)
    return (x_norm + 1.0) / 2.0 * (x_max - x_min) + x_min

def denormalize_prediction(y_pred_norm, ds_out, var_names):
    """
    y_pred_norm: (sample, lev, 4)
    return: dict of real-valued predictions
    """
    out = {}
    for i, v in enumerate(var_names):
        vmin = np.asarray(ds_out[f"{v}_min"].values)
        vmax = np.asarray(ds_out[f"{v}_max"].values)
        if vmin.ndim == 1 and vmin.size == 1:
            vmin = vmin.item()
        if vmax.ndim == 1 and vmax.size == 1:
            vmax = vmax.item()
        out[v] = reverse_norm_m11(y_pred_norm[:, :, i], vmin, vmax)
    return out

def denormalize_truth(y_true_norm, ds_out, var_names):
    out = {}
    for i, v in enumerate(var_names):
        vmin = np.asarray(ds_out[f"{v}_min"].values)
        vmax = np.asarray(ds_out[f"{v}_max"].values)

        if vmin.ndim == 1 and vmin.size == 1:
            vmin = vmin.item()
        if vmax.ndim == 1 and vmax.size == 1:
            vmax = vmax.item()
        out[v] = reverse_norm_m11(y_true_norm[:, :, i], vmin, vmax)
    return out

#======================================================
#2.build dataset
#======================================================
def build_dataset_from_normalized_nc(
    input_file,
    output_file,
    ntimes_input=5,
    dynamic_vars=("U", "V", "T", "Q"),
    static_vars=("PHIS", "LANDFRAC"),
    add_geo_features=True,
    train_ratio=0.6,
    val_ratio=0.2,
):
    ds_in = xr.open_dataset(input_file, decode_times=False)
    ds_out = xr.open_dataset(output_file, decode_times=False)

    # -------- 动态输入 --------
    x_list = []
    for v in dynamic_vars:
        if v not in ds_in:
            raise KeyError(f"{v} 不在输入文件中")
        x_list.append(ds_in[v])   # (time, lev, grid)

    x_dyn = xr.concat(x_list, dim="var").assign_coords(var=list(dynamic_vars))
    x_dyn = x_dyn.transpose("time", "lev", "grid", "var")   # (time, lev, grid, var)

    # -------- 输出 --------
    y_list = []
    for v in dynamic_vars:
        if v not in ds_out:
            raise KeyError(f"{v} 不在输出文件中")
        y_list.append(ds_out[v])

    y_all = xr.concat(y_list, dim="var").assign_coords(var=list(dynamic_vars))
    y_all = y_all.transpose("time", "lev", "grid", "var")   # (time, lev, grid, var)

    # -------- 静态输入 --------
    static_np = None
    used_static = []
    tmp = []
    for v in static_vars:
        if v in ds_in:
            da = ds_in[v]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            tmp.append(da)
            used_static.append(v)

    if len(tmp) > 0:
        static_da = xr.concat(tmp, dim="svar").assign_coords(svar=used_static)
        static_da = static_da.transpose("grid", "svar")   # (grid, svar)
        static_np = static_da.values.astype(np.float32)

    # -------- 位置特征 --------
    geo_np = None
    if add_geo_features:
        lat_name_candidates = ["lat_grid", "grid_lat", "sample_lat", "lat_sel"]
        lon_name_candidates = ["lon_grid", "grid_lon", "sample_lon", "lon_sel"]

        lat_grid = None
        lon_grid = None

        for name in lat_name_candidates:
            if name in ds_in:
                lat_grid = ds_in[name].values
                break
        for name in lon_name_candidates:
            if name in ds_in:
                lon_grid = ds_in[name].values
                break

        if (lat_grid is not None) and (lon_grid is not None):
            lat_rad = np.deg2rad(lat_grid)
            lon_rad = np.deg2rad(lon_grid)
            geo_np = np.stack(
                [
                    lat_grid / 90.0,
                    lon_grid / 180.0,
                    np.sin(lat_rad),
                    np.cos(lat_rad),
                    np.sin(lon_rad),
                    np.cos(lon_rad),
                ],
                axis=-1
            ).astype(np.float32)

    x_np = x_dyn.values   # (time, lev, grid, var)
    y_np = y_all.values   # (time, lev, grid, var)

    ntime, nlev, ngrid, nvar = x_np.shape
    if ntime < ntimes_input:
        raise ValueError(f"ntime={ntime} < ntimes_input={ntimes_input}")

    X_list = []
    Y_list = []
    time_index = []
    grid_index = []

    for t in range(ntimes_input - 1, ntime):
        # 输入时间窗
        x_win = x_np[t - ntimes_input + 1:t + 1]  # (window, lev, grid, var)
        y_t   = y_np[t]                            # (lev, grid, var)

        # -> (grid, window, lev, var)
        x_win = np.transpose(x_win, (2, 0, 1, 3))

        # 静态特征复制到每个时间步和每层
        feats = [x_win]
        if static_np is not None:
            s = np.repeat(static_np[:, None, None, :], ntimes_input, axis=1)
            s = np.repeat(s, nlev, axis=2)   # (grid, window, lev, static)
            feats.append(s)
        if geo_np is not None:
            g = np.repeat(geo_np[:, None, None, :], ntimes_input, axis=1)
            g = np.repeat(g, nlev, axis=2)   # (grid, window, lev, geo)
            feats.append(g)

        x_win = np.concatenate(feats, axis=-1)   # (grid, window, lev, feat)

        # 输出 -> (grid, lev, var)
        y_t = np.transpose(y_t, (1, 0, 2))

        X_list.append(x_win.astype(np.float32))
        Y_list.append(y_t.astype(np.float32))
        time_index.append(np.full((ngrid,), t))
        grid_index.append(np.arange(ngrid))

    X = np.concatenate(X_list, axis=0)  # (sample, window, lev, feat)
    Y = np.concatenate(Y_list, axis=0)  # (sample, lev, 4)
    time_index = np.concatenate(time_index, axis=0)
    grid_index = np.concatenate(grid_index, axis=0)

    # ---- 按时间块切分，避免泄漏 ----
    valid_times = np.arange(ntimes_input - 1, ntime)
    n_valid = len(valid_times)
    n_train_t = int(n_valid * train_ratio)
    n_val_t   = int(n_valid * val_ratio)

    train_times = valid_times[:n_train_t]
    val_times   = valid_times[n_train_t:n_train_t + n_val_t]
    test_times  = valid_times[n_train_t + n_val_t:]

    train_mask = np.isin(time_index, train_times)
    val_mask   = np.isin(time_index, val_times)
    test_mask  = np.isin(time_index, test_times)

    meta = {
        "nlev": nlev,
        "n_input_feat": X.shape[-1],
        "n_output_vars": Y.shape[-1],
        "time_index": time_index,
        "grid_index": grid_index,
        "used_dynamic_vars": list(dynamic_vars),
        "used_static_vars": used_static,
    }

    return (
        X[train_mask], Y[train_mask],
        X[val_mask],   Y[val_mask],
        X[test_mask],  Y[test_mask],
        ds_in, ds_out, meta
    )

#======================================================
#3.train and evaluate
#======================================================
def plot_training_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    hist = history.history

    # -------- Loss 曲线 --------
    if "loss" in hist:
        plt.figure(figsize=(8, 5))
        plt.plot(hist["loss"], label="train_loss", linewidth=1.8)
        if "val_loss" in hist:
            plt.plot(hist["val_loss"], label="val_loss", linewidth=1.8)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training / Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
        plt.close()

    # -------- RMSE 曲线 --------
    # Keras metrics 里通常是 mean_squared_error
    if "mean_squared_error" in hist:
        train_rmse = np.sqrt(np.array(hist["mean_squared_error"]))
        plt.figure(figsize=(8, 5))
        plt.plot(train_rmse, label="train_rmse", linewidth=1.8)

        if "val_mean_squared_error" in hist:
            val_rmse = np.sqrt(np.array(hist["val_mean_squared_error"]))
            plt.plot(val_rmse, label="val_rmse", linewidth=1.8)

        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Training / Validation RMSE")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rmse_curve.png"), dpi=150)
        plt.close()
import h5py
import xarray as xr


def weights_file_to_nc(h5_file, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(h5_file):
        print(f"[skip] h5 file not found: {h5_file}")
        return

    print(f"Converting weights from: {h5_file}")

    # 先列出顶层 groups
    with h5py.File(h5_file, "r") as f:
        groups = [name for name in f.keys()]

    print("Top-level groups:", groups)

    exported = 0

    for group in groups:
        if ("conv" not in group) and ("dense" not in group):
            continue

        # 你原来的旧写法是 group=group+"/"+group
        # 这里保留并加 try/except
        group_candidates = [
            f"{group}/{group}",
            group,
        ]

        opened = False
        for g in group_candidates:
            try:
                ds = xr.open_dataset(h5_file, group=g)
                print(f"[ok] group={g}")
                print(ds)

                out_nc = os.path.join(out_dir, f"Model_weights_{group}.nc")
                ds.to_netcdf(out_nc)
                print(f"[saved] {out_nc}")
                exported += 1
                opened = True
                break
            except Exception as e:
                print(f"[skip] failed to open group={g} in {h5_file}: {e}")

        if not opened:
            print(f"[warn] layer group {group} could not be exported.")

    print(f"Finished exporting {exported} layers from {h5_file}")

def compile_and_train(model, X_train, Y_train, X_val, Y_val,
                      lr=1e-4, batch_size=256, epochs=100, out_dir="./ResCNN_results"):
    os.makedirs(out_dir, exist_ok=True)
    if model == "2d_conv":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=Res.weighted_mse,#loss=tf.keras.losses.Huber(),
            metrics=[tf.keras.metrics.MeanSquaredError()]
        )
    else:
        model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
        )
    best_h5=os.path.join(out_dir,"best_model.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=4, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_h5,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

    plot_training_history(history, out_dir)

    # 保存最终模型
    final_h5 = os.path.join(out_dir, "final_model.h5")
    model.save(final_h5)
    print(f"Saved final model to: {final_h5}")

    # =========================
    # 新增：导出逐层 nc 权重文件
    # =========================
    weights_nc_dir_best = os.path.join(out_dir, "weights_nc_best")
    weights_nc_dir_final = os.path.join(out_dir, "weights_nc_final")

    print("Exporting best_model.h5 layer weights to nc ...")
    weights_file_to_nc(best_h5, weights_nc_dir_best)

    print("Exporting final_model.h5 layer weights to nc ...")
    weights_file_to_nc(final_h5, weights_nc_dir_final)

    return history


def plot_bar_metric(metric_dict, title, ylabel, out_png):
    names = list(metric_dict.keys())
    vals = [metric_dict[k] for k in names]

    plt.figure(figsize=(7, 5))
    plt.bar(names, vals)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_profile(profile_dict, lev_values, title, xlabel, out_png):
    plt.figure(figsize=(6, 8))
    for name, prof in profile_dict.items():
        plt.plot(prof, lev_values, marker="o", label=name)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel("lev")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_scatter_true_pred(y_true, y_pred, var_name, out_png, max_points=20000):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    if len(yt) > max_points:
        idx = np.random.choice(len(yt), size=max_points, replace=False)
        yt = yt[idx]
        yp = yp[idx]

    vmin = min(yt.min(), yp.min())
    vmax = max(yt.max(), yp.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, s=4, alpha=0.3)
    plt.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.2)
    plt.xlabel(f"True {var_name}")
    plt.ylabel(f"Predicted {var_name}")
    plt.title(f"{var_name}: True vs Predicted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def evaluate_model(model, X_test, Y_test, ds_out, out_dir, var_names=("U","V","T","Q")):
    os.makedirs(out_dir, exist_ok=True)

    Y_pred = model.predict(X_test, batch_size=256)

    metrics_out = {}

    # =========================
    # 1. overall 指标（归一化空间）
    # =========================
    metrics_out["overall_r2_norm"] = float(r2_score(Y_test.reshape(-1), Y_pred.reshape(-1)))
    metrics_out["overall_rmse_norm"] = float(np.sqrt(mean_squared_error(Y_test.reshape(-1), Y_pred.reshape(-1))))
    metrics_out["overall_mae_norm"] = float(mean_absolute_error(Y_test.reshape(-1), Y_pred.reshape(-1)))

    # =========================
    # 2. 按变量评估（归一化空间）
    # =========================
    by_var_norm = {}
    for i, v in enumerate(var_names):
        yt = Y_test[:, :, i].reshape(-1)
        yp = Y_pred[:, :, i].reshape(-1)
        by_var_norm[v] = {
            "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
        }
    metrics_out["by_variable_norm"] = by_var_norm

    # RMSE / R2 柱状图
    plot_bar_metric(
        {v: by_var_norm[v]["rmse"] for v in var_names},
        title="RMSE by Variable (Normalized Space)",
        ylabel="RMSE",
        out_png=os.path.join(out_dir, "rmse_by_variable_norm.png")
    )
    plot_bar_metric(
        {v: by_var_norm[v]["r2"] for v in var_names},
        title="R2 by Variable (Normalized Space)",
        ylabel="R2",
        out_png=os.path.join(out_dir, "r2_by_variable_norm.png")
    )

    # =========================
    # 3. 分层评估（归一化空间）
    # =========================
    r2_by_level = {}
    rmse_by_level = {}

    for i, v in enumerate(var_names):
        r2_lev = []
        rmse_lev = []
        for k in range(Y_test.shape[1]):
            yt = Y_test[:, k, i]
            yp = Y_pred[:, k, i]

            rmse_lev.append(float(np.sqrt(mean_squared_error(yt, yp))))
            if np.std(yt) == 0:
                r2_lev.append(np.nan)
            else:
                r2_lev.append(float(r2_score(yt, yp)))

        r2_by_level[v] = r2_lev
        rmse_by_level[v] = rmse_lev

    metrics_out["r2_by_level_norm"] = r2_by_level
    metrics_out["rmse_by_level_norm"] = rmse_by_level

    lev_values = ds_out["lev"].values if "lev" in ds_out.coords else np.arange(Y_test.shape[1])

    plot_profile(
        r2_by_level,
        lev_values,
        title="R2 Profile by Level (Normalized Space)",
        xlabel="R2",
        out_png=os.path.join(out_dir, "r2_profile_norm.png")
    )
    plot_profile(
        rmse_by_level,
        lev_values,
        title="RMSE Profile by Level (Normalized Space)",
        xlabel="RMSE",
        out_png=os.path.join(out_dir, "rmse_profile_norm.png")
    )

    # =========================
    # 4. 散点图（归一化空间）
    # =========================
    for i, v in enumerate(var_names):
        plot_scatter_true_pred(
            Y_test[:, :, i],
            Y_pred[:, :, i],
            var_name=v,
            out_png=os.path.join(out_dir, f"scatter_{v}_norm.png")
        )

    # =========================
    # 5. 反归一化后评估
    # =========================
    y_true_real = denormalize_truth(Y_test, ds_out, var_names=var_names)
    y_pred_real = denormalize_prediction(Y_pred, ds_out, var_names=var_names)

    by_var_real = {}
    for v in var_names:
        yt = y_true_real[v].reshape(-1)
        yp = y_pred_real[v].reshape(-1)
        by_var_real[v] = {
            "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
        }

    metrics_out["by_variable_real"] = by_var_real

    plot_bar_metric(
        {v: by_var_real[v]["rmse"] for v in var_names},
        title="RMSE by Variable (Real Space)",
        ylabel="RMSE",
        out_png=os.path.join(out_dir, "rmse_by_variable_real.png")
    )
    plot_bar_metric(
        {v: by_var_real[v]["r2"] for v in var_names},
        title="R2 by Variable (Real Space)",
        ylabel="R2",
        out_png=os.path.join(out_dir, "r2_by_variable_real.png")
    )

    for v in var_names:
        plot_scatter_true_pred(
            y_true_real[v],
            y_pred_real[v],
            var_name=f"{v}_real",
            out_png=os.path.join(out_dir, f"scatter_{v}_real.png")
        )
    # =========================
    # 6. 保存结果
    # =========================
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(out_dir, "Y_pred_norm.npy"), Y_pred)
    np.save(os.path.join(out_dir, "Y_test_norm.npy"), Y_test)

    return metrics_out

# =========================================================
# 4. main
# =========================================================
data_path=r"/data2/share/llj/lwh/ML_code/lwh_NN_model/normalized_output/"
inputdata=r"no_nudging_2deg_2010-01_days_01_ntimesinput_2_inputdata_Resampling_selectnum_600_randomseed_10000_trop_0.5.subtrop_0.4.polar_0.1_nudging_plotlev_normalized.nc"
outputdata=r"test_nudging_2deg_2010-01_days_01_outputdata_Resampling_selectnum_600_randomseed_10000_trop_0.5.subtrop_0.4.polar_0.1_nudging_plotlev_normalized.nc"
def build_run_name(args):
    lr_str = f"{args.learning_rate:.0e}" if args.learning_rate < 1e-3 else str(args.learning_rate)
    run_name = (
        f"model_{args.model_type}"
        f"_seq{args.ntimes_input}"
        f"_lr{lr_str}"
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}_vertnorm_newresRNN"
    )
    return run_name
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default=os.path.join(data_path,inputdata))
    parser.add_argument("--output_file", type=str,
                        default=os.path.join(data_path,outputdata))
    parser.add_argument("--model_type", type=str, default="rescnn", choices=["mlp", "rescnn", "gru","cnn","2d_conv"])
    parser.add_argument("--ntimes_input", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()
    run_name = build_run_name(args)
    out_dir = os.path.join(data_path, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print("run_name =", run_name)
    print("out_dir  =", out_dir)


    print("Loading and building dataset ...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test, ds_in, ds_out, meta = \
        build_dataset_from_normalized_nc(
            input_file=args.input_file,
            output_file=args.output_file,
            ntimes_input=args.ntimes_input,
            dynamic_vars=("U","V","T","Q"),
            static_vars=("PHIS","LANDFRAC"),
            add_geo_features=True,
        )

    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_val  :", X_val.shape)
    print("Y_val  :", Y_val.shape)
    print("X_test :", X_test.shape)
    print("Y_test :", Y_test.shape)
    print("meta   :", meta)

    ntimes_input = X_train.shape[1]
    nlev = X_train.shape[2]
    nfeat = X_train.shape[3]
    out_vars = Y_train.shape[-1]

    print("Building model ...")
    if args.model_type == "mlp":
        model = Res.build_mlp_model(ntimes_input, nlev, nfeat, out_vars=out_vars)
    elif args.model_type == "rescnn":
        model = Res.build_rescnn_model(ntimes_input, nlev, nfeat, out_vars=out_vars)
    elif args.model_type == "gru":
        model = Res.build_gru_model(ntimes_input, nlev, nfeat, out_vars=out_vars)
    elif args.model_type == "cnn":
        model = Res.zm_model( ntimes_input, nlev, nfeat,out_vars=out_vars,
            num_filter=128,kernel_size=3,num_blocks=6,fn_size=16,bn=False)
    elif args.model_type == "2d_conv":
        model = Res.build_2dconv_model( nlev, nfeat, out_vars=out_vars)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model.summary()

    print("Training ...")
    compile_and_train(
        model,
        X_train, Y_train,
        X_val, Y_val,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        out_dir=out_dir
    )

    print("Evaluating ...")
    metrics = evaluate_model(
        model,
        X_test, Y_test,
        ds_out,
        out_dir=out_dir,
        var_names=("U","V","T","Q")
    )
    config_to_save = vars(args).copy()
    config_to_save["run_name"] = run_name
    config_to_save["out_dir"] = out_dir

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)
    print("Done.")



if __name__ == "__main__":
    main()