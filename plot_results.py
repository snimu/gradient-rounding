
import ast
from typing import Literal

import colorsys
import matplotlib.pyplot as plt
import polars as pl
import numpy as np


def close_plt() -> None:
    plt.cla()
    plt.clf()
    plt.close()


def series_to_array(series: pl.Series) -> np.ndarray:
    try:
        return np.array(ast.literal_eval(series[0]))
    except SyntaxError:
        return np.array(ast.literal_eval(series))


def format_num_params(num_params: int, round_to_digits: int = 1) -> str:
    if num_params < 1_000:
        pnum = str(round(num_params, max(0, round_to_digits)))
        scalar = ""
    elif num_params < 1_000_000:
        pnum = f"{round(num_params/1_000, max(0, round_to_digits))}"
        scalar = "k"
    elif num_params < 1_000_000_000:
        pnum = f"{round(num_params/1_000_000, max(0, round_to_digits))}"
        scalar = "M"
    else:
        pnum = f"{round(num_params/1_000_000_000, max(0, round_to_digits))}"
        scalar = "B"

    before_dot = pnum.split(".")[0]
    after_dot = pnum.split(".")[1] if "." in pnum else ""
    after_dot = "" if after_dot and (round_to_digits <= 0) else after_dot
    after_dot = "" if after_dot and (int(after_dot) == 0) else after_dot
    after_dot = "." + after_dot if after_dot else ""

    return f"{before_dot}{after_dot}{scalar}"


def load_xs_ys_avg_y(
        file: str,
        model_scale: float | None = None,
        depth: int | None = None,
        width: int | None = None,
        gradient_rounding_digits: int | None = None,
        num_params: int | None = None,
        linear_value: bool | None = None,
        num_heads: int | None = None,
        run_num: int | None = None,
        seed: int | None = None,
        to_plot: Literal["val_loss", "train_loss", "val_accs", "train_accs", "val_pplxs", "train_pplxs"] = "val_loss",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, and average y from a CSV file."""
    filters = (pl.col("seed").ge(0))  # initial condition -> always true

    if model_scale is not None:
        filters &= (pl.col("model_scale") == model_scale)
    if depth is not None:
        filters &= (pl.col("depth") == depth)
    if width is not None:
        filters &= (pl.col("width") == width)
    if gradient_rounding_digits is not None:
        filters &= (pl.col("gradient_rounding_digits") == gradient_rounding_digits)
    if num_params is not None:
        filters &= (pl.col("num_params") == num_params)
    if linear_value is not None:
        filters &= (pl.col("linear_value") == linear_value)
    if num_heads is not None:
        filters &= (pl.col("num_heads") == num_heads)
    if run_num is not None:
        filters &= (pl.col("run_num") == run_num)
    if seed is not None:
        filters &= (pl.col("seed") == seed)

    df = pl.scan_csv(file).filter(filters).collect()
    df.sort("run_num")
    arrays = [series_to_array(df[to_plot][i]) for i in range(len(df[to_plot]))]

    if plot_over == "step":
        return load_steps_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "epoch":
        return load_epochs_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "token":
        return load_tokens_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "time_sec":
        return load_time_ys_avg_ys(df, arrays, to_plot)
    else:
        raise ValueError(f"{plot_over} not a valid x-value")


def load_steps_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_len = min([len(a) for a in arrays])
    ys = np.array([list(a[:min_len]) for a in arrays])
    num_datapoints = len(ys[0])

    if "train" in to_plot:
        xs = ((np.arange(num_datapoints) + 1) * 12.5).astype(int)
    elif "val" in to_plot:
        xs = (np.arange(num_datapoints) + 1) * 50

    avg_ys = np.mean(ys, axis=0)

    return xs, ys, avg_ys


def load_epochs_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs_str = "epochs_train" if "train" in to_plot else "epochs_val"
    xs = [series_to_array(df[epochs_str][i]) for i in range(len(df[epochs_str]))]
    return interpolate_linearly(xs, arrays)


def load_tokens_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens_str = "tokens_seen_train" if "train" in to_plot else "tokens_seen_val"
    xs = [series_to_array(df[tokens_str][i]) for i in range(len(df[tokens_str]))]
    return interpolate_linearly(xs, arrays)


def load_time_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert "val" in to_plot, "Only validation data has time data"
    time_str = "cumulative_time"
    xs = [series_to_array(df[time_str][i]) for i in range(len(df[time_str]))]
    return interpolate_linearly(xs, arrays)


def interpolate_linearly(
        xs: list[np.ndarray], ys: list[np.ndarray], num_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Determine the maximum x value across all datasets
    max_x = max(x_vals.max() for x_vals in xs)
    
    # Generate a single set of new x values for all datasets
    new_x_vals = np.linspace(0, max_x, num_samples)

    new_ys = []
    for x_vals, y_vals in zip(xs, ys):
        # Interpolate y to the common set of new x values
        new_y_vals = np.interp(new_x_vals, x_vals, y_vals)
        new_ys.append(new_y_vals)

    # Convert new_ys to a 2D numpy array for easy manipulation
    new_ys = np.array(new_ys)
    
    # Calculate the average y values across all datasets
    avg_ys = np.nanmean(new_ys, axis=0)

    return new_x_vals, new_ys, avg_ys


def get_unique_settings(file: str, targets: list[str]) -> list[str | int | float | bool]:
    settings = []
    
    # Load the unique combinations of the targets
    combinations = (
        pl.scan_csv(file)
        .select(*[pl.col(target) for target in targets])
        .collect()
        .unique()
    )
    # Sort combinations alphabetically by content, target by target (for consistency in plotting)
    for target in targets:
        combinations = combinations.sort(target)
    # Create a list of settings
    for features in zip(
            *[combinations[target] for target in targets]
    ):
        settings.append(tuple(features))

    return settings


def generate_distinct_colors(n):
    """
    Generates n visually distinct colors.

    Parameters:
        n (int): The number of distinct colors to generate.

    Returns:
        list: A list of n visually distinct colors in hex format.
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Fixing saturation and lightness/value to 0.9 for bright colors
        # You can adjust these values for different color variations
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    
    return colors


def unique_num_params(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("num_params")
        .collect()
        ["num_params"]
        .unique()
        .to_numpy()
    )


def unique_widths(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("width")
        .collect()
        ["width"]
        .unique()
        .to_numpy()
    )


def unique_depths(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("depth")
        .collect()
        ["depth"]
        .unique()
        .to_numpy()
    )


def unique_gradient_rounding_digits(file: str) -> np.ndarray:
    return (
        pl.scan_csv(file)
        .select("gradient_rounding_digits")
        .collect()
        ["gradient_rounding_digits"]
        .unique()
        .to_numpy()
    )


def plot_performance(
        file: str,
        depth: int | None = 8,
        width: int | None = 384,
        num_heads: int | None = None,
        gradient_rounding_digits: int | list[int] | None = 16,
        from_percentage: float = 0.0,
        to_plot: Literal["val_loss", "train_loss", "val_accs", "train_accs", "val_pplxs"] = "val_loss",
        plot_over: Literal["step", "epoch", "epoch_unique_token", "token", "time_sec"] = "epoch",
        show: bool = True,
        loglog: bool = False,
        plot_all: bool = False,
) -> None:
    settings = get_unique_settings(file, ["num_heads", "depth", "width", "gradient_rounding_digits"])

    if num_heads is not None:
        settings = [(nh, d, w, rd) for nh, d, w, rd in settings if nh == num_heads]
    if depth is not None:
        settings = [(nh, d, w, rd) for nh, d, w, rd in settings if d == depth]
    if width is not None:
        settings = [(nh, d, w, rd) for nh, d, w, rd in settings if w == width]
    if gradient_rounding_digits is not None:
        gradient_rounding_digits = [gradient_rounding_digits] if isinstance(gradient_rounding_digits, int) else gradient_rounding_digits
        settings = [(nh, d, w, rd) for nh, d, w, rd in settings if rd in gradient_rounding_digits]

    colors = generate_distinct_colors(len(settings))

    for color, (num_heads_, depth_, width_, gradient_rounding_digits_) in zip(colors, settings):
            xs, ys, avg_ys = load_xs_ys_avg_y(
                file,
                depth=depth_,
                width=width_,
                num_heads=num_heads_,
                linear_value=False,
                gradient_rounding_digits=gradient_rounding_digits_,
                to_plot=to_plot,
                plot_over=plot_over,
            )

            if from_percentage > 0.0:
                from_idx = int(from_percentage * len(xs))
                xs = xs[from_idx:]
                ys = ys[:, from_idx:]
                avg_ys = avg_ys[from_idx:]

            if plot_all:
                for y in ys:
                    if loglog:
                        plt.loglog(xs, y, color=color, alpha=0.2)
                    else:
                        plt.plot(xs, y, color=color, alpha=0.2)

            num_params = pl.scan_csv(file).filter(
                (pl.col("num_heads") == num_heads_)
                & (pl.col("depth") == depth_)
                & (pl.col("width") == width_)
            ).collect()["num_params"][0]
            
            label = (
                f"digits={gradient_rounding_digits_}, "
                f"depth={depth_}, width={width_}, #params={format_num_params(num_params)}"
            )
            if loglog:
                plt.loglog(xs, avg_ys, color=color if plot_all else None, label=label)
            else:
                plt.plot(xs, avg_ys, color=color if plot_all else None, label=label)


    fig = plt.gcf()
    fig.set_size_inches(12, 7)

    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.grid()
    plt.title(f"{to_plot} vs {plot_over}")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        name = f"{to_plot}_vs_{plot_over}_from_percentage_{from_percentage}"
        if depth is not None:
            name += f"_depth_{depth}"
        if width is not None:
            name += f"_width_{width}"
        if num_heads is not None:
            name += f"_num_heads_{num_heads}"
        if gradient_rounding_digits is not None:
            name += f"_digits_{min(gradient_rounding_digits)}_to_{max(gradient_rounding_digits)}"
        if len(ys) > 1:
            name += f"_{len(ys)}_tries"
        plt.savefig(f"results/images/{name}.png", dpi=300)
    close_plt()  # in case you call this function multiple times with different settings


def heatmap_l2_dist_losses_by_size(
        file: str,
        to_plot: Literal["val_loss", "train_loss", "val_accs", "train_accs", "val_pplxs"] = "val_loss",
        min_digits: int = 0,
        show: bool = True,
) -> None:
    digits = unique_gradient_rounding_digits(file)
    max_digits = max(digits)
    digits = [d for d in digits if d >= min_digits and d != max_digits]
    param_nums = unique_num_params(file)

    results = np.zeros((len(param_nums), len(digits)))
    for pidx, num_params in enumerate(param_nums):
        for didx, digit in enumerate(digits):
            xs1, _, avg_ys1 = load_xs_ys_avg_y(
                file,
                num_params=num_params,
                gradient_rounding_digits=digit,
                to_plot=to_plot,
                plot_over="epoch",
            )
            xs2, _, avg_ys2 = load_xs_ys_avg_y(
                file,
                num_params=num_params,
                gradient_rounding_digits=max_digits,
                to_plot=to_plot,
                plot_over="epoch",
            )
            # Make the two arrays the same length
            xs = xs1 if xs1[-1] < xs2[-1] else xs2
            avg_ys1 = np.interp(xs, xs1, avg_ys1)
            avg_ys2 = np.interp(xs, xs2, avg_ys2)
            l2_dist = np.linalg.norm(avg_ys1 - avg_ys2)
            results[pidx, didx] = l2_dist

    fig, ax = plt.subplots()
    cax = ax.matshow(results, cmap="viridis")
    cbar = fig.colorbar(cax)
    cbar.set_label(f"L2 distance to {max_digits} gradient rounding digits")

    yticklabels = []
    for i in range(len(param_nums)):
        # Write the values in the heatmap in white
        for j in range(len(digits)):
            results_text = f"{results[i, j]:.2f}"
            while len(results_text) > 4:  # > x.xx
                results_text = results_text[:-1]
            if results_text.endswith("."):
                results_text = results_text[:-1]
            ax.text(j, i, results_text, ha="center", va="center", color="white")
        
        # Complete the yticklabels in the format "depth, width (#params)"
        width = pl.scan_csv(file).filter(pl.col("num_params") == param_nums[i]).collect()["width"][0]
        depth = pl.scan_csv(file).filter(pl.col("num_params") == param_nums[i]).collect()["depth"][0]
        yticklabels.append(f"{depth} / {width} / {format_num_params(param_nums[i])}")

    ax.set_xticks(np.arange(len(digits)))
    ax.set_yticks(np.arange(len(param_nums)))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticklabels(digits)
    ax.set_yticklabels(yticklabels)
    plt.xlabel("digits")
    plt.ylabel("depth / width / #params")

    # Increase plot-size
    fig = plt.gcf()
    fig.set_size_inches(10, 7)

    # Reduce whitespace
    fig.subplots_adjust(left=0.1, right=0.87, top=0.9, bottom=0.1)

    plt.title(f"L2 distance of {to_plot} between n and {max_digits} rounding digits")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(
            f"results/images/l2_dist_{to_plot}"
            f"_{len(param_nums)}_modelsizes_"
            f"{max_digits}_max_digits_{min_digits}_min_digits.png", 
            dpi=300,
        )


def heatmap_l2_dist_to_largest_model(
        file: str,
        to_plot: Literal["val_loss", "train_loss", "val_accs", "train_accs", "val_pplxs"] = "val_loss",
        min_digits: int = 0,
        show: bool = True,
        measure: Literal["L1", "L2"] = "L2",
) -> None:
    digits = unique_gradient_rounding_digits(file)
    digits = [d for d in digits if d >= min_digits]
    param_nums = unique_num_params(file)
    max_param_nums = max(param_nums)
    param_nums = [p for p in param_nums if p != max_param_nums]

    results = np.zeros((len(param_nums), len(digits)))
    for pidx, num_params in enumerate(param_nums):
        for didx, digit in enumerate(digits):
            xs1, _, avg_ys1 = load_xs_ys_avg_y(
                file,
                num_params=num_params,
                gradient_rounding_digits=digit,
                to_plot=to_plot,
                plot_over="epoch",
            )
            xs2, _, avg_ys2 = load_xs_ys_avg_y(
                file,
                num_params=max_param_nums,
                gradient_rounding_digits=digit,
                to_plot=to_plot,
                plot_over="epoch",
            )
            # Make the two arrays the same length
            xs = xs1 if xs1[-1] < xs2[-1] else xs2
            avg_ys1 = np.interp(xs, xs1, avg_ys1)
            avg_ys2 = np.interp(xs, xs2, avg_ys2)
            if measure == "L1":
                l2_dist = np.linalg.norm(avg_ys1 - avg_ys2, ord=1)
            elif measure == "L2":
                l2_dist = np.linalg.norm(avg_ys1 - avg_ys2)
            else:
                raise ValueError(f"Measure {measure} not supported")
            results[pidx, didx] = l2_dist

    fig, ax = plt.subplots()
    cax = ax.matshow(results, cmap="viridis")
    cbar = fig.colorbar(cax)
    cbar.set_label(f"L2 distance to largest model ({format_num_params(max_param_nums, 0)})")

    yticklabels = []
    for i in range(len(param_nums)):
        # Write the values in the heatmap in white
        for j in range(len(digits)):
            results_text = f"{results[i, j]:.2f}"
            while len(results_text) > 4:  # > x.xx
                results_text = results_text[:-1]
            if results_text.endswith("."):
                results_text = results_text[:-1]
            ax.text(j, i, results_text, ha="center", va="center", color="white")
        
        # Complete the yticklabels in the format "depth, width (#params)"
        width = pl.scan_csv(file).filter(pl.col("num_params") == param_nums[i]).collect()["width"][0]
        depth = pl.scan_csv(file).filter(pl.col("num_params") == param_nums[i]).collect()["depth"][0]
        yticklabels.append(f"{depth} / {width} / {format_num_params(param_nums[i])}")

    ax.set_xticks(np.arange(len(digits)))
    ax.set_yticks(np.arange(len(param_nums)))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticklabels(digits)
    ax.set_yticklabels(yticklabels)
    plt.xlabel("digits")
    plt.ylabel("depth / width / #params")

    # Increase plot-size
    fig = plt.gcf()
    fig.set_size_inches(10, 6)

    # Reduce whitespace
    fig.subplots_adjust(left=0.1, right=0.87, top=0.9, bottom=0.1)

    plt.title(f"L2 distance of {to_plot} to {format_num_params(max_param_nums)} params")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(
            f"results/images/l2_dist_{to_plot}_to_largest_model"
            f"_{len(param_nums)}_modelsizes_"
            f"{len(digits)}_digits_"
            f"{max_param_nums}_max_param_nums.png", 
            dpi=300,
        )
    close_plt()


def plot_fourier_transform_of_loss_curves_heatmap(
        file: str,
        to_plot: Literal["val_loss", "train_loss", "val_accs", "train_accs", "val_pplxs"] = "val_loss",
        plot_over: Literal["step", "epoch", "epoch_unique_token", "token", "time_sec"] = "epoch",
        min_digits: int = 0,
        draw_numbers: bool = False,
        show: bool = True,
):
    param_nums = unique_num_params(file)
    gradient_rounding_digits = unique_gradient_rounding_digits(file)
    gradient_rounding_digits = [d for d in gradient_rounding_digits if d >= min_digits]

    fft_means = []
    for num_params in param_nums:
        fft_means_for_num_params = []
        for digits in gradient_rounding_digits:
            xs, ys, avg_ys = load_xs_ys_avg_y(
                file,
                num_params=num_params,
                gradient_rounding_digits=digits,
                to_plot=to_plot,
                plot_over=plot_over,
            )

            # Fourier transform
            fft_values = np.fft.fft(avg_ys)
            fft_freq = np.fft.fftfreq(len(xs))
            amplitudes = np.abs(fft_values)
            fft_mean = np.sum(amplitudes * np.abs(fft_freq)) / np.sum(amplitudes)
            fft_means_for_num_params.append(fft_mean)
        fft_means.append(fft_means_for_num_params)

    fig, ax = plt.subplots()
    cax = ax.matshow(np.array(fft_means), cmap="viridis")
    cbar = fig.colorbar(cax)
    cbar.set_label("Bla")

    yticklabels = []
    for i in range(len(param_nums)):
        # Write the values in the heatmap in white
        if draw_numbers:
            for j in range(len(gradient_rounding_digits)):
                results_text = f"{fft_means[i][j]:.2f}"
                while len(results_text) > 4:  # > x.xx
                    results_text = results_text[:-1]
                if results_text.endswith("."):
                    results_text = results_text[:-1]
                ax.text(j, i, results_text, ha="center", va="center", color="white")
        
        # Complete the yticklabels in the format "depth, width (#params)"
        width = pl.scan_csv(file).filter(pl.col("num_params") == param_nums[i]).collect()["width"][0]
        depth = pl.scan_csv(file).filter(pl.col("num_params") == param_nums[i]).collect()["depth"][0]
        yticklabels.append(f"{depth} / {width} / {format_num_params(param_nums[i])}")

    ax.set_xticks(np.arange(len(gradient_rounding_digits)))
    ax.set_yticks(np.arange(len(param_nums)))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticklabels(gradient_rounding_digits)
    ax.set_yticklabels(yticklabels)
    plt.xlabel("digits")
    plt.ylabel("depth / width / #params")

    # Increase plot-size
    fig = plt.gcf()
    fig.set_size_inches(7, 7)

    # Reduce whitespace
    fig.subplots_adjust(left=0.1, right=0.87, top=0.9, bottom=0.1)

    plt.title(f"Mean frequency of {to_plot} (calculated with FFT)")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(
            f"results/images/fft_mean_{to_plot}"
            f"_{len(param_nums)}_modelsizes_"
            f"{len(gradient_rounding_digits)}_digits"
            f"_{min_digits}_min_digits.png", 
            dpi=300,
        )


def plot_mean_fourier_freq_over_gradient_rounding_digits(
        file: str,
        to_plot: Literal["val_loss", "train_loss", "val_accs", "train_accs", "val_pplxs"] = "val_loss",
        plot_over: Literal["step", "epoch", "epoch_unique_token", "token", "time_sec"] = "epoch",
        min_digits: int = 0,
        show: bool = True,
):
    param_nums = unique_num_params(file)
    gradient_rounding_digits = unique_gradient_rounding_digits(file)
    gradient_rounding_digits = [d for d in gradient_rounding_digits if d >= min_digits]

    fft_means = []
    for num_params in param_nums:
        fft_means_for_num_params = []
        for digits in gradient_rounding_digits:
            xs, ys, avg_ys = load_xs_ys_avg_y(
                file,
                num_params=num_params,
                gradient_rounding_digits=digits,
                to_plot=to_plot,
                plot_over=plot_over,
            )

            # Fourier transform
            fft_values = np.fft.fft(avg_ys)
            fft_freq = np.fft.fftfreq(len(xs))
            amplitudes = np.abs(fft_values)
            fft_mean = np.sum(amplitudes * np.abs(fft_freq)) / np.sum(amplitudes)
            fft_means_for_num_params.append(fft_mean)
        fft_means.append(fft_means_for_num_params)

    for i, num_params in enumerate(param_nums):
        depth = pl.scan_csv(file).filter(pl.col("num_params") == num_params).collect()["depth"][0]
        width = pl.scan_csv(file).filter(pl.col("num_params") == num_params).collect()["width"][0]
        plt.plot(gradient_rounding_digits, fft_means[i], label=f"{depth=}, {width=} ({format_num_params(num_params)})")

    fig = plt.gcf()
    fig.set_size_inches(12, 7)
    plt.xlabel("#digits")
    plt.ylabel("Mean frequency")
    plt.legend()
    plt.grid()
    plt.title(f"Mean frequency of {to_plot} (calculated with FFT) over #digits")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(
            f"results/images/fft_mean_{to_plot}"
            f"_{len(param_nums)}_modelsizes_"
            f"{len(gradient_rounding_digits)}_digits"
            f"_{min_digits}_min_digits.png", 
            dpi=300,
        )
    close_plt()


if __name__ == "__main__":
    file_digits = "results/results_digits.csv"
    file_10_tries = "results/results_10_tries.csv"
    file_digits_and_scale = "results/results_digits_and_scale.csv"
    
    # plot_performance(
    #     file=file_digits_and_scale,
    #     depth=None,
    #     width=None,
    #     gradient_rounding_digits=16,
    #     num_heads=1,
    #     from_percentage=0.6,
    #     to_plot="train_loss",
    #     plot_over="token",
    #     show=True,
    #     loglog=False,
    #     plot_all=False,
    # )
    # heatmap_l2_dist_losses_by_size(
    #     file=file_digits_and_scale,
    #     to_plot="val_loss",
    #     min_digits=0,
    #     show=False,
    # )
    # plot_fourier_transform_of_loss_curves_heatmap(
    #     file=file_digits_and_scale,
    #     to_plot="val_loss",
    #     plot_over="epoch",
    #     min_digits=7,
    #     show=False,
    # )
    # plot_mean_fourier_freq_over_gradient_rounding_digits(
    #     file=file_digits_and_scale,
    #     to_plot="val_loss",
    #     plot_over="epoch",
    #     min_digits=7,
    #     show=True,
    # )
    heatmap_l2_dist_to_largest_model(
        file=file_digits_and_scale,
        to_plot="val_loss",
        min_digits=6,
        show=True,
        measure="L1",
    )
