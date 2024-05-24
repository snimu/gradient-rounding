
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


def format_num_params(num_params: int) -> str:
    if num_params < 1_000:
        return str(num_params)
    elif num_params < 1_000_000:
        return f"{num_params/1_000:.1f}k"
    elif num_params < 1_000_000_000:
        return f"{num_params/1_000_000:.1f}M"
    else:
        return f"{num_params/1_000_000_000:.1f}B"


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
        to_plot: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs", "train_pplxs"] = "val_loss",
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


def unique_num_params(file: str) -> list[int]:
    return (
        pl.scan_csv(file)
        .select("num_params")
        .collect()
        ["num_params"]
        .unique()
        .to_numpy()
    )


def unique_widths(file: str) -> list[int]:
    return (
        pl.scan_csv(file)
        .select("width")
        .collect()
        ["width"]
        .unique()
        .to_numpy()
    )


def unique_depths(file: str) -> list[int]:
    return (
        pl.scan_csv(file)
        .select("depth")
        .collect()
        ["depth"]
        .unique()
        .to_numpy()
    )


def plot_performance(
        file: str,
        depth: int | None = 8,
        width: int | None = 384,
        num_heads: int | None = None,
        gradient_rounding_digits: int | None = 16,
        to_plot: Literal["val_loss", "train_losses", "val_accs", "train_accs", "val_pplxs"] = "val_loss",
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
        settings = [(nh, d, w, rd) for nh, d, w, rd in settings if rd == gradient_rounding_digits]

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
        # You should probably adjust the filename
        plt.savefig(f"{to_plot}_vs_{plot_over}.png", dpi=300)
    close_plt()  # in case you call this function multiple times with different settings


if __name__ == "__main__":
    file_digits = "results/results_digits.csv"
    file_10_tries = "results/results_10_tries.csv"
    
    plot_performance(
        file=file_digits,
        depth=8,
        width=384,
        gradient_rounding_digits=None,
        num_heads=1,
        to_plot="val_loss",
        plot_over="token",
        show=True,
        loglog=False,
        plot_all=False,
    )