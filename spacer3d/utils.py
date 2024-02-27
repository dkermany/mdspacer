import os
import numpy as np
import cupy as cp
import seaborn as sns
import pandas as pd
from glob import glob
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def calculate_pi(data, p=0.95):
    """
    Calculate the 95% prediction interval for the given data.

    Parameters:
    data (array-like): A list or array of numerical data points.

    Returns:
    dict: Lower and upper bounds of the 95% prediction interval.
    """
    # Calculate mean and standard error of the mean
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # Number of observations
    n = len(data)

    bounds = {}
    for conf in [0.95, 0.99, 0.999]:
        t_score = stats.t.ppf((1 + conf) / 2., n-1)
        margin_of_error = t_score * std * np.sqrt(1 + 1/n)
        bounds[str(conf)] = {
            "lower": mean - margin_of_error,
            "upper": mean + margin_of_error,
        }

    return bounds

def calculate_percentile_range(data):
    """
    Calculate the 95%, 99%, 99.9%, and 99.99% percentile intervals for the given data.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'Radius (r)', 'K(r) value', and possibly 
                         'simulation number' columns.

    Returns:
    dict: Lower and upper bounds of the percentile intervals for each radius.
    """
    radii = data[data["Line"] == 1]["Radius (r)"].tolist()
    
    bounds = {}
    for radius in radii:
        subset = data[data["Radius (r)"] == radius]["K(r)"].tolist()

        # Define the confidence levels and their corresponding percentile values
        percentiles = {
            0.95: (2.5, 97.5),
            0.99: (0.5, 99.5),
            0.999: (0.05, 99.95),
            0.9999: (0.005, 99.995)
        }

        bounds[radius] = {}
        for conf, (lower_perc, upper_perc) in percentiles.items():
            lower_bound = np.percentile(subset, lower_perc)
            upper_bound = np.percentile(subset, upper_perc)
            bounds[radius][str(conf)] = {
                "lower": lower_bound,
                "upper": upper_bound,
            }
    
    return bounds

def get_equivalent_color(cf, alpha, cb=(255, 255, 255)):
    """
    Calculate the RGB values of an opaque color in decimal format (0-1) that visually matches a semi-transparent color
    when placed over a specified background color. This function automatically handles both regular RGB (0-255) and
    decimal RGB (0-1) formats for the input colors.

    Parameters:
    - cf (tuple): The foreground color in RGB format, either in regular (0-255) or decimal (0-1).
    - alpha (float): The alpha transparency level of the foreground color, ranging from 0 (completely transparent)
      to 1 (completely opaque).
    - cb (tuple, optional): The background color in RGB format, either in regular (0-255) or decimal (0-1). Defaults to white (255, 255, 255).

    Returns:
    - list of float: The RGB values of the equivalent opaque color in decimal format that visually matches the semi-transparent
      foreground color when viewed against the background color.

    Example:
    >>> get_equivalent_color((0, 0, 255), 0.5)
    [0.5, 0.5, 1.0]
    """
    # Check if the input colors are in the regular RGB (0-255) format and convert them to decimal format if necessary
    cf_decimal = [f / 255. if max(cf) > 1 else f for f in cf]
    cb_decimal = [b / 255. if max(cb) > 1 else b for b in cb]

    # Calculate the equivalent color in decimal RGB format
    result_decimal = [alpha * f + (1 - alpha) * b for f, b in zip(cf_decimal, cb_decimal)]

    return result_decimal
    
def pi_range_to_df(data: dict):
    """
    Converts a nested dictionary containing radius, confidence intervals ('pi'), and their corresponding
    lower and upper bounds into a pandas DataFrame.

    Parameters:
    - data (dict): A nested dictionary where the first level keys are radii, the second level keys are
      confidence intervals (e.g., '0.95'), and the values are dictionaries with 'lower' and 'upper' bounds.

    Returns:
    - pandas.DataFrame: A DataFrame with columns for 'radius', 'pi' (confidence interval), 'lower', and 'upper' bounds.

    Example:
    >>> data = {1: {'0.95': {'lower': 0, 'upper': 10}, '0.99': {'lower': -5, 'upper': 15}}}
    >>> pi_range_to_df(data)
    # Returns a DataFrame with each row representing a combination of radius, pi, lower, and upper values.
    """
    # Flatten the dictionary into a list of rows
    rows = []
    for radius, pi in data.items():
        for pi, bounds in pi.items():
            rows.append({
                "radius": radius,
                "pi": pi,
                "lower": bounds["lower"],
                "upper": bounds["upper"],
            })

    # Convert the list of rows into a DataFrame
    return pd.DataFrame(rows)

def plot_interval(df, color, ax, pi="0.95"):
    """
    Plots confidence intervals (lower and upper bounds) for a specified 'pi' value from a DataFrame
    onto a given axis ('ax'), including filling the area between these bounds.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to plot, expected to have columns
      'radius', 'pi', 'lower', and 'upper'.
    - color (str): The color to use for both lines and the fill between them.
    - ax (matplotlib.axes.Axes): The Matplotlib axis object where the plot will be drawn.
    - pi (str, optional): The confidence interval level to plot (e.g., '0.95'). Defaults to '0.95'.

    Returns:
    - None: This function plots on the provided Matplotlib axis and does not return a value.

    Example:
    >>> plot_interval(df, 'blue', ax, pi="0.99")
    # This will plot the lower and upper bounds for the '0.99' confidence interval on the provided axis.
    """
    for col_name in ["lower", "upper"]:
        sns.lineplot(
            data=df[df["pi"] == pi],
            x="radius",
            y=col_name,
            color=get_equivalent_color(color, 0.4),
            linewidth=0.5,
            ax=ax,
        )
    
    # Now, fill between the lines
    # Extracting the 'radius', 'lower', and 'upper' values for the "0.95" confidence interval
    radii = df[df["pi"] == pi]["radius"]
    lower = df[df["pi"] == pi]["lower"]
    upper = df[df["pi"] == pi]["upper"]

    # Filling between the 'lower' and 'upper' lines on the second subplot
    ax.fill_between(radii, lower, upper, color=get_equivalent_color(color, 0.2))  # Adjust the color and alpha as needed
    label = float(pi) * 100
    label = int(label) if label.is_integer() else label
    return mpatches.Patch(color=get_equivalent_color(color, 0.2), label=f"{label}% Interval")

def plot_background_intervals(df, color, ax):
    max_pi = str(df["pi"].astype(float).max())

    # Extracting the 'radius', 'lower', and 'upper' values for the specified confidence interval
    radii = df[df["pi"] == max_pi]["radius"]
    lower = df[df["pi"] == max_pi]["lower"]
    upper = df[df["pi"] == max_pi]["upper"]

    # Get current y-axis limits
    ymin, ymax = ax.get_ylim()

    # Filling from the bottom of the graph to the bottom-most line
    ax.fill_between(radii, ymin, lower, color=get_equivalent_color(color, 0.2))

    # Filling from the top of the graph to the topmost line
    ax.fill_between(radii, upper, ymax, color=get_equivalent_color(color, 0.2))
    return mpatches.Patch(color=get_equivalent_color(color, 0.2), label=f">99.99% Interval")

def normalize(rstats, rand_rstats):
    def min_max(data, min_val, max_val):
        """
        Normalize the data to have a lower bound of -1 and upper bound of 1
        with respect to the 95% interval min and max values
    
        Parameters:
        data (array-like): A list or array of numerical data points.
        min_val (float): min value
        max_val (float): max value
    
        Returns:
        array: Normalized data with bounds [-1, 1].
        """
        normalized_data = 2 * ((data - min_val) / (max_val - min_val)) - 1
        return normalized_data

    normalized_K = []
    for radius in rstats["Radius (r)"]:
        subset = rand_rstats[rand_rstats["Radius (r)"] == radius]["K(r)"].tolist()
        bounds = calculate_pi(subset)
        lower_bound_95, upper_bound_95 = bounds["0.95"]["lower"], bounds["0.95"]["upper"]
        if abs(lower_bound_95 - upper_bound_95) < 1:
            val = 0.
        else:
            val = min_max(rstats[rstats["Radius (r)"] == radius]["K(r)"].tolist()[0], lower_bound_95, upper_bound_95)
        normalized_K.append(val)
    return normalized_K

def normalize_w_intervals(rstats, rand_rstats):
    def min_max(data, min_val, max_val):
        """
        Normalize the data to have a lower bound of -1 and upper bound of 1
        with respect to the 95% interval min and max values.

        Parameters:
        data (float): A numerical data point.
        min_val (float): min value of the 95% interval.
        max_val (float): max value of the 95% interval.

        Returns:
        float: Normalized data with bounds [-1, 1].
        """
        # Avoid division by zero
        if max_val == min_val:
            return 0
        return 2 * ((data - min_val) / (max_val - min_val)) - 1

    # Lists to store data before concatenation
    normalized_k_rows = []
    intervals_rows = []

    # Calculate percentile ranges for random simulations
    percentile_ranges = calculate_percentile_range(rand_rstats)

    for radius in rstats["Radius (r)"].unique():
        bounds_95 = percentile_ranges[radius]["0.95"]
        
        # Normalize the K function value for this radius
        K_value = rstats[rstats["Radius (r)"] == radius]["K(r)"].iloc[0]
        normalized_K = min_max(K_value, bounds_95["lower"], bounds_95["upper"])

        # Normalize the theoretical K function for this radius
        theoretical_K = rstats[rstats["Radius (r)"] == radius]["theoretical_K"].iloc[0]
        normalized_theoretical_K = min_max(theoretical_K, bounds_95["lower"], bounds_95["upper"])

        # Append to the list for normalized K values
        normalized_k_rows.append({"radius": radius, "normalized_K": normalized_K, "theoretical_K": normalized_theoretical_K})

        # Iterate through each confidence level to populate intervals_df
        for conf in ["0.95", "0.99", "0.999", "0.9999"]:
            bounds = percentile_ranges[radius][conf]
            lower = min_max(bounds["lower"], bounds_95["lower"], bounds_95["upper"])
            upper = min_max(bounds["upper"], bounds_95["lower"], bounds_95["upper"])
            
            # Append to the list for interval bounds
            intervals_rows.append({
                "radius": radius,
                "pi": conf,
                "lower": lower,
                "upper": upper
            })
    
    # Concatenate lists into DataFrames
    normalized_k_df = pd.concat([pd.DataFrame([row]) for row in normalized_k_rows], ignore_index=True)
    intervals_df = pd.concat([pd.DataFrame([row]) for row in intervals_rows], ignore_index=True)
    
    return normalized_k_df, intervals_df

def get_interval_pairs(df):
    interval_dfs = {}
    
    # Confidence levels to create separate DataFrames for
    conf_levels = ["95", "99", "999", "9999"]
    
    for conf in conf_levels:
        # Selecting columns for the current confidence level
        cols = ["radius", f"lower_{conf}", f"upper_{conf}"]
        interval_dfs[conf] = df[cols].copy()
        interval_dfs[conf].columns = ["radius", "lower", "upper"]

    return interval_dfs

def _plot_all_intervals(pi_df, ax, palette):
    patches = []
    for i, pi in enumerate(["0.9999", "0.999", "0.99", "0.95"]):
        patch = plot_interval(pi_df, color=palette[i+1], ax=ax, pi=pi)
        patches.append(patch)
    patch = plot_background_intervals(pi_df, color=palette[0], ax=ax)
    patches.insert(0, patch)
    return patches

def _plot_normalized_graph(rstats, rand_rstats, ax, palette):
    normalized_k_df, intervals_df = normalize_w_intervals(rstats, rand_rstats)
    sns.lineplot(data=normalized_k_df, x="radius", y="theoretical_K", ax=ax, label=r"Theoretical $\mathit{K}$ Function", color="#888", linewidth=2, linestyle="dotted", zorder=98)
    sns.lineplot(data=normalized_k_df, x="radius", y="normalized_K", ax=ax, alpha=1, zorder=99, label=r"Observed $\mathit{K}$ Function")
    patches = _plot_all_intervals(intervals_df, ax=ax, palette=palette)
    return patches
    
    # paired_intervals = get_interval_pairs(normalized_df)

def _config_legend(patches, ax):
    patch_labels = [p.get_label() for p in patches]

    # Retrieve existing handles and labels from the axis
    line_handles, line_labels = ax.get_legend_handles_labels()
    all_handles = line_handles + patches
    all_labels = line_labels + patch_labels

    ax.legend(handles=all_handles, labels=all_labels, loc="upper left", fontsize="12")

def plot_process(rstats_path, save=False, output_folder="./ripley_plots"):
    # palette = sns.color_palette("colorblind")
    palette = sns.color_palette("rocket_r")
    # palette = sns.color_palette("husl", 4)

    def get_rstats_files(path):
        return glob(f"{path}/*.csv")

    def plot(rstats, rand_rstats, ax):
        a = sns.lineplot(data=rand_rstats, x="Radius (r)", y="K(r)", hue="Line", ax=ax[0], alpha=1)
        a.get_legend().remove()
        ax[0].set(xlabel=None, ylabel=r"$\mathit{K}$(r)")

        pi_df = pi_range_to_df(calculate_percentile_range(rand_rstats))
        patches = _plot_all_intervals(pi_df, ax=ax[1], palette=palette)
        _config_legend(patches, ax=ax[1])
        ax[1].set(xlabel=None, ylabel=None)

        # Calculate theoretical K values and add to rstats
        rstats['theoretical_K'] = ((4/3) * np.pi * rstats['Radius (r)']**3) + rstats["K(r)"].min()

        sns.lineplot(data=rstats, x="Radius (r)", y="theoretical_K", ax=ax[2], label=r"Theoretical $\mathit{K}$ Function", color="#888", linewidth=2, linestyle="dotted", zorder=98)
        sns.lineplot(data=rstats, x="Radius (r)", y="K(r)", ax=ax[2], alpha=1, zorder=99, label=r"Observed $\mathit{K}$ Function")
        patches = _plot_all_intervals(pi_df, ax=ax[2], palette=palette)

        _config_legend(patches, ax=ax[2])        
        ax[2].set(xlabel=None, ylabel=None)

        # draw_normalized_graph(rstats, rand_rstats, ax=ax[3])
        patches = _plot_normalized_graph(rstats, rand_rstats, ax=ax[3], palette=palette)
        _config_legend(patches, ax=ax[3])
        ax[3].set(xlabel=None, ylabel=r"$\mathit{K}$$_{Norm}$")
        
    rstats_files = get_rstats_files(rstats_path)
    print("Loaded:", rstats_files)
    
    # filter our monte carlo results
    rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
    u_rstats_files = [f for f in rstats_files if "univariate" in f]
    m_rstats_files = [f for f in rstats_files if "multivariate" in f]

    # Update the font size for the plot
    plt.rcParams.update({"font.size": 16})
    
    # Create a subplot with 3 rows and 1 column
    f, axes = plt.subplots(1,4, sharex=True, figsize=(19,6))
    for ax in axes:
        ax.tick_params(axis="both", labelsize="16")
    f.tight_layout(pad=1.4)

    f.supxlabel("Radius (μm)", y=0.00)
    f.subplots_adjust(bottom=0.12)
    
    for i, filename in enumerate(u_rstats_files):
        fullpath = os.path.join(rstats_path, f"{filename}.csv")
        
        prefix, _, date, id, mode, label, _ = filename.split("_")
        
        # Construct the path for the random CSV file
        rand_fullpath = os.path.join(rstats_path, f"FV10__{date}_{id}_random_{mode}_{label}_rstats.csv")
    
        # Load the CSV file and random CSV file into DataFrames
        rstats = pd.read_csv(fullpath)
        rand_rstats = pd.read_csv(rand_fullpath)
        plot(rstats, rand_rstats, ax=axes)

        if save:
            create_directory(output_folder)
            plt.savefig(os.path.join(output_folder, f"{filename}_process.svg"))
        
def plot_individuals(rstats_path, save=False, output_folder="./ripley_results/"):
    palette = sns.color_palette("rocket_r")

    def get_rstats_files(path):
        return glob(f"{path}/*.csv")

    def group_items(l, n=4):
        """Groups every 'n' items in the list into sublists."""
        return [l[i:i + n] for i in range(0, len(l), n)]

    def rearrange_sublists(lst, order):
        """Rearranges elements within each sublist of a list based on a specified order."""
        return [[sublist[i] for i in order] for sublist in lst]
        
    rstats_files = get_rstats_files(rstats_path)
    print("Loaded:", rstats_files)

    # filter our monte carlo results
    rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
    u_rstats_files = [f for f in rstats_files if "univariate" in f]
    m_rstats_files = [f for f in rstats_files if "multivariate" in f]

    grouped_u_rstats_files = rearrange_sublists(group_items(sorted(u_rstats_files)), [2,1,0,3])
    # Update the font size for the plot
    plt.rcParams.update({"font.size": 20})
    
    # Create a subplot with 3 rows and 1 column
    f, axes = plt.subplots(4,4, sharex=True, figsize=(15,15))
    f.tight_layout(pad=-0.35)

    f.supxlabel("Radius (μm)", x=0.51, y=-0.001)
    f.supylabel("K$_{Norm}$", x=-0.001)
    f.subplots_adjust(bottom=0.06, left=0.06)
    
    # Loop over each univariate filename
    for i, group in enumerate(grouped_u_rstats_files):
        for j, filename in enumerate(group):
            # Construct the path for the CSV file
            fullpath = os.path.join(rstats_path, f"{filename}.csv")
            
            prefix, _, date, id, mode, label, _ = filename.split("_")
            
            # Construct the path for the random CSV file
            rand_fullpath = os.path.join(rstats_path, f"FV10__{date}_{id}_random_{mode}_{label}_rstats.csv")
    
            # Load the CSV file and random CSV file into DataFrames
            rstats = pd.read_csv(fullpath)
            rand_rstats = pd.read_csv(rand_fullpath)

            if i == 0:
                title=["Tumor", "NG2+", "Branches", "TVC"]
            else:
                title=[None]*4
    
            # Calculate theoretical K values and add to rstats
            rstats['theoretical_K'] = ((4/3) * np.pi * rstats['Radius (r)']**3) + rstats["K(r)"].min()

            # draw_normalized_graph(rstats, rand_rstats, ax=axes[i,j], title=title[j])
            patches = _plot_normalized_graph(rstats, rand_rstats, ax=axes[i,j], palette=palette)
            axes[i, j].legend().remove()
            axes[i, j].set(xlabel="")
            axes[i, j].set(ylabel="")
            axes[i, j].tick_params(axis="both", labelsize="14")

            # _config_legend(patches, ax=axes[i, j])
    if save:
        plt.savefig(os.path.join(output_folder, f"{filename}_individual.svg"))

    # Create a new figure for the legend
    # fig_legend = plt.figure(figsize=(3, 2))
    # ax_legend = fig_legend.add_subplot(111)
    # Make the subplot area transparent and remove the axes
    # ax_legend.axis('off')
    # fig_legend.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Draw the legend on the new figure
    # ax_legend.legend(*axes[0,0].get_legend_handles_labels(), loc='center')

    # Save the figure containing only the legend
    # fig_legend.savefig('/Users/danielkermany/Desktop/legend_only.svg')
    # plt.savefig("/Users/danielkermany/Desktop/S6.svg", bbox_inches="tight")

def _draw_combined_graph(df, title=None):
    # Set the tick label format to plain
    plt.ticklabel_format(style="plain")

    # Update the font size for the plot
    plt.rcParams.update({"font.size": 20})

    # Create a subplot with 3 rows and 1 column
    f, ax = plt.subplots(figsize=(7,7))
    
    # Plot the K(r) values for the data and the random data
    l_ax = sns.lineplot(data=df, x="Radius (r)", y="K_norm", hue="Sample", style="Sample", ax=ax, legend="brief", zorder=3)

    for line in plt.gca().get_lines():
        label = line.get_label()
        if label == "Average":
            line.set_alpha(1.0)
        else:
            line.set_alpha(0.5)

    if len(l_ax. lines) > 0:
        average_line = l_ax.lines[0]
        average_line.set_linewidth(2)
        average_line.set_alpha(1.0)
        average_line.set_zorder(5)
    
    # # Get the current x-axis limits after plotting the data
    # ax = plt.gca()
    xlims = ax.get_xlim()
    
    # Get the current y-axis limits after plotting the data
    ylims = ax.get_ylim()
    
    # Fill between y=1 and y=-1, extending to the edges of the graph
    ax.fill_between(xlims, -1, 1, color='#ffe5ce', alpha=0.9)
    
    # Adding thicker lines for the top and bottom borders, extending to the edges
    ax.hlines(1, *xlims, colors='#ffbb80', linewidth=1.5)
    ax.hlines(-1, *xlims, colors='#ffbb80', linewidth=1.5)
    
    # Set the x and y-axis limits to ensure the fill and borders extend to the edges
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    ax.set(xlabel="")
    ax.set(ylabel="")

    # Manually set the y-axis limits to include the -1 to 1 range
    max_val = max(df["K_norm"])
    min_val = min(df["K_norm"])
    
    new_ylims = (min(-1-(0.05*max_val), min_val), max(1+(0.05*max_val), max_val))  # This includes both your data and the -1 to 1 interval
    #print("new ylims", new_ylims)
    ax.set_ylim(new_ylims)

    plt.legend(prop={'size': 14})  # Set the font size to 10
    ax.title.set_text(title)

def plot_combined_univariate(rstats_path):
    def get_rstats_files(path):
        return glob(f"{path}/*.csv")

    def group_items(l, n=4):
        """Groups every 'n' items in the list into sublists."""
        return [l[i:i + n] for i in range(0, len(l), n)]

    def rearrange_sublists(lst, order):
        """Rearranges elements within each sublist of a list based on a specified order."""
        return [[sublist[i] for i in order] for sublist in lst]
        
    rstats_files = get_rstats_files(rstats_path)
    print("Loaded:", rstats_files)

    # filter our monte carlo results
    rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
    u_rstats_files = sorted([f for f in rstats_files if "univariate" in f])
    m_rstats_files = sorted([f for f in rstats_files if "multivariate" in f])

    types = ["tumor", "ng2", "branch", "tvc"]
    for t in types:
        t_files = [i for i in u_rstats_files if t in i]

        df = pd.DataFrame()
        df["Radius (r)"] = pd.Series(np.arange(2,100))
        for i, filename in enumerate(t_files):
            # Construct the path for the CSV file
            fullpath = os.path.join(rstats_path, f"{filename}.csv")
            prefix, _, date, id, mode, label, _ = filename.split("_")
            
            # Construct the path for the random CSV file
            rand_fullpath = os.path.join(rstats_path, f"FV10__{date}_{id}_random_{mode}_{label}_rstats.csv")
    
            # Load the CSV file and random CSV file into DataFrames
            rstats = pd.read_csv(fullpath)
            rand_rstats = pd.read_csv(rand_fullpath)

            K_norm = normalize(rstats, rand_rstats)
            df[f"Sample {i+1}"] = pd.Series(K_norm)

        df['Average'] = df.drop('Radius (r)', axis=1).mean(axis=1)
        df_long = pd.melt(df, id_vars=["Radius (r)"], value_vars=["Average"]+[f"Sample {i+1}" for i in range(8)],
                          var_name="Sample", value_name="K_norm")
        _draw_combined_graph(df_long, title=t)

def plot_combined_multivariate(rstats_path):
    def get_rstats_files(path):
        return glob(f"{path}/*.csv")

    def group_items(l, n=4):
        """Groups every 'n' items in the list into sublists."""
        return [l[i:i + n] for i in range(0, len(l), n)]

    def rearrange_sublists(lst, order):
        """Rearranges elements within each sublist of a list based on a specified order."""
        return [[sublist[i] for i in order] for sublist in lst]
        
    rstats_files = get_rstats_files(rstats_path)
    print("Loaded:", rstats_files)

    # filter our monte carlo results
    rstats_files = [os.path.splitext(os.path.basename(f))[0] for f in rstats_files if "random" not in f]
    m_rstats_files = sorted([f for f in rstats_files if "multivariate" in f])

    anchor = ["tumor", "ng2"]
    types = ["ng2", "branch", "tvc"]
    titles = ["Tumor-NG2 Relationship", "Tumor-Branch Relationship", "Tumor-Tortuous Vessel Relationship", "NG2-Branch Relationship",
              "NG2-Tortuous Vessel Relationship"]
    i_cnt = 0
    for a in anchor:
        for t in types:
            if a != t: 
                a_files = [i for i in m_rstats_files if a in i and t in i]

                df = pd.DataFrame()
                df["Radius (r)"] = pd.Series(np.arange(2,100))
                for i, filename in enumerate(a_files):
                    # Construct the path for the CSV file
                    fullpath = os.path.join(rstats_path, f"{filename}.csv")
                    prefix, _, date, id, mode, anchor, target, _ = filename.split("_")
            
                    # Construct the path for the random CSV file
                    rand_fullpath = os.path.join(rstats_path, f"FV10__{date}_{id}_random_{mode}_{anchor}_{target}_rstats.csv")
            
                    # Load the CSV file and random CSV file into DataFrames
                    rstats = pd.read_csv(fullpath)
                    rand_rstats = pd.read_csv(rand_fullpath)
        
                    K_norm = normalize(rstats, rand_rstats)
                    df[f"Sample {i+1}"] = pd.Series(K_norm)

                df['Average'] = df.drop('Radius (r)', axis=1).mean(axis=1)
        
                df_long = pd.melt(df, id_vars=["Radius (r)"], value_vars=["Average"]+[f"Sample {i+1}" for i in range(8)],
                                  var_name="Sample", value_name="K_norm")
                _draw_combined_graph(df_long, title=titles[i_cnt])
                i_cnt += 1

def create_directory(path):
    """
    Creates directory at path if not exists
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    if isinstance(arr, cp.ndarray):
        nonzero = cp.nonzero
    elif isinstance(arr, np.ndarray):
        nonzero = np.nonzero
    else:
        raise ValueError("arr needs to be np or cp ndarray type")

    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in nonzero(arr))
    return arr[slices]

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
    point1 (array-like): An array-like object representing the first point.
    point2 (array-like): An array-like object representing the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    # Use numpy's linalg.norm function to calculate the Euclidean distance
    return np.linalg.norm(np.array(point1)-np.array(point2))

def replace_np_values(arr: np.ndarray, map: dict) -> np.ndarray:
    fn = np.vectorize(lambda x: map.get(x, 0) * 255)
    return fn(arr)

