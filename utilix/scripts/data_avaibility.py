import argparse
import sys
import platform
import strax
import straxen
import pandas as pd
import numpy as np
from datetime import datetime


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Process Strax context and calculate availability percentages."
    )

    parser.add_argument(
        "--container",
        type=str,
        help="Container name for setting up the environment.",
        required=True,
    )

    parser.add_argument(
        "--cutax_location",
        type=str,
        help="Cutax location read it from setup.sh of container",
        required=True,
    )

    parser.add_argument(
        "--context",
        choices=["online", "offline"],
        required=True,
        help="Choose the Strax context: online or offline.",
    )

    parser.add_argument(
        "--global_config",
        type=str,
        help="Global config for offline context (required for offline).",
    )

    parser.add_argument(
        "--include_tags", type=str, nargs="*", help='Tags to include, e.g., "*sr0*"'
    )

    parser.add_argument(
        "--exclude_tags",
        type=str,
        nargs="*",
        default=[
            "flash",
            "ramp_up",
            "ramp_down",
            "anode_off",
            "abandon",
            "hot_spot",
            "missing_one_pmt",
            "messy",
            "bad",
        ],
        help="Tags to exclude (default: predefined tags)",
    )

    parser.add_argument(
        "--plugins",
        type=str,
        nargs="*",
        default=None,
        help="Plugins to include for availability calculation (if not provided, will be determined by the --check_peaks flag). "
        "For using i pass, i.e., peak_basics event_basics",
    )

    parser.add_argument(
        "--time-range",
        type=str,
        nargs=2,
        metavar=("START_DATE", "END_DATE"),
        help='Time range for filtering, format: YYYY-MM-DD YYYY-MM-DD (e.g., "2023-01-01 2023-12-31")',
    )

    def str2bool(value):
        """Convert string to boolean for argparse."""
        if isinstance(value, bool):
            return value
        if value.lower() in ("true", "t", "yes", "y", "1"):
            return True
        elif value.lower() in ("false", "f", "no", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

    parser.add_argument(
        "--check_peaks",
        type=str2bool,
        nargs="?",
        const=True,  # If argument is given without value, default to True
        default=False,  # Default to False if not provided
        help="Check above peaks if True, below peaks if False. "
        "Below peaks: lone_hits, peaklets, merged_s2s, hitlets_nv. "
        "Above peaks: peak_basics, event_basics.",
    )

    return parser.parse_args()


# Function to initialize Strax context
def initialize_straxen(
    context_type, global_config, container, cutax=None, output_folder="./strax_data"
):

    # Initialize the context arguments
    context_args = {"output_folder": output_folder}

    print("")
    print("Login node:\n", platform.node())

    # Handle Midway or Dali configurations
    if "midway" in platform.node():
        if container <= "2023.05.2":
            context_args.update(
                {"_rucio_local_path": "/project/lgrandi/rucio", "include_rucio_local": True}
            )
    elif "dali" in platform.node():
        context_args.update(
            {
                "_auto_append_rucio_local": False,
                "_rucio_local_path": "/dali/lgrandi/rucio",
                "include_rucio_local": True,
            }
        )

    if context_type == "online":
        st = straxen.contexts.xenonnt_online(**context_args)
    elif context_type == "offline":
        if not global_config:
            raise ValueError("Global config is required for offline context.")
        st = cutax.contexts.xenonnt_offline(xedocs_version=global_config, **context_args)

    if "midway" in platform.node():
        st.storage.append(
            strax.DataDirectory("/project2/lgrandi/xenonnt/processed/", readonly=True)
        )
        st.storage.append(strax.DataDirectory("/project/lgrandi/xenonnt/processed/", readonly=True))

    print("")
    straxen.print_versions()

    print("\nStorage")
    for item in st.storage:
        print(f"- {item}")

    return st

def safe_is_stored(st, r, p):
    try:
        return st.is_stored(r, p)
    except (strax.DataCorrupted, strax.DataNotAvailable) as e:
        print(f"Error for run {r}: {e}")
        return False


# Function to calculate percentage of True values in the dataframe
def calculate_percentage(df, st, plugins):
    modes = df["mode"].unique()
    percentages = []

    for mode in modes:
        mode_df = df[df["mode"] == mode]
        mode_percentages = {"Mode": mode}

        for p in plugins:
            is_stored = np.array([safe_is_stored(st, r, p) for r in mode_df["name"]])
            tot_length = len(is_stored)
            _true = np.count_nonzero(is_stored)
            mode_percentages[f"{p}_available"] = (
                f"{_true}/{tot_length} ({100 * _true / tot_length:.2f}%)"
            )

        percentages.append(mode_percentages)

    return pd.DataFrame(percentages)


def main():
    print("<code>")
    # Get the current date
    current_date = datetime.today().date()
    print("\nToday's date is:", current_date)

    args = parse_args()

    print("")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # For `offline` context we try to install cutax
    if args.context == "offline":
        print("")
        print(f"Setting cutax: {args.cutax_location}")
        sys.path.append(args.cutax_location)
        import cutax

        st = initialize_straxen(args.context, args.global_config, args.container, cutax)
    else:
        st = initialize_straxen(args.context, args.global_config, args.container)

    # Prepare arguments for `select_runs`
    select_runs_kwargs = {"exclude_tags": args.exclude_tags}

    # Only add include_tags if provided
    if args.include_tags:
        select_runs_kwargs["include_tags"] = args.include_tags

    # Select runs
    selection = st.select_runs(**select_runs_kwargs)

    # Apply time filtering if --time-range is provided
    if args.time_range:
        start_date, end_date = pd.to_datetime(args.time_range)
        # Ensure column is in datetime format
        selection["start"] = pd.to_datetime(selection["start"])
        selection = selection[(selection["start"] >= start_date) & (selection["start"] <= end_date)]

    # Calculate and display the percentage table
    if args.plugins != None:
        percentage_df = calculate_percentage(selection, st, args.plugins)
    elif args.check_peaks:
        percentage_df = calculate_percentage(
            selection, st, ["lone_hits", "peaklets", "merged_s2s", "hitlets_nv"]
        )
    elif not args.check_peaks:
        percentage_df = calculate_percentage(selection, st, ["peak_basics", "event_basics"])

    print("")
    print(percentage_df)
    print("")
    print("</code>")


if __name__ == "__main__":
    main()
