import argparse
import strax
import straxen
import pandas as pd
import sys
import numpy as np

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Process Strax context and calculate availability percentages.')
    
    parser.add_argument('--context', choices=['online', 'offline'], required=True, 
                        help='Choose the Strax context: online or offline.')
    
    parser.add_argument('--global_config', type=str, 
                        help='Global config for offline context (required for offline).')
    
    parser.add_argument('--include_tags', type=str, help='Tags to include, e.g., "*sr0*"')
    
    parser.add_argument('--exclude_tags', type=str, nargs='*', 
                        default=['flash', 'ramp_up', 'ramp_down', 'anode_off', 'abandon', 'hot_spot', 'missing_one_pmt', 'messy', 'bad'],
                        help='Tags to exclude (default: predefined tags)')
    
    parser.add_argument('--plugins', type=str, nargs='*', 
                        default=['peak_basics', 'event_basics'],
                        help='Plugins to include for availability calculation (default: peak_basics, event_basics)')
    
    parser.add_argument('--time-range', type=str, nargs=2, metavar=('START_DATE', 'END_DATE'),
                        help='Time range for filtering, format: YYYY-MM-DD YYYY-MM-DD (e.g., "2023-01-01 2023-12-31")')
    
    return parser.parse_args()

# Function to initialize Strax context
def initialize_strax(context_type, global_config, output_folder='./strax_data'):
    
    if context_type == 'online':
        st = straxen.contexts.xenonnt(output_folder=output_folder)
        
    elif context_type == 'offline':
        if not global_config:
            raise ValueError("Global config is required for offline context.")
        st = straxen.contexts.xenonnt(global_config, output_folder=output_folder)
    
    print('')
    straxen.print_versions()
    print('')
    
    return st

# Function to calculate percentage of True values in the dataframe
def calculate_percentage(df, st, plugins):
    modes = df['mode'].unique()
    percentages = []
    
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        mode_percentages = {'Mode': mode}
        
        for p in plugins:
            is_stored = np.array([st.is_stored(r, p) for r in mode_df['name']])
            tot_length = len(is_stored)
            _true = np.count_nonzero(is_stored)  # Faster way to count True values
            mode_percentages[f"{p}_available"] = f'{_true}/{tot_length} ({100 * _true / tot_length:.2f}%)'
        
        percentages.append(mode_percentages)
    
    return pd.DataFrame(percentages)

def main():
    args = parse_args()

    # Initialize Strax context
    st = initialize_strax(args.context, args.global_config)
    
    # Prepare arguments for `select_runs`
    select_runs_kwargs = {'exclude_tags': args.exclude_tags}
    # Only add include_tags if provided
    if args.include_tags:  
        select_runs_kwargs['include_tags'] = args.include_tags

    # Select runs
    selection = st.select_runs(**select_runs_kwargs)

    # Apply time filtering if --time-range is provided
    if args.time_range:
        start_date, end_date = pd.to_datetime(args.time_range)
        # Ensure column is in datetime format
        selection['start'] = pd.to_datetime(selection['start'])  
        selection = selection[(selection['start'] >= start_date) & (selection['start'] <= end_date)]
    
    # Calculate and display the percentage table
    percentage_df = calculate_percentage(selection, st, args.plugins)
    print(percentage_df)

if __name__ == '__main__':
    main()
