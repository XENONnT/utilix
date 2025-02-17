# XEFIND - Data Availability Checker

## Overview
`xefind` is a command-line tool for checking data availability in the XENON experiment database. It is part of the `utilix` package and provides functionalities to query run data, verify storage locations, and calculate livetime.

## Installation
Since `xefind` is included in `utilix`, ensure you have `utilix` installed. Refer to the main `utilix` documentation for installation instructions.

## Usage
Once `utilix` is installed, you can run `xefind` directly from the command line:

```bash
xefind --help
```

This will display the available options and usage instructions.

## Command-Line Options
`xefind` supports multiple options for querying data:

### 1. Checking Data Availability from a File
```bash
xefind <data_type> --filename <path_to_runlist>
```
- `data_type`: The type of data to check (e.g., `peaklets`, `event_info`)
- `--filename`: Path to a file containing run IDs, one per line

Example:
```bash
xefind peaklets --filename runs.txt
```

### 2. Checking Data Availability for a Specific Source in a Science Run
```bash
xefind <data_type> --source <source_name> --science_run <run_name>
```
- `--source`: Source of the data (e.g., `none`, `th-232`)
- `--science_run`: Science run to check (`sr0`, `sr1`, `sr2`, etc.)

Example:
```bash
xefind peaklets --source none --science_run sr1
```

### 3. Checking Data Availability for a Specific Run ID
```bash
xefind <data_type> --run_id <run_id>
```
Example:
```bash
xefind event_info --run_id 123456
```

### 4. Saving Run IDs to a File
If using `--source` and `--science_run`, you can save the retrieved run IDs:
```bash
xefind <data_type> --source <source> --science_run <run> --save_runs
```
This will create a `runlists` folder in the script directory and store the run IDs.

### 5. Checking Data Availability with Livetime Calculation
```bash
xefind <data_type> --source <source> --science_run <run> --livetime
```
Instead of counting the number of runs, this computes the total livetime (in days).

### 6. Specifying an Additional Storage Location
```bash
xefind <data_type> --source <source> --science_run <run> --extra_location <location>
```
This checks an extra data storage location beyond the default ones (`UC_DALI_USERDISK`, `UC_MIDWAY_USERDISK`).

### 7. Debug Mode
Enable debug logging with:
```bash
xefind <data_type> --debug
```

## Example Use Cases
- Checking `event_info` availability for `sr1`:
  ```bash
  xefind event_info --source none --science_run sr1
  ```
- Checking `peaklets` availability from a file:
  ```bash
  xefind peaklets --filename my_runs.txt
  ```
- Checking `event_info` for a single run:
  ```bash
  xefind event_info --run_id 987654
  ```
- Checking `event_info` availability and saving runs:
  ```bash
  xefind event_info --source none --science_run sr1 --save_runs
  ```

## Notes
- If no lineage hash is found for a given context and environment, `xefind` will attempt to infer it using predefined environment versions.
- If no run IDs are provided, the script will raise an error.

## Further Information
For more details on `utilix` installation and configuration, refer to the official `utilix` documentation.

