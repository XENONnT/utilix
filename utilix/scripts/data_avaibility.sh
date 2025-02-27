#!/bin/bash

# Default usage message
usage() {
    echo "Usage: $0 --container <container_name> [other arguments for Python script]"
    exit 1
}

# Ensure at least two arguments are provided (container and other args)
if [ $# -lt 2 ]; then
    usage
fi

# Initialize variables
CONTAINER=""
OTHER_ARGS=()

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --container)
            CONTAINER="$2"
            shift 2  # Move past --container and its value
            ;;
        *)
            OTHER_ARGS+=("$1")  # Store all other arguments
            shift
            ;;
    esac
done

# Ensure container name is provided
if [ -z "$CONTAINER" ]; then
    echo "Error: --container argument is required."
    usage
fi

# Set up the environment
current_dir=$(pwd)

# Run everything inside a subshell to avoid modifying the parent shell
(
    # Source the setup script for the container
    source /cvmfs/xenon.opensciencegrid.org/releases/nT/${CONTAINER}/setup.sh

    _CUTAX_LOCATION="${CUTAX_LOCATION}/cutax"
    
    # Run the Python script with all arguments, including --container
    python3 "$current_dir/data_avaibility.py" --container "$CONTAINER" --cutax_location "${_CUTAX_LOCATION}" "${OTHER_ARGS[@]}"
)
