#!/bin/bash

# Default usage message
usage() {
    echo "Usage: $0 --container <container_name> [other arguments for Python script]"
    exit 1
}

# Check if at least one argument is provided
if [ $# -lt 2 ]; then
    usage
fi

# Parse arguments
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

shift
# Run everything inside a subshell to avoid modifying the parent shell
(
    source /cvmfs/xenon.opensciencegrid.org/releases/nT/${CONTAINER}/setup.sh
    python3 /home/gvolta/XENONnT/DataAvaibility/data_avaibility.py "${OTHER_ARGS[@]}"
)
