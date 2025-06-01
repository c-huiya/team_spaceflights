# !/usr/bin/env bash
set -e
# "set -e" tells Bash to exit if any command fails.

# Build/rebuild Docker image whenever the code/requirements change:
docker build -t spaceflights-image .

# Run the container. Inside the container, `kedro run` is the default CMD.
# The part "--rm" ensures Docker delete the container as soon as it exits so there won't be alot of unused containers.
docker run --rm spaceflights-image
