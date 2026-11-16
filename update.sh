#!/usr/bin/env bash
# Updates this folder to the latest Polyvox Studio release. Models, datasets (data), voices (lora)
# and the Python environment stay; stop the studio first. The work is done by tools/update.sh,
# run from a temporary copy so that it can replace this folder's files.
set -e
here=$(cd "$(dirname "$0")" && pwd)
updater=$(mktemp)
cp "$here/tools/update.sh" "$updater"
UPDATER_SELF=$updater exec bash "$updater" "$here" "$@"
