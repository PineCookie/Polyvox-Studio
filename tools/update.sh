#!/usr/bin/env bash
# Updates a Polyvox Studio folder to the latest release. update.sh runs this from a temporary copy
# (so this file can be replaced too). Only the program files are replaced: models, data (datasets),
# lora (voices), output and .venv (the Python environment) stay as they are; the next run.sh
# installs what the new version needs.
#   tools/update.sh FOLDER [--archive RELEASE.tar.gz] [--yes]
set -euo pipefail
repo=PineCookie/Polyvox-Studio  # also in src/polyvox/updates.py
keep=" models data lora output .venv "

folder=$1
shift
archive="" yes=""
while [ $# -gt 0 ]; do
  case $1 in
    --archive) archive=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2 ;;
    --yes) yes=1; shift ;;
    *) echo "Unknown option $1"; exit 2 ;;
  esac
done
folder=$(cd "$folder" && pwd)
if [ -e "$folder/.git" ]; then echo "This folder is a git checkout: update it with git pull."; exit 1; fi
if [ ! -f "$folder/pyproject.toml" ]; then echo "$folder doesn't look like a Polyvox Studio folder."; exit 1; fi
# Unpacked inside the folder: every mv below is then a rename on one filesystem, which happens
# whole or not at all (across filesystems mv copies, and a failure could split a folder).
work="$folder/.update-work"
rm -rf "$work"
mkdir "$work"
trap 'rm -rf "$work"; if [ -n "${UPDATER_SELF:-}" ]; then rm -f "$UPDATER_SELF"; fi' EXIT
current=$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$folder/pyproject.toml" | head -n 1)
# A running studio uses these files (and would serve half-replaced ones).
if pgrep -f "$folder/.venv/" >/dev/null 2>&1; then
  echo "Polyvox Studio is running: stop it first (Ctrl+C in its terminal), then run update.sh again."
  exit 1
fi

if [ -n "$archive" ]; then
  label=$(basename "$archive")
else
  echo "Looking for the latest release..."
  json=$(curl -fsSL -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$repo/releases/latest")
  tag=$(printf '%s' "$json" | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)
  page=$(printf '%s' "$json" | sed -n 's/.*"html_url": *"\([^"]*\/releases\/tag\/[^"]*\)".*/\1/p' | head -n 1)
  url=$(printf '%s' "$json" | grep -o '"browser_download_url": *"[^"]*\.tar\.gz"' | sed 's/.*"\(http[^"]*\)"$/\1/' | head -n 1)
  latest=${tag#v}
  if [ "$(printf '%s\n%s\n' "$current" "$latest" | sort -V | tail -n 1)" = "$current" ]; then
    echo "Polyvox Studio $current is up to date."
    exit 0
  fi
  if [ -z "$url" ]; then echo "Release $tag has no .tar.gz."; exit 1; fi
  label="$latest ($page)"
fi
echo "Polyvox Studio $current -> $label"
if [ -z "$yes" ]; then
  read -r -p "Update now? Models, datasets and voices stay. [Y/n] " answer
  case $answer in [nN]*) exit 0 ;; esac
fi
if [ -z "$archive" ]; then
  archive="$work/release.tar.gz"
  echo "Downloading $(basename "$url")..."
  curl -fL --progress-bar -o "$archive" "$url"
fi
mkdir "$work/new"
tar -xzf "$archive" -C "$work/new"
shopt -s dotglob nullglob
new="$work/new"
dirs=("$work"/new/*/)
if [ ! -f "$new/pyproject.toml" ] && [ ${#dirs[@]} -eq 1 ]; then new=${dirs[0]%/}; fi
if [ ! -f "$new/pyproject.toml" ]; then echo "$archive holds no Polyvox Studio."; exit 1; fi

# The old program files go to a backup first (put back on any error), then the new ones in.
names=()
for path in "$new"/*; do
  name=$(basename "$path")
  case $keep in *" $name "*) continue ;; esac
  case $name in .update-*) continue ;; esac
  names+=("$name")
done
backup="$folder/.update-backup"
rm -rf "$backup"
mkdir "$backup"
moved=() installed=()
restore() {
  trap - ERR
  echo "Updating failed; putting the old files back."
  for name in "${installed[@]}"; do rm -rf "${folder:?}/$name"; done
  for name in "${moved[@]}"; do mv "$backup/$name" "$folder/$name"; done
  rm -rf "$backup"
  exit 1
}
trap restore ERR
for name in "${names[@]}"; do
  if [ -e "$folder/$name" ]; then mv "$folder/$name" "$backup/$name"; moved+=("$name"); fi
done
for name in "${names[@]}"; do mv "$new/$name" "$folder/$name"; installed+=("$name"); done
trap - ERR
rm -rf "$backup"
echo "Updated. Start ./run.sh: it first installs what the new version needs (a few minutes)."
