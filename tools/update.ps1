# Updates a Polyvox Studio folder to the latest release. update.bat runs this from a temporary copy
# (so this file can be replaced too). Only the program files are replaced: models, data (datasets),
# lora (voices), output and .venv (the Python environment) stay as they are; the next run.bat
# installs what the new version needs.
param(
    [Parameter(Mandatory)] [string] $Folder,
    [string] $Archive = '',  # a release .zip to install instead of downloading the latest
    [switch] $Yes            # don't ask before updating
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest's progress display slows it down a lot
$Repo = 'PineCookie/Polyvox-Studio'       # also in src/polyvox/updates.py
$Keep = @('models', 'data', 'lora', 'output', '.venv')

$Folder = (Resolve-Path $Folder).Path.TrimEnd('\')
if (Test-Path (Join-Path $Folder '.git')) {
    Write-Host 'This folder is a git checkout: update it with git pull.'
    exit 1
}
$pyproject = Join-Path $Folder 'pyproject.toml'
if (-not (Test-Path $pyproject)) {
    Write-Host "$Folder doesn't look like a Polyvox Studio folder."
    exit 1
}
$current = [regex]::Match((Get-Content $pyproject -Raw), '(?m)^version\s*=\s*"([^"]+)"').Groups[1].Value

# A running studio uses these files (and would serve half-replaced ones).
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('polyvox.exe', 'python.exe') -and $_.CommandLine -and $_.CommandLine.Contains($Folder)
}
if ($running) {
    Write-Host 'Polyvox Studio is running: close it first (its window, or Ctrl+C in it), then run update.bat again.'
    exit 1
}

# Unpacked inside the folder: every move below is then a rename on one drive, which happens whole or
# not at all (Move-Item falls back to moving files one by one, so a failure could split a folder).
$work = Join-Path $Folder '.update-work'
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory $work | Out-Null

function Rename-Whole([string] $from, [string] $to) {
    # A file open inside (a virus scanner reading new files, an editor, a terminal in a subfolder)
    # blocks renaming a folder on Windows: try again for a few seconds before giving up.
    for ($try = 1; ; $try++) {
        try {
            if (Test-Path $from -PathType Container) { [IO.Directory]::Move($from, $to) } else { [IO.File]::Move($from, $to) }
            return
        } catch {
            if ($try -ge 20) { throw }
            Start-Sleep -Milliseconds 500
        }
    }
}
try {
    if ($Archive) {
        $zip = (Resolve-Path $Archive).Path
        $label = Split-Path $zip -Leaf
    } else {
        Write-Host 'Looking for the latest release...'
        $release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ Accept = 'application/vnd.github+json' }
        $latest = $release.tag_name.TrimStart('v')
        if ([version]$latest -le [version]$current) {
            Write-Host "Polyvox Studio $current is up to date."
            exit 0
        }
        $asset = $release.assets | Where-Object { $_.name -like '*.zip' } | Select-Object -First 1
        if (-not $asset) { throw "Release $($release.tag_name) has no .zip." }
        $label = "$latest ($($release.html_url))"
    }
    Write-Host "Polyvox Studio $current -> $label"
    if (-not $Yes) {
        $answer = Read-Host 'Update now? Models, datasets and voices stay. [Y/n]'
        if ($answer -match '^\s*[nN]') { exit 0 }
    }
    if (-not $Archive) {
        $zip = Join-Path $work $asset.name
        Write-Host ("Downloading {0} ({1:N0} MB)..." -f $asset.name, ($asset.size / 1MB))
        Invoke-WebRequest $asset.browser_download_url -OutFile $zip
    }
    $unpacked = Join-Path $work 'new'
    # .NET's extraction directly: Windows PowerShell 5.1's Expand-Archive fails on some zips (these too).
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($zip, $unpacked)
    $new = $unpacked
    $dirs = @(Get-ChildItem $unpacked -Directory)
    if (-not (Test-Path (Join-Path $unpacked 'pyproject.toml')) -and $dirs.Count -eq 1) { $new = $dirs[0].FullName }
    if (-not (Test-Path (Join-Path $new 'pyproject.toml'))) { throw "$zip holds no Polyvox Studio." }

    # The old program files go to a backup first (put back on any error), then the new ones in.
    $items = @(Get-ChildItem $new -Force | Where-Object { $Keep -notcontains $_.Name -and $_.Name -notlike '.update-*' })
    $backup = Join-Path $Folder '.update-backup'
    if (Test-Path $backup) { Remove-Item -Recurse -Force $backup }
    New-Item -ItemType Directory $backup | Out-Null
    $moved = @()
    $installed = @()
    $phase = 'backup'
    try {
        foreach ($item in $items) {
            $target = Join-Path $Folder $item.Name
            if (Test-Path $target) {
                Rename-Whole $target (Join-Path $backup $item.Name)
                $moved += $item.Name
            }
        }
        $phase = 'install'
        foreach ($item in $items) {
            Rename-Whole $item.FullName (Join-Path $Folder $item.Name)
            $installed += $item.Name
        }
    } catch {
        Write-Host "Updating failed ($($_.Exception.Message.Trim())); putting the old files back."
        if ($phase -eq 'install') {
            foreach ($name in $installed) { Remove-Item -Recurse -Force (Join-Path $Folder $name) }
        }
        foreach ($name in $moved) { Rename-Whole (Join-Path $backup $name) (Join-Path $Folder $name) }
        Remove-Item -Recurse -Force $backup
        Write-Host 'Nothing was changed. If a program has a file of this folder open (an editor, a terminal in it), close it and run update.bat again.'
        exit 1
    }
    Remove-Item -Recurse -Force $backup
    Write-Host 'Updated. Start run.bat: it first installs what the new version needs (a few minutes).'
} finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
