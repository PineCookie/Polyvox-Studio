# Points uv.lock's download addresses at mirrors in mainland China, or back (run.bat calls this;
# run.sh does the same with sed). The mirrors keep the original paths, so only the host part
# changes: versions and hashes stay as locked, and uv checks every file against its hash.
# PyPI -> Tsinghua University, PyTorch's CUDA / CPU builds -> Nanjing University (ROCm stays).
param(
    [Parameter(Mandatory)] [string] $Lock,
    [Parameter(Mandatory)] [ValidateSet('mirror', 'official')] [string] $To
)
$ErrorActionPreference = 'Stop'
$text = [IO.File]::ReadAllText($Lock)
if ($To -eq 'mirror') {
    $new = $text -replace 'https://files\.pythonhosted\.org/packages/', 'https://pypi.tuna.tsinghua.edu.cn/packages/' `
        -replace 'https://download-r2\.pytorch\.org/whl/(cu\d+|cpu)/', 'https://mirrors.nju.edu.cn/pytorch/whl/$1/'
} else {
    $new = $text -replace 'https://pypi\.tuna\.tsinghua\.edu\.cn/packages/', 'https://files.pythonhosted.org/packages/' `
        -replace 'https://mirrors\.nju\.edu\.cn/pytorch/whl/', 'https://download-r2.pytorch.org/whl/'
}
if ($new -ne $text) {
    # Written aside, then swapped in: an interrupted write never leaves half a lock.
    [IO.File]::WriteAllText("$Lock.tmp", $new, [Text.UTF8Encoding]::new($false))
    Move-Item -Force "$Lock.tmp" $Lock
}
