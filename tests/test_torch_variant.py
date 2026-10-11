"""The run scripts' choice of PyTorch build."""

import pytest

from polyvox import torch_variant
from polyvox.torch_variant import choose


@pytest.mark.parametrize(
    ("gpus", "variant", "note"),
    [
        (((13, 0), [(8, 9), (12, 0)]), "cu130", False),  # current driver, Ada + Blackwell
        (((13, 0), []), "cu130", False),  # capabilities not reported: trust the driver
        (((13, 0), [(6, 1)]), "cu126", False),  # Pascal: not in CUDA 13
        (((12, 8), [(8, 6)]), "cu126", False),  # older driver
        (((12, 8), [(12, 0)]), "cu130", True),  # Blackwell on an old driver: update it
        (((12, 2), [(7, 5)]), "cpu", True),  # driver too old for both builds
        ((None, [(12, 0)]), "cu130", False),  # driver version not reported: by the GPU
        ((None, [(6, 1)]), "cu126", False),
        (None, "cpu", True),  # no NVIDIA GPU
    ],
)
def test_choice(gpus, variant, note):
    chosen, message = choose(gpus)
    assert chosen == variant and (message is not None) == note


def test_amd_gpus_get_rocm_unless_an_nvidia_one_is_there_too():
    assert choose(None, amd=True) == ("rocm", None)
    assert choose(((13, 0), [(8, 9)]), amd=True)[0] == "cu130"


def test_override(monkeypatch, capsys):
    monkeypatch.setenv(torch_variant.ENV, "CPU")
    torch_variant.main()
    assert capsys.readouterr().out.strip() == "cpu"

    monkeypatch.setenv(torch_variant.ENV, "cu999")
    with pytest.raises(SystemExit, match="must be one of"):
        torch_variant.main()
