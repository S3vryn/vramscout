from vramscout.gpu import _parse_nvidia_smi_line


def test_parse_nvidia_smi_line():
    gpu = _parse_nvidia_smi_line("0, NVIDIA RTX 5090, 32607, 2048, 30559")
    assert gpu.index == 0
    assert gpu.name == "NVIDIA RTX 5090"
    assert round(gpu.total_gib, 2) == round(32607 / 1024, 2)
    assert round(gpu.free_gib, 2) == round(30559 / 1024, 2)
