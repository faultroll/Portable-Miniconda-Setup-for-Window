# envcheck_torch.py

class TestResult:
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.status = "NOT RUN"
        self.detail = ""
    def run(self, previous_failed=False):
        if previous_failed:
            self.status = "SKIP"
            self.detail = "Previous dependency failed"
            return
        try:
            self.detail = self.func()
            self.status = "PASS"
        except Exception as e:
            self.status = "FAIL"
            self.detail = str(e)
    def __str__(self):
        s = f"{self.name}: {self.status}"
        if self.detail:
            s += f" ({self.detail})"
        return s

# ----------------------------
# Test functions
# ----------------------------
def test_core_packages():
    import torch, torchvision, einops, tqdm, numpy, scipy, thop, onnx
    return ""  # no detail needed
def test_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    # Detect CUDA origin (Conda/System/Unknown)
    # import os
    # cuda_home = os.environ.get("CUDA_HOME", "")
    # print(cuda_home)
    # import ctypes
    # cuda_dlls = {}
    # for lib in ["cudart", "cublas", "cudnn"]:
    #     path = ctypes.util.find_library(lib)
    #     cuda_dlls[lib] = path
    # print(cuda_dlls)
    import dllist
    loaded = dllist.dllist()
    cuda_keys = ["cudart", "cublas", "cudnn"]
    cuda_dlls = {k: next((p for p in loaded if k in p.lower()), None) for k in cuda_keys}
    return f"CUDA available, version={torch.version.cuda}, origin=({cuda_dlls})"
def test_onnx_export():
    import io
    import torch
    import torch.nn as nn
    model = nn.Linear(3, 2)
    dummy_input = torch.randn(1, 3)
    f = io.BytesIO()
    torch.onnx.export(model, dummy_input, f, opset_version=13)
    f.seek(0)
    import onnx
    onnx_model = onnx.load_model_from_string(f.read())
    onnx.checker.check_model(onnx_model)
    return ""
def test_model_metrics():
    import torch
    import torch.nn as nn
    from thop import profile
    model = nn.Linear(3, 2)
    flops, params = profile(model, inputs=(torch.randn(1, 3),))
    return f"FLOPs={flops}, Params={params}"

# ----------------------------
# Table of tests
# ----------------------------
tests = [
    ("PyTorch Core Packages", test_core_packages),
    ("CUDA Availability", test_cuda),
    ("ONNX Availability", test_onnx_export),
    ("Model Metrics", test_model_metrics),
]

# ----------------------------
# Run tests
# ----------------------------
results = []
previous_failed = False
print("\n=== PyTorch Environment Test ===")
for idx, (name, func) in enumerate(tests, 1):
    test = TestResult(name, func)
    test.run(previous_failed)
    if test.status == "FAIL":
        previous_failed = True  # skip dependent tests
    print(f"[{idx}/{len(tests)}] {test}")
    results.append(test)

# ----------------------------
# Summary
# ----------------------------
passed = sum(1 for t in results if t.status == "PASS")
failed = sum(1 for t in results if t.status == "FAIL")
skipped = sum(1 for t in results if t.status == "SKIP")
print(f"\nSummary: {passed} PASS, {failed} FAIL, {skipped} SKIP\n")
