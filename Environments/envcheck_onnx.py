# envcheck_onnx.py
# ONNX Environment Test (onnx + onnxruntime-gpu + onnxsim + GPU + FLOPs analysis)

class TestResult:
    def __init__(self, name):
        self.name = name
        self.subresults = []
        self.status = "NOT RUN"
    def add_subtest(self, desc, func):
        self.subresults.append({"desc": desc, "func": func, "status": "NOT RUN", "detail": ""})
    def run(self):
        any_fail = False
        for sr in self.subresults:
            try:
                ret = sr["func"]()
                sr["status"] = "PASS"
                if ret:
                    sr["detail"] = str(ret)
            except Exception as e:
                sr["status"] = "FAIL"
                sr["detail"] = str(e)
                any_fail = True
            print(f"    [{sr['status']}] {sr['desc']}" + (f" ({sr['detail']})" if sr['detail'] else ""))
        self.status = "FAIL" if any_fail else "PASS"
        return self.status
    def summary(self):
        print(f"{self.name}: {self.status}")
        for sr in self.subresults:
            print(f"    [{sr['status']}] {sr['desc']}" + (f" ({sr['detail']})" if sr['detail'] else ""))
    def __str__(self):
        return f"{self.name}: {self.status}"

# ----------------------------
# ONNX Subtests
# ----------------------------
def core_packages_tests():
    def loadpkgs():
        import onnx
        import onnxruntime as ort
        import onnxsim
        import tqdm
        import numpy, scipy
        return ort.__version__
    result = TestResult("ONNX Core Packages")
    result.add_subtest("import packages", loadpkgs)
    return result
def cuda_tests():
    def gpu_available():
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(f"CUDAExecutionProvider not available, got {providers}")
        return f"Available providers: {providers}"
    def dll_path():
        # We need to start an InferenceSession, ort.preload_dlls(cuda=True, cudnn=True) is not available in current version
        # import onnxruntime as ort
        # print(ort.__version__)
        # print(ort.get_device())
        # session = ort.InferenceSession("xxx.onnx", providers=['CUDAExecutionProvider'])
        # print(session.get_providers())
        import dllist
        loaded = dllist.dllist()
        keys = ["cudart", "cublas", "cudnn"]
        cuda_dlls = {k: next((p for p in loaded if k in p.lower()), "Not loaded") for k in keys}
        return cuda_dlls
    result = TestResult("CUDA Availability")
    result.add_subtest("GPU check", gpu_available)
    result.add_subtest("DLL path", dll_path)
    return result
def model_metrics_tests():
    def dummy_flops():
        import onnx
        import onnx_tool as ot
        # Build a minimal ONNX model and analyze flops
        from onnx import helper, TensorProto
        import numpy as np
        # --- Layer 1 ---
        node1 = helper.make_node("MatMul", ["X", "W1"], ["H"])
        W1_init = np.random.rand(3, 4).astype(np.float32)  # 3x4
        W1 = helper.make_tensor("W1", TensorProto.FLOAT, W1_init.shape, W1_init.flatten())
        # --- Layer 2 ---
        node2 = helper.make_node("MatMul", ["H", "W2"], ["Y"])
        W2_init = np.random.rand(4, 2).astype(np.float32)  # 4x2
        W2 = helper.make_tensor("W2", TensorProto.FLOAT, W2_init.shape, W2_init.flatten())
        # --- Inputs / Outputs ---
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
        # --- Build graph ---
        graph = helper.make_graph(
            nodes=[node1, node2],
            name="dummy",
            inputs=[X],
            outputs=[Y],
            initializer=[W1, W2]
        )
        model = helper.make_model(graph)
        # --- Profile with onnx-tool ---
        # ot.model_profile(model)
        m = ot.Model(model)
        m.graph.shape_infer({'X': np.zeros((1, 3))})
        m.graph.profile()
        m.graph.print_node_map()
        macs = m.graph.macs[0]
        params = m.graph.params
        flops = macs * 2
        return f"FLOPs={flops}, Params={params}"
    result = TestResult("Model Metrics")
    result.add_subtest("Dummy model Profile", dummy_flops)
    return result

# ----------------------------
# Run all tests
# ----------------------------
all_tests = [
    core_packages_tests(),
    cuda_tests(),
    model_metrics_tests()
]

print("\n=== ONNX Environment Test ===")
for idx, t in enumerate(all_tests, 1):
    print(f"[{idx}/{len(all_tests)}] Start testing {t.name} ...")
    t.run()

# Summary
print("\n=== ONNX Environment Summary ===")
passed = sum(t.status=="PASS" for t in all_tests)
failed = sum(t.status=="FAIL" for t in all_tests)
skipped = sum(t.status=="SKIP" for t in all_tests)
print(f"{passed} PASS, {failed} FAIL, {skipped} SKIP\n")
for t in all_tests:
    t.summary()
