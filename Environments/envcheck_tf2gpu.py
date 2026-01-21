# envcheck_tf2gpu.py
# Checks TensorFlow 2.x environment compatibility (eager execution, tf.keras)

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
    import tensorflow as tf
    import tensorflow_addons as tfa
    import numpy, scipy, tqdm, h5py
    # TF2 eager + tf.keras check
    model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(3,))])
    x = tf.random.normal((1, 3))
    _ = model(x)
    # TFA smoke test
    y = tf.constant([[1., 2.]])
    _ = tfa.activations.gelu(y)
    return f"TF={tf.__version__}, Keras=tensorflow"
def test_cuda():
    import tensorflow as tf
    if not tf.test.is_built_with_cuda():
        raise RuntimeError("TensorFlow not built with CUDA")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No GPU visible to TensorFlow")
    # Ensure op runs on GPU
    with tf.device("/GPU:0"):
        a = tf.random.normal((1024, 1024))
        b = tf.matmul(a, a)
    import dllist
    loaded = dllist.dllist()
    keys = ["cudart", "cublas", "cudnn"]
    cuda_dlls = {k: next((p for p in loaded if k in p.lower()), None) for k in keys}
    return f"GPU OK, CUDA DLLs={cuda_dlls}"
def test_onnx_export():
    import tensorflow as tf
    import tf2onnx
    import onnx
    model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(3,))])
    spec = (tf.TensorSpec((None, 3), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.checker.check_model(onnx_model)
    return ""
def test_model_metrics():
    import tensorflow as tf
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
    model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(3,))])
    concrete = tf.function(model).get_concrete_function(tf.TensorSpec((1, 3), tf.float32))
    opts = ProfileOptionBuilder.float_operation()
    flops = profile(concrete.graph, options=opts).total_float_ops
    params = model.count_params()
    return f"FLOPs={flops}, Params={params}"

# ----------------------------
# Table of tests
# ----------------------------
tests = [
    ("TensorFlow Core Packages", test_core_packages),
    ("CUDA Availability", test_cuda),
    ("ONNX Export", test_onnx_export),
    ("Model Metrics", test_model_metrics),
]

# ----------------------------
# Run tests
# ----------------------------
results = []
previous_failed = False
print("\n=== TensorFlow 2 GPU Environment Test ===")
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
passed = sum(t.status == "PASS" for t in results)
failed = sum(t.status == "FAIL" for t in results)
skipped = sum(t.status == "SKIP" for t in results)
print(f"\nSummary: {passed} PASS, {failed} FAIL, {skipped} SKIP\n")
