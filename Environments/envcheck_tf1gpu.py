# envcheck_tf1gpu.py
# TF1 Compatibility Test (using tf.compat.v1)
# Standalone Keras + tf2onnx + GPU + FLOPs

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
# TF1 Subtests
# ----------------------------
def core_packages_tests():
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    import keras
    import tensorflow_addons as tfa
    import numpy, scipy, tqdm, h5py
    def tf_placeholder_dense():
        tf.reset_default_graph()
        sess = tf.Session()
        a = tf.placeholder(tf.float32, shape=(None, 3))
        b = tf.layers.dense(a, 2)
        sess.run(tf.global_variables_initializer())
        _ = sess.run(b, feed_dict={a:[[1.,2.,3.]]})
        sess.close()
        tf.reset_default_graph()
    def keras_old():
        backend = keras.backend.backend()
        if backend != "tensorflow":
            raise RuntimeError(f"Keras backend mismatch: {backend}")
        m = keras.Sequential([keras.layers.Dense(2, input_shape=(3,))])
        # _ = m.predict([[1.,2.,3.]])
        import numpy as np
        _ = m.predict(np.array([[1.,2.,3.]]))
    def tfa_smoke():
        y = tf.constant([[1.,2.]])
        _ = tfa.activations.gelu(y)
    result = TestResult("TensorFlow Core Packages")
    result.add_subtest("TF1 placeholder + dense", tf_placeholder_dense)
    result.add_subtest("Standalone Keras", keras_old)
    result.add_subtest("TensorFlow Addons smoke test", tfa_smoke)
    return result
def cuda_tests():
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    def gpu_visible():
        if not tf.test.is_built_with_cuda():
            raise RuntimeError("TF not built with CUDA")
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("No GPU visible")
        with tf.device("/GPU:0"):
            a = tf.random_normal((1024,1024))
            b = tf.matmul(a, a)
    def dll_path():
        import dllist
        loaded = dllist.dllist()
        keys = ["cudart", "cublas", "cudnn"]
        cuda_dlls = {k: next((p for p in loaded if k in p.lower()), None) for k in keys}
        return cuda_dlls
    result = TestResult("CUDA Availability")
    result.add_subtest("GPU visible and runnable", gpu_visible)
    result.add_subtest("DLL path", dll_path)
    return result
def onnx_tests():
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    import tf2onnx
    import onnx
    def tf1_to_onnx():
        tf.reset_default_graph() # reuse=tf.AUTO_REUSE
        sess = tf.Session()
        a = tf.placeholder(tf.float32, shape=(None, 3), name="input")
        b = tf.layers.dense(a, 2, name="dense")
        sess.run(tf.global_variables_initializer())
        # --- freeze variables into constants ---
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            output_node_names=["dense/BiasAdd"]
        )
        # --- convert frozen graph to ONNX ---
        onnx_model, _ = tf2onnx.convert.from_graph_def(
            frozen_graph_def,
            input_names=["input:0"],
            output_names=["dense/BiasAdd:0"],
            opset=13
        )
        onnx.checker.check_model(onnx_model)
        sess.close()
        tf.reset_default_graph()
        return "ONNX export via frozen GraphDef OK"
    # def keras_to_onnx():
    #     import tensorflow.compat.v1 as tf
    #     tf.disable_eager_execution()
    #     import keras
    #     import tf2onnx
    #     import onnx
    #     model = keras.Sequential([keras.layers.Dense(2, input_shape=(3,))])
    #     spec = (tf.placeholder(tf.float32, shape=(None,3), name="input"),)
    #     onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    #     onnx.checker.check_model(onnx_model)
    result = TestResult("ONNX Export")
    result.add_subtest("TF1 graph -> ONNX", tf1_to_onnx)
    # result.add_subtest("Keras model -> ONNX", keras_to_onnx)
    return result
def model_metrics_tests():
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
    def flops_params():
        tf.reset_default_graph()
        sess = tf.Session()
        a = tf.placeholder(tf.float32, shape=(None,3))
        b = tf.layers.dense(a, 2)
        sess.run(tf.global_variables_initializer())
        opts = ProfileOptionBuilder.float_operation()
        flops = profile(sess.graph, options=opts).total_float_ops
        params = sum([v.get_shape().num_elements() for v in tf.trainable_variables()])
        sess.close()
        tf.reset_default_graph()
        return f"FLOPs={flops}, Params={params}"
    result = TestResult("Model Metrics")
    result.add_subtest("FLOPs and Params", flops_params)
    return result

# ----------------------------
# Run all tests
# ----------------------------
all_tests = [
    core_packages_tests(),
    cuda_tests(),
    onnx_tests(),
    model_metrics_tests()
]

print("\n=== TensorFlow 1 Compatibility Test ===")
for idx, t in enumerate(all_tests, 1):
    print(f"[{idx}/{len(all_tests)}] Start testing {t.name} ...")
    t.run()

# Summary
print("\n=== TensorFlow 1 Compatibility Summary ===")
passed = sum(t.status=="PASS" for t in all_tests)
failed = sum(t.status=="FAIL" for t in all_tests)
skipped = sum(t.status=="SKIP" for t in all_tests)
print(f"{passed} PASS, {failed} FAIL, {skipped} SKIP\n")
for t in all_tests:
    t.summary()
