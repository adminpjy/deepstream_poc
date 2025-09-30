# test_gpu.py
import onnxruntime as ort
from onnxruntime import SessionOptions

print("Available providers:", ort.get_available_providers())

# 显式指定只使用 CUDA，排除 TensorRT 干扰
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

try:
    # 使用 SessionOptions 捕获更详细错误
    sess_options = SessionOptions()
    sess_options.log_severity_level = 0  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
    sess_options.log_verbosity_level = 1

    session = ort.InferenceSession(
        "models/hat.onnx",
        sess_options=sess_options,
        providers=providers
    )
    print("✅ Provider used:", session.get_providers())
except Exception as e:
    print("❌ Failed to create session with CUDA:")
    print(e)

    # 尝试只用 CPU
    try:
        session = ort.InferenceSession("/workspace/pt/yolov11n-face.onnx", providers=['CPUExecutionProvider'])
        print("✅ Fallback to CPUExecutionProvider")
    except Exception as e2:
        print("❌ Even CPU failed:", e2)