"""ONNX Runtime session creation with a provider policy: auto = CUDA -> CPU. Each session records which provider it
actually got and why a fallback happened, so `HindiOCR.ready()` can report it."""
import os
import onnxruntime as ort


def _preload_cuda_dlls():
    """onnxruntime-gpu loads CUDA / cuDNN from the nvidia-*-cu12 pip wheels when asked to."""
    try:
        ort.preload_dlls()
    except Exception:
        pass


def run_options(info):
    """Per-run options: on CUDA, give unused arena chunks back after each run (fights fragmentation from varying shapes)."""
    ro = ort.RunOptions()
    if info.get("provider") == "cuda":
        ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    return ro


def make_session(model_path, policy="auto", cuda_mem_limit_gb=2):
    """-> (session, info) where info = {"provider": "cuda"|"cpu", "fallbacks": [reasons]}."""
    policy = "cpu" if os.environ.get("HINDI_OCR_CPU") == "1" else (policy or "auto")
    available = ort.get_available_providers()
    fallbacks = []
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if policy in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        _preload_cuda_dlls()
        opts = {"gpu_mem_limit": int(cuda_mem_limit_gb * (1 << 30)),   # default kNextPowerOfTwo growth: fewer fragmentation failures on big det inputs
                "cudnn_conv_algo_search": "HEURISTIC"}
        try:
            sess = ort.InferenceSession(str(model_path), so, providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"])
            if "CUDAExecutionProvider" in sess.get_providers():
                return sess, {"provider": "cuda", "fallbacks": fallbacks}
            fallbacks.append("cuda provider not active after session creation")
        except Exception as e:
            fallbacks.append(f"cuda provider failed ({type(e).__name__}: {str(e)[:120]})")
    elif policy in ("auto", "cuda"):
        fallbacks.append("CUDAExecutionProvider not available in this onnxruntime build")
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    return sess, {"provider": "cpu", "fallbacks": fallbacks}
