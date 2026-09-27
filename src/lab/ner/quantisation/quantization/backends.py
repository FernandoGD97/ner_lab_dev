"""Optional backend capability reporting without importing optional packages."""
import importlib.util
def available_backends(): return {name: importlib.util.find_spec(name) is not None for name in ('torch','onnx','onnxruntime','bitsandbytes')}
