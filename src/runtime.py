from __future__ import annotations

import numpy as np
import onnxruntime as ort


def create_onnx_session(onnx_path: str, providers: list[str] | None = None) -> ort.InferenceSession:
    providers = providers or ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=options, providers=providers)


def run_onnx(session: ort.InferenceSession, batch_np: np.ndarray, input_name: str = "input"):
    return session.run(None, {input_name: batch_np.astype(np.float32)})[0]
