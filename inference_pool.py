"""Lease independent generation state while sharing immutable model weights."""
from __future__ import annotations

import copy
import queue
import threading
from contextlib import contextmanager


class InferencePool:
    def __init__(self, models):
        self.models = tuple(models)
        if not self.models:
            raise ValueError("An inference pool requires at least one model")
        self._available = queue.Queue()
        for model in self.models:
            self._available.put(model)

    @property
    def available(self):
        return self._available.qsize()

    @contextmanager
    def lease(self):
        model = self._available.get()
        try:
            yield model
        finally:
            self._available.put(model)


def serialize_codec_decode(model):
    """Reuse the large codec workspace across slots without serializing talkers."""
    import torch

    tokenizer = model.speech_tokenizer
    decode = tokenizer.decode
    lock = threading.Lock()

    def guarded_decode(*args, **kwargs):
        with lock:
            result = decode(*args, **kwargs)
            # Keep the lock until CUDA has released temporary decoder tensors;
            # otherwise the next thread can allocate another full workspace.
            torch.cuda.synchronize()
            return result

    tokenizer.decode = guarded_decode


def validate_fast_slots(models):
    """Fail startup if any active slots could write the same generation state."""
    if len(models) < 2:
        return
    states = [
        [id(model.model.model.talker) for model in models],
        [id(model.talker_graph.static_cache) for model in models],
        [id(model.predictor_graph.static_cache) for model in models],
        [model.talker_graph.input_buf.data_ptr() for model in models],
        [model.talker_graph.output_buf.data_ptr() for model in models],
        [model.predictor_graph.input_buf.data_ptr() for model in models],
        [model.predictor_graph.output_tokens.data_ptr() for model in models],
    ]
    if any(len(set(values)) != len(models) for values in states):
        raise RuntimeError("Inference slots share mutable generation state")

def fork_fast_model(original):
    """Copy mutable talker state and graphs; retain the original Parameter objects.

    The talker wrapper itself stores rope_deltas, in addition to the KV/cache
    buffers on its CUDA graphs. A separate FasterQwen wrapper alone is therefore
    insufficient. deepcopy with a Parameter memo isolates the module state
    without allocating another copy of the model weights.
    """
    from faster_qwen3_tts import FasterQwen3TTS
    from faster_qwen3_tts.predictor_graph import PredictorGraph
    from faster_qwen3_tts.talker_graph import TalkerGraph

    original_talker = original.model.model.talker
    memo = {id(parameter): parameter for parameter in original_talker.parameters()}
    talker = copy.deepcopy(original_talker, memo)
    original_parameters = dict(original_talker.named_parameters())
    for name, parameter in talker.named_parameters():
        if parameter is not original_parameters[name]:
            raise RuntimeError(f"Slot fork duplicated model weights: {name}")

    root = copy.copy(original.model.model)
    root._modules = dict(root._modules)
    root._parameters = dict(root._parameters)
    root._buffers = dict(root._buffers)
    root.talker = talker
    base = copy.copy(original.model)
    base.model = root

    config = root.config.talker_config
    predictor = talker.code_predictor
    old_predictor = original.predictor_graph
    predictor_graph = PredictorGraph(
        predictor, predictor.model.config, config.hidden_size,
        device=original.device, dtype=original.dtype,
        do_sample=old_predictor.do_sample, top_k=old_predictor.top_k,
        top_p=old_predictor.top_p, temperature=old_predictor.temperature,
    )
    talker_graph = TalkerGraph(
        talker.model, config, device=original.device, dtype=original.dtype,
        max_seq_len=original.max_seq_len,
    )
    worker = FasterQwen3TTS(
        base, predictor_graph, talker_graph, device=original.device,
        dtype=original.dtype, max_seq_len=original.max_seq_len,
    )
    # Cached prompts are read-only during generation; each slot owns its dict.
    worker._voice_prompt_cache = dict(original._voice_prompt_cache)
    return worker
