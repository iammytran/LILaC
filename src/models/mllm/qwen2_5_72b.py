import argparse
import os
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.utils.utils import REPO_ROOT


#
# NOTE: The class/file are named "72B" for historical reasons but the
# underlying checkpoint is configurable here. For machines that can't fit
# Qwen2.5-72B-Instruct + 4×GPU tensor_parallel, the default below uses the
# 7B variant. Override at run time via the ``QWEN_TEXT_LLM`` env var to
# switch back to the 72B checkpoint (set it to an absolute path or repo id):
#
#     QWEN_TEXT_LLM=/root/lilac/models/Qwen2.5-72B-Instruct ./run.sh
#
_DEFAULT_QWEN_PATH = f"{REPO_ROOT}/models/Qwen2.5-7B-Instruct"
QWEN_2_5_72B_PATH = os.environ.get("QWEN_TEXT_LLM", _DEFAULT_QWEN_PATH)


def _auto_tensor_parallel_size() -> int:
    """Pick the largest tensor_parallel_size that (a) <= number of visible
    GPUs and (b) divides the model's attention-head count.

    Qwen2.5-7B has 28 heads -> divisors {1,2,4,7,14,28}; safe set {1,2,4}.
    Qwen2.5-72B has 64 heads -> divisors {1,2,4,8,16,32,64}; safe set {1,2,4}.
    Either way TP in {1,2,4} works.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip():
        n_visible = len([t for t in cvd.split(",") if t.strip()])
    else:
        try:
            import torch
            n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 1
        except Exception:
            n_visible = 1
    if n_visible >= 4:
        return 4
    if n_visible >= 2:
        return 2
    return 1


class Qwen2_5_72B:
    """
    Light wrapper around a Qwen2.5-Instruct text model with a batch-friendly
    ``infer`` method. API intentionally matches ``Qwen2_5_VL`` (no multimodal
    kwargs). Despite the name, the checkpoint path is configurable via
    ``QWEN_2_5_72B_PATH`` above — defaults to Qwen2.5-7B-Instruct.
    """

    def __init__(
        self,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        system_prompt: str = "You are a helpful assistant.",
    ):
        if tensor_parallel_size is None:
            tensor_parallel_size = _auto_tensor_parallel_size()
            print(f"[Qwen2_5_72B] auto tensor_parallel_size={tensor_parallel_size} "
                  f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})")
        if gpu_memory_utilization is None:
            # 7B fits comfortably at 0.7 (F-014). 72B needs ~35 GB/GPU just for
            # weights at TP=4 + room for KV cache, so push to 0.92. Override
            # explicitly via QWEN_GPU_MEM_UTIL if needed.
            env_util = os.environ.get("QWEN_GPU_MEM_UTIL")
            if env_util:
                gpu_memory_utilization = float(env_util)
            elif "72B" in QWEN_2_5_72B_PATH:
                gpu_memory_utilization = 0.92
            else:
                gpu_memory_utilization = 0.7
            print(f"[Qwen2_5_72B] gpu_memory_utilization={gpu_memory_utilization}")
        # Chat template / tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_2_5_72B_PATH, trust_remote_code = True
        )
        # vLLM engine
        self.llm = LLM(
            model=QWEN_2_5_72B_PATH,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=30000,
            max_num_seqs=32,  # Reduce from default 256 to something manageable
        )
        self.system_prompt = system_prompt

    # ────────────────────────────────────────────────────────────────────────
    # Public inference API
    # -----------------------------------------------------------------------

    def infer(
        self,
        objects: List[Dict[str, Any]],
        *,
        max_tokens: int = 256,
        batch_size: int = 4,
        temperature: float = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
    ) -> List[str]:
        """
        Args
        ----
        objects : List[Dict]
            Each dict must have a `"text"` field with the prompt content.
            Any other keys are ignored.
        Returns
        -------
        List[str] – generated responses, same order as inputs.
        """
        self.sampling_params = SamplingParams(
            temperature = temperature,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            max_tokens = max_tokens
        )

        results: List[str] = []

        for start in tqdm(range(0, len(objects), batch_size), desc = "Inference Batches"):
            
            batch_objs = objects[start : start + batch_size]

            llm_inputs = []
            for obj in batch_objs:
                # Build standard chat message list
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": obj["text"]},
                ]
                # Convert to single prompt string via chat template
                prompt = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
                llm_inputs.append({"prompt": prompt})

            batch_outputs = self.llm.generate(llm_inputs, sampling_params = self.sampling_params)
            for out in batch_outputs:
                results.append(out.outputs[0].text)

        return results


# ───────────────────────────────────────────────────────────────────────────────
# CLI demo (optional)
# ───────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    
    
    qwen = Qwen2_5_72B()
    
    prompts = [
        {"text": "What is the capital of France?"},
        {"text": "What is the capital of Japan?"},
        {"text": "What is the capital of South Korea?"},
        {"text": "What is the capital of China?"},
        {"text": "What is the capital of the United States?"},
        {"text": "What is the capital of Canada?"},
        {"text": "What is the capital of Australia?"},
        {"text": "What is the capital of Germany?"},
        {"text": "What is the capital of Italy?"},
        {"text": "What is the capital of Spain?"},
    ] * 100

    outputs = qwen.infer(prompts, batch_size = 32, max_tokens = 50)

    for i, out in enumerate(outputs, 1):
        print(f"\n===== Output {i} =====\n{out}\n")

    pass