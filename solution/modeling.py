"""QLoRA setup and complete adapter/projector persistence."""
import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

PROJECTOR = ("visual.merger", "visual.deepstack_merger_list")
LLM_PATTERN = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
VIT_PATTERN = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"


def is_projector(name):
    return any(part in name for part in PROJECTOR)


def load_base(model_id):
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA: select a Colab T4 GPU runtime")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True,
                              # Transformers 5.19 matches full module prefixes;
                              # "visual" alone does not match "model.visual".
                              llm_int8_skip_modules=["visual", "model.visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    for name, module in model.named_modules():
        if ("visual" in name or name == "lm_head") and "4bit" in type(module).__name__.lower():
            raise RuntimeError(f"Unexpected quantized vision/output module: {name}")
    return model, processor


def apply_setup(model, config):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    # PEFT preparation promotes all unquantized parameters. Keep frozen vision
    # and tied output/embedding storage in fp16; LoRA trainable weights use fp32.
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "visual" in name or "embed_tokens" in name or "lm_head" in name:
            param.data = param.data.to(torch.float16)
    regex = LLM_PATTERN
    expected = 252
    if config["lora"]["targets"] == "llm+vit":
        regex = f"(?:{LLM_PATTERN}|{VIT_PATTERN})"
        expected = 348
    matches = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    if len(matches) != expected:
        raise RuntimeError(f"LoRA module count {len(matches)} != {expected}; inspect architecture")
    lc = config["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"], target_modules=regex, bias="none", task_type="CAUSAL_LM"))
    if config["train_projector"]:
        for name, param in model.named_parameters():
            if is_projector(name):
                param.data = param.data.float()
                param.requires_grad = True
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model, matches


def save_checkpoint(model, processor, directory, run):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    weights = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
               if is_projector(n) and p.requires_grad}
    if weights:
        torch.save(weights, directory / "extra_projector.pt")
    (directory / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")


def load_checkpoint(model, directory, trainable=False):
    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    model = PeftModel.from_pretrained(model, directory, is_trainable=trainable)
    if run["config"].get("train_projector"):
        path = directory / "extra_projector.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing trained projector: {path}")
        weights = torch.load(path, map_location="cpu", weights_only=True)
        params = dict(model.named_parameters())
        expected = {n for n in params if is_projector(n)}
        if set(weights) != expected:
            raise ValueError("Projector checkpoint keys do not match the model")
        for name, tensor in weights.items():
            if tensor.shape != params[name].shape:
                raise ValueError(f"Projector shape mismatch: {name}")
            params[name].data = tensor.to(device=params[name].device, dtype=torch.float32)
            params[name].requires_grad = trainable
    return model, run
