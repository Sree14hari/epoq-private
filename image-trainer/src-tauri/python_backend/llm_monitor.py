"""
llm_monitor.py – EPOQ LLM Co-Pilot Daemon
==========================================
This script is spawned ONCE by Rust on application startup.
It stays alive for the lifetime of the app and communicates
exclusively through stdin/stdout using newline-delimited JSON.

Input  (from Rust, one JSON object per line):
  System metric update:
    {"type": "system_update", "epoch": 3, "loss": 0.42, "accuracy": 0.87,
     "ram_mb": 6200, "vram_mb": 3800, "vram_total_mb": 8192}

  User chat message:
    {"type": "user_chat", "message": "Why is my loss spiking?"}

Output (to Rust, one JSON object per line):
  {"tag": "advice"|"alert"|"chat", "title": "...", "body": "..."}
"""

import sys
import json
import os
import traceback
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI argument parsing (called before anything else)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EPOQ LLM Co-Pilot Daemon")
    parser.add_argument(
        "--model", default="",
        help="Absolute path to the .gguf model file. Overrides auto-discovery."
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Model bootstrap
# ---------------------------------------------------------------------------

def _find_model(cli_model: str) -> str:
    """
    Locate the .gguf model file.
    Priority:
      1. --model CLI argument
      2. EPOQ_LLM_MODEL_PATH environment variable
      3. models/ directory next to this script
    """
    if cli_model and Path(cli_model).is_file():
        return cli_model

    env_path = os.environ.get("EPOQ_LLM_MODEL_PATH", "").strip()
    if env_path and Path(env_path).is_file():
        return env_path

    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    if models_dir.is_dir():
        for f in models_dir.iterdir():
            if f.suffix.lower() == ".gguf":
                return str(f)

    return ""



def _load_model(model_path: str):
    """
    Load the model with llama-cpp-python.
    n_gpu_layers=-1  →  offload every layer to GPU (fastest).
    Falls back to CPU-only (n_gpu_layers=0) if CUDA isn't available.
    """
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError:
        _emit({"tag": "alert",
               "title": "LLM Unavailable",
               "body": "llama-cpp-python is not installed. Run: pip install llama-cpp-python"})
        return None

    if not model_path:
        _emit({"tag": "alert",
               "title": "Model Not Found",
               "body": "Place a .gguf model file inside python_backend/models/ and restart."})
        return None

    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,   # full GPU offload
            n_ctx=2048,
            verbose=False,
        )
        _emit({"tag": "advice",
               "title": "LLM Ready",
               "body": f"Co-pilot loaded: {Path(model_path).name} (GPU offload active)"})
        return llm
    except Exception:
        # GPU load failed → retry CPU-only
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=0,
                n_ctx=2048,
                verbose=False,
            )
            _emit({"tag": "advice",
                   "title": "LLM Ready (CPU)",
                   "body": f"Co-pilot loaded on CPU: {Path(model_path).name}"})
            return llm
        except Exception as e:
            _emit({"tag": "alert",
                   "title": "LLM Load Failed",
                   "body": str(e)})
            return None


import time

# ---------------------------------------------------------------------------
# Output helper  (with per-title cooldown to prevent spam)
# ---------------------------------------------------------------------------

# Minimum seconds between emitting the same insight title
_COOLDOWN: dict[str, float] = {}
_COOLDOWN_SECONDS = {
    "alert":  5 * 60,   # 5 minutes between identical alert titles
    "advice": 10 * 60,  # 10 minutes between identical advice titles
}

def _emit(payload: dict):
    """Write a single-line JSON object to stdout, respecting per-title coolbacks."""
    tag   = payload.get("tag", "chat")
    title = payload.get("title", "")

    if tag in _COOLDOWN_SECONDS:
        now  = time.time()
        last = _COOLDOWN.get(title, 0)
        cooldown = _COOLDOWN_SECONDS[tag]
        if now - last < cooldown:
            return   # too soon — suppress duplicate
        _COOLDOWN[title] = now

    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """You are EPOQ Co-Pilot, an expert AI training assistant embedded in a PyTorch image classification trainer.
Current training snapshot:
  - Epoch        : {epoch}
  - Train Loss   : {loss:.4f}
  - Accuracy     : {accuracy:.2%}
  - RAM used     : {ram_mb} MB
  - VRAM used    : {vram_mb} / {vram_total_mb} MB

Analyze the snapshot. Detect issues like overfitting, VRAM pressure, or stalled learning.
Respond ONLY with a single JSON object on one line, no markdown, no prose outside JSON.
Format: {{"tag": "advice" or "alert", "title": "<short title>", "body": "<one or two sentence insight>"}}
If everything looks healthy, use tag=advice. If there is a critical issue, use tag=alert."""

_CHAT_SYSTEM = """You are EPOQ Co-Pilot, an expert AI model training assistant.
Answer the user's question about their training run concisely and helpfully.
Respond ONLY with a single JSON object on one line, no markdown outside JSON.
Format: {{"tag": "chat", "title": "Co-Pilot", "body": "<your answer>"}}"""


def _build_system_prompt(data: dict) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        epoch=data.get("epoch", "?"),
        loss=float(data.get("loss", 0)),
        accuracy=float(data.get("accuracy", 0)),
        ram_mb=data.get("ram_mb", 0),
        vram_mb=data.get("vram_mb", 0),
        vram_total_mb=data.get("vram_total_mb", 0),
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

_MAX_TOKENS = 200
_last_system_context: str = ""


import re

def _extract_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from the model's raw output.
    Handles:
      - Double-braced escapes: {{...}} -> {...}
      - Surrounding prose / code fences
      - Bare text fallback
    """
    # Strip code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    # Collapse double-braces that the model copies from the prompt template
    cleaned = raw.replace("{{", "{").replace("}}", "}")

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract the first {...} block via regex
    m = re.search(r'\{[^{}]*"tag"[^{}]*\}', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Give up — return the raw text as a chat body
    return {"tag": "chat", "title": "Co-Pilot", "body": cleaned}


def _infer(llm, system_prompt: str, user_message: str) -> dict:
    """
    Run a chat completion and attempt to parse the model's response as JSON.
    Falls back to a raw body if the model doesn't produce valid JSON.
    """
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=_MAX_TOKENS,
            temperature=0.2,
            stop=["\n\n"],
        )
        raw = response["choices"][0]["message"]["content"].strip()
        return _extract_json(raw)
    except Exception as e:
        return {"tag": "alert", "title": "Inference Error", "body": str(e)}


# ---------------------------------------------------------------------------
# Main daemon loop
# ---------------------------------------------------------------------------

def main():
    global _last_system_context

    args = _parse_args()
    model_path = _find_model(args.model)
    llm = _load_model(model_path)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            _emit({"tag": "alert", "title": "Parse Error",
                   "body": f"Could not parse input: {raw_line[:80]}"})
            continue

        msg_type = payload.get("type", "")

        # ------------------------------------------------------------------
        # Case 1: Periodic system metric update from the Rust aggregator
        # ------------------------------------------------------------------
        if msg_type == "system_update":
            epoch    = int(payload.get("epoch", 0))
            loss     = float(payload.get("loss", 0))
            accuracy = float(payload.get("accuracy", 0))

            # Skip analysis when training hasn't started yet
            if epoch == 0 and loss == 0.0 and accuracy == 0.0:
                continue

            if llm is None:
                # No model loaded – still emit a deterministic rule-based alert
                _rule_based_check(payload)
                continue

            system_prompt = _build_system_prompt(payload)
            _last_system_context = system_prompt  # remember for follow-up chats
            result = _infer(llm, system_prompt,
                            "Analyze these metrics and give me your most important insight.")
            _emit(result)

        # ------------------------------------------------------------------
        # Case 2: User typed a message in the Co-Pilot chat sidebar
        # ------------------------------------------------------------------
        elif msg_type == "user_chat":
            message = payload.get("message", "")
            if not message:
                continue

            if llm is None:
                _emit({"tag": "chat", "title": "Co-Pilot Offline",
                       "body": "The LLM model is not loaded. Check alerts for details."})
                continue

            # Blend the latest training context into the system prompt
            combined_system = _CHAT_SYSTEM
            if _last_system_context:
                combined_system += (
                    "\n\nLatest training snapshot for context:\n" + _last_system_context
                )

            result = _infer(llm, combined_system, message)
            result["tag"] = "chat"  # force tag for chat responses
            _emit(result)

        else:
            _emit({"tag": "alert", "title": "Unknown Payload",
                   "body": f"Unrecognised message type: '{msg_type}'"})


# ---------------------------------------------------------------------------
# Rule-based fallback (used when no LLM is loaded)
# ---------------------------------------------------------------------------

def _rule_based_check(data: dict):
    """Emit simple deterministic insights without the LLM."""
    vram       = data.get("vram_mb", 0)
    vram_total = data.get("vram_total_mb", 1)
    loss       = float(data.get("loss", 0))
    accuracy   = float(data.get("accuracy", 0))
    epoch      = int(data.get("epoch", 0))

    # Don't fire when training hasn't produced any metrics yet
    if epoch == 0 and loss == 0.0 and accuracy == 0.0:
        return

    vram_pct = vram / vram_total if vram_total > 0 else 0

    if vram_pct > 0.92:
        _emit({"tag": "alert",
               "title": "VRAM Critical",
               "body": f"VRAM usage is {vram_pct:.0%}. Consider reducing batch size to avoid OOM."})
    elif loss > 2.0 and epoch > 3:
        _emit({"tag": "alert",
               "title": "High Loss",
               "body": f"Loss is {loss:.3f} after epoch {epoch}. Check your learning rate or data pipeline."})
    elif accuracy > 0.99 and epoch < 5:
        _emit({"tag": "advice",
               "title": "Suspiciously High Accuracy",
               "body": "Accuracy is very high early on — verify your validation split isn't leaking into training data."})
    else:
        _emit({"tag": "advice",
               "title": "Training Nominal",
               "body": f"Epoch {epoch} — Loss {loss:.4f} | Accuracy {accuracy:.2%}. Looking healthy."})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _emit({"tag": "alert", "title": "Daemon Crash",
               "body": traceback.format_exc().replace("\n", " ")})
        sys.exit(1)
