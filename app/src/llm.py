from src.config import get_secret


class LLMClient:
    """Wraps two open-model backends behind one method, chat(messages) -> str.

    hf_api: Hugging Face Inference API (free account; needs HF_TOKEN in the environment).
    local:  Transformers on your own machine or Colab GPU (no token needed for public models).
    """

    def __init__(self, cfg: dict):
        self.backend = cfg["backend"]
        self.model_id = cfg["model_id"]
        self.max_new_tokens = cfg.get("max_new_tokens", 400)
        self.temperature = cfg.get("temperature", 0.2)

        if self.backend == "hf_api":
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(model=self.model_id, token=get_secret("HF_TOKEN", required=True))
        elif self.backend == "local":
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map=cfg.get("device_map", "auto"), torch_dtype="auto"
            )
            self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        else:
            raise ValueError(f"Backend desconocido: {self.backend}")

    def chat(self, messages: list[dict]) -> str:
        if self.backend == "hf_api":
            resp = self.client.chat_completion(
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=max(self.temperature, 0.01),
            )
            return resp.choices[0].message.content.strip()

        tokenizer = self.pipe.tokenizer
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            return_full_text=False,
        )
        return out[0]["generated_text"].strip()