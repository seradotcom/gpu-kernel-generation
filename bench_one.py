"""
One real generate-and-test cycle.
Run from the repo ROOT with the venv active:
    python bench_one.py

It: builds the prompt -> asks Qwen on Kaggle (/generate) -> parses JSON ->
translates to TTIR locally -> sends TTIR to /benchmark on the T4 -> saves results.json
"""
import requests, json, time
import core.config                       # puts your local MLIR bindings on the path
from core.schemas import MlirResponse
from core.prompt_builder import PromptBuilder
from core.mlir_translator import MLIRTranslator

# ====== EDIT THIS each Kaggle session (copy the PUBLIC URL the notebook printed) ======
NGROK_URL = "https://handled-baked-copied.ngrok-free.dev/"
# ======================================================================================

TEST_NAME = "vec_sum_kernel"   # must match a key in KERNEL_TESTS in the notebook

USER_PROMPT = (
    "Write a Triton kernel that adds two vectors A and B element-wise into C. "
    "Use a single block of exactly 256 elements (tensor<256xf32>). "
    "Do NOT use program_id or masking; operate on offsets 0..255 directly. "
    "Follow the vec_sum_kernel example structure exactly."
)

HEADERS = {"ngrok-skip-browser-warning": "true"}


def build_chatml(system_p, user_p):
    return (f"<|im_start|>system\n{system_p}<|im_end|>\n"
            f"<|im_start|>user\n{user_p}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def call_generate(url, prompt, schema_dict, poll=8, max_wait=600):
    r = requests.post(f"{url}/generate",
                      json={"prompt": prompt, "max_tokens": 2048, "schema_dict": schema_dict},
                      headers=HEADERS, timeout=30)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print("job dispatched:", job_id)
    waited = 0
    while waited < max_wait:
        time.sleep(poll); waited += poll
        s = requests.get(f"{url}/status/{job_id}", headers=HEADERS, timeout=30).json()
        if s.get("status") == "done":
            return s["response"]
        if s.get("status") == "error":
            raise RuntimeError(f"remote generation error: {s.get('error')}")
        print(f"  pending... {waited}s")
    raise TimeoutError("generation timed out")


def clean_json(raw):
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def main():
    url = NGROK_URL.rstrip("/")
    schema = MlirResponse.model_json_schema()
    system_p = PromptBuilder().build_prompt(USER_PROMPT, schema)
    prompt = build_chatml(system_p, USER_PROMPT)

    print("Generating with Qwen (constrained decoding)...")
    raw = call_generate(url, prompt, schema)
    print("--- raw response (first 400 chars) ---")
    print(raw[:400])
    print("---------------------------------------")

    try:
        obj = MlirResponse(**json.loads(clean_json(raw)))
    except Exception as e:
        row = {"name": TEST_NAME, "status": "gen_failed", "error": repr(e)}
        print(json.dumps(row, indent=2))
        json.dump([row], open("results.json", "w"), indent=2)
        print("wrote results.json (generation/parse failed)")
        return

    ttir = MLIRTranslator().translate_to_module(obj.code)
    print("translated to TTIR, length:", len(ttir))

    resp = requests.post(f"{url}/benchmark",
                         json={"name": TEST_NAME, "ttir": ttir},
                         headers=HEADERS, timeout=120)
    row = resp.json()
    print(json.dumps(row, indent=2))
    json.dump([row], open("results.json", "w"), indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()
