from vllm import LLM, SamplingParams

#TODO: update it to your chosen epoch
trained_model_path = "models/torchtune/llama3_2_3B/lora_single_device/epoch_1"
# trained_model_path = "/home/cine/Documents/tune/models/Llama-3.2-3B-Instruct"

def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)

#TODO: update it to your chosen epoch
llm = LLM(
    model=trained_model_path,
    load_format="safetensors",
    kv_cache_dtype="auto",
)
sampling_params = SamplingParams(max_tokens=16, temperature=0.5)

conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {
        "role": "user",
        "content": "Write an essay about the importance of higher education.",
    },
]
outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
print_outputs(outputs)



#   Attempting uninstall: pillow
#     Found existing installation: pillow 11.0.0
#     Uninstalling pillow-11.0.0:
#       Successfully uninstalled pillow-11.0.0
#   Attempting uninstall: numpy
#     Found existing installation: numpy 2.2.1
#     Uninstalling numpy-2.2.1:
#       Successfully uninstalled numpy-2.2.1
#   Attempting uninstall: tiktoken
#     Found existing installation: tiktoken 0.8.0
#     Uninstalling tiktoken-0.8.0:
#       Successfully uninstalled tiktoken-0.8.0
# Successfully installed aiohttp-cors-0.7.0 airportsdata-20241001 annotated-types-0.7.0 anyio-4.7.0 astor-0.8.1 blake3-1.0.0 cachetools-5.5.0 cloudpickle-3.1.0 colorful-0.5.6 compressed-tensors-0.8.1 depyf-0.18.0 diskcache-5.6.3 distlib-0.3.9 distro-1.9.0 einops-0.8.0 fastapi-0.115.6 gguf-0.10.0 google-api-core-2.24.0 google-auth-2.37.0 googleapis-common-protos-1.66.0 grpcio-1.68.1 h11-0.14.0 httpcore-1.0.7 httptools-0.6.4 httpx-0.28.1 importlib_metadata-8.5.0 iniconfig-2.0.0 interegular-0.3.3 jiter-0.8.2 jsonschema-4.23.0 jsonschema-specifications-2024.10.1 lark-1.2.2 linkify-it-py-2.0.3 lm-format-enforcer-0.10.9 markdown-it-py-3.0.0 mdit-py-plugins-0.4.2 mdurl-0.1.2 memray-1.15.0 mistral_common-1.5.1 msgpack-1.1.0 msgspec-0.19.0 numpy-1.26.4 nvidia-ml-py-12.560.30 openai-1.58.1 opencensus-0.11.4 opencensus-context-0.1.3 opencv-python-headless-4.10.0.84 outlines-0.1.11 outlines_core-0.1.26 partial-json-parser-0.2.1.1.post4 pillow-10.4.0 pluggy-1.5.0 prometheus-fastapi-instrumentator-7.0.0 prometheus_client-0.21.1 proto-plus-1.25.0 protobuf-5.29.2 py-cpuinfo-9.0.0 py-spy-0.4.0 pyasn1-0.6.1 pyasn1-modules-0.4.1 pycountry-24.6.1 pydantic-2.10.4 pydantic-core-2.27.2 pytest-8.3.4 python-dotenv-1.0.1 ray-2.40.0 referencing-0.35.1 rich-13.9.4 rpds-py-0.22.3 rsa-4.9 smart-open-7.1.0 sniffio-1.3.1 starlette-0.41.3 textual-1.0.0 tiktoken-0.7.0 tomli-2.2.1 uc-micro-py-1.0.3 uvicorn-0.34.0 uvloop-0.21.0 virtualenv-20.28.0 vllm-0.6.6.post1 watchfiles-1.0.3 websockets-14.1 wrapt-1.17.0 xformers-0.0.28.post3 xgrammar-0.1.8 zipp-3.21.0