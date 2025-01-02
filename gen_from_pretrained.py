from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

#TODO: update it to your chosen epoch
trained_model_path = "models/torchtune/llama3_2_3B/lora_single_device/epoch_1"

# Define the model and adapter paths

## Use this if you want to use the original model from huggingface
# original_model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(original_model_name)

# # OSError: You are trying to access a gated repo.
# # Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct.
# # 403 Client Error. (Request ID: Root=1-6775688b-377f1c1433c725d674e15302;aad41fb7-4adf-4ddc-986e-e0512b9f4639)
# # Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json.
# # Your request to access model meta-llama/Llama-3.2-3B-Instruct has been rejected by the repo's authors.

# # To Avoid this error, we can use local model
original_model_name = '/home/cine/Documents/tune/models/Llama-3.2-3B-Instruct'
model = AutoModelForCausalLM.from_pretrained(original_model_name)

# huggingface will look for adapter_model.safetensors and adapter_config.json
peft_model = PeftModel.from_pretrained(model, trained_model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(original_model_name)

# Function to generate text
def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "tell me a joke: '"
print("Base model output:", generate_text(peft_model, tokenizer, prompt))