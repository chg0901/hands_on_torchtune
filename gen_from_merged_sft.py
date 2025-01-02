
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

print(transformers.__version__)

#TODO: update it to your chosen epoch
trained_model_path = "models/torchtune/llama3_2_3B/lora_single_device/epoch_1"
# trained_model_path = "/home/cine/Documents/tune/models/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=trained_model_path,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(trained_model_path, safetensors=True)


# Function to generate text
def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "tell me a joke"
print("Base model output:", generate_text(model, tokenizer, prompt))

prompt = "Complete the sentence: 'Once upon a time..."
print("Base model output:", generate_text(model, tokenizer, prompt))