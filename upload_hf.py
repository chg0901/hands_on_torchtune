import huggingface_hub
api = huggingface_hub.HfApi()

#TODO: update it to your chosen epoch
trained_model_path = "models/torchtune/llama3_2_3B/lora_single_device/epoch_1"

username = huggingface_hub.whoami()["name"]
repo_name = "llama3_2_3B_Instruct_alphaca_epoch_1"

# if the repo doesn't exist
repo_id = huggingface_hub.create_repo(repo_name).repo_id

# if it already exists
repo_id = f"{username}/{repo_name}"

api.upload_folder(
    folder_path=trained_model_path,
    repo_id=repo_id,
    repo_type="model",
    create_pr=False
)