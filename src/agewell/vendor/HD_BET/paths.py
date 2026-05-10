import os

# AgeWell vendors HD-BET but stores the parameter file as a DVC-tracked model
# asset. Avoid runtime downloads by resolving the folder from the service env.
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
folder_with_parameter_files = os.environ.get(
    "HDBET_PARAMS_DIR",
    os.path.join(repo_root, "models", "brainiac", "hdbet"),
)
