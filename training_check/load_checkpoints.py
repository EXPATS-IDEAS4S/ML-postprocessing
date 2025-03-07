import torch
from vissl.models.base_ssl_model import BaseSSLMultiInputOutputModel
from vissl.models.trunks.resnext import ResNeXt  # or appropriate module

# Recreate the model
model = BaseSSLMultiInputOutputModel()

exit()

path_checkpoints = '/data1/runs/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/checkpoints/'

# Load the checkpoint
checkpoint = torch.load(f"{path_checkpoints}model_final_checkpoint_phase799.torch", map_location="cpu")

# Load state_dict into the model
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

# Set the model to evaluation mode
model.eval()
