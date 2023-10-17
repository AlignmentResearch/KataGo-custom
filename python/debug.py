"""TODO(tomtseng) write a header comment

NOTE(tomtseng) on FP16: Enabling FP16 consists of invoking a model wrapped `with
torch.cuda.amp.autocast()`, or at least that's what train.py does. I'm not sure
if tracing with autocast is supported---I get an error when tracing KataGo
models with autocast. I'm also not sure if tracing with autocast is needed in
order to have the resulting TorchScript model be autocastable.
"""
import os

import torch

import data_processing_pytorch
import load_model
import modelconfigs
from model_pytorch import Model

BATCH_SIZE = 2
DESTINATION_PATH = "/tmp/tt.pt"
GPU_ID = 1
RANK = 0
TRAIN_DATA = [
    "/nas/ucb/k8/go-attack/victimplay/ttseng-cyclic-vs-b18-s6201m-20230517-130803/selfplay/t0-s9737216-d2331818/tdata/0B45CABDA2418864.npz"
]
WORLD_SIZE = 1

if torch.cuda.is_available():
    device = torch.device("cuda", GPU_ID)
else:
    print("No GPU, using CPU")
    device = torch.device("cpu")

model = Model(modelconfigs.config_of_name["vit"], pos_len=19)
model.initialize()
model.eval()
model.to(device)

input_batch = next(
    data_processing_pytorch.read_npz_training_data(
        npz_files=TRAIN_DATA,
        batch_size=BATCH_SIZE,
        world_size=WORLD_SIZE,
        rank=RANK,
        pos_len=model.pos_len,
        device=device,
        randomize_symmetries=True,
        model_config=model.config,
    )
)

with torch.no_grad():
    print(model(input_batch["binaryInputNCHW"], input_batch["globalInputNC"]))

with torch.no_grad():
    traced_script_module = torch.jit.trace(
        func=model,
        example_inputs=(input_batch["binaryInputNCHW"], input_batch["globalInputNC"]),
    )
traced_script_module.cpu()
traced_script_module.save(DESTINATION_PATH)
print("Model saved to", DESTINATION_PATH)

model = torch.jit.load(DESTINATION_PATH)
device = torch.device("cuda", 2)
model.to(device)
with torch.no_grad():
    print(model(input_batch["binaryInputNCHW"].to(device), input_batch["globalInputNC"].to(device)))
