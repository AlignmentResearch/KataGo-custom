import os

import torch

import data_processing_pytorch
import modelconfigs
from model_pytorch import Model

BATCH_SIZE = 2
GPU_ID = 3
POS_LEN = 19
RANK = 0
TRAIN_DATA = [
        "/nas/ucb/k8/go-attack/victimplay/ttseng-cyclic-vs-b18-s6201m-20230517-130803/selfplay/t0-s9737216-d2331818/tdata/0B45CABDA2418864.npz"
]
WORLD_SIZE = 1

if torch.cuda.is_available():
    device = torch.device("cuda", GPU_ID)
else:
    print("WARNING: No GPU, using CPU")
    device = torch.device("cpu")

model_config = modelconfigs.config_of_name["b1c6nbt"]
model = Model(model_config, POS_LEN)
model.initialize()
model.to(device)

input_batch = next(data_processing_pytorch.read_npz_training_data(
    npz_files=TRAIN_DATA,
    batch_size=BATCH_SIZE,
    world_size=WORLD_SIZE,
    rank=RANK,
    pos_len=POS_LEN,
    device=device,
    randomize_symmetries=True,
    model_config=model_config
))

traced_script_module = torch.jit.trace(
        func=model, 
        example_inputs=(input_batch["binaryInputNCHW"], input_batch["globalInputNC"]),
)
traced_script_module.save("/nas/ucb/ttseng/go_attack/torch-script/traced-test-model.pt")
