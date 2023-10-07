import os

import torch

import data_processing_pytorch
import load_model
import modelconfigs
from model_pytorch import Model

BATCH_SIZE = 1
DESTINATION_PATH = "/nas/ucb/ttseng/go_attack/torchscript/traced-test-model.pt"
GPU_ID = 1
RANK = 0
TRAIN_DATA = [
    "/nas/ucb/k8/go-attack/victimplay/ttseng-cyclic-vs-b18-s6201m-20230517-130803/selfplay/t0-s9737216-d2331818/tdata/0B45CABDA2418864.npz"
]
USE_SWA = True
WORLD_SIZE = 1

if torch.cuda.is_available():
    device = torch.device("cuda", GPU_ID)
else:
    print("No GPU, using CPU")
    device = torch.device("cpu")


# TODO argparse stuff instead of hardcoding

model, swa_model, _ = load_model.load_model(
    checkpoint_file="/nas/ucb/ttseng/go_attack/victim-weights/kata1-b18c384nbt-s7529928448-d3667707199/model.ckpt",
    use_swa=USE_SWA,
    device=device,
)
config = model.config
pos_len = model.pos_len
if swa_model is not None:
    model = swa_model
model.eval()

input_batch = next(
    data_processing_pytorch.read_npz_training_data(
        npz_files=TRAIN_DATA,
        batch_size=BATCH_SIZE,
        world_size=WORLD_SIZE,
        rank=RANK,
        pos_len=pos_len,
        device=device,
        randomize_symmetries=True,
        model_config=config,
    )
)

traced_script_module = torch.jit.trace(
    func=model,
    example_inputs=(input_batch["binaryInputNCHW"], input_batch["globalInputNC"]),
)
traced_script_module.cpu()
traced_script_module.save(DESTINATION_PATH)
print("Model saved to", DESTINATION_PATH)
