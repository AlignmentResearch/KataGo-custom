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


class EvalModel(torch.nn.Module):
    """Removes outputs that are only used during training.

    I (tomtseng) doubt that JIT is smart enough to optimize out parameters and
    calculations that are no longer used as a result of removing outputs, but we
    might as well remove unnecessary outputs regardless so that in the future we
    can choose to manually remove those parameters and calculations without
    changing the interface of the TorchScript model.
    """

    def __init__(self, model):
        super(EvalModel, self).__init__()
        self.model = model

        module = (
            model.module
            if isinstance(model, torch.optim.swa_utils.AveragedModel)
            else model
        )
        has_optimistic_head = module.policy_head.conv2p.weight.shape[0] > 5
        # Optimistic policy head exists at channel 5 and we should output it along
        # with the self policy output at channel 0.
        policy_output_channels = [0, 5] if has_optimistic_head else [0]
        # We need to register policy_output_channels this as a buffer instead
        # of defining+using it in forward(). Otherwise the TorchScript tracer
        # saves it as a tensor with a fixed device, i.e., it won't get moved
        # when the TorchScript model moves devices. Since tensor indices like
        # policy_output_channels should be on the same device that the indexed
        # tensor is on, an error will then occur if the TorchScript model is
        # executed on a different device than the device it was traced on.
        self.register_buffer(
            "policy_output_channels",
            torch.tensor(policy_output_channels),
            persistent=False,
        )

    def forward(self, input_spatial, spatial_global):
        # The output of self.model() is a tuple of tuples, where the first tuple
        # is the main head output.
        policy, value, miscvalue, moremiscvalue, ownership, _, _, _, _ = self.model(
            input_spatial, spatial_global
        )[0]
        return (
            policy[:, self.policy_output_channels],
            value,
            miscvalue[:, :4],
            moremiscvalue[:, :2],
            ownership,
        )


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
model = EvalModel(model)
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

with torch.no_grad():
    traced_script_module = torch.jit.trace(
        func=model,
        example_inputs=(input_batch["binaryInputNCHW"], input_batch["globalInputNC"]),
    )
traced_script_module.cpu()
traced_script_module.save(DESTINATION_PATH)
print("Model saved to", DESTINATION_PATH)
