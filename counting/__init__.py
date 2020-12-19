from functools import partial

from counting.inference_train_FLOPs import (
    RigL_train_FLOPs,
    SET_train_FLOPs,
    SNFS_train_FLOPs,
    Pruning_train_FLOPs, model_inference_FLOPs,
)

registry = {
    "RigL": RigL_train_FLOPs,
    "SET": SET_train_FLOPs,
    "SNFS": SNFS_train_FLOPs,
    "Pruning": Pruning_train_FLOPs,
}
wrn_22_2_FLOPs = partial(
    model_inference_FLOPs, model_name="wrn-22-2", input_size=(1, 3, 32, 32)
)
resnet50_FLOPs = partial(
    model_inference_FLOPs, model_name="resnet50", input_size=(1, 3, 32, 32)
)
