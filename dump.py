#!/usr/bin/env python3
import os

import numpy as np
import torch


# 1) Define a trivial Add model
class AddModel(torch.nn.Module):

    def forward(self, x, y):
        return x + y


# 2) Create two random signed-8 inputs covering full int8 range
#    shape = [1,5,5,5]
in0_int8 = torch.randint(-128, 128, (1, 5, 5, 5), dtype = torch.int8)
in1_int8 = torch.randint(-128, 128, (1, 5, 5, 5), dtype = torch.int8)

# Convert to float32 (so ONNX Add stays float32)
input0 = in0_int8.to(torch.float32)
input1 = in1_int8.to(torch.float32)

# 3) Run the model once
model = AddModel()
output = model(input0, input1)

output_dir = "add_out"

# 4) Export the ONNX
onnx_filename = "network.onnx"
torch.onnx.export(
    model,
    (input0, input1),
    os.path.join(output_dir, onnx_filename),
    input_names = ["onnx::Add_0", "onnx::Add_1"],
    output_names = ["2"],
    opset_version = 11,
)
print(f"ONNX model written to: {os.path.join(output_dir, onnx_filename)}")

# 5) Convert to NumPy
np_input0 = input0.numpy()
np_input1 = input1.numpy()
np_output = output.numpy()

# 6) Save inputs.npz & outputs.npz
np.savez(os.path.join(output_dir, "inputs.npz"), **{
    "onnx::Add_0": np_input0,
    "onnx::Add_1": np_input1,
})
np.savez(os.path.join(output_dir, "outputs.npz"), **{
    "2": np_output,
})

print("Saved inputs.npz (keys: onnx::Add_0, onnx::Add_1)")
print("Saved outputs.npz (key: 2)")
