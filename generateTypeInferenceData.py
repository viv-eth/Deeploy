import os

import numpy as np
import onnx
from onnx import TensorProto, helper

# 1) Prepare a test directory
test_dir = 'test_add_chain_no_cast'
os.makedirs(test_dir, exist_ok = True)

# 2) Define your three inputs and one output
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 5, 5, 5])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 5, 5, 5])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 5, 5, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 5, 5, 5])

# 3) Build two Add nodes:
node1 = helper.make_node('Add', ['A', 'B'], ['X'], name = 'Add1')
node2 = helper.make_node('Add', ['X', 'C'], ['Y'], name = 'Add2')

graph = helper.make_graph([node1, node2], 'add_chain_graph', [A, B, C], [Y])
model = helper.make_model(graph, producer_name = 'add_chain_test')
onnx.save(model, os.path.join(test_dir, 'network.onnx'))

# 4) Generate data with strict ranges:
#    A_data in int16 range [-32768, 32767]
A_data = np.random.randint(-32768, 32768, size = (1, 5, 5, 5), dtype = np.int16).astype(np.float32)

#    B_data in int8 range  [-128, 127]
B_data = np.random.randint(-128, 128, size = (1, 5, 5, 5), dtype = np.int8).astype(np.float32)

#    C_data in int32 range [-2**31, 2**31-1]
C_data = np.random.randint(-2**31, 2**31, size = (1, 5, 5, 5), dtype = np.int32).astype(np.float32)

# 5) Perform the adds in 32-bit to compute expected output
#    (to avoid overflow in numpy we upcast)
X = A_data.astype(np.int32) + B_data.astype(np.int32)
Y_data = X + C_data.astype(np.int32)

# 6) Save inputs.npz and outputs.npz
np.savez(os.path.join(test_dir, 'inputs.npz'), A = A_data, B = B_data, C = C_data)
np.savez(os.path.join(test_dir, 'outputs.npz'), Y = Y_data.astype(np.int32))

print(f"Written ONNX + NPZ to ./{test_dir}")
