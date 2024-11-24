import numpy as np
import os
from onnx import TensorProto
import onnxruntime as ort
from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test
from onnx.helper import make_tensor_value_info, make_tensor_sequence_value_info, make_node
from e2e_testing.storage import TestTensors

class SplitToSequenceModel(BuildAModel):
    def construct_i_o_value_info(self):
        
        X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 6])  
        Y = make_tensor_value_info("Y", TensorProto.INT64, [2])  

        Z = make_tensor_sequence_value_info("Z", TensorProto.FLOAT, [2, 6])  
        self.input_vi = [X, Y]
        self.output_vi = [Z]

    def construct_nodes(self):
        concat_node = make_node(
            op_type="SplitToSequence",
            inputs=["X", "Y"],
            outputs=["Z"],
            axis=0,  # Splitting along axis 0
        )
        self.node_list = [concat_node]

    def construct_inputs(self) -> TestTensors:
        """Override to generate specific inputs for SplitToSequence with valid split sizes."""
        default_inputs = super().construct_inputs() 

        tensors = list(default_inputs.data)

        x_tensor = tensors[0] 
        y_tensor = tensors[1] 
        
        if sum(y_tensor) != x_tensor.shape[0]:
            raise ValueError(f"Sum of split sizes in Y ({sum(y_tensor)}) does not match X's first dimension ({x_tensor.shape[0]})")

        rng = np.random.default_rng(19)

        x_tensor = rng.random((3, 6), dtype=np.float32)
        tensors[0] = x_tensor

        
        y_tensor = rng.integers(1, 3, size=(2,), dtype=np.int64)  
        tensors[1] = y_tensor

        default_sample_inputs = TestTensors(tuple(tensors))
        return default_sample_inputs


register_test(SplitToSequenceModel, "split_to_sequence_test")