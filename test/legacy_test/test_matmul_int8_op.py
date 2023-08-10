#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import logging
import sys

import numpy as np
from test_sparse_attention_op import get_cuda_version

import paddle
from paddle.fluid import core

paddle.disable_static()



# def reference_matmul_int8(X, Y, transpose_X=False, transpose_Y=False):
#     """Reference forward implementation using np.matmul."""
#     # np.matmul does not support the transpose flags, so we manually
#     # transpose X and Y appropriately.
#     if transpose_X:
#         if X.ndim == 1:
#             X = X.reshape((X.size,))
#         elif X.ndim == 2:
#             X = X.T
#         else:
#             dim = list(range(len(X.shape)))
#             dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
#             X = np.transpose(X, tuple(dim))
#     if transpose_Y:
#         if Y.ndim == 1:
#             Y = Y.reshape((Y.size,))
#         else:
#             dim = list(range(len(Y.shape)))
#             dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
#             Y = np.transpose(Y, tuple(dim))

#     Out = np.matmul(X, Y)
#     return Out

# class TestMatMulInt8Op(OpTest):
#     """
#     case 1
#     """

#     def config(self):
#         self.x_shape = (100,)
#         self.y_shape = (100,)
#         self.trans_x = False
#         self.trans_y = False

#     def init_kernel_type(self):
#         self.dtype = "int8"

#     def setUp(self):
#         self.init_kernel_type()
#         self.config()
#         # TOOD(yinshangfei)
#         self.op_type = "matmul_int8"
#         self.python_api = paddle.tensor.matmul_int8
#         x = np.random.random(self.x_shape).astype(self.dtype)
#         y = np.random.random(self.y_shape).astype(self.dtype)
#         # -0.1 ~ 0.1
#         x = -0.1 + 0.2 * x
#         y = -0.1 + 0.2 * y
#         result = reference_matmul_int8(x, y, self.trans_x, self.trans_y)
#         result = result.astype(self.dtype)
#         self.inputs = {
#             'X': x,
#             'Y': y,
#         }
#         self.attrs = {'trans_x': self.trans_x, 'trans_y': self.trans_y}
#         self.outputs = {'Out': result}

#     def test_check_output(self):
#         self.check_output(
#             check_cinn=self.check_cinn if hasattr(self, 'check_cinn') else True
#         )
        
#         self.assert_equal('0', '1')

#     # TODO(check_grad)






# TODO(yinshangfei) bias
@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or get_cuda_version() < 11020
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "MatmulInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
)
class TestMatmulInt8(unittest.TestCase):
    """
    Test matmul int8
    Only NT (Non-Transposed-A and Transposed-B) is supported
    """

    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = True
        # self.m = 8
        # self.k = 64
        # self.n = 64
        self.x_shape = (8, 64)
        self.y_shape = (64, 64)
        self.trans_x = False
        self.trans_y = True

    def setUp(self):
        self.config()
        self.input_a_np = np.random.randint(-127, 127, self.x_shape).astype(
            'int32'
        )
        self.input_b_np = np.random.randint(-127, 127, self.y_shape).astype(
            'int32'
        )
        if self.trans_x:
            if input_a_np.ndim == 1:
                self.input_a = paddle.to_tensor(
                    self.input_a_np.reshape((self.input_a_np.size, )), 
                    dtype=self.dtype)
            elif input_a_np.ndim == 2:
                self.input_a = paddle.to_tensor(
                    self.input_a_np.T, dtype=self.dtype)
            else:
                dim = list(range(len(self.input_a_np.shape)))
                dim[-1], dim[len(self.input_a_np.shape) - 2] = dim[len(self.input_a_np.shape) - 2], dim[-1]
                self.input_a = paddle.to_tensor(
                    self.input_a_np.transpose(input_a_np.size, tuple(dim)), 
                    dtype=self.dtype)
        else:
            self.input_a = paddle.to_tensor(self.input_a_np, dtype=self.dtype)
        if self.trans_y:
            if input_b_np.ndim == 1:
                self.input_b = paddle.to_tensor(
                    self.input_b_np.reshape((self.input_b_np.size, )), 
                    dtype=self.dtype)
            elif input_b_np.ndim == 2:
                self.input_b = paddle.to_tensor(
                    self.input_b_np.T, dtype=self.dtype)
            else:
                dim = list(range(len(self.input_b_np.shape)))
                dim[-1], dim[len(self.input_b_np.shape) - 2] = dim[len(self.input_b_np.shape) - 2], dim[-1]
                self.input_b = paddle.to_tensor(
                    self.input_b_np.transpose(input_b_np.size, tuple(dim)), 
                    dtype=self.dtype)
        else:
            self.input_b = paddle.to_tensor(self.input_b_np, dtype=self.dtype)
        

    def get_reference_out(self):
        out = np.dot(self.input_a_np, self.input_b_np)
        return out

    def get_op_out(self):
        out = paddle._C_ops.matmul_int8(self.input_a, self.input_b, slef.trans_x, self.trans_y)
        return out.numpy()

    def test_matmul_int8(self):

        self.assert_equal('0', '1')
        
        out_real = self.get_op_out()
        out_expect = self.get_reference_out()
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )

class TestMatmulInt8Op2(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = True
        self.x_shape = (100,)
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True

        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        try:
            print("AA2")
            logging.getLogger().info("BB2")
        finally:
            logger.removeHandler(stream_handler)


if __name__ == '__main__':
    unittest.main()
