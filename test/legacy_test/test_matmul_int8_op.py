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






# class TestMatmulInt8Multi(unittest.TestCase):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.m_range_min = 1
#         self.m_rang_max = 25
#         self.n_range_min = 1
#         self.n_rang_max = 25
#         self.k_range_min = 1
#         self.k_rang_max = 25
#         self.trans_x_range = (True, False)
#         self.trans_y_range = (True, False)
#     def setUp(self):
#         pass

#     def init_input_np(self, shape):
#         return np.random.randint(-127, 127, shape).astype(
#             'int32'
#         )
#     def init_input_tensor(self, x):
#         return paddle.to_tensor(x, dtype=self.dtype)
        
#     def get_reference_out(self, x, y):
#         return np.dot(x, y)

#     def get_op_out(self, a, b, trans_x, trans_y):
#         out = paddle._C_ops.matmul_int8(a, b, trans_x, trans_y)
#         return out.numpy()

#     def check(self, real, expect, atol, rtol):
#         for i  in range(real.shape[0]):
#             for j in range(real.shape[1]):
#                 if np.abs(expect[i, j] - real[i, j]) > atol:
#                     print("error atol\n")
#                 if np.abs((expect[i, j] - real[i, j]) / expect[i, j]) > rtol:
#                     print("error rtol\n")

#     def test_matmul_int8(self):
#         self.config()
#         for trans_x in self.trans_x_range:
#             for trans_y in self.trans_y_range:
#                 for m in range(self.m_range_min, self.m_rang_max + 1):
#                     for n in range(self.n_range_min, self.n_rang_max + 1):
#                         for k in range(self.k_range_min, self.k_rang_max + 1):
#                             x_shape = (m, k)
#                             y_shape = (k, n)
#                             if trans_x:
#                                 x_shape = (k, m)
#                             if trans_y:
#                                 y_shape = (n, k)

#                             x_np = self.init_input_np(x_shape)
#                             y_np = self.init_input_np(y_shape)
#                             x = self.init_input_tensor(x_np)
#                             y = self.init_input_tensor(y_np)

#                             if trans_x:
#                                 x_np = x_np.T
#                             if trans_y:
#                                 y_np = y_np.T

#                             out_real = self.get_op_out(x, y, trans_x, trans_y)
#                             out_expect = self.get_reference_out(x_np, y_np)
#                             print(x_shape)
#                             print(y_shape)
#                             print(trans_x)
#                             print(trans_y)
#                             self.check(out_real, out_expect, self.atol, self.rtol)

#         assert 1 == 0





























# TODO(yinshangfei) bias
# @unittest.skipIf(
#     not core.is_compiled_with_cuda()
#     or get_cuda_version() < 11020
#     or paddle.device.cuda.get_device_capability()[0] < 8,
#     "MatmulInt8 requires CUDA >= 11.2 and CUDA_ARCH >= 8",
# )
class TestMatmulInt8(unittest.TestCase):

    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.bias = False
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
        self.input_a = paddle.to_tensor(self.input_a_np, dtype=self.dtype)
        self.input_b = paddle.to_tensor(self.input_b_np, dtype=self.dtype)

        if self.trans_x:
            if self.input_a_np.ndim == 1:
                self.input_a_np = self.input_a_np.reshape((self.input_a_np.size, ))
            elif self.input_a_np.ndim == 2:
                self.input_a_np = self.input_a_np.T
            else:
                dim = list(range(len(self.input_a_np.shape)))
                dim[-1], dim[len(self.input_a_np.shape) - 2] = dim[len(self.input_a_np.shape) - 2], dim[-1]
                self.input_a_np = np.transpose(self.input_a_np, tuple(dim))
        if self.trans_y:
            if self.input_b_np.ndim == 1:
                self.input_b_np = self.input_b_np.reshape((self.input_b_np.size, ))
            elif self.input_b_np.ndim == 2:
                self.input_b_np = self.input_b_np.T
            else:
                dim = list(range(len(self.input_b_np.shape)))
                dim[-1], dim[len(self.input_b_np.shape) - 2] = dim[len(self.input_b_np.shape) - 2], dim[-1]
                self.input_b_np = np.transpose(self.input_b_np, tuple(dim))
                print("transY")
                print(np.transpose(self.input_b_np, tuple(dim)).shape)

        print(self.input_a_np.shape)
        print(self.input_b_np.shape)

        print("-------------------------------------------")
        

    def get_reference_out(self):
        out = np.matmul(self.input_a_np, self.input_b_np)
        
        print("out.shape")
        print(out.shape)
        return out

    def get_op_out(self):
        print("trans_y {}".format(self.trans_y))
        out = paddle._C_ops.matmul_int8(self.input_a, self.input_b, self.trans_x, self.trans_y)
        return out.numpy()

    def test_matmul_int8(self):
        out_real = self.get_op_out()
        out_expect = self.get_reference_out()
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


# class TestMatmulInt8Op2(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (100,)
#         self.y_shape = (1, 3, 2, 100)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8Op3(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (4,)
#         self.y_shape = (1, 1, 4, 100)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatmulInt8Op4(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (100,)
#         self.y_shape = (1, 2, 100, 4)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatmulInt8Op5(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (1, 1, 100, 4)
#         self.y_shape = (100,)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8Op6(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (1, 2, 104, 4)
#         self.y_shape = (104,)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8Op7(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (1, 2, 4, 100)
#         self.y_shape = (100,)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatmulInt8Op8(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (1, 1, 4, 100)
#         self.y_shape = (1, 1, 100, 4)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatmulInt8Op9(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (1, 1, 4, 100)
#         self.y_shape = (2, 1, 8, 100)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8Op10(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (1, 1, 24, 4)
#         self.y_shape = (1, 2, 4, 24)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatmulInt8Op11(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (2, 1, 4, 100)
#         self.y_shape = (1, 1, 100, 4)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatmulInt8Op12(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (2, 1, 4, 24)
#         self.y_shape = (1, 1, 4, 24)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8Op13(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (2, 2, 12, 12)
#         self.y_shape = (2, 2, 12, 12)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8Op14(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (3, 1, 8, 8)
#         self.y_shape = (1, 2, 8, 8)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8Op15(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (3, 1, 8, 8)
#         self.y_shape = (1, 2, 8, 8)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatmulInt8Op16(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = 100
#         self.y_shape = (1, 2, 2, 100, 4)
#         self.trans_x = False
#         self.trans_y = False
   
# class TestMatmulInt8Op17(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         # TODO
#         self.x_shape = (2, 4, 100)
#         self.y_shape = 100
#         self.trans_x = False
#         self.trans_y = False

# class TestMatmulInt8OpBroadcast1(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (3, 1, 12, 12)
#         self.y_shape = (1, 2, 12, 12)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8OpBroadcast2(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (3, 1, 12, 12)
#         self.y_shape = (1, 2, 12, 12)
#         self.trans_x = False
#         self.trans_y = True

#######################################################################
# test12 TRUE TRUE K

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (1, 32)
#         self.y_shape = (16, 1)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (2, 32)
#         self.y_shape = (16, 2)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (3, 32)
#         self.y_shape = (16, 3)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (4, 32)
#         self.y_shape = (16, 4)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (5, 32)
#         self.y_shape = (16, 5)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (6, 32)
#         self.y_shape = (16, 6)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (8, 32)
#         self.y_shape = (16, 8)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (9, 32)
#         self.y_shape = (16, 9)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (10, 32)
#         self.y_shape = (16, 10)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (12, 32)
#         self.y_shape = (16, 12)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (16, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (17, 32)
#         self.y_shape = (16, 17)
#         self.trans_x = True
#         self.trans_y = True


#######################################################################
# test11 TRUE TRUE N

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (1, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (2, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (3, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (4, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (5, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (6, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (8, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (9, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (10, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (12, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (16, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (17, 16)
#         self.trans_x = True
#         self.trans_y = True


#######################################################################
#test10 True True M

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 1)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 2)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 3)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 4)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 5)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 6)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 8)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 9)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 10)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 12)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 17)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = True


#######################################################################
#test9 True False K

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (1, 16)
#         self.y_shape = (1, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (2, 16)
#         self.y_shape = (2, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (3, 16)
#         self.y_shape = (3, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (4, 16)
#         self.y_shape = (4, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (5, 16)
#         self.y_shape = (5, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (6, 16)
#         self.y_shape = (6, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (8, 16)
#         self.y_shape = (8, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (9, 16)
#         self.y_shape = (9, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (10, 16)
#         self.y_shape = (10, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (12, 16)
#         self.y_shape = (12, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (17, 16)
#         self.y_shape = (17, 32)
#         self.trans_x = True
#         self.trans_y = False


#######################################################################
#test8 True False N
# N=1 or N%4==0

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 1)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 2)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 3)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 4)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 5)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 6)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 8)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 9)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 10)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 12)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 16)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 17)
#         self.trans_x = True
#         self.trans_y = False

       
######################################################################
# test7 True False M
# M=1 or M%4==0

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 1)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 2)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 3)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 4)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 5)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 6)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 8)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 9)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 10)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 12)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 16)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False

# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (32, 17)
#         self.y_shape = (32, 32)
#         self.trans_x = True
#         self.trans_y = False


######################################################################
# test6 False True K
# K%4==0

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 1)
#         self.y_shape = (32, 1)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 2)
#         self.y_shape = (32, 2)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 3)
#         self.y_shape = (32, 3)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 4)
#         self.y_shape = (32, 4)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 5)
#         self.y_shape = (32, 5)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 6)
#         self.y_shape = (32, 6)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 8)
#         self.y_shape = (32, 8)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 9)
#         self.y_shape = (32, 9)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 10)
#         self.y_shape = (32, 10)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 12)
#         self.y_shape = (32, 12)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (32, 16)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 17)
#         self.y_shape = (32, 17)
#         self.trans_x = False
#         self.trans_y = True


######################################################################
# test5 False True N
# N没有要求

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (1, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (2, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (3, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (4, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (5, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (6, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (8, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (9, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (10, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (12, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (17, 32)
#         self.trans_x = False
#         self.trans_y = True


######################################################################
# test4 False True M
# 没有要求

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (1, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (2, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (3, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (4, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (5, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (6, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (8, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (9, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (10, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (12, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (17, 32)
#         self.y_shape = (16, 32)
#         self.trans_x = False
#         self.trans_y = True


######################################################################
# test3 False False K 
# 8_1 4 7 10 11

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 1)
#         self.y_shape = (1, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 2)
#         self.y_shape = (2, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 3)
#         self.y_shape = (3, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 4)
#         self.y_shape = (4, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 5)
#         self.y_shape = (5, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 6)
#         self.y_shape = (6, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 8)
#         self.y_shape = (8, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 9)
#         self.y_shape = (9, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 10)
#         self.y_shape = (10, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 12)
#         self.y_shape = (12, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 17)
#         self.y_shape = (17, 16)
#         self.trans_x = False
#         self.trans_y = False


######################################################################
# test2 False False N 1或4的倍数
# 8_1 7 10 11

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 1)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 2)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 3)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 4)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 5)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 6)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 8)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 9)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 10)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 12)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 17)
#         self.trans_x = False
#         self.trans_y = False

######################################################################
# test1 False False M M没有要求

# class TestMatmulInt8_1(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (1, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_2(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (2, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_3(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (3, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_4(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (4, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_5(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (5, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_6(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (6, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_7(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (8, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_8(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (9, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_9(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (10, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_10(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (12, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_11(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False
# class TestMatmulInt8_12(TestMatmulInt8):

#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (17, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = False
#         self.trans_y = False

# class TestMatmulInt8Op2(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (2, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op3(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (3, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op4(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (4, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op5(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (5, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op6(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (6, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op7(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (7, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op8(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (8, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op9(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (9, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op10(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (10, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op11(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (11, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op12(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (12, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op13(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (13, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op14(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (14, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op15(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (15, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op16(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (16, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op17(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (17, 16)
#         self.trans_x = True
#         self.trans_y = True

# class TestMatmulInt8Op18(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 16)
#         self.y_shape = (18, 16)
#         self.trans_x = True
#         self.trans_y = True
# test true false
# class TestMatmulInt8Op2(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (4, 2)
#         self.y_shape = (4, 12)
#         self.trans_x = True
#         self.trans_y = False

# test true true
# class TestMatmulInt8Op3(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (4, 2)
#         self.y_shape = (12, 4)
#         self.trans_x = True
#         self.trans_y = True


# test false false
# class TestMatmulInt8Op2(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (2, 4)
#         self.y_shape = (12, 4)
#         self.trans_x = False
#         self.trans_y = True

# class TestMatmulInt8Op3(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (3, 4)
#         self.y_shape = (4, 3)
#         self.trans_x = False
#         self.trans_y = False
        
# class TestMatmulInt8Op4(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (4, 4)
#         self.y_shape = (4, 4)
#         self.trans_x = False
#         self.trans_y = False
        
# class TestMatmulInt8Op5(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (5, 4)
#         self.y_shape = (4, 5)
#         self.trans_x = False
#         self.trans_y = False
        
# class TestMatmulInt8Op6(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (6, 4)
#         self.y_shape = (4, 6)
#         self.trans_x = False
#         self.trans_y = False
        
# class TestMatmulInt8Op7(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (7, 4)
#         self.y_shape = (4, 7)
#         self.trans_x = False
#         self.trans_y = False
        
# class TestMatmulInt8Op8(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (8, 4)
#         self.y_shape = (4, 8)
#         self.trans_x = False
#         self.trans_y = False
        
# class TestMatmulInt8Op9(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (9, 4)
#         self.y_shape = (4, 9)
#         self.trans_x = False
#         self.trans_y = False
        
# class TestMatmulInt8Op10(TestMatmulInt8):
#     def config(self):
#         self.dtype = 'int8'
#         self.rtol = 1e-5
#         self.atol = 1e-2
#         self.bias = False
#         self.x_shape = (16, 4)
#         self.y_shape = (4, 16)
#         self.trans_x = False
#         self.trans_y = False


if __name__ == '__main__':
    unittest.main()
