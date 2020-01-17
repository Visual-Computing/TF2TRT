# Copyright 2020, Visual Computing Group at HTW. All Rights Reserved.
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
# ==============================================================================

import abc
import numpy as np
import tensorflow as tf


class TFProtobufParser(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, graph_def, trt_builder):
        """

        :param graph_def:
        :param trt_builder: TRTBuilder
        """
        self._trt_builder = trt_builder

        # Get the nodes of the Directed Acyclic Graph (DAG) in Topological Order
        # https://www.geeksforgeeks.org/python-program-for-topological-sorting/
        self._topological_nodes = self._topological_sort(graph_def.node, trt_builder.registered_input_names)

    @property
    def topological_nodes(self):
        """
        All nodes in topological order

        Returns
        -------
        nodes : pb_nodes[]
            array of nodes
        """
        return self._topological_nodes

    def parse(self):
        """
        Parse the pb file
        https://github.com/Azure/aml-real-time-ai/tree/master/pythonlib/amlrealtimeai/external/tensorflow/core/framework

        :return: bool
        """

        for node in self._topological_nodes:

            if "Placeholder" == node.op:
                self._parse_placeholder(node)
            elif "Const" == node.op:
                self._parse_constant(node)
            elif "Shape" == node.op:
                self._parse_shape(node)
            elif "Cast" == node.op:
                self._parse_cast(node)
            elif "FusedBatchNorm" == node.op:
                self._parse_fused_batch_norm(node)
            elif "FusedBatchNormV2" == node.op:
                self._parse_fused_batch_norm(node)
            elif "Pad" == node.op:
                self._parse_padding(node)
            elif "Conv2D" == node.op:
                self._parse_conv2d(node)
            elif "Relu" == node.op:
                self._parse_relu(node)
            elif "Selu" == node.op:
                self._parse_selu(node)
            elif "MaxPool" == node.op:
                self._parse_maxpool(node)
            elif "Add" == node.op:
                self._parse_add(node)
            elif "Sub" == node.op:
                self._parse_sub(node)
            elif "Mul" == node.op:
                self._parse_mul(node)
            elif "RealDiv" == node.op:
                self._parse_div(node)
            elif "Maximum" == node.op:
                self._parse_maximum(node)
            elif "Minimum" == node.op:
                self._parse_minimum(node)
            elif "Rsqrt" == node.op:
                self._parse_rsqrt(node)
            elif "Sqrt" == node.op:
                self._parse_sqrt(node)
            elif "Square" == node.op:
                self._parse_square(node)
            elif "StridedSlice" == node.op:
                self._parse_strided_slice(node)
            elif "Slice" == node.op:
                self._parse_slice(node)
            elif "Max" == node.op:
                self._parse_reduce_max(node)
            elif "Mean" == node.op:
                self._parse_reduce_mean(node)
            elif "Sum" == node.op:
                self._parse_reduce_sum(node)
            elif "ExpandDims" == node.op:
                self._parse_expand_dims(node)
            elif "Squeeze" == node.op:
                self._parse_squeeze(node)
            elif "Reshape" == node.op:
                self._parse_reshape(node)
            elif "Transpose" == node.op:
                self._parse_transpose(node)
            elif "ConcatV2" == node.op:
                self._parse_concatv2(node)
            elif "Pack" == node.op:
                self._parse_pack(node)
            elif "MatMul" == node.op:
                self._parse_matmul(node)
            elif "BiasAdd" == node.op:
                self._parse_bias_add(node)
            elif "Softmax" == node.op:
                self._parse_softmax(node)
            else:
                raise Exception("Could not parse node", node)

        return True

    def _parse_placeholder(self, pb_placeholder_node):
        """
        Parse the placeholder node. A name and dtype needs to be present.

        :param pb_placeholder_node:
        """
        name = pb_placeholder_node.name
        dtype = self._parse_dtype_attr(pb_placeholder_node.attr['dtype'].type)
        shape = self._parse_shape_attr(pb_placeholder_node.attr['shape'].shape)

        self._trt_builder.add_placeholder(name=name, dtype=dtype, shape=shape)

    def _parse_constant(self, pb_const_node):
        """
        Parse the constant node. A name and value tensor needs to be present.

        :param pb_const_node:
        """
        
        name = pb_const_node.name
        np_array = tf.make_ndarray(pb_const_node.attr['value'].tensor)

        # scalars will be stored in a 1D array with a single element
        if len(np_array.shape) == 0:
            np_array = np.array([np_array])

        self._trt_builder.add_constant(name, np_array)

    def _parse_shape(self, pb_shape_node):
        """
        Parse the shape node. A name and value tensor needs to be present.

        :param pb_shape_node:
        """

        name = pb_shape_node.name
        input_names = pb_shape_node.input

        self._trt_builder.add_shape(name, input_names[0])

    def _parse_cast(self, pb_cast_node):
        """
        Parse the cast node. A name and value tensor needs to be present.

        :param pb_cast_node:
        """

        name = pb_cast_node.name
        input_names = pb_cast_node.input
        dtype = self._parse_dtype_attr(pb_cast_node.attr['DstT'].type)

        self._trt_builder.add_cast(name, input_names[0], dtype)

    def _parse_fused_batch_norm(self, pb_fused_batch_norm_node):
        """
        Parse the fused batch norm node. A name, epsilon value and five inputs need to be present.

        :param pb_fused_batch_norm_node:
        """

        name = pb_fused_batch_norm_node.name
        input_names = pb_fused_batch_norm_node.input
        eps = pb_fused_batch_norm_node.attr["epsilon"].f

        self._trt_builder.add_fused_batch_norm(name, input_tensor_name=input_names[0],
                                               scale_weights_name=input_names[1], bias_weights_name=input_names[2],
                                               mean_weights_name=input_names[3], variance_weights_name=input_names[4],
                                               eps=eps)

    def _parse_padding(self, pb_pad_node):
        """
        Parse the padding node. A name and two inputs must be present.

        :param pb_pad_node:
        """
        name = pb_pad_node.name
        input_names = pb_pad_node.input

        self._trt_builder.add_padding(name, input_names[0], input_names[1])

    def _parse_conv2d(self, pb_conv2d_node):
        """
        Parse the conv2d node. A name, two inputs, data_format, padding and strides must be present.

        :param pb_conv2d_node:
        """

        name = pb_conv2d_node.name
        input_names = pb_conv2d_node.input
        padding_type = pb_conv2d_node.attr['padding'].s.decode("utf-8")
        data_format = pb_conv2d_node.attr['data_format'].s.decode("utf-8")
        strides = np.array(pb_conv2d_node.attr['strides'].list.i)

        assert len(input_names) == 2, "conv2d with bias not supported yet"

        self._trt_builder.add_conv2d(name, input_names[0], input_names[1], data_format, padding_type, strides)

    def _parse_relu(self, pb_relu_node):
        """
        Parse the relu node. A name and an input must be present.

        :param pb_relu_node:
        """

        name = pb_relu_node.name
        input_names = pb_relu_node.input

        self._trt_builder.add_relu(name, input_names[0])

    def _parse_selu(self, pb_selu_node):
        """
        Parse the relu node. A name and an input must be present.

        :param pb_selu_node:
        """

        name = pb_selu_node.name
        input_names = pb_selu_node.input

        # default values
        alpha = 1.67326319217681884765625
        beta = 1.05070102214813232421875

        self._trt_builder.add_selu(name, input_names[0], alpha, beta)

    def _parse_maxpool(self, pb_maxpool_node):
        """
        Parse the maxpool node. A name, padding, ksize and strides must be present.

        :param pb_maxpool_node:
        """
        name = pb_maxpool_node.name
        input_names = pb_maxpool_node.input

        padding_type = pb_maxpool_node.attr['padding'].s.decode("utf-8")
        kernel_size = np.array(pb_maxpool_node.attr['ksize'].list.i)
        strides = np.array(pb_maxpool_node.attr['strides'].list.i)

        self._trt_builder.add_maxpool(name, input_names[0], padding_type, kernel_size, strides)

    def _parse_add(self, pb_add_node):
        """
        Parse the addition node. A name and inputs must be present.

        :param pb_add_node:
        """
        name = pb_add_node.name
        input_names = [s for s in pb_add_node.input]

        self._trt_builder.add_addition(name, input_names)

    def _parse_sub(self, pb_sub_node):
        """
        Parse the substraction node. A name and inputs must be present.

        :param pb_sub_node:
        """
        name = pb_sub_node.name
        input_names = [s for s in pb_sub_node.input]

        self._trt_builder.add_substraction(name, input_names)

    def _parse_mul(self, pb_mul_node):
        """
        Parse the multiplication node. A name and inputs must be present.

        :param pb_mul_node:
        """
        name = pb_mul_node.name
        input_names = [s for s in pb_mul_node.input]

        self._trt_builder.add_multiplication(name, input_names)

    def _parse_div(self, pb_div_node):
        """
        Parse the division node. A name and inputs must be present.

        :param pb_div_node:
        """
        name = pb_div_node.name
        input_names = [s for s in pb_div_node.input]

        self._trt_builder.add_division(name, input_names)

    def _parse_maximum(self, pb_maximum_node):
        """
        Parse the maximum node. A name and inputs must be present.

        :param pb_maximum_node:
        """
        name = pb_maximum_node.name
        input_names = [s for s in pb_maximum_node.input]

        self._trt_builder.add_maximum(name, input_names)

    def _parse_minimum(self, pb_minimum_node):
        """
        Parse the minimum node. A name and inputs must be present.

        :param pb_minimum_node:
        """
        name = pb_minimum_node.name
        input_names = [s for s in pb_minimum_node.input]

        self._trt_builder.add_minimum(name, input_names)

    def _parse_square(self, pb_square_node):
        """
        Parse the square node. A name and input must be present.

        :param pb_square_node:
        """

        name = pb_square_node.name
        input_names = pb_square_node.input

        self._trt_builder.add_square(name, input_names[0])

    def _parse_sqrt(self, pb_sqrt_node):
        """
        Parse the sqrt node. A name and input must be present.

        :param pb_sqrt_node:
        """

        name = pb_sqrt_node.name
        input_names = pb_sqrt_node.input

        self._trt_builder.add_sqrt(name, input_names[0])

    def _parse_recip(self, pb_recip_node):
        """
        Parse the recip node. A name and input must be present.

        :param pb_recip_node:
        """

        name = pb_recip_node.name
        input_names = pb_recip_node.input

        self._trt_builder.add_recip(name, input_names[0])

    def _parse_rsqrt(self, pb_rsqrt_node):
        """
        Parse the rsqrt node. A name and input must be present.

        :param pb_rsqrt_node:
        """

        name = pb_rsqrt_node.name
        input_names = pb_rsqrt_node.input

        self._trt_builder.add_rsqrt(name, input_names[0])

    def _parse_strided_slice(self, pb_strided_slice_node):
        """
        Parse the strided slice node. A name and inputs must be present.

        :param pb_strided_slice_node:
        """

        name = pb_strided_slice_node.name
        input_names = pb_strided_slice_node.input

        self._trt_builder.add_strided_slice(name, input_names[0], input_names[1], input_names[2], input_names[3])

    def _parse_slice(self, pb_slice_node):
        """
        Parse the slice node. A name and inputs must be present.

        :param pb_slice_node:
        """

        name = pb_slice_node.name
        input_names = pb_slice_node.input

        self._trt_builder.add_slice(name, input_names[0], input_names[1], input_names[2])

    def _parse_reduce_max(self, pb_reduce_max_node):
        """
        Parse the reduce maximum node. A name and inputs must be present.

        :param pb_reduce_max_node:
        """

        name = pb_reduce_max_node.name
        input_names = pb_reduce_max_node.input
        keep_dims = pb_reduce_max_node.attr["keep_dims"].b

        self._trt_builder.add_reduce_max(name, input_names[0], input_names[1], keep_dims)

    def _parse_reduce_mean(self, pb_reduce_mean_node):
        """
        Parse the reduce mean node. A name and inputs must be present.

        :param pb_reduce_mean_node:
        """

        name = pb_reduce_mean_node.name
        input_names = pb_reduce_mean_node.input
        keep_dims = pb_reduce_mean_node.attr["keep_dims"].b

        self._trt_builder.add_reduce_mean(name, input_names[0], input_names[1], keep_dims)

    def _parse_reduce_min(self, pb_reduce_min_node):
        """
        Parse the reduce minimum node. A name and inputs must be present.

        :param pb_reduce_min_node:
        """

        name = pb_reduce_min_node.name
        input_names = pb_reduce_min_node.input
        keep_dims = pb_reduce_min_node.attr["keep_dims"].b

        self._trt_builder.add_reduce_min(name, input_names[0], input_names[1], keep_dims)

    def _parse_reduce_prod(self, pb_reduce_prod_node):
        """
        Parse the reduce product node. A name and inputs must be present.

        :param pb_reduce_prod_node:
        """

        name = pb_reduce_prod_node.name
        input_names = pb_reduce_prod_node.input
        keep_dims = pb_reduce_prod_node.attr["keep_dims"].b

        self._trt_builder.add_reduce_prod(name, input_names[0], input_names[1], keep_dims)

    def _parse_reduce_sum(self, pb_reduce_sum_node):
        """
        Parse the reduce sum node. A name and inputs must be present.

        :param pb_reduce_sum_node:
        """

        name = pb_reduce_sum_node.name
        input_names = pb_reduce_sum_node.input
        keep_dims = pb_reduce_sum_node.attr["keep_dims"].b

        self._trt_builder.add_reduce_sum(name, input_names[0], input_names[1], keep_dims)

    def _parse_reshape(self, pb_reshape_node):
        """
        Parse the reshape  node. A name and inputs must be present.

        :param pb_reshape_node:
        """

        name = pb_reshape_node.name
        input_names = pb_reshape_node.input

        self._trt_builder.add_reshape(name, input_names[0], input_names[1])

    def _parse_expand_dims(self, pb_expand_dims_node):
        """
        Parse the expand dims node. A name and inputs must be present.

        :param pb_expand_dims_node:
        """

        name = pb_expand_dims_node.name
        input_names = pb_expand_dims_node.input

        self._trt_builder.add_expand_dims(name, input_names[0], input_names[1])

    def _parse_squeeze(self, pb_squeeze_node):
        """
        Parse the squeeze node. A name, squeeze_dims and inputs must be present.

        :param pb_squeeze_node:
        """

        name = pb_squeeze_node.name
        input_names = pb_squeeze_node.input
        squeeze_dims = np.array(pb_squeeze_node.attr['squeeze_dims'].list.i)

        self._trt_builder.add_squeeze(name, input_names[0], squeeze_dims)

    def _parse_transpose(self, pb_transpose_node):
        """
        Parse the transpose node. A name and inputs must be present.

        :param pb_transpose_node:
        """

        name = pb_transpose_node.name
        input_names = pb_transpose_node.input

        self._trt_builder.add_transpose(name, input_names[0], input_names[1])

    def _parse_concatv2(self, pb_concat_node):
        """
        Parse the concatv2 node. A name, image count and inputs must be present.

        :param pb_concat_node:
        """

        name = pb_concat_node.name
        input_names = [s for s in pb_concat_node.input]
        input_count = pb_concat_node.attr["N"].i

        self._trt_builder.add_concatv2(name, input_names[:input_count], input_names[input_count])

    def _parse_pack(self, pb_pack_node):
        """
        Parse the pack node. A name, image count, axis and inputs must be present.

        :param pb_pack_node:
        """

        name = pb_pack_node.name
        input_names = [s for s in pb_pack_node.input]
        axis = pb_pack_node.attr["axis"].i

        self._trt_builder.add_pack(name, input_names, axis)

    def _parse_matmul(self, pb_matmul_node):
        """
        Parse the matmul node. A name, transpose_a, transpose_b inputs must be present.

        :param pb_matmul_node:
        """

        name = pb_matmul_node.name
        input_names = pb_matmul_node.input
        transpose_a = pb_matmul_node.attr["transpose_a"].b
        transpose_b = pb_matmul_node.attr["transpose_b"].b

        self._trt_builder.add_matmul(name, input_names[0], input_names[1], transpose_a, transpose_b)

    def _parse_bias_add(self, pb_bias_add_node):
        """
        Parse the bias add node. A name and inputs must be present.

        :param pb_bias_add_node:
        """

        name = pb_bias_add_node.name
        input_names = pb_bias_add_node.input

        self._trt_builder.add_bias_add(name, input_names[0], input_names[1])

    def _parse_softmax(self, pb_softmax_node):
        """
        Parse the softmax node. A name and input must be present.

        :param pb_softmax_node:
        """

        name = pb_softmax_node.name
        input_names = pb_softmax_node.input

        self._trt_builder.add_softmax(name, input_names[0])

    @staticmethod
    def _parse_shape_attr(pb_shape):
        """
        Convert a tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto to the corresponding trt version
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto

        :param pb_shape:
        :return: array<int>
        """
        output_shape = []
        for pb_shape_dim in pb_shape.dim:
            output_shape.append(pb_shape_dim.size)

        return output_shape

    @staticmethod
    def _parse_dtype_attr(pb_dtype):
        """
        Convert the dtype number to the corresponding trt version
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto

        :param pb_dtype:
        :return: numpy_dtype
        """
        if pb_dtype == tf.float32.as_datatype_enum:
            return np.float32
        elif pb_dtype == tf.int32.as_datatype_enum:
            return np.int32
        elif pb_dtype == tf.int8.as_datatype_enum:
            return np.int8
        elif pb_dtype == tf.float16.as_datatype_enum:
            return np.float16

        return

    @staticmethod
    def _topological_recursive_sort(parent, name_to_node, visited, stack):
        """
        A recursive function used by _topological_sort()
        :param parent:
        :param name_to_node:
        :param visited:
        :param stack:
        :return:
        """

        # Mark the current node as visited.
        visited[parent.name] = True

        # Recursive for all the vertices adjacent to this vertex
        for node_name in parent.input:
            node_name = node_name.rsplit(':')[0]
            if node_name not in visited:
                node = name_to_node[node_name]
                TFProtobufParser._topological_recursive_sort(node, name_to_node, visited, stack)

        # Push current vertex to stack which stores result
        stack.append(parent)

    @staticmethod
    def _topological_sort(nodes, input_node_names):
        """
        The function to do Topological Sort. It uses recursive _topological_recursive_sort()
        :param nodes:
        :return:
        """

        # mapping from node names to the actual node
        name_to_node = {}
        for node in nodes:
            name_to_node[node.name] = node

        # Mark all the vertices as not visited
        visited = {}
        stack = []

        # Collect the inputs first to have a fix binding order
        for input_node_name in input_node_names:
            if input_node_name not in visited:
                TFProtobufParser._topological_recursive_sort(name_to_node[input_node_name], name_to_node, visited,
                                                             stack)

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for node in nodes:
            if node.name not in visited:
                TFProtobufParser._topological_recursive_sort(node, name_to_node, visited, stack)

        return stack
