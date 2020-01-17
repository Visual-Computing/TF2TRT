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
import tensorrt as trt


class TRTNetworkBuilder(object):
    """
    Similar to
    https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/ImporterContext.hpp
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, network):
        """

        :param network: tensorrt.INetworkDefinition
        """

        # network and its input and output nodes
        self._network = network
        self._registered_input_node_names = []
        self._registered_input_node_shapes = []
        self._registered_output_node_names = []

        # ops and outputs of the resulting graph
        self._ops = {}
        self._ops_output = {}

    @property
    def registered_output_names(self):
        """
        List of all registered output node names

        Returns
        -------
        output_node_names : array<string>
        """
        return self._registered_output_node_names

    @property
    def registered_input_names(self):
        """
        List of all registered input node names

        Returns
        -------
        input_node_names : array<string>
        """
        return self._registered_input_node_names

    @property
    def registered_input_shapes(self):
        """
        List of all registered input node shapes

        Returns
        -------
        input_node_shapes : array<array<int>>
        """
        return self._registered_input_node_shapes

    @property
    def network(self):
        """
        List of all registered output node names

        Returns
        -------
        network : tensorrt.INetworkDefinition
        """
        return self._network

    @property
    def ops(self):
        """
        Map of all network layer names to their corresponding operation.

        :return: array<tensorrt.ILayer>
        """
        return self._ops

    @property
    def ops_output(self):
        """
        Map of all network layer names to the corresponding output of the layer.

        :return: array<tensorrt.ITensor>
        """
        return self._ops_output

    def register_output(self, output_node_name):
        """
        Register an output node name of a TRT network.

        :param output_node_name: string
        :return:
        """
        self._registered_output_node_names.append(output_node_name)

    def register_outputs(self, output_node_names):
        """
        Register a list of output node names of a TRT network.

        :param output_node_names: array<string>
        :return:
        """
        for output_node_name in output_node_names:
            self.register_output(output_node_name)

    def register_input(self, input_node_name, input_node_shape):
        """
        Register an input name of a TRT network with the associated Dimensions.

        :param input_node_name: string
        :param input_node_shape: array<int>
        :return:
        """

        assert np.prod(input_node_shape) < (1 << 30), "The total volume of the input "+input_node_name+" must be less than 2^30 elements"

        self._registered_input_node_names.append(input_node_name)
        self._registered_input_node_shapes.append(input_node_shape)

    def register_inputs(self, input_node_names, input_node_shapes):
        """
        Register a list of input names of a TRT network with the associated Dimensions.

        :param input_node_names: array<string>
        :param input_node_shapes: array<array<int>>
        :return:
        """
        for name, shape in zip(input_node_names, input_node_shapes):
            self.register_input(name, shape)

    @staticmethod
    def print_network(network):
        """
        Print the information of a network.

        :param network: tensorrt::INetworkDefinition
        :return:
        """
        print("name", network.name)
        print("num_layers", network.num_layers)
        print("num_inputs", network.num_inputs)
        print("num_outputs", network.num_outputs)
        print("has_implicit_batch_dimension", network.has_implicit_batch_dimension)
        print("has_explicit_precision", network.has_explicit_precision)

    @staticmethod
    def print_layer(layer):
        """
        Print the information of a network layer.

        :param layer: tensorrt.ILayer
        :return:
        """
        print("name", layer.name)
        print("type", layer.type)
        print("num_inputs", layer.num_inputs)
        print("num_outputs", layer.num_outputs)
        print("precision", layer.precision)
        print("precision_is_set", layer.precision_is_set)

    @staticmethod
    def print_tensor(tensor):
        """
        Print the information of a tensor.

        :param tensor: tensorrt.ITensor
        :return:
        """
        print("name", tensor.name)
        print("shape", tensor.shape)
        print("dtype", tensor.dtype)
        print("dynamic_range", tensor.dynamic_range)
        print("location ", tensor.location)
        print("is_network_input", tensor.is_network_input)
        print("is_network_output", tensor.is_network_output)

    def get_input_shape(self, input_node_name):
        """
        Return the registered inpute shape for the given name. Otherwise none.

        :param input_node_name:
        :return:
        """

        for i in range(len(self._registered_input_node_names)):
            if input_node_name == self._registered_input_node_names[i]:
                return self._registered_input_node_shapes[i]
        return None

    def get_layer_output(self, layer_name):
        """
        Get the output of a layer

        :param layer_name:
        :return:
        """

        if layer_name in self._ops_output:
            return self._ops_output[layer_name]
        else:
            return self._ops_output[layer_name.rsplit(':')[0]]

    def get_layer_weights(self, layer_name):
        """
        Get the weights of a constant layer as a numpy array

        :param layer_name: string
        :return:
        """

        layer = self._ops[layer_name]
        if isinstance(layer, trt.IConstantLayer):
            return np.reshape(self._ops[layer_name].weights, self._ops_output[layer_name].shape)
        elif isinstance(layer, trt.IIdentityLayer):
            cast_to_dtype = trt.nptype(layer.precision)
            const_layer_name = layer.get_input(0).name
            weights = self.get_layer_weights(const_layer_name)
            return weights.astype(cast_to_dtype)
        else:
            return layer.get_output(0)
            #raise Exception("No valid layer "+layer_name+" to retrieve weights")

    def add_placeholder(self, name, dtype, shape=None):
        """
        Add a placeholder op to the network.

        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/ModelImporter.cpp#L36

        :param name:
        :param dtype: numpy dtype
        :param shape:
        :return: placerholder tensor
        """

        overwrite_shape = self.get_input_shape(name)
        if overwrite_shape:
            shape = overwrite_shape
        placerholder_tensor = self._network.add_input(name=name, dtype=TRTNetworkBuilder._to_dtype(dtype), shape=shape)
        self._remember_op_output(placerholder_tensor, name)
        return placerholder_tensor

    def add_constant(self, name, np_array):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L562

        :param name:
        :param np_array:
        :return:
        """

        const_layer = self._network.add_constant(shape=np_array.shape, weights=np_array)
        self._remember_op_and_output(const_layer, name)
        return const_layer

    def add_shape(self, name, input_tensor_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1533

        :param name:
        :param input_tensor_name:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)
        shape_layer = self._network.add_shape(input=input_tensor)
        self._remember_op_and_output(shape_layer, name)
        return shape_layer

    def add_cast(self, name, input_tensor_name, dtype):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L278

        :param name:
        :param input_tensor_name:
        :param dtype: numpy dtype
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)

        cast_layer = self._network.add_identity(input_tensor)
        cast_layer.precision = TRTNetworkBuilder._to_dtype(dtype)
        self._remember_op_and_output(cast_layer, name)
        return cast_layer

    def add_fused_batch_norm(self, name, input_tensor_name, scale_weights_name, bias_weights_name, mean_weights_name, variance_weights_name, eps):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L204

        :param name:
        :param input_tensor_name:
        :param scale_weights_name:
        :param bias_weights_name:
        :param mean_weights_name:
        :param variance_weights_name:
        :param eps:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)
        scale_weights = self._ops[scale_weights_name].weights
        bias_weights = self._ops[bias_weights_name].weights
        mean_weights = self._ops[mean_weights_name].weights
        variance_weights = self._ops[variance_weights_name].weights

        combined_scale_weights = scale_weights / np.sqrt(variance_weights + eps)
        combined_bias_weights = bias_weights - mean_weights * combined_scale_weights

        batch_norm_layer = self._network.add_scale(input=input_tensor, mode=trt.ScaleMode.CHANNEL,
                                                   shift=combined_bias_weights, scale=combined_scale_weights)
        self._remember_op_and_output(batch_norm_layer, name)
        return batch_norm_layer

    def add_padding(self, name, input_tensor_name, padding_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1321

        :param name:
        :param input_tensor_name:
        :param padding_name:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)
        pad = self.get_layer_weights(padding_name)

        # just use the last two padding dims in a channel first setup to get width/height paddings
        pre_padding = trt.DimsHW(pad[-2:, 0])
        post_padding = trt.DimsHW(pad[-2:, 1])

        # create layer
        pad_layer = self._network.add_padding(input=input_tensor, pre_padding=pre_padding, post_padding=post_padding)
        return self._remember_op_and_output(pad_layer, name)

    def add_conv2d(self, name, input_tensor_name, weights_name, data_format, padding_type, strides):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L332

        :param name:
        :param input_tensor_name:
        :param weights_name:
        :param data_format:
        :param padding_type:
        :param strides:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)
        weights = self.get_layer_weights(weights_name)

        # Check that the number of spatial dimensions and the kernel shape matches up.
        nb_spatial_dims = len(input_tensor.shape) - 2
        assert nb_spatial_dims == len(weights.shape) - 2, "input tensor and weights do not have the same rank"

        # Check that the data of the weights is in NCHW
        assert 'NCHW' in data_format, "conv2d is in "+data_format+", not in NCHW"

        # check for valid padding in pooling layers
        assert padding_type in ["VALID", "SAME"], "Conv2d only supports valid or same padding not "+padding_type

        # Create empty bias arrays
        bias = trt.Weights(type=TRTNetworkBuilder._to_dtype(weights.dtype))
        #if len(input_names) == 3:
        #    bias = self.get_layer_weights(bias_name)

        # weight are stored in RSCK where K is the number of output feature maps,
        # C the number of input channels, and R and S are the height and width of the filter.
        num_output_maps = weights.shape[-1]
        kernel_shape = trt.DimsHW(weights.shape[:2])

        # Cannot construct Weights object from non-contiguous array. Please use numpy.ascontiguousarray.
        weights = weights.transpose([3, 2, 0, 1])
        weights = np.ascontiguousarray(weights, dtype=weights.dtype)
        weights = trt.Weights(a=weights)

        # create layer
        conv2d_layer = self._network.add_convolution(input=input_tensor, num_output_maps=num_output_maps, kernel_shape=kernel_shape, kernel=weights, bias=bias)
        conv2d_layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_DOWN if padding_type == "VALID" else trt.PaddingMode.SAME_UPPER
        #conv2d_layer.pre_padding = trt.DimsHW([1, 1])
        #conv2d_layer.post_padding = trt.DimsHW([1, 1])
        conv2d_layer.stride = trt.DimsHW(strides[-2:])

        self._remember_op_and_output(conv2d_layer, name)
        return conv2d_layer

    def add_relu(self, name, input_tensor_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1487
        :param name:
        :param input_tensor_name:
        :return:
        """
        return self.add_activation_func(name, input_tensor_name, trt.ActivationType.RELU)

    def add_selu(self, name, input_tensor_name, alpha=None, beta=None):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1525

        :param name:
        :param input_tensor_name:
        :param alpha:
        :param beta:
        :return:
        """

        # default values
        if alpha is None:
            alpha = 1.67326319217681884765625
        if beta is None:
            beta = 1.05070102214813232421875

        return self.add_activation_func(name, input_tensor_name, trt.ActivationType.SELU, alpha, beta)

    def add_activation_func(self, name, input_tensor_name, activation_type, alpha=None, beta=None):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L60

        :param name:
        :param input_tensor_name:
        :param activation_type:
        :param alpha: optional
        :param beta: optional
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)

        # create layer
        activation_layer = self._network.add_activation(input=input_tensor, type=activation_type)
        if alpha:
            activation_layer.alpha = alpha
        if beta:
            activation_layer.beta = beta

        return self._remember_op_and_output(activation_layer, name)

    def add_maxpool(self, name, input_tensor_name, padding_type, kernel_size, strides):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1277

        :param name:
        :param input_tensor_name:
        :param padding_type:
        :param kernel_size:
        :param strides:
        :return:
        """
        return self.add_pooling_func(name, input_tensor_name, trt.PoolingType.MAX, padding_type, kernel_size, strides)

    def add_pooling_func(self, name, input_tensor_name, pooling_type, padding_type, kernel_size, strides, blend_factor=None):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L1002

        :param name:
        :param input_tensor_name:
        :param pooling_type:
        :param padding_type:
        :param kernel_size:
        :param strides:
        :param blend_factor: optional
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)

        # check for valid padding in pooling layers
        assert padding_type in ["VALID", "SAME"], "Pooling only supports valid or same padding"

        # 2D windows size
        window_size = trt.DimsHW(kernel_size[-2:])

        # create layer
        pooling_layer = self._network.add_pooling(input=input_tensor, type=pooling_type, window_size=window_size)
        pooling_layer.stride = trt.DimsHW(strides[-2:])
        pooling_layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_DOWN if padding_type == "VALID" else trt.PaddingMode.SAME_UPPER
        if blend_factor:
            pooling_layer.blend_factor = blend_factor

        self._remember_op_and_output(pooling_layer, name)
        return pooling_layer

    def add_addition(self, name, input_tensor_names):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L109

        :param name:
        :param input_tensor_names:
        :return:
        """
        return self.add_elementwise_func(name, input_tensor_names, trt.ElementWiseOperation.SUM)

    def add_substraction(self, name, input_tensor_names):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1922

        :param name:
        :param input_tensor_names:
        :return:
        """
        return self.add_elementwise_func(name, input_tensor_names, trt.ElementWiseOperation.SUB)

    def add_multiplication(self, name, input_tensor_names):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1311

        :param name:
        :param input_tensor_names:
        :return:
        """
        return self.add_elementwise_func(name, input_tensor_names, trt.ElementWiseOperation.PROD)

    def add_division(self, name, input_tensor_names):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L617

        :param name:
        :param input_tensor_names:
        :return:
        """
        return self.add_elementwise_func(name, input_tensor_names, trt.ElementWiseOperation.DIV)

    def add_maximum(self, name, input_tensor_names):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1272

        :param name:
        :param input_tensor_names:
        :return:
        """
        return self.add_elementwise_func(name, input_tensor_names, trt.ElementWiseOperation.MAX)

    def add_minimum(self, name, input_tensor_names):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1306

        :param name:
        :param input_tensor_names:
        :return:
        """
        return self.add_elementwise_func(name, input_tensor_names, trt.ElementWiseOperation.MIN)

    def add_square(self, name, input_tensor_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1897

        :param name:
        :param input_tensor_name:
        :return:
        """

        return self.add_elementwise_func(name, [input_tensor_name, input_tensor_name], trt.ElementWiseOperation.PROD)

    def add_elementwise_func(self, name, input_tensor_names, binary_op):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L593

        :param name:
        :param input_tensor_names:
        :param binary_op:
        :return:
        """

        # convert to list
        if isinstance(input_tensor_names, list) is False:
            input_tensor_names = [input_tensor_names]

        # collect all input tensors
        input_tensors = []
        for input_name in input_tensor_names:
            input_tensors.append(self.get_layer_output(input_name))

        # need at least two inputs
        assert len(input_tensors) >= 2, "Not enough inputs for elementwise op: " + str(binary_op)

        # find max rank
        max_nb_dims = -1
        for input_tensor in input_tensors:
            max_nb_dims = max(max_nb_dims, len(input_tensor.shape))

        # broadcast input tensors
        input_tensors_broadcasted = []
        for input_tensor in input_tensors:
            input_tensor_broadcasted = self._broadcastTensor(input_tensor, max_nb_dims,
                                                             name + "_reshape_" + input_tensor.name)

            assert len(input_tensor_broadcasted.shape) == max_nb_dims, \
                "Failed to broadcast tensor " + input_tensor.name + " for elementwise op " + name
            input_tensors_broadcasted.append(input_tensor_broadcasted)

        # Use the first tensor input as the base for the elementwise operation
        combined = input_tensors_broadcasted[0]
        elementwise_layer = None
        for i in range(1, len(input_tensors_broadcasted)):
            combined_name = name
            if len(input_tensors_broadcasted) > 2:
                combined_name += "_combine" + str(i)

            # create layer
            elementwise_layer = self._network.add_elementwise(combined, input_tensors_broadcasted[i], op=binary_op)
            self._remember_op_and_output(elementwise_layer, combined_name)

        return elementwise_layer

    def add_sqrt(self, name, input_tensor_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1897

        :param name:
        :param input_tensor_name:
        :return:
        """
        return self.add_unary_func(name, input_tensor_name, trt.UnaryOperation.SQRT)

    def add_recip(self, name, input_tensor_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1378

        :param name:
        :param input_tensor_name:
        :return:
        """
        return self.add_unary_func(name, input_tensor_name, trt.UnaryOperation.RECIP)

    def add_rsqrt(self, name, input_tensor_name):
        """
        Special op not supported by tensorrt. Create a sqrt op followed by a recip op.
        The recip will get the name of the rsqrt node.

        :param name:
        :param input_tensor_name:
        :return:
        """

        # create sqrt and recip layerlayer
        self.add_unary_func(name+"_sqrt", input_tensor_name, trt.UnaryOperation.SQRT)
        return self.add_unary_func(name, name + "_sqrt", trt.UnaryOperation.RECIP)

    def add_unary_func(self, name, input_tensor_name, unary_op):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L1283

        :param name:
        :param input_tensor_name:
        :param unary_op:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)

        # create layer
        unary_layer = self._network.add_unary(input=input_tensor,  op=unary_op)
        self._remember_op_and_output(unary_layer, name)
        return unary_layer

    def add_slice(self, name, input_tensor_name, slice_begin_name, slice_size_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1684

        :param name:
        :param input_tensor_name:
        :param slice_begin_name:
        :param slice_size_name:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)
        slice_begin = self.get_layer_weights(slice_begin_name)
        slice_size = self.get_layer_weights(slice_size_name)
        stride = np.ones(shape=slice_size.shape, dtype=np.int32)

        return self._add_slice(name, input_tensor, slice_begin, slice_size, stride)

    def add_strided_slice(self, name, input_tensor_name, slice_begin_name, slice_end_name, slice_stride_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1684

        :param name:
        :param input_tensor_name:
        :param slice_begin_name:
        :param slice_end_name:
        :param slice_stride_name:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)
        slice_begin = self.get_layer_weights(slice_begin_name)
        slice_end = self.get_layer_weights(slice_end_name)
        slice_stride = self.get_layer_weights(slice_stride_name)
        slice_size = np.floor((slice_end - slice_begin) / slice_stride).astype(slice_end.dtype)

        return self._add_slice(name, input_tensor, slice_begin, slice_size, slice_stride)

    def _add_slice(self, name, input_tensor, slice_begin, slice_size, slice_stride):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1684

        :param name:
        :param input_tensor:
        :param slice_begin:
        :param slice_size:
        :param slice_stride:
        :return:
        """

        inputs = [input_tensor]
        rank = len(input_tensor.shape)
        params = [slice_begin, slice_size, slice_stride]
        param_names = ["slice_begin", "slice_size", "slice_stride"]

        # use all non np.ndarray parameters as tensors and feed them as input in the slice layer
        # https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-601/tensorrt-api/python_api/infer/Graph/Layers.html#tensorrt.ISliceLayer
        # https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1616
#        for i in reversed(range(3)):
#
#            # check which param is not an numpy array
#            if isinstance(params[i], np.ndarray) is False:
#
#                # convert every other param with the same or lower index to a trt.ITensor
#                for j in range(i+1):
#
#                    # convert all params to trt.ITensor
#                    param_weight_or_tensor = params[j]
#                    if isinstance(param_weight_or_tensor, np.ndarray):
#
#                        # expand the weights to the same rank as the input tensor
#                        while len(param_weight_or_tensor.shape) < rank:
#                            param_weight_or_tensor = np.expand_dims(param_weight_or_tensor, 0)
#
#                        param_layer = self.add_constant(name+"_constant_"+param_names[j], param_weight_or_tensor)
#                        param_weight_or_tensor = param_layer.get_output(0)
#
#                    # TODO might need to _broadcastTensor the existing ITensors
#
#                    # copy all params to the input list
#                    inputs.append(param_weight_or_tensor)
#                    params[j] = trt.Dims([rank])
#                break

        # create the slice layer
        slice_layer = self._network.add_slice(input=input_tensor, start=params[0], shape=params[1], stride=params[2])
#        for i in range(len(inputs)):
#            slice_layer.set_input(i, inputs[i])

        self._remember_op_and_output(slice_layer, name)
        return slice_layer

    def add_reduce_max(self, name, input_tensor_name, axis_tensor_name, keep_dims):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1458

        :param name:
        :param input_tensor_name:
        :param axis_tensor_name:
        :param keep_dims:
        :return:
        """
        return self.add_reduce_func(name, input_tensor_name, axis_tensor_name, keep_dims, trt.ReduceOperation.MAX)

    def add_reduce_mean(self, name, input_tensor_name, axis_tensor_name, keep_dims):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1462

        :param name:
        :param input_tensor_name:
        :param axis_tensor_name:
        :param keep_dims:
        :return:
        """
        return self.add_reduce_func(name, input_tensor_name, axis_tensor_name, keep_dims, trt.ReduceOperation.AVG)

    def add_reduce_min(self, name, input_tensor_name, axis_tensor_name, keep_dims):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1466

        :param name:
        :param input_tensor_name:
        :param axis_tensor_name:
        :param keep_dims:
        :return:
        """
        return self.add_reduce_func(name, input_tensor_name, axis_tensor_name, keep_dims, trt.ReduceOperation.MIN)

    def add_reduce_prod(self, name, input_tensor_name, axis_tensor_name, keep_dims):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1470

        :param name:
        :param input_tensor_name:
        :param axis_tensor_name:
        :param keep_dims:
        :return:
        """
        return self.add_reduce_func(name, input_tensor_name, axis_tensor_name, keep_dims, trt.ReduceOperation.PROD)

    def add_reduce_sum(self, name, input_tensor_name, axis_tensor_name, keep_dims):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1474

        :param name:
        :param input_tensor_name:
        :param axis_tensor_name:
        :param keep_dims:
        :return:
        """
        return self.add_reduce_func(name, input_tensor_name, axis_tensor_name, keep_dims, trt.ReduceOperation.SUM)

    def add_reduce_func(self, name, input_tensor_name, axis_tensor_name, keep_dims, reduce_op):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1383

        :param name:
        :param input_tensor_name:
        :param axis_tensor_name:
        :param keep_dims:
        :param reduce_op:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)
        reduction_indices = self.get_layer_weights(axis_tensor_name)

        # TensorRT 6.0 does not accept INT32 inputs into the reduce layer.
        assert input_tensor.dtype != trt.tensorrt.DataType.INT32, "Reduce layer does not accept INT32 inputs."

        ndim = len(input_tensor.shape)

        # convert to bit mask
        axis_mask = 0
        for axis in reduction_indices:
            axis = TRTNetworkBuilder._check_axis(axis, ndim)
            axis_mask |= 1 << axis

        # create layer
        reduce_layer = self._network.add_reduce(input=input_tensor, op=reduce_op, axes=axis_mask, keep_dims=keep_dims)
        self._remember_op_and_output(reduce_layer, name)
        return reduce_layer

    def _add_reshape_layer(self, input_tensor, new_shape, name=None):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1492

        :param input_tensor:
        :param new_shape:
        :param name:
        :return:
        """

        if not name:
            name = "reshape_"+input_tensor.name

        reshape_layer = self._network.add_shuffle(input=input_tensor)
        reshape_layer.reshape_dims = new_shape
        self._remember_op_and_output(reshape_layer, name)
        return reshape_layer

    def add_reshape(self, name, input_tensor_name, shape_tensor_name):
        """
        Add a reshape layer

        :param name:
        :param input_tensor_name:
        :param shape_tensor_name:
        :return:
        """
        input_tensor = self.get_layer_output(input_tensor_name)
        shape = self.get_layer_weights(shape_tensor_name)
        return self._add_reshape_layer(input_tensor, shape, name)

    def add_expand_dims(self, name, input_tensor_name, shape_tensor_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L1291
        https://www.tensorflow.org/api_docs/python/tf/expand_dims

        :param name:
        :param input_tensor_name:
        :param shape_tensor_name:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)
        expand_dim = self._ops[shape_tensor_name].weights[0]

        if expand_dim < 0:
            expand_dim += len(input_tensor.shape) + 1
        new_shape = np.insert(input_tensor.shape, expand_dim, 1)
        # TODO ist das korrekt sollten die weights nicht eher eine axis sein

        return self._add_reshape_layer(input_tensor, new_shape, name)

    def add_squeeze(self, name, input_tensor_name, squeeze_dims):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1902
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L1193

        :param name:
        :param input_tensor_name:
        :param squeeze_dims:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)
        shape = input_tensor.shape
        rank = len(shape)
        print("input shape", shape, "rank", rank, "for layer", name)

        axes = []
        for axis in squeeze_dims:
            axes.append(TRTNetworkBuilder._check_axis(axis, rank))

        new_shape = []
        for i, dim in enumerate(input_tensor.shape):
            if i not in axes:
                new_shape.append(dim)
        print("new_shape", new_shape, "shape", shape, "squeeze_dims", squeeze_dims)
        return self._add_reshape_layer(input_tensor, new_shape, name)




    def add_transpose(self, name, input_tensor_name, perm_tensor_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L2003
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L1255

        :param name:
        :param input_tensor_name:
        :param perm_tensor_name:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)
        perm = self._ops[perm_tensor_name].weights
        shape = input_tensor.shape

        # create new layer
        transpose_layer = self._network.add_shuffle(input=input_tensor)

        # If a transpose is required, add transpose property to the shuffle layer.
        if TRTNetworkBuilder._is_transpose_required(shape, perm):
            transpose_layer.first_transpose = perm

        # Else, the transpose can be simplified to a reshape.
        else:
            new_shape = []
            for i in range(len(shape)):
                new_shape.append(shape[perm[i]])
            transpose_layer.reshape_dims = new_shape

        self._remember_op_and_output(transpose_layer, name)
        return transpose_layer

    @staticmethod
    def _is_transpose_required(shape, perm):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L761

        :param shape:
        :param perm:
        :return:
        """

        ndim = len(shape)
        prev_significant_dim = 0
        for dst_i in range(ndim):

            src_i = perm[dst_i]
            dim_i = shape[src_i]
            if dim_i != 1:
                # For transposes on dynamically shaped tensors, we must return true.
                if dim_i == -1:
                    return True
                elif src_i < prev_significant_dim:
                    return True
                prev_significant_dim = src_i
        return False

    def _broadcastTensor(self, input_tensor, num_dims, name=None):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L282

        :param input_tensor:
        :param num_dims:
        :param name:
        :return:
        """

        if not name:
            name = "broadcast_"+input_tensor.name

        # check if the shape is already bigger
        if len(input_tensor.shape) >= num_dims:
            return input_tensor

        output_shape = input_tensor.shape
        while len(output_shape) < num_dims:
            output_shape = np.insert(output_shape, 0, 1)

        return self._add_reshape_layer(input_tensor, output_shape, name).get_output(0)

    def add_concatv2(self, name, input_tensor_names, axis_tensor_name):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L307

        :param name:
        :param input_tensor_names:
        :param axis_tensor_name:
        :return:
        """

        input_tensors = []
        for input_tensor_name in input_tensor_names:
            input_tensors.append(self.get_layer_output(input_tensor_name))
        axis = self._ops[axis_tensor_name].weights[0]

        # create layer
        concat_layer = self._network.add_concatenation(input_tensors)
        concat_layer.axis = axis
        concat_layer.name = name

        self._remember_op_and_output(concat_layer, name)
        return concat_layer

    def add_pack(self, name, input_tensor_names, axis):
        """
        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L307
        https://www.tensorflow.org/api_docs/python/tf/stack

        :param name:
        :param input_tensor_names:
        :param axis:
        :return:
        """

        input_tensors = []
        for input_tensor_name in input_tensor_names:
            input_tensor = self.get_layer_output(input_tensor_name)
            new_shape = np.insert(input_tensor.shape, axis, 1)
            reshape_layer = self._add_reshape_layer(input_tensor, new_shape, name + "_expand_dim_" + input_tensor_name)
            input_tensors.append(reshape_layer.get_output(0))

        # create layer
        concat_layer = self._network.add_concatenation(input_tensors)
        concat_layer.axis = axis
        concat_layer.name = name

        self._remember_op_and_output(concat_layer, name)
        return concat_layer

    def add_matmul(self, name, input_tensor_a_name, input_tensor_b_name, transpose_a, transpose_b):
        """

        Similar to
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1253
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L745

        :param name:
        :param input_tensor_a_name:
        :param input_tensor_b_name:
        :param transpose_a:
        :param transpose_b:
        :return:
        """

        input_tensor1 = self.get_layer_output(input_tensor_a_name)
        input_tensor2 = self.get_layer_output(input_tensor_b_name)

        op0 = trt.MatrixOperation.NONE
        if len(input_tensor1.shape) == 1:
            op0 = trt.MatrixOperation.VECTOR
        elif transpose_a:
            op0 = trt.MatrixOperation.TRANSPOSE

        op1 = trt.MatrixOperation.NONE
        if len(input_tensor2.shape) == 1:
            op1 = trt.MatrixOperation.VECTOR
        elif transpose_b:
            op1 = trt.MatrixOperation.TRANSPOSE

        # create layer
        matmul_layer = self._network.add_matrix_multiply(input0=input_tensor1, op0=op0, input1=input_tensor2, op1=op1)
        self._remember_op_and_output(matmul_layer, name)
        return matmul_layer

    def add_bias_add(self, name, input_tensor_name, bias_tensor_name):
        """
        Add a bias addition layer.

        :param name:
        :param input_tensor_name:
        :param bias_tensor_name:
        :return:
        """

        conv_tensor = self.get_layer_output(input_tensor_name)

        conv_op = self._ops[input_tensor_name]
        if isinstance(conv_op, trt.IConvolutionLayer):

            # change previous conv2d bias values
            conv_op.bias = self.get_layer_weights(bias_tensor_name)

            # add an identity layer to have a separate node of this bias node
            bias_add_layer = self._network.add_identity(conv_tensor)

        else:

            # naive bias add
            bias = self.get_layer_output(bias_tensor_name)
            bias = self._broadcastTensor(bias, num_dims=len(conv_tensor.shape), name=name+"_broadcast_bias")
            bias_add_layer = self._network.add_elementwise(conv_tensor, bias, op=trt.ElementWiseOperation.SUM)

        self._remember_op_and_output(bias_add_layer, name)
        return bias_add_layer

    def add_softmax(self, name, input_tensor_name):
        """
        Similar to:
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/builtin_op_importers.cpp#L1748

        :param name:
        :param input_tensor_name:
        :return:
        """

        input_tensor = self.get_layer_output(input_tensor_name)

        assert len(input_tensor.shape) == 2, "Softmax("+name+") is only supported for rank 2 input tensors"

        softmax_layer = self._network.add_softmax(input_tensor)
        softmax_layer.axes = (1 << 1)
        self._remember_op_and_output(softmax_layer, name)
        return softmax_layer

    @staticmethod
    def _check_axis(axis, nb_dims):
        """
        Converts a negative axis to a positive one.

        Similar to:
        https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/onnx2trt_utils.cpp#L392

        :param axis:
        :param nb_dims:
        :return:
        """

        # Support negative indexing
        if axis < 0:
            axis += nb_dims

        assert 0 <= axis < nb_dims, "Axis "+str(axis)+" is smaller than -"+str(nb_dims)+". Impossible."
        return axis

    @staticmethod
    def _to_dtype(numpy_dtype):
        """
        Convert the numpy dtype to the corresponding trt version
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto

        :param numpy_dtype:
        :return:
        """
        if numpy_dtype == np.float32:
            return trt.tensorrt.DataType.FLOAT
        elif numpy_dtype == np.int32:
            return trt.tensorrt.DataType.INT32
        elif numpy_dtype == np.int8:
            return trt.tensorrt.DataType.INT8
        elif numpy_dtype == np.float16:
            return trt.tensorrt.DataType.HALF

        return

    def _remember_op_output(self, tensor, name):
        """
        Add the op output to the internal ops_output map.

        :param tensor: tensorrt.ITensor
        :param name: string
        :return: tensor
        """

        assert isinstance(tensor, trt.ITensor), "tensor("+str(type(tensor))+") is not a TensorRT tensor"

        # check if this is an output tensor
        if name in self._registered_output_node_names:
            self._network.mark_output(tensor)

        print("add",name,tensor.shape)

        tensor.name = name
        self._ops_output[name] = tensor
        return tensor

    def _remember_op_and_output(self, op, name):
        """
        Add the op and first output of the op to the internal ops and ops_output map.

        :param op: tensorrt.ILayer
        :param name: string
        :return: first output of the op: tensorrt.ITensor
        """

        assert isinstance(op, trt.ILayer), "op("+str(type(op))+") is not a TensorRT layer"

        op.name = name
        self._ops[name] = op

        return self._remember_op_output(op.get_output(0), name)
