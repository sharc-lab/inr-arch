#include "top.h"

void test_activations() {
#pragma HLS inline off

    F_TYPE activation_in_fixed;

    F_TYPE activation_out_elu_kernel_fixed = activation_elu<F_TYPE>(activation_in_fixed);
    F_TYPE activation_out_hardtanh_kernel_fixed = activation_hardtanh<F_TYPE>(activation_in_fixed);
    F_TYPE activation_out_leakyrelu_kernel_fixed = activation_leakyrelu<F_TYPE>(activation_in_fixed);
    F_TYPE activation_out_relu_kernel_fixed = activation_relu<F_TYPE>(activation_in_fixed);
    F_TYPE activation_out_gelu_kernel_fixed = activation_gelu<F_TYPE>(activation_in_fixed);
    F_TYPE activation_out_sigmoid_kernel_fixed = activation_sigmoid<F_TYPE>(activation_in_fixed);
    F_TYPE activation_out_silu_kernel_fixed = activation_silu<F_TYPE>(activation_in_fixed);
    F_TYPE activation_out_tanh_kernel_fixed = activation_tanh<F_TYPE>(activation_in_fixed);
    F_TYPE activation_out_softsign_kernel_fixed = activation_softsign<F_TYPE>(activation_in_fixed);
    F_TYPE activation_out_sin_kernel_fixed = activation_sin<F_TYPE>(activation_in_fixed);
    F_TYPE activation_out_cos_kernel_fixed = activation_cos<F_TYPE>(activation_in_fixed);
}

void test_siren() {
#pragma HLS inline off

    // architecture parameters
    const int dim_in = 2;
    const int dim_hidden = 256;
    const int dim_out = 3;
    const int num_hidden_layers = 4;

    // model parameters
    F_TYPE input_layer_weight_fixed[dim_hidden][dim_in];
    F_TYPE input_layer_bias_fixed[dim_hidden];
    F_TYPE input_layer_w0_fixed[1];
    F_TYPE hidden_layers_weight_fixed[num_hidden_layers][dim_hidden][dim_hidden];
    F_TYPE hidden_layers_bias_fixed[num_hidden_layers][dim_hidden];
    F_TYPE hidden_layers_w0_fixed[num_hidden_layers];
    F_TYPE output_layer_weight_fixed[dim_out][dim_hidden];
    F_TYPE output_layer_bias_fixed[dim_out];

    // model input
    F_TYPE x_in_fixed[dim_in];

    // kernel
    F_TYPE x_out_kernel_fixed[dim_out];
    SirenNet<F_TYPE, dim_in, dim_out, dim_hidden, num_hidden_layers, activation_sigmoid, 1, 4, 1> siren_net;
    siren_net.load_params(
        input_layer_weight_fixed, input_layer_bias_fixed, input_layer_w0_fixed[0],
        hidden_layers_weight_fixed, hidden_layers_bias_fixed, hidden_layers_w0_fixed,
        output_layer_weight_fixed, output_layer_bias_fixed);

    siren_net.forward(x_in_fixed, x_out_kernel_fixed);
}

void test_elementwise() {
#pragma HLS inline off

    const int SIZE = 32;

    F_TYPE a_in_1d_fixed[SIZE];
    F_TYPE b_in_1d_fixed[SIZE];

    F_TYPE a_in_2d_fixed[SIZE][SIZE];
    F_TYPE b_in_2d_fixed[SIZE][SIZE];

    F_TYPE a_in_3d_fixed[SIZE][SIZE][SIZE];
    F_TYPE b_in_3d_fixed[SIZE][SIZE][SIZE];

    F_TYPE c_out_1d_add_hw[SIZE];
    F_TYPE c_out_1d_mult_hw[SIZE];

    F_TYPE c_out_2d_add_hw[SIZE][SIZE];
    F_TYPE c_out_2d_mult_hw[SIZE][SIZE];

    F_TYPE c_out_3d_add_hw[SIZE][SIZE][SIZE];
    F_TYPE c_out_3d_mult_hw[SIZE][SIZE][SIZE];

    const int block_size = 4;

    // 1D
    ElementWiseAdd1D<F_TYPE, SIZE, block_size> add_1d;
    ElementWiseMul1D<F_TYPE, SIZE, block_size> mult_1d;
    add_1d.forward(a_in_1d_fixed, b_in_1d_fixed, c_out_1d_add_hw);
    mult_1d.forward(a_in_1d_fixed, b_in_1d_fixed, c_out_1d_mult_hw);

    // 2D
    ElementWiseAdd2D<F_TYPE, SIZE, SIZE, block_size> add_2d;
    ElementWiseMul2D<F_TYPE, SIZE, SIZE, block_size> mult_2d;
    add_2d.forward(a_in_2d_fixed, b_in_2d_fixed, c_out_2d_add_hw);
    mult_2d.forward(a_in_2d_fixed, b_in_2d_fixed, c_out_2d_mult_hw);

    // 3D
    ElementWiseAdd3D<F_TYPE, SIZE, SIZE, SIZE, block_size> add_3d;
    ElementWiseMul3D<F_TYPE, SIZE, SIZE, SIZE, block_size> mult_3d;
    add_3d.forward(a_in_3d_fixed, b_in_3d_fixed, c_out_3d_add_hw);
    mult_3d.forward(a_in_3d_fixed, b_in_3d_fixed, c_out_3d_mult_hw);
}

const int SIZE = 32;

void test_broadcast(
    F_TYPE in_1d_fixed[SIZE],
    F_TYPE out_1d_fixed_0[SIZE],
    F_TYPE out_1d_fixed_1[SIZE],
    F_TYPE out_1d_fixed_2[SIZE]
) {
#pragma HLS inline off

    Broadcast_1d<F_TYPE, SIZE> broadcast_1d;
    broadcast_1d.broadcast_1d_3(in_1d_fixed, out_1d_fixed_0, out_1d_fixed_1, out_1d_fixed_2);
}

void top() {
    // test_activations();
    // test_siren();
    // test_elementwise();
    // test_broadcast();
}