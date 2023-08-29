#include "./main.h"

bool test_linear_layer() {

    const int in_size = 32;
    const int out_size = 16;

    // load input data
    float linear_in_float[in_size];
    F_TYPE linear_in_fixed[in_size];

    load_data_1d<in_size>("./test_data/linear_layer_in.bin", linear_in_float);
    cast_1d<in_size, float, F_TYPE>(linear_in_float, linear_in_fixed);

    // load parameters
    float linear_weight_float[out_size][in_size];
    float linear_bias_float[out_size];

    F_TYPE linear_weight_fixed[out_size][in_size];
    F_TYPE linear_bias_fixed[out_size];

    load_data_2d<out_size, in_size>("./test_data/linear_layer_weight.bin", linear_weight_float);
    load_data_1d<out_size>("./test_data/linear_layer_bias.bin", linear_bias_float);

    cast_2d<out_size, in_size, float, F_TYPE>(linear_weight_float, linear_weight_fixed);
    cast_1d<out_size, float, F_TYPE>(linear_bias_float, linear_bias_fixed);

    // load output data
    float linear_out_gold_float[out_size];
    load_data_1d<out_size>("./test_data/linear_layer_out.bin", linear_out_gold_float);

    // kernel data
    float linear_out_kernel_float[out_size];
    F_TYPE linear_out_kernel_fixed[out_size];

    // kernel execution
    Linear<F_TYPE, in_size, out_size> linear_kernel;
    linear_kernel.load_params(linear_weight_fixed, linear_bias_fixed);
    linear_kernel.forward(linear_in_fixed, linear_out_kernel_fixed);

    cast_1d<out_size, F_TYPE, float>(linear_out_kernel_fixed, linear_out_kernel_float);

    // compare output
    bool pass = true;
    float test_delta = 1e-3;
    for (int i = 0; i < out_size; i++) {
        // fabs
        float diff = fabs(linear_out_kernel_float[i] - linear_out_gold_float[i]);
        if (diff > test_delta) {
            pass = false;
        }
    }

    return pass;
}

bool test_activations() {

    const int in_size = 32;

    // load input data
    float activation_in_float[in_size];
    F_TYPE activation_in_fixed[in_size];

    load_data_1d<in_size>("./test_data/activation_in.bin", activation_in_float);
    cast_1d<in_size, float, F_TYPE>(activation_in_float, activation_in_fixed);

    // load output data
    float activation_out_elu_gold_float[in_size];
    float activation_out_hardtanh_gold_float[in_size];
    float activation_out_leakyrelu_gold_float[in_size];
    float activation_out_relu_gold_float[in_size];
    float activation_out_gelu_gold_float[in_size];
    float activation_out_sigmoid_gold_float[in_size];
    float activation_out_silu_gold_float[in_size];
    float activation_out_tanh_gold_float[in_size];
    float activation_out_softsign_gold_float[in_size];
    float activation_out_sin_gold_float[in_size];
    float activation_out_cos_gold_float[in_size];

    load_data_1d<in_size>("./test_data/activation_out_elu.bin", activation_out_elu_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_hardtanh.bin", activation_out_hardtanh_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_leakyrelu.bin", activation_out_leakyrelu_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_relu.bin", activation_out_relu_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_gelu.bin", activation_out_gelu_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_sigmoid.bin", activation_out_sigmoid_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_silu.bin", activation_out_silu_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_tanh.bin", activation_out_tanh_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_softsign.bin", activation_out_softsign_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_sin.bin", activation_out_sin_gold_float);
    load_data_1d<in_size>("./test_data/activation_out_cos.bin", activation_out_cos_gold_float);

    // kernel data
    float activation_out_elu_kernel_float[in_size];
    float activation_out_hardtanh_kernel_float[in_size];
    float activation_out_leakyrelu_kernel_float[in_size];
    float activation_out_relu_kernel_float[in_size];
    float activation_out_gelu_kernel_float[in_size];
    float activation_out_sigmoid_kernel_float[in_size];
    float activation_out_silu_kernel_float[in_size];
    float activation_out_tanh_kernel_float[in_size];
    float activation_out_softsign_kernel_float[in_size];
    float activation_out_sin_kernel_float[in_size];
    float activation_out_cos_kernel_float[in_size];
    F_TYPE activation_out_elu_kernel_fixed[in_size];
    F_TYPE activation_out_hardtanh_kernel_fixed[in_size];
    F_TYPE activation_out_leakyrelu_kernel_fixed[in_size];
    F_TYPE activation_out_relu_kernel_fixed[in_size];
    F_TYPE activation_out_gelu_kernel_fixed[in_size];
    F_TYPE activation_out_sigmoid_kernel_fixed[in_size];
    F_TYPE activation_out_silu_kernel_fixed[in_size];
    F_TYPE activation_out_tanh_kernel_fixed[in_size];
    F_TYPE activation_out_softsign_kernel_fixed[in_size];
    F_TYPE activation_out_sin_kernel_fixed[in_size];
    F_TYPE activation_out_cos_kernel_fixed[in_size];

    // kernel execution
    for (int i = 0; i < in_size; i++) {
        activation_out_elu_kernel_fixed[i] = activation_elu<F_TYPE>(activation_in_fixed[i]);
        activation_out_hardtanh_kernel_fixed[i] = activation_hardtanh<F_TYPE>(activation_in_fixed[i]);
        activation_out_leakyrelu_kernel_fixed[i] = activation_leakyrelu<F_TYPE>(activation_in_fixed[i]);
        activation_out_relu_kernel_fixed[i] = activation_relu<F_TYPE>(activation_in_fixed[i]);
        activation_out_gelu_kernel_fixed[i] = activation_gelu<F_TYPE>(activation_in_fixed[i]);
        activation_out_sigmoid_kernel_fixed[i] = activation_sigmoid<F_TYPE>(activation_in_fixed[i]);
        activation_out_silu_kernel_fixed[i] = activation_silu<F_TYPE>(activation_in_fixed[i]);
        activation_out_tanh_kernel_fixed[i] = activation_tanh<F_TYPE>(activation_in_fixed[i]);
        activation_out_softsign_kernel_fixed[i] = activation_softsign<F_TYPE>(activation_in_fixed[i]);
        activation_out_sin_kernel_fixed[i] = activation_sin<F_TYPE>(activation_in_fixed[i]);
        activation_out_cos_kernel_fixed[i] = activation_cos<F_TYPE>(activation_in_fixed[i]);
    }

    cast_1d<in_size, F_TYPE, float>(activation_out_elu_kernel_fixed, activation_out_elu_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_hardtanh_kernel_fixed, activation_out_hardtanh_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_leakyrelu_kernel_fixed, activation_out_leakyrelu_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_relu_kernel_fixed, activation_out_relu_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_gelu_kernel_fixed, activation_out_gelu_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_sigmoid_kernel_fixed, activation_out_sigmoid_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_silu_kernel_fixed, activation_out_silu_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_tanh_kernel_fixed, activation_out_tanh_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_softsign_kernel_fixed, activation_out_softsign_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_sin_kernel_fixed, activation_out_sin_kernel_float);
    cast_1d<in_size, F_TYPE, float>(activation_out_cos_kernel_fixed, activation_out_cos_kernel_float);

    // compare output
    bool pass = true;
    float test_delta = 1e-3;
    for (int i = 0; i < in_size; i++) {
        float diff_elu = fabs(activation_out_elu_gold_float[i] - activation_out_elu_kernel_float[i]);
        float diff_hardtanh = fabs(activation_out_hardtanh_gold_float[i] - activation_out_hardtanh_kernel_float[i]);
        float diff_leakyrelu = fabs(activation_out_leakyrelu_gold_float[i] - activation_out_leakyrelu_kernel_float[i]);
        float diff_relu = fabs(activation_out_relu_gold_float[i] - activation_out_relu_kernel_float[i]);
        float diff_gelu = fabs(activation_out_gelu_gold_float[i] - activation_out_gelu_kernel_float[i]);
        float diff_sigmoid = fabs(activation_out_sigmoid_gold_float[i] - activation_out_sigmoid_kernel_float[i]);
        float diff_silu = fabs(activation_out_silu_gold_float[i] - activation_out_silu_kernel_float[i]);
        float diff_tanh = fabs(activation_out_tanh_gold_float[i] - activation_out_tanh_kernel_float[i]);
        float diff_softsign = fabs(activation_out_softsign_gold_float[i] - activation_out_softsign_kernel_float[i]);
        float diff_sin = fabs(activation_out_sin_gold_float[i] - activation_out_sin_kernel_float[i]);
        float diff_cos = fabs(activation_out_cos_gold_float[i] - activation_out_cos_kernel_float[i]);

        if (diff_elu > test_delta) {
            pass = false;
        }
        if (diff_hardtanh > test_delta) {
            pass = false;
        }
        if (diff_leakyrelu > test_delta) {
            pass = false;
        }
        if (diff_relu > test_delta) {
            pass = false;
        }
        if (diff_gelu > test_delta) {
            pass = false;
        }
        if (diff_sigmoid > test_delta) {
            pass = false;
        }
        if (diff_silu > test_delta) {
            pass = false;
        }
        if (diff_tanh > test_delta) {
            pass = false;
        }
        if (diff_softsign > test_delta) {
            pass = false;
        }
        if (diff_sin > test_delta) {
            pass = false;
        }
        if (diff_cos > test_delta) {
            pass = false;
        }
    }

    return pass;
}

bool test_siren_net() {

    // architecture parameters

    const int dim_in = 2;
    const int dim_hidden = 256;
    const int dim_out = 3;
    const int num_hidden_layers = 4;

    // model parameters

    float input_layer_weight[dim_hidden][dim_in];
    float input_layer_bias[dim_hidden];
    float input_layer_w0[1];
    float hidden_layers_weight[num_hidden_layers][dim_hidden][dim_hidden];
    float hidden_layers_bias[num_hidden_layers][dim_hidden];
    float hidden_layers_w0[num_hidden_layers];
    float output_layer_weight[dim_out][dim_hidden];
    float output_layer_bias[dim_out];

    load_data_2d<dim_hidden, dim_in>("test_data/siren_input_layer_weight.bin", input_layer_weight);
    load_data_1d<dim_hidden>("test_data/siren_input_layer_bias.bin", input_layer_bias);
    load_data_1d<1>("test_data/siren_input_layer_w0.bin", input_layer_w0);
    load_data_3d<num_hidden_layers, dim_hidden, dim_hidden>("test_data/siren_hidden_layers_weight.bin", hidden_layers_weight);
    load_data_2d<num_hidden_layers, dim_hidden>("test_data/siren_hidden_layers_bias.bin", hidden_layers_bias);
    load_data_1d<num_hidden_layers>("test_data/siren_hidden_layers_w0.bin", hidden_layers_w0);
    load_data_2d<dim_out, dim_hidden>("test_data/siren_output_layer_weight.bin", output_layer_weight);
    load_data_1d<dim_out>("test_data/siren_output_layer_bias.bin", output_layer_bias);

    F_TYPE input_layer_weight_fixed[dim_hidden][dim_in];
    F_TYPE input_layer_bias_fixed[dim_hidden];
    F_TYPE input_layer_w0_fixed[1];
    F_TYPE hidden_layers_weight_fixed[num_hidden_layers][dim_hidden][dim_hidden];
    F_TYPE hidden_layers_bias_fixed[num_hidden_layers][dim_hidden];
    F_TYPE hidden_layers_w0_fixed[num_hidden_layers];
    F_TYPE output_layer_weight_fixed[dim_out][dim_hidden];
    F_TYPE output_layer_bias_fixed[dim_out];

    cast_2d<dim_hidden, dim_in, float, F_TYPE>(input_layer_weight, input_layer_weight_fixed);
    cast_1d<dim_hidden, float, F_TYPE>(input_layer_bias, input_layer_bias_fixed);
    cast_1d<1, float, F_TYPE>(input_layer_w0, input_layer_w0_fixed);
    cast_3d<num_hidden_layers, dim_hidden, dim_hidden, float, F_TYPE>(hidden_layers_weight, hidden_layers_weight_fixed);
    cast_2d<num_hidden_layers, dim_hidden, float, F_TYPE>(hidden_layers_bias, hidden_layers_bias_fixed);
    cast_1d<num_hidden_layers, float, F_TYPE>(hidden_layers_w0, hidden_layers_w0_fixed);
    cast_2d<dim_out, dim_hidden, float, F_TYPE>(output_layer_weight, output_layer_weight_fixed);
    cast_1d<dim_out, float, F_TYPE>(output_layer_bias, output_layer_bias_fixed);

    // model input
    float x_in[dim_in];
    F_TYPE x_in_fixed[dim_in];
    load_data_1d<dim_in>("test_data/siren_x_in.bin", x_in);
    cast_1d<dim_in, float, F_TYPE>(x_in, x_in_fixed);

    // golden output
    float x_out_golden[dim_out];
    load_data_1d<dim_out>("test_data/siren_x_out.bin", x_out_golden);

    // kernel
    float x_out_kernel_float[dim_out];
    F_TYPE x_out_kernel_fixed[dim_out];
    SirenNet<F_TYPE, dim_in, dim_out, dim_hidden, num_hidden_layers, activation_sigmoid, 1, 1, 1> siren_net;
    siren_net.load_params(
        input_layer_weight_fixed, input_layer_bias_fixed, input_layer_w0_fixed[0],
        hidden_layers_weight_fixed, hidden_layers_bias_fixed, hidden_layers_w0_fixed,
        output_layer_weight_fixed, output_layer_bias_fixed);

    siren_net.forward(x_in_fixed, x_out_kernel_fixed);

    cast_1d<dim_out, F_TYPE, float>(x_out_kernel_fixed, x_out_kernel_float);

    // compare
    bool pass = true;
    float test_delta = 1e-3;
    for (int i = 0; i < dim_out; i++) {
        float diff = fabs(x_out_kernel_float[i] - x_out_golden[i]);
        if (diff > test_delta) {
            pass = false;
        }
    }

    return pass;
}

bool test_elementwise(){

    const int SIZE=32;

    // serialize_numpy(a_in_1d_np, test_data_dir / "a_in_1d.bin")
    // serialize_numpy(b_in_1d_np, test_data_dir / "b_in_1d.bin")
    // serialize_numpy(c_out_1d_add_np, test_data_dir / "c_out_1d_add.bin")
    // serialize_numpy(c_out_1d_mult_np, test_data_dir / "c_out_1d_mult.bin")

    // serialize_numpy(a_in_2d_np, test_data_dir / "a_in_2d.bin")
    // serialize_numpy(b_in_2d_np, test_data_dir / "b_in_2d.bin")
    // serialize_numpy(c_out_2d_add_np, test_data_dir / "c_out_2d_add.bin")
    // serialize_numpy(c_out_2d_mult_np, test_data_dir / "c_out_2d_mult.bin")

    // serialize_numpy(a_in_3d_np, test_data_dir / "a_in_3d.bin")
    // serialize_numpy(b_in_3d_np, test_data_dir / "b_in_3d.bin")
    // serialize_numpy(c_out_3d_add_np, test_data_dir / "c_out_3d_add.bin")
    // serialize_numpy(c_out_3d_mult_np, test_data_dir / "c_out_3d_mult.bin")

    float a_in_1d[SIZE];
    float b_in_1d[SIZE];
    float c_out_1d_add[SIZE];
    float c_out_1d_mult[SIZE];

    float a_in_2d[SIZE][SIZE];
    float b_in_2d[SIZE][SIZE];
    float c_out_2d_add[SIZE][SIZE];
    float c_out_2d_mult[SIZE][SIZE];

    float a_in_3d[SIZE][SIZE][SIZE];
    float b_in_3d[SIZE][SIZE][SIZE];
    float c_out_3d_add[SIZE][SIZE][SIZE];
    float c_out_3d_mult[SIZE][SIZE][SIZE];

    load_data_1d<SIZE>("test_data/a_in_1d.bin", a_in_1d);
    load_data_1d<SIZE>("test_data/b_in_1d.bin", b_in_1d);
    load_data_1d<SIZE>("test_data/c_out_1d_add.bin", c_out_1d_add);
    load_data_1d<SIZE>("test_data/c_out_1d_mult.bin", c_out_1d_mult);

    load_data_2d<SIZE, SIZE>("test_data/a_in_2d.bin", a_in_2d);
    load_data_2d<SIZE, SIZE>("test_data/b_in_2d.bin", b_in_2d);
    load_data_2d<SIZE, SIZE>("test_data/c_out_2d_add.bin", c_out_2d_add);
    load_data_2d<SIZE, SIZE>("test_data/c_out_2d_mult.bin", c_out_2d_mult);

    load_data_3d<SIZE, SIZE, SIZE>("test_data/a_in_3d.bin", a_in_3d);
    load_data_3d<SIZE, SIZE, SIZE>("test_data/b_in_3d.bin", b_in_3d);
    load_data_3d<SIZE, SIZE, SIZE>("test_data/c_out_3d_add.bin", c_out_3d_add);
    load_data_3d<SIZE, SIZE, SIZE>("test_data/c_out_3d_mult.bin", c_out_3d_mult);

    F_TYPE a_in_1d_fixed[SIZE];
    F_TYPE b_in_1d_fixed[SIZE];

    F_TYPE a_in_2d_fixed[SIZE][SIZE];
    F_TYPE b_in_2d_fixed[SIZE][SIZE];

    F_TYPE a_in_3d_fixed[SIZE][SIZE][SIZE];
    F_TYPE b_in_3d_fixed[SIZE][SIZE][SIZE];

    cast_1d<SIZE, float, F_TYPE>(a_in_1d, a_in_1d_fixed);
    cast_1d<SIZE, float, F_TYPE>(b_in_1d, b_in_1d_fixed);

    cast_2d<SIZE, SIZE, float, F_TYPE>(a_in_2d, a_in_2d_fixed);
    cast_2d<SIZE, SIZE, float, F_TYPE>(b_in_2d, b_in_2d_fixed);

    cast_3d<SIZE, SIZE, SIZE, float, F_TYPE>(a_in_3d, a_in_3d_fixed);
    cast_3d<SIZE, SIZE, SIZE, float, F_TYPE>(b_in_3d, b_in_3d_fixed);

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

    float c_out_1d_add_hw_float[SIZE];
    float c_out_1d_mult_hw_float[SIZE];

    float c_out_2d_add_hw_float[SIZE][SIZE];
    float c_out_2d_mult_hw_float[SIZE][SIZE];

    float c_out_3d_add_hw_float[SIZE][SIZE][SIZE];
    float c_out_3d_mult_hw_float[SIZE][SIZE][SIZE];

    cast_1d<SIZE, F_TYPE, float>(c_out_1d_add_hw, c_out_1d_add_hw_float);
    cast_1d<SIZE, F_TYPE, float>(c_out_1d_mult_hw, c_out_1d_mult_hw_float);

    cast_2d<SIZE, SIZE, F_TYPE, float>(c_out_2d_add_hw, c_out_2d_add_hw_float);
    cast_2d<SIZE, SIZE, F_TYPE, float>(c_out_2d_mult_hw, c_out_2d_mult_hw_float);

    cast_3d<SIZE, SIZE, SIZE, F_TYPE, float>(c_out_3d_add_hw, c_out_3d_add_hw_float);
    cast_3d<SIZE, SIZE, SIZE, F_TYPE, float>(c_out_3d_mult_hw, c_out_3d_mult_hw_float);

    // compare results
    bool pass = true;
    float eps = 1e-3;
    // 1d
    for (int i = 0; i < SIZE; i++) {
        if (fabs(c_out_1d_add_hw_float[i] - c_out_1d_add[i]) > eps) {
            pass = false;
            break;
        }
        if (fabs(c_out_1d_mult_hw_float[i] - c_out_1d_mult[i]) > eps) {
            pass = false;
            break;
        }
    }
    // 2d
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (fabs(c_out_2d_add_hw_float[i][j] - c_out_2d_add[i][j]) > eps) {
                pass = false;
                break;
            }
            if (fabs(c_out_2d_mult_hw_float[i][j] - c_out_2d_mult[i][j]) > eps) {
                pass = false;
                break;
            }
        }
    }
    // 3d
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                if (fabs(c_out_3d_add_hw_float[i][j][k] - c_out_3d_add[i][j][k]) > eps) {
                    pass = false;
                    break;
                }
                if (fabs(c_out_3d_mult_hw_float[i][j][k] - c_out_3d_mult[i][j][k]) > eps) {
                    pass = false;
                    break;
                }
            }
        }
    }

    return pass;
}

bool test_shape_edits(){

    const int SIZE = 32;

    // serialize_numpy(in_1d_np, test_data_dir / "in_1d.bin")
    // serialize_numpy(in_2d_np, test_data_dir / "in_2d.bin")
    // serialize_numpy(in_3d_np, test_data_dir / "in_3d.bin")

    // serialize_numpy(in_2d_transpose_np, test_data_dir / "in_2d_transpose.bin")

    // serialize_numpy(in_1d_unsqueeze_neg_2_np, test_data_dir / "in_1d_unsqueeze_neg_2.bin")
    // serialize_numpy(in_1d_unsqueeze_neg_1_np, test_data_dir / "in_1d_unsqueeze_neg_1.bin")
    // serialize_numpy(in_1d_unsqueeze_0_np, test_data_dir / "in_1d_unsqueeze_0.bin")
    // serialize_numpy(in_1d_unsqueeze_1_np, test_data_dir / "in_1d_unsqueeze_1.bin")

    // serialize_numpy(in_2d_select_dim_1_select_0_np, test_data_dir / "in_2d_select_dim_1_select_0.bin")
    // serialize_numpy(in_2d_select_dim_1_select_1_np, test_data_dir / "in_2d_select_dim_1_select_1.bin")
    // serialize_numpy(in_2d_select_dim_1_select_2_np, test_data_dir / "in_2d_select_dim_1_select_2.bin")

    // load data
    float in_1d[SIZE];
    float in_2d[SIZE][SIZE];
    float in_3d[SIZE][SIZE][SIZE];

    float in_2d_transpose[SIZE][SIZE];

    float in_1d_unsqueeze_neg_2[1][SIZE];
    float in_1d_unsqueeze_neg_1[SIZE][1];
    float in_1d_unsqueeze_0[1][SIZE];
    float in_1d_unsqueeze_1[SIZE][1];

    float in_2d_select_dim_1_select_0[SIZE];
    float in_2d_select_dim_1_select_1[SIZE];
    float in_2d_select_dim_1_select_2[SIZE];

    load_data_1d<SIZE>("test_data/in_1d.bin", in_1d);
    load_data_2d<SIZE, SIZE>("test_data/in_2d.bin", in_2d);
    load_data_3d<SIZE, SIZE, SIZE>("test_data/in_3d.bin", in_3d);

    load_data_2d<SIZE, SIZE>("test_data/in_2d_transpose.bin", in_2d_transpose);

    load_data_2d<1, SIZE>("test_data/in_1d_unsqueeze_neg_2.bin", in_1d_unsqueeze_neg_2);
    load_data_2d<SIZE, 1>("test_data/in_1d_unsqueeze_neg_1.bin", in_1d_unsqueeze_neg_1);
    load_data_2d<1, SIZE>("test_data/in_1d_unsqueeze_0.bin", in_1d_unsqueeze_0);
    load_data_2d<SIZE, 1>("test_data/in_1d_unsqueeze_1.bin", in_1d_unsqueeze_1);

    load_data_1d<SIZE>("test_data/in_2d_select_dim_1_select_0.bin", in_2d_select_dim_1_select_0);
    load_data_1d<SIZE>("test_data/in_2d_select_dim_1_select_1.bin", in_2d_select_dim_1_select_1);
    load_data_1d<SIZE>("test_data/in_2d_select_dim_1_select_2.bin", in_2d_select_dim_1_select_2);


    F_TYPE in_1d_fixed[SIZE];
    F_TYPE in_2d_fixed[SIZE][SIZE];
    F_TYPE in_3d_fixed[SIZE][SIZE][SIZE];

    F_TYPE in_2d_transpose_kernel_fixed[SIZE][SIZE];

    F_TYPE in_1d_unsqueeze_neg_2_kernel_fixed[1][SIZE];
    F_TYPE in_1d_unsqueeze_neg_1_kernel_fixed[SIZE][1];
    F_TYPE in_1d_unsqueeze_0_kernel_fixed[1][SIZE];
    F_TYPE in_1d_unsqueeze_1_kernel_fixed[SIZE][1];

    F_TYPE in_2d_select_dim_1_select_0_kernel_fixed[SIZE];
    F_TYPE in_2d_select_dim_1_select_1_kernel_fixed[SIZE];
    F_TYPE in_2d_select_dim_1_select_2_kernel_fixed[SIZE];

    cast_1d<SIZE>(in_1d, in_1d_fixed);
    cast_2d<SIZE, SIZE>(in_2d, in_2d_fixed);
    cast_3d<SIZE, SIZE, SIZE>(in_3d, in_3d_fixed);

    Transpose_2<F_TYPE, SIZE, SIZE> transpose;
    transpose.forward(in_2d_fixed, in_2d_transpose_kernel_fixed);

    Unsqueeze_1_2<F_TYPE, SIZE, -2> unsqueeze_neg_2;
    unsqueeze_neg_2.forward_dim_neg_2(in_1d_fixed, in_1d_unsqueeze_neg_2_kernel_fixed);

    Unsqueeze_1_2<F_TYPE, SIZE, -1> unsqueeze_neg_1;
    unsqueeze_neg_1.foward_dim_neg_1(in_1d_fixed, in_1d_unsqueeze_neg_1_kernel_fixed);

    Unsqueeze_1_2<F_TYPE, SIZE, 0> unsqueeze_0;
    unsqueeze_0.forward_dim_0(in_1d_fixed, in_1d_unsqueeze_0_kernel_fixed);

    Unsqueeze_1_2<F_TYPE, SIZE, 1> unsqueeze_1;
    unsqueeze_1.forward_dim_1(in_1d_fixed, in_1d_unsqueeze_1_kernel_fixed);

    Select_2_1<F_TYPE, SIZE, SIZE, 1, 0> select_dim_1_select_0;
    select_dim_1_select_0.forward_dim_1(in_2d_fixed, in_2d_select_dim_1_select_0_kernel_fixed);

    Select_2_1<F_TYPE, SIZE, SIZE, 1, 1> select_dim_1_select_1;
    select_dim_1_select_1.forward_dim_1(in_2d_fixed, in_2d_select_dim_1_select_1_kernel_fixed);

    Select_2_1<F_TYPE, SIZE, SIZE, 1, 2> select_dim_1_select_2;
    select_dim_1_select_2.forward_dim_1(in_2d_fixed, in_2d_select_dim_1_select_2_kernel_fixed);

    float in_2d_transpose_kernel_float[SIZE][SIZE];

    float in_1d_unsqueeze_neg_2_kernel_float[1][SIZE];
    float in_1d_unsqueeze_neg_1_kernel_float[SIZE][1];
    float in_1d_unsqueeze_0_kernel_float[1][SIZE];
    float in_1d_unsqueeze_1_kernel_float[SIZE][1];

    float in_2d_select_dim_1_select_0_kernel_float[SIZE];
    float in_2d_select_dim_1_select_1_kernel_float[SIZE];
    float in_2d_select_dim_1_select_2_kernel_float[SIZE];

    cast_2d<SIZE, SIZE>(in_2d_transpose_kernel_fixed, in_2d_transpose_kernel_float);

    cast_2d<1, SIZE>(in_1d_unsqueeze_neg_2_kernel_fixed, in_1d_unsqueeze_neg_2_kernel_float);
    cast_2d<SIZE, 1>(in_1d_unsqueeze_neg_1_kernel_fixed, in_1d_unsqueeze_neg_1_kernel_float);
    cast_2d<1, SIZE>(in_1d_unsqueeze_0_kernel_fixed, in_1d_unsqueeze_0_kernel_float);
    cast_2d<SIZE, 1>(in_1d_unsqueeze_1_kernel_fixed, in_1d_unsqueeze_1_kernel_float);

    cast_1d<SIZE>(in_2d_select_dim_1_select_0_kernel_fixed, in_2d_select_dim_1_select_0_kernel_float);
    cast_1d<SIZE>(in_2d_select_dim_1_select_1_kernel_fixed, in_2d_select_dim_1_select_1_kernel_float);
    cast_1d<SIZE>(in_2d_select_dim_1_select_2_kernel_fixed, in_2d_select_dim_1_select_2_kernel_float);

    bool pass = true;
    float eps = 1e-3;

    pass &= compare_data_2d<SIZE, SIZE>(in_2d_transpose, in_2d_transpose_kernel_float, eps);

    pass &= compare_data_2d<1, SIZE>(in_1d_unsqueeze_neg_2, in_1d_unsqueeze_neg_2_kernel_float, eps);
    pass &= compare_data_2d<SIZE, 1>(in_1d_unsqueeze_neg_1, in_1d_unsqueeze_neg_1_kernel_float, eps);
    pass &= compare_data_2d<1, SIZE>(in_1d_unsqueeze_0, in_1d_unsqueeze_0_kernel_float, eps);
    pass &= compare_data_2d<SIZE, 1>(in_1d_unsqueeze_1, in_1d_unsqueeze_1_kernel_float, eps);

    pass &= compare_data_1d<SIZE>(in_2d_select_dim_1_select_0, in_2d_select_dim_1_select_0_kernel_float, eps);
    pass &= compare_data_1d<SIZE>(in_2d_select_dim_1_select_1, in_2d_select_dim_1_select_1_kernel_float, eps);
    pass &= compare_data_1d<SIZE>(in_2d_select_dim_1_select_2, in_2d_select_dim_1_select_2_kernel_float, eps);

    return pass;

}


bool test_broadcast(){
    const int SIZE = 16;

    float in_1d[SIZE];
    float in_2d[SIZE][SIZE];
    float in_3d[SIZE][SIZE][SIZE];

    // assign random data
    for(int i = 0; i < SIZE; i++){
        in_1d[i] = (float)rand() / (float)RAND_MAX;
    }

    for(int i = 0; i < SIZE; i++){
        for(int j = 0; j < SIZE; j++){
            in_2d[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }

    for(int i = 0; i < SIZE; i++){
        for(int j = 0; j < SIZE; j++){
            for(int k = 0; k < SIZE; k++){
                in_3d[i][j][k] = (float)rand() / (float)RAND_MAX;
            }
        }
    }

    float out_1d_0[SIZE];
    float out_1d_1[SIZE];
    float out_1d_2[SIZE];

    float out_2d_0[SIZE][SIZE];
    float out_2d_1[SIZE][SIZE];
    float out_2d_2[SIZE][SIZE];

    float out_3d_0[SIZE][SIZE][SIZE];
    float out_3d_1[SIZE][SIZE][SIZE];
    float out_3d_2[SIZE][SIZE][SIZE];

    // data in out should be the same

    for(int i = 0; i < SIZE; i++){
        out_1d_0[i] = in_1d[i];
        out_1d_1[i] = in_1d[i];
        out_1d_2[i] = in_1d[i];
    }

    for(int i = 0; i < SIZE; i++){
        for(int j = 0; j < SIZE; j++){
            out_2d_0[i][j] = in_2d[i][j];
            out_2d_1[i][j] = in_2d[i][j];
            out_2d_2[i][j] = in_2d[i][j];
        }
    }

    for(int i = 0; i < SIZE; i++){
        for(int j = 0; j < SIZE; j++){
            for(int k = 0; k < SIZE; k++){
                out_3d_0[i][j][k] = in_3d[i][j][k];
                out_3d_1[i][j][k] = in_3d[i][j][k];
                out_3d_2[i][j][k] = in_3d[i][j][k];
            }
        }
    }

    // cast input to fixed point
    F_TYPE in_1d_fixed[SIZE];
    F_TYPE in_2d_fixed[SIZE][SIZE];
    F_TYPE in_3d_fixed[SIZE][SIZE][SIZE];

    cast_1d<SIZE>(in_1d, in_1d_fixed);
    cast_2d<SIZE, SIZE>(in_2d, in_2d_fixed);
    cast_3d<SIZE, SIZE, SIZE>(in_3d, in_3d_fixed);

    F_TYPE out_1d_0_kernel_fixed[SIZE];
    F_TYPE out_1d_1_kernel_fixed[SIZE];
    F_TYPE out_1d_2_kernel_fixed[SIZE];

    F_TYPE out_2d_0_kernel_fixed[SIZE][SIZE];
    F_TYPE out_2d_1_kernel_fixed[SIZE][SIZE];
    F_TYPE out_2d_2_kernel_fixed[SIZE][SIZE];

    F_TYPE out_3d_0_kernel_fixed[SIZE][SIZE][SIZE];
    F_TYPE out_3d_1_kernel_fixed[SIZE][SIZE][SIZE];
    F_TYPE out_3d_2_kernel_fixed[SIZE][SIZE][SIZE];

    // run kernel
    // broadcast_1d<F_TYPE, SIZE>(in_1d_fixed, out_1d_0_kernel_fixed, out_1d_1_kernel_fixed, out_1d_2_kernel_fixed);
    // broadcast_2d<F_TYPE, SIZE, SIZE, 3>(in_2d_fixed, out_2d_0_kernel_fixed, out_2d_1_kernel_fixed, out_2d_2_kernel_fixed);
    // broadcast_3d<F_TYPE, SIZE, SIZE, SIZE, 3>(in_3d_fixed, out_3d_0_kernel_fixed, out_3d_1_kernel_fixed, out_3d_2_kernel_fixed);

    Broadcast_1d<F_TYPE, SIZE> broadcast_1d;
    Broadcast_2d<F_TYPE, SIZE, SIZE> broadcast_2d;
    Broadcast_3d<F_TYPE, SIZE, SIZE, SIZE> broadcast_3d;

    broadcast_1d.broadcast_1d_3(in_1d_fixed, out_1d_0_kernel_fixed, out_1d_1_kernel_fixed, out_1d_2_kernel_fixed);
    broadcast_2d.broadcast_2d_3(in_2d_fixed, out_2d_0_kernel_fixed, out_2d_1_kernel_fixed, out_2d_2_kernel_fixed);
    broadcast_3d.broadcast_3d_3(in_3d_fixed, out_3d_0_kernel_fixed, out_3d_1_kernel_fixed, out_3d_2_kernel_fixed);

    // cast output to float
    float out_1d_0_kernel_float[SIZE];
    float out_1d_1_kernel_float[SIZE];
    float out_1d_2_kernel_float[SIZE];

    float out_2d_0_kernel_float[SIZE][SIZE];
    float out_2d_1_kernel_float[SIZE][SIZE];
    float out_2d_2_kernel_float[SIZE][SIZE];

    float out_3d_0_kernel_float[SIZE][SIZE][SIZE];
    float out_3d_1_kernel_float[SIZE][SIZE][SIZE];
    float out_3d_2_kernel_float[SIZE][SIZE][SIZE];

    cast_1d<SIZE>(out_1d_0_kernel_fixed, out_1d_0_kernel_float);
    cast_1d<SIZE>(out_1d_1_kernel_fixed, out_1d_1_kernel_float);
    cast_1d<SIZE>(out_1d_2_kernel_fixed, out_1d_2_kernel_float);

    cast_2d<SIZE, SIZE>(out_2d_0_kernel_fixed, out_2d_0_kernel_float);
    cast_2d<SIZE, SIZE>(out_2d_1_kernel_fixed, out_2d_1_kernel_float);
    cast_2d<SIZE, SIZE>(out_2d_2_kernel_fixed, out_2d_2_kernel_float);

    cast_3d<SIZE, SIZE, SIZE>(out_3d_0_kernel_fixed, out_3d_0_kernel_float);
    cast_3d<SIZE, SIZE, SIZE>(out_3d_1_kernel_fixed, out_3d_1_kernel_float);
    cast_3d<SIZE, SIZE, SIZE>(out_3d_2_kernel_fixed, out_3d_2_kernel_float);

    bool pass = true;
    float eps = 1e-3;

    pass &= compare_data_1d<SIZE>(out_1d_0, out_1d_0_kernel_float, eps);
    pass &= compare_data_1d<SIZE>(out_1d_1, out_1d_1_kernel_float, eps);
    pass &= compare_data_1d<SIZE>(out_1d_2, out_1d_2_kernel_float, eps);

    pass &= compare_data_2d<SIZE, SIZE>(out_2d_0, out_2d_0_kernel_float, eps);
    pass &= compare_data_2d<SIZE, SIZE>(out_2d_1, out_2d_1_kernel_float, eps);
    pass &= compare_data_2d<SIZE, SIZE>(out_2d_2, out_2d_2_kernel_float, eps);

    pass &= compare_data_3d<SIZE, SIZE, SIZE>(out_3d_0, out_3d_0_kernel_float, eps);
    pass &= compare_data_3d<SIZE, SIZE, SIZE>(out_3d_1, out_3d_1_kernel_float, eps);
    pass &= compare_data_3d<SIZE, SIZE, SIZE>(out_3d_2, out_3d_2_kernel_float, eps);

    return pass;
}

bool test_mm(){

    const int size_0 = 32;
    const int size_1 = 16;
    const int size_2 = 8;

    float a[size_0][size_1];
    float b[size_1][size_2];
    float c[size_0][size_2];

    load_data_2d<size_0, size_1>("test_data/mm_a.bin", a);
    load_data_2d<size_1, size_2>("test_data/mm_b.bin", b);
    load_data_2d<size_0, size_2>("test_data/mm_c.bin", c);

    // cast input to fixed
    F_TYPE a_fixed[size_0][size_1];
    F_TYPE b_fixed[size_1][size_2];

    cast_2d<size_0, size_1>(a, a_fixed);
    cast_2d<size_1, size_2>(b, b_fixed);

    F_TYPE c_kernel_fixed[size_0][size_2];

    // run kernel
    MM<F_TYPE, size_0, size_1, size_2> mm;
    mm.forward(a_fixed, b_fixed, c_kernel_fixed);

    // cast output to float
    float c_kernel_float[size_0][size_2];
    cast_2d<size_0, size_2>(c_kernel_fixed, c_kernel_float);

    bool pass = true;
    float eps = 1e-3;

    pass &= compare_data_2d<size_0, size_2>(c, c_kernel_float, eps);

    return pass;
}

bool test_elementwise_const(){
    
    const int SIZE = 32;

    const float CONST = 2.5;
    const F_TYPE CONST_FIXED = F_TYPE(CONST);

    // serialize_numpy(elementwise_const_in_1d_np, test_data_dir / "elementwise_const_in_1d.bin")
    // serialize_numpy(elementwise_const_in_2d_np, test_data_dir / "elementwise_const_in_2d.bin")
    // serialize_numpy(elementwise_const_in_3d_np, test_data_dir / "elementwise_const_in_3d.bin")

    // serialize_numpy(elementwise_const_add_out_1d_np, test_data_dir / "elementwise_const_add_out_1d.bin")
    // serialize_numpy(elementwise_const_add_out_2d_np, test_data_dir / "elementwise_const_add_out_2d.bin")
    // serialize_numpy(elementwise_const_add_out_3d_np, test_data_dir / "elementwise_const_add_out_3d.bin")

    // serialize_numpy(elementwise_const_mul_out_1d_np, test_data_dir / "elementwise_const_mul_out_1d.bin")
    // serialize_numpy(elementwise_const_mul_out_2d_np, test_data_dir / "elementwise_const_mul_out_2d.bin")
    // serialize_numpy(elementwise_const_mul_out_3d_np, test_data_dir / "elementwise_const_mul_out_3d.bin")

    float in_1d[SIZE];
    float in_2d[SIZE][SIZE];
    float in_3d[SIZE][SIZE][SIZE];

    float out_1d_add[SIZE];
    float out_2d_add[SIZE][SIZE];
    float out_3d_add[SIZE][SIZE][SIZE];

    float out_1d_mul[SIZE];
    float out_2d_mul[SIZE][SIZE];
    float out_3d_mul[SIZE][SIZE][SIZE];

    load_data_1d<SIZE>("test_data/elementwise_const_in_1d.bin", in_1d);
    load_data_2d<SIZE, SIZE>("test_data/elementwise_const_in_2d.bin", in_2d);
    load_data_3d<SIZE, SIZE, SIZE>("test_data/elementwise_const_in_3d.bin", in_3d);

    load_data_1d<SIZE>("test_data/elementwise_const_add_out_1d.bin", out_1d_add);
    load_data_2d<SIZE, SIZE>("test_data/elementwise_const_add_out_2d.bin", out_2d_add);
    load_data_3d<SIZE, SIZE, SIZE>("test_data/elementwise_const_add_out_3d.bin", out_3d_add);

    load_data_1d<SIZE>("test_data/elementwise_const_mul_out_1d.bin", out_1d_mul);
    load_data_2d<SIZE, SIZE>("test_data/elementwise_const_mul_out_2d.bin", out_2d_mul);
    load_data_3d<SIZE, SIZE, SIZE>("test_data/elementwise_const_mul_out_3d.bin", out_3d_mul);

    // cast input to fixed
    F_TYPE in_1d_fixed[SIZE];
    F_TYPE in_2d_fixed[SIZE][SIZE];
    F_TYPE in_3d_fixed[SIZE][SIZE][SIZE];

    cast_1d<SIZE>(in_1d, in_1d_fixed);
    cast_2d<SIZE, SIZE>(in_2d, in_2d_fixed);
    cast_3d<SIZE, SIZE, SIZE>(in_3d, in_3d_fixed);

    F_TYPE out_1d_add_kernel_fixed[SIZE];
    F_TYPE out_2d_add_kernel_fixed[SIZE][SIZE];
    F_TYPE out_3d_add_kernel_fixed[SIZE][SIZE][SIZE];

    F_TYPE out_1d_mul_kernel_fixed[SIZE];
    F_TYPE out_2d_mul_kernel_fixed[SIZE][SIZE];
    F_TYPE out_3d_mul_kernel_fixed[SIZE][SIZE][SIZE];

    // run kernel
    ElementWiseAdd1DConst < F_TYPE, SIZE> elementwise_add_1d_const;
    ElementWiseAdd2DConst < F_TYPE, SIZE, SIZE> elementwise_add_2d_const;
    ElementWiseAdd3DConst < F_TYPE, SIZE, SIZE, SIZE> elementwise_add_3d_const;

    ElementWiseMul1DConst < F_TYPE, SIZE> elementwise_mul_1d_const;
    ElementWiseMul2DConst < F_TYPE, SIZE, SIZE> elementwise_mul_2d_const;
    ElementWiseMul3DConst < F_TYPE, SIZE, SIZE, SIZE> elementwise_mul_3d_const;

    elementwise_add_1d_const.forward(in_1d_fixed, out_1d_add_kernel_fixed, CONST_FIXED);
    elementwise_add_2d_const.forward(in_2d_fixed, out_2d_add_kernel_fixed, CONST_FIXED);
    elementwise_add_3d_const.forward(in_3d_fixed, out_3d_add_kernel_fixed, CONST_FIXED);

    elementwise_mul_1d_const.forward(in_1d_fixed, out_1d_mul_kernel_fixed, CONST_FIXED);
    elementwise_mul_2d_const.forward(in_2d_fixed, out_2d_mul_kernel_fixed, CONST_FIXED);
    elementwise_mul_3d_const.forward(in_3d_fixed, out_3d_mul_kernel_fixed, CONST_FIXED);

    // cast output to float
    float out_1d_add_kernel[SIZE];
    float out_2d_add_kernel[SIZE][SIZE];
    float out_3d_add_kernel[SIZE][SIZE][SIZE];

    float out_1d_mul_kernel[SIZE];
    float out_2d_mul_kernel[SIZE][SIZE];
    float out_3d_mul_kernel[SIZE][SIZE][SIZE];

    cast_1d<SIZE>(out_1d_add_kernel_fixed, out_1d_add_kernel);
    cast_2d<SIZE, SIZE>(out_2d_add_kernel_fixed, out_2d_add_kernel);
    cast_3d<SIZE, SIZE, SIZE>(out_3d_add_kernel_fixed, out_3d_add_kernel);

    cast_1d<SIZE>(out_1d_mul_kernel_fixed, out_1d_mul_kernel);
    cast_2d<SIZE, SIZE>(out_2d_mul_kernel_fixed, out_2d_mul_kernel);
    cast_3d<SIZE, SIZE, SIZE>(out_3d_mul_kernel_fixed, out_3d_mul_kernel);

    // compare
    float eps = 1e-3;
    bool pass = true;

    pass &= compare_data_1d<SIZE>(out_1d_add_kernel, out_1d_add, eps);
    pass &= compare_data_2d<SIZE, SIZE>(out_2d_add_kernel, out_2d_add, eps);
    pass &= compare_data_3d<SIZE, SIZE, SIZE>(out_3d_add_kernel, out_3d_add, eps);

    pass &= compare_data_1d<SIZE>(out_1d_mul_kernel, out_1d_mul, eps);
    pass &= compare_data_2d<SIZE, SIZE>(out_2d_mul_kernel, out_2d_mul, eps);
    pass &= compare_data_3d<SIZE, SIZE, SIZE>(out_3d_mul_kernel, out_3d_mul, eps);

    return pass;
}

bool test_neg() {

    const int SIZE = 32;

    // load data
    float in_1d[SIZE];
    float in_2d[SIZE][SIZE];
    float in_3d[SIZE][SIZE][SIZE];

    float out_1d[SIZE];
    float out_2d[SIZE][SIZE];
    float out_3d[SIZE][SIZE][SIZE];

    load_data_1d<SIZE>("test_data/neg_in_1d.bin", in_1d);
    load_data_2d<SIZE, SIZE>("test_data/neg_in_2d.bin", in_2d);
    load_data_3d<SIZE, SIZE, SIZE>("test_data/neg_in_3d.bin", in_3d);

    load_data_1d<SIZE>("test_data/neg_out_1d.bin", out_1d);
    load_data_2d<SIZE, SIZE>("test_data/neg_out_2d.bin", out_2d);
    load_data_3d<SIZE, SIZE, SIZE>("test_data/neg_out_3d.bin", out_3d);

    // cast input to fixed
    F_TYPE in_1d_fixed[SIZE];
    F_TYPE in_2d_fixed[SIZE][SIZE];
    F_TYPE in_3d_fixed[SIZE][SIZE][SIZE];

    cast_1d<SIZE>(in_1d, in_1d_fixed);
    cast_2d<SIZE, SIZE>(in_2d, in_2d_fixed);
    cast_3d<SIZE, SIZE, SIZE>(in_3d, in_3d_fixed);

    F_TYPE out_1d_kernel_fixed[SIZE];
    F_TYPE out_2d_kernel_fixed[SIZE][SIZE];
    F_TYPE out_3d_kernel_fixed[SIZE][SIZE][SIZE];

    // run kernel
    Neg1D<F_TYPE, SIZE> neg_1d;
    Neg2D<F_TYPE, SIZE, SIZE> neg_2d;
    Neg3D<F_TYPE, SIZE, SIZE, SIZE> neg_3d;

    neg_1d.forward(in_1d_fixed, out_1d_kernel_fixed);
    neg_2d.forward(in_2d_fixed, out_2d_kernel_fixed);
    neg_3d.forward(in_3d_fixed, out_3d_kernel_fixed);

    // cast output to float
    float out_1d_kernel[SIZE];
    float out_2d_kernel[SIZE][SIZE];
    float out_3d_kernel[SIZE][SIZE][SIZE];

    cast_1d<SIZE>(out_1d_kernel_fixed, out_1d_kernel);
    cast_2d<SIZE, SIZE>(out_2d_kernel_fixed, out_2d_kernel);
    cast_3d<SIZE, SIZE, SIZE>(out_3d_kernel_fixed, out_3d_kernel);

    // compare
    float eps = 1e-3;
    bool pass = true;
    pass &= compare_data_1d<SIZE>(out_1d_kernel, out_1d, eps);
    pass &= compare_data_2d<SIZE, SIZE>(out_2d_kernel, out_2d, eps);
    pass &= compare_data_3d<SIZE, SIZE, SIZE>(out_3d_kernel, out_3d, eps);

    return pass;
}

int main() {
    printf("#######################\n");
    printf("### inr_hw_lib_test ###\n");
    printf("#######################\n");

    bool results_test_linear_layer = test_linear_layer();
    if (results_test_linear_layer) {
        printf("test_linear_layer: PASS\n");
    } else {
        printf("test_linear_layer: FAIL\n");
    }

    bool results_test_activations = test_activations();
    if (results_test_activations) {
        printf("test_activations: PASS\n");
    } else {
        printf("test_activations: FAIL\n");
    }

    bool results_test_siren_net = test_siren_net();
    if (results_test_siren_net) {
        printf("test_siren_net: PASS\n");
    } else {
        printf("test_siren_net: FAIL\n");
    }

    bool results_test_elementwise = test_elementwise();
    if (results_test_elementwise) {
        printf("test_elementwise: PASS\n");
    } else {
        printf("test_elementwise: FAIL\n");
    }

    bool results_test_shape = test_shape_edits();
    if (results_test_shape) {
        printf("test_shape_edits: PASS\n");
    } else {
        printf("test_shape_edits: FAIL\n");
    }

    bool results_test_broadcast = test_broadcast();
    if (results_test_broadcast) {
        printf("test_broadcast: PASS\n");
    } else {
        printf("test_broadcast: FAIL\n");
    }

    bool results_test_mm = test_mm();
    if (results_test_mm) {
        printf("test_mm: PASS\n");
    } else {
        printf("test_mm: FAIL\n");
    }

    bool results_test_elementwise_const = test_elementwise_const();
    if (results_test_elementwise_const) {
        printf("test_elementwise_const: PASS\n");
    } else {
        printf("test_elementwise_const: FAIL\n");
    }

    bool results_test_neg = test_neg();
    if (results_test_neg) {
        printf("test_neg: PASS\n");
    } else {
        printf("test_neg: FAIL\n");
    }
}