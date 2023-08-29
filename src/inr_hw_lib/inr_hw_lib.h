#pragma once

// Array Helpers //

template <const int M, typename T>
void print_1d(T in[M]) {
    printf("[% 2.5f, % 2.5f, % 2.5f ... % 2.5f, % 2.5f, % 2.5f]\n",
           float(in[0]), (float)in[1], (float)in[2],
           float(in[M - 3]), (float)in[M - 2], (float)in[M - 1]);
}

template <const int M, const int N, typename T>
void print_2d(T in[M][N]) {
    print_1d<N, T>(in[0]);
    print_1d<N, T>(in[1]);
    print_1d<N, T>(in[2]);
    printf("[.............................................................]\n");
    printf("[.............................................................]\n");
    printf("[.............................................................]\n");
    print_1d<N, T>(in[M - 3]);
    print_1d<N, T>(in[M - 2]);
    print_1d<N, T>(in[M - 1]);
}

template <const int M, const int N, const int O, typename T>
void print_3d(T in[M][N][O]) {
    printf("[\n");
    print_2d<N, O, T>(in[0]);
    printf("                               .\n");
    printf("                               .\n");
    printf("                               .\n");
    print_2d<N, O, T>(in[M - 1]);
    printf("]\n");
}

template <const int M, typename T>
void copy_1d(T from[M], T to[M]) {
    for (int i = 0; i < M; i++) {
        to[i] = from[i];
    }
}

template <const int M, const int N, typename T>
void copy_2d(T from[M][N], T to[M][N]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            to[i][j] = from[i][j];
        }
    }
}

template <const int M, const int N, const int O, typename T>
void copy_3d(T from[M][N][O], T to[M][N][O]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                to[i][j][k] = from[i][j][k];
            }
        }
    }
}

template <const int M, typename T_in, typename T_out>
void cast_1d(T_in in[M], T_out out[M]) {
    for (int i = 0; i < M; i++) {
        out[i] = (T_out)in[i];
    }
}

template <const int M, const int N, typename T_in, typename T_out>
void cast_2d(T_in in[M][N], T_out out[M][N]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[i][j] = (T_out)in[i][j];
        }
    }
}

template <const int M, const int N, const int O, typename T_in, typename T_out>
void cast_3d(T_in in[M][N][O], T_out out[M][N][O]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                out[i][j][k] = (T_out)in[i][j][k];
            }
        }
    }
}

template <const int M, typename T = float>
void load_data_1d(const char *fp, T arr[M]) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), M, f);
    fclose(f);
}

template <const int M, const int N, typename T = float>
void load_data_2d(const char *fp, T arr[M][N]) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), M * N, f);
    fclose(f);
}

template <const int M, const int N, const int O, typename T = float>
void load_data_3d(const char *fp, T arr[M][N][O]) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), M * N * O, f);
    fclose(f);
}

template <const int M, typename T = float>
void load_data_var_1d(const char *fp, T arr[M], int i) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), i, f);
    fclose(f);
}

template <const int M, const int N, typename T = float>
void load_data_var_2d(const char *fp, T arr[M][N], int i, int j) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), i * j, f);
    fclose(f);
}

template <const int M, const int N, const int O, typename T = float>
void load_data_var_3d(const char *fp, T arr[M][N][O], int i, int j, int k) {
    FILE *f;
    f = fopen(fp, "r");
    fread(arr, sizeof(T), i * j * k, f);
    fclose(f);
}

template <const int M, typename T = float>
bool compare_data_1d(T arr1[M], T arr2[M], float eps) {
    for (int i = 0; i < M; i++) {
        if (std::abs(arr1[i] - arr2[i]) > eps) {
            return false;
        }
    }
    return true;
}

template <const int M, const int N, typename T = float>
bool compare_data_2d(T arr1[M][N], T arr2[M][N], float eps) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (std::abs(arr1[i][j] - arr2[i][j]) > eps) {
                return false;
            }
        }
    }
    return true;
}

template <const int M, const int N, const int O, typename T = float>
bool compare_data_3d(T arr1[M][N][O], T arr2[M][N][O], float eps) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                if (std::abs(arr1[i][j][k] - arr2[i][j][k]) > eps) {
                    return false;
                }
            }
        }
    }
    return true;
}

// compute_mae_1d
template <const int M, typename T = float>
float compute_mae_1d(T arr1[M], T arr2[M]) {
    float mae = 0;
    for (int i = 0; i < M; i++) {
        mae += std::abs(arr1[i] - arr2[i]);
    }
    return mae / M;
}

// compute_mae_2d
template <const int M, const int N, typename T = float>
float compute_mae_2d(T arr1[M][N], T arr2[M][N]) {
    float mae = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            mae += std::abs(arr1[i][j] - arr2[i][j]);
        }
    }
    return mae / (M * N);
}

// compute_mae_3d
template <const int M, const int N, const int O, typename T = float>
float compute_mae_3d(T arr1[M][N][O], T arr2[M][N][O]) {
    float mae = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                mae += std::abs(arr1[i][j][k] - arr2[i][j][k]);
            }
        }
    }
    return mae / (M * N * O);
}


// Activation_1d //

template <typename T>
T activation_elu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

    const T alpha = T(1.0);

    if (x > 0) {
        return x;
    } else {
        return alpha * (m_exp(x) - T(1.0));
    }
}

template <typename T>
T activation_hardtanh(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

    const T min_val = T(-1.0);
    const T max_val = T(1.0);

    if (x < min_val) {
        return min_val;
    } else if (x > max_val) {
        return max_val;
    } else {
        return x;
    }
}

template <typename T>
T activation_leakyrelu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

    const T negative_slope = T(0.1);

    if (x >= 0) {
        return x;
    } else {
        return x * negative_slope;
    }
}

template <typename T>
T activation_relu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

template <typename T>
T activation_gelu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    const T sqrt_2 = m_sqrt(T(2));
    const T sqrt_2_recip = T(1) / sqrt_2;
    const T one_half = T(0.5);
    T out = x * one_half * (T(1) + m_erf(x * sqrt_2_recip));
    return out;
}

template <typename T>
T activation_gelu_approx_1(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    const T sqrt_2_div_pi = m_sqrt(T(2) / m_pi());
    const T c = T(0.044715);
    const T one_half = T(0.5);
    T out = one_half * x * (1 + m_tanh(sqrt_2_div_pi * (x + c * m_pow(x, F_TYPE(3)))));
    return out;
}

template <typename T>
T activation_gelu_approx_2(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    const F_TYPE c = T(1.702);
    F_TYPE sigmod_arg = c * x;
    F_TYPE sigmod_out = T(1.0) / (T(1.0) + m_exp(-sigmod_arg));
    F_TYPE out = x * sigmod_arg;
    return out;
}

template <typename T>
T activation_sigmoid(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    return T(1.0) / (T(1.0) + m_exp(-x));
}

template <typename T>
T activation_silu(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE
    return x * (T(1.0) / (T(1.0) + m_exp(-x)));
}

template <typename T>
T activation_tanh(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

#if __FLOATING_POINT_MODEL__
    T out = m_tanh(x);
    return out;
#else
    T out = m_tanh(x);
    T out_fixed = (hls::signbit(x) != hls::signbit(out)) ? T(-out) : out;
    return out_fixed;
#endif
}

template <typename T>
T activation_softsign(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

    return x / (T(1.0) + m_abs(x));
}

template <typename T>
T activation_sin(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

    return m_sin(x);
}

template <typename T>
T activation_cos(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

    return m_cos(x);
}

template <typename T>
T activation_identity(T x) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

    return x;
}

// enum for activation function
enum ACTIVATION_FUNCTION {
    ELU,
    HARDTANH,
    LEAKYRELU,
    RELU,
    GELU,
    GELU_APPROX_1,
    GELU_APPROX_2,
    SIGMOID,
    SILU,
    TANH,
    SOFTSIGN,
    SIN,
    COS,
    IDENTITY
};

template <
    typename T,
    const int size,
    T (*activation_func)(T),
    const int block_size = 1>
class Activation_1d {

    static_assert(size > 0, "size must be greater than 0");
    static_assert(block_size > 0, "block_size must be greater than 0");
    static_assert(size % block_size == 0, "size must be divisible by block_size");

public:
    void forward(T x_in[size], T x_out[size]) {
#pragma HLS INLINE off
#pragma HLS array_partition variable = x_in cyclic factor = block_size dim = 1
#pragma HLS array_partition variable = x_out cyclic factor = block_size dim = 1
        for (int i = 0; i < size; i += block_size) {
#pragma HLS PIPELINE
            for (int j = 0; j < block_size; j++) {
                x_out[i + j] = activation_func(x_in[i + j]);
            }
        }
    }
};

template <
    typename T,
    const int size_0,
    const int size_1,
    T (*activation_func)(T),
    const int block_size_0 = 1,
    const int block_size_1 = 1>
class Activation_2d {
    
    static_assert(size_0 > 0, "size_0 must be greater than 0");
    static_assert(size_1 > 0, "size_1 must be greater than 0");
    static_assert(block_size_0 > 0, "block_size_0 must be greater than 0");
    static_assert(block_size_1 > 0, "block_size_1 must be greater than 0");
    static_assert(size_0 % block_size_0 == 0, "size_0 must be divisible by block_size_0");
    static_assert(size_1 % block_size_1 == 0, "size_1 must be divisible by block_size_1");

public:
    void forward(T x_in[size_0][size_1], T x_out[size_0][size_1]) {
#pragma HLS INLINE off
#pragma HLS array_partition variable = x_in cyclic factor = block_size_0 dim = 1
#pragma HLS array_partition variable = x_out cyclic factor = block_size_0 dim = 1
        for (int i = 0; i < size_0; i += block_size_0) {
            for (int j = 0; j < size_1; j += block_size_1) {
#pragma HLS PIPELINE
                for (int k = 0; k < block_size_0; k++) {
                    for (int l = 0; l < block_size_1; l++) {
                        x_out[i + k][j + l] = activation_func(x_in[i + k][j + l]);
                    }
                }
            }
        }
    }
};

template <
    typename T,
    const int size_0,
    const int size_1,
    const int size_2,
    T (*activation_func)(T),
    const int block_size_0 = 1,
    const int block_size_1 = 1,
    const int block_size_2 = 1>
class Activation_3d {

    static_assert(size_0 > 0, "size_0 must be greater than 0");
    static_assert(size_1 > 0, "size_1 must be greater than 0");
    static_assert(size_2 > 0, "size_2 must be greater than 0");
    static_assert(block_size_0 > 0, "block_size_0 must be greater than 0");
    static_assert(block_size_1 > 0, "block_size_1 must be greater than 0");
    static_assert(block_size_2 > 0, "block_size_2 must be greater than 0");
    static_assert(size_0 % block_size_0 == 0, "size_0 must be divisible by block_size_0");
    static_assert(size_1 % block_size_1 == 0, "size_1 must be divisible by block_size_1");
    static_assert(size_2 % block_size_2 == 0, "size_2 must be divisible by block_size_2");

public:
    void forward(T x_in[size_0][size_1][size_2], T x_out[size_0][size_1][size_2]) {
#pragma HLS INLINE off
#pragma HLS array_partition variable = x_in cyclic factor = block_size_0 dim = 1
#pragma HLS array_partition variable = x_out cyclic factor = block_size_0 dim = 1

        for (int i = 0; i < size_0; i += block_size_0) {
            for (int j = 0; j < size_1; j += block_size_1) {
                for (int k = 0; k < size_2; k += block_size_2) {
#pragma HLS PIPELINE
                    for (int l = 0; l < block_size_0; l++) {
                        for (int m = 0; m < block_size_1; m++) {
                            for (int n = 0; n < block_size_2; n++) {
                                x_out[i + l][j + m][k + n] = activation_func(x_in[i + l][j + m][k + n]);
                            }
                        }
                    }
                }
            }
        }
    }
};


// Linear //

template <typename T,
          const int in_size,
          const int out_size,
          const int block_size_in = 1,
          const int block_size_out = 1>
class Linear {

    static_assert(in_size > 0, "in_size must be greater than 0");
    static_assert(out_size > 0, "out_size must be greater than 0");
    static_assert(block_size_in > 0, "block_size_in must be greater than 0");
    static_assert(block_size_out > 0, "block_size_out must be greater than 0");
    static_assert(in_size % block_size_in == 0, "in_size must be divisible by block_size_in");
    static_assert(out_size % block_size_out == 0, "out_size must be divisible by block_size_out");

public:
    T weight[out_size][in_size] = {0};
    T bias[out_size] = {0};

    void load_params(T weight_in[out_size][in_size],
                     T bias_in[out_size]) {
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < in_size; j++) {
                weight[i][j] = weight_in[i][j];
            }
            bias[i] = bias_in[i];
        }
    }

    void forward(T input[in_size], T output[out_size]) {

#pragma HLS INLINE off
        // #pragma HLS DATAFLOW

#pragma HLS array_partition variable = input cyclic factor = block_size_in dim = 1
#pragma HLS array_partition variable = output cyclic factor = block_size_out dim = 1

#pragma HLS array_partition variable = weight cyclic factor = block_size_out dim = 1
#pragma HLS array_partition variable = weight cyclic factor = block_size_in dim = 2
#pragma HLS array_partition variable = bias cyclic factor = block_size_out dim = 1

        T temp_sum[block_size_out];
#pragma HLS ARRAY_PARTITION variable = temp_sum complete

        // BLOCK_OUT
        for (int i = 0; i < out_size; i += block_size_out) {
            // BLOCK_IN
            for (int j = 0; j < in_size; j += block_size_in) {
#pragma HLS PIPELINE
                // TEMP_SUM_ZERO_LOOP
                for (int k = 0; k < block_size_out; k++) {
#pragma HLS UNROLL
                    temp_sum[k] = 0;
                }
                // SUM_OUTER
                for (int k = 0; k < block_size_out; k++) {
#pragma HLS UNROLL
                    // SUM_INNER
                    for (int l = 0; l < block_size_in; l++) {
#pragma HLS UNROLL
                        temp_sum[k] += weight[i + k][j + l] * input[j + l];
                    }
                }
                // WRITE_LOOP
                for (int k = 0; k < block_size_out; k++) {
#pragma HLS UNROLL
                    // check if first block itteration
                    // if first block itteration, write bias
                    if (j == 0) {
                        output[i + k] = bias[i + k];
                    }
                    output[i + k] += temp_sum[k];
                }
            }
        }
    }
};

// Normalization //

template <typename T,
          const int size,
          const int block_size = 1>
class LayerNorm {
    static_assert(size > 0, "size must be greater than 0");
    static_assert(block_size > 0, "block_size must be greater than 0");
    static_assert(size % block_size == 0, "size must be divisible by block_size");

public:
    T scale[size] = {T(0.0)};
    T bias[size] = {T(0.0)};

    void load_params(T scale_in[size], T bias_in[size]) {
        for (int i = 0; i < size; i++) {
            scale[i] = scale_in[i];
            bias[i] = bias_in[i];
        }
    }
    void forward(T x_in[size], T x_out[size]) {
#pragma HLS INLINE off
        // #pragma HLS DATAFLOW

#pragma HLS array_partition variable = x_in cyclic factor = block_size dim = 1
#pragma HLS array_partition variable = x_out cyclic factor = block_size dim = 1

#pragma HLS array_partition variable = scale cyclic factor = block_size dim = 1
#pragma HLS array_partition variable = bias cyclic factor = block_size dim = 1

        T mean = T(0.0);
        T variance = T(0.0);

        // MEAN
        for (int i = 0; i < size; i += block_size) {
#pragma HLS PIPELINE
            for (int j = 0; j < block_size; j++) {
                mean += x_in[i + j];
            }
        }
        mean /= size;

        // VARIANCE
        for (int i = 0; i < size; i += block_size) {
#pragma HLS PIPELINE
            for (int j = 0; j < block_size; j++) {
                variance += (x_in[i + j] - mean) * (x_in[i + j] - mean);
            }
        }
        variance /= size;

        // NORMALIZE
        for (int i = 0; i < size; i += block_size) {
#pragma HLS PIPELINE
            for (int j = 0; j < block_size; j++) {
                x_out[i + j] = scale[i + j] * (x_in[i + j] - mean) / sqrt(variance + T(1e-12)) + bias[i + j];
            }
        }
    }
};

// Architecture //

template <
    typename T,
    const int size,
    const int block_size = 1>
class SirenSine {

    static_assert(size > 0, "size must be greater than 0");
    static_assert(block_size > 0, "block_size must be greater than 0");
    static_assert(size % block_size == 0, "size must be divisible by block_size");

public:
    T w0 = T(1.0);

    void load_params(T w0_in) {
        w0 = w0_in;
    };

    void forward(T x_in[size], T x_out[size]) {
#pragma HLS inline off
        // #pragma HLS DATAFLOW

#pragma HLS array_partition variable = x_in cyclic factor = block_size dim = 1
#pragma HLS array_partition variable = x_out cyclic factor = block_size dim = 1
        for (int i = 0; i < size; i += block_size) {
#pragma HLS PIPELINE
            for (int j = 0; j < block_size; j++) {
                x_out[i + j] = activation_sin(x_in[i + j] * w0);
            }
        }
    };
};

template <typename T,
          const int in_size,
          const int out_size,
          const int block_size_in = 1,
          const int block_size_out = 1>
class SirenLayer {
public:
    // #pragma HLS ARRAY_PARTITION variable = weight type = cyclic factor = block_size_out dim = 1
    // #pragma HLS ARRAY_PARTITION variable = weight type = cyclic factor = block_size_in dim = 2
    // #pragma HLS ARRAY_PARTITION variable = bias type = cyclic factor = block_size_out dim = 1

    Linear<T, in_size, out_size, block_size_in, block_size_out> linear;
    SirenSine<T, out_size, block_size_out> sine;

    void load_params(T weight_in[out_size][in_size], T bias_in[out_size], T w0_in) {
        linear.load_params(weight_in, bias_in);
        sine.load_params(w0_in);
    };

    void forward(T x_in[in_size], T x_out[out_size]) {
#pragma HLS inline off
        // #pragma HLS DATAFLOW

        T x_out_temp[out_size];

        linear.forward(x_in, x_out_temp);
        sine.forward(x_out_temp, x_out);
    };
};

template <
    typename T,
    const int model_in_dim,
    const int model_out_dim,
    const int model_hidden_dim,
    const int num_hidden_layers,
    T (*activation_func)(T) = activation_identity,
    int p_in = 1,
    int p_hidden = 1,
    int p_out = 1>
class SirenNet {

    // parameter checks //
    static_assert(model_in_dim > 0, "model_in_dim must be greater than 0");
    static_assert(model_out_dim > 0, "model_out_dim must be greater than 0");
    static_assert(model_hidden_dim > 0, "model_hidden_dim must be greater than 0");
    static_assert(num_hidden_layers > 0, "num_hidden_layers must be greater than 0");

    static_assert(p_in > 0, "p_in must be greater than 0");
    static_assert(p_hidden > 0, "p_hidden must be greater than 0");
    static_assert(p_out > 0, "p_out must be greater than 0");

    static_assert(model_in_dim % p_in == 0, "model_in_dim must be divisible by p_in");
    static_assert(model_hidden_dim % p_hidden == 0, "model_hidden_dim must be divisible by p_hidden");
    static_assert(model_out_dim % p_out == 0, "model_out_dim must be divisible by p_out");

public:
    SirenLayer<T, model_hidden_dim, model_hidden_dim, p_hidden, p_hidden> hidden_layers[num_hidden_layers];
    SirenLayer<T, model_in_dim, model_hidden_dim, p_in, p_hidden> input_layer;
    Linear<T, model_hidden_dim, model_out_dim, p_hidden, p_out> output_layer;
    Activation_1d<T, model_out_dim, activation_func, p_out> output_activation;

    void load_params(
        T input_layer_weight[model_hidden_dim][model_in_dim],
        T input_layer_bias[model_hidden_dim],
        T input_layer_w0,
        T hidden_layers_weight[num_hidden_layers][model_hidden_dim][model_hidden_dim],
        T hidden_layers_bias[num_hidden_layers][model_hidden_dim],
        T hidden_layers_w0[num_hidden_layers],
        T output_layer_weight[model_out_dim][model_hidden_dim],
        T output_layer_bias[model_out_dim]) {
        input_layer.load_params(input_layer_weight, input_layer_bias, input_layer_w0);
        for (int i = 0; i < num_hidden_layers; i++) {
            hidden_layers[i].load_params(hidden_layers_weight[i], hidden_layers_bias[i], hidden_layers_w0[i]);
        }
        output_layer.load_params(output_layer_weight, output_layer_bias);
    };

    void forward(T x_in[model_in_dim],
                 T x_out[model_out_dim]) {
#pragma HLS inline off
        // #pragma HLS DATAFLOW

        T input_layer_buffer[model_hidden_dim];
        T hidden_layer_buffers[num_hidden_layers][model_hidden_dim];
        T output_layer_buffer[model_out_dim];

        input_layer.forward(x_in, input_layer_buffer);
        for (int i = 0; i < num_hidden_layers; i++) {
            if (i == 0) {
                hidden_layers[i].forward(input_layer_buffer, hidden_layer_buffers[i]);
            } else {
                hidden_layers[i].forward(hidden_layer_buffers[i - 1], hidden_layer_buffers[i]);
            }
        }
        output_layer.forward(hidden_layer_buffers[num_hidden_layers - 1], output_layer_buffer);
        output_activation.forward(output_layer_buffer, x_out);
    };
};

// element wise multiplication 1d
template <
    typename T,
    const int size,
    const int block_size = 1>
class ElementWiseMul1D {
public:
    void forward(
        T a_in[size],
        T b_in[size],
        T c_out[size]) {
#pragma HLS inline off
        // #pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size dim = 1
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size dim = 1

        for (int i = 0; i < size; i += block_size) {
#pragma HLS PIPELINE
            for (int j = 0; j < block_size; j++) {
#pragma HLS UNROLL
                c_out[i + j] = a_in[i + j] * b_in[i + j];
            }
        }
    };
};

// element wise multiplication 2d
template <
    typename T,
    const int size_1,
    const int size_2,
    const int block_size_1 = 1,
    const int block_size_2 = 1>
class ElementWiseMul2D {
public:
    void forward(
        T a_in[size_1][size_2],
        T b_in[size_1][size_2],
        T c_out[size_1][size_2]) {
#pragma HLS inline off
        // #pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size_1 dim = 1
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size_1 dim = 1

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size_2 dim = 2
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size_2 dim = 2

        for (int i = 0; i < size_1; i += block_size_1) {
            for (int j = 0; j < size_2; j += block_size_2) {
#pragma HLS PIPELINE
                for (int k = 0; k < block_size_1; k++) {
                    for (int l = 0; l < block_size_2; l++) {
#pragma HLS UNROLL
                        c_out[i + k][j + l] = a_in[i + k][j + l] * b_in[i + k][j + l];
                    }
                }
            }
        }
    };
};

// element wise multiplication 3d
template <
    typename T,
    const int size_1,
    const int size_2,
    const int size_3,
    const int block_size_1 = 1,
    const int block_size_2 = 1,
    const int block_size_3 = 1>
class ElementWiseMul3D {
public:
    void forward(
        T a_in[size_1][size_2][size_3],
        T b_in[size_1][size_2][size_3],
        T c_out[size_1][size_2][size_3]) {
#pragma HLS inline off
        // #pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size_1 dim = 1
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size_1 dim = 1

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size_2 dim = 2
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size_2 dim = 2

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size_3 dim = 3
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size_3 dim = 3

        for (int i = 0; i < size_1; i += block_size_1) {
            for (int j = 0; j < size_2; j += block_size_2) {
                for (int k = 0; k < size_3; k += block_size_3) {
#pragma HLS PIPELINE
                    for (int l = 0; l < block_size_1; l++) {
                        for (int m = 0; m < block_size_2; m++) {
                            for (int n = 0; n < block_size_3; n++) {
#pragma HLS UNROLL
                                c_out[i + l][j + m][k + n] = a_in[i + l][j + m][k + n] * b_in[i + l][j + m][k + n];
                            }
                        }
                    }
                }
            }
        }
    };
};

/// add 1d
template <
    typename T,
    const int size,
    const int block_size = 1>
class ElementWiseAdd1D {
public:
    void forward(
        T a_in[size],
        T b_in[size],
        T c_out[size]) {
#pragma HLS inline off
        // #pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size dim = 1
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size dim = 1

        for (int i = 0; i < size; i += block_size) {
#pragma HLS PIPELINE
            for (int j = 0; j < block_size; j++) {
#pragma HLS UNROLL
                c_out[i + j] = a_in[i + j] + b_in[i + j];
            }
        }
    };
};

/// add 2d
template <
    typename T,
    const int size_1,
    const int size_2,
    const int block_size_1 = 1,
    const int block_size_2 = 1>
class ElementWiseAdd2D {
public:
    void forward(
        T a_in[size_1][size_2],
        T b_in[size_1][size_2],
        T c_out[size_1][size_2]) {
#pragma HLS inline off
        // #pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size_1 dim = 1
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size_1 dim = 1

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size_2 dim = 2
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size_2 dim = 2

        for (int i = 0; i < size_1; i += block_size_1) {
            for (int j = 0; j < size_2; j += block_size_2) {
#pragma HLS PIPELINE
                for (int k = 0; k < block_size_1; k++) {
                    for (int l = 0; l < block_size_2; l++) {
#pragma HLS UNROLL
                        c_out[i + k][j + l] = a_in[i + k][j + l] + b_in[i + k][j + l];
                    }
                }
            }
        }
    };
};

// add 3d
template <
    typename T,
    const int size_1,
    const int size_2,
    const int size_3,
    const int block_size_1 = 1,
    const int block_size_2 = 1,
    const int block_size_3 = 1>
class ElementWiseAdd3D {
public:
    void forward(
        T a_in[size_1][size_2][size_3],
        T b_in[size_1][size_2][size_3],
        T c_out[size_1][size_2][size_3]) {
#pragma HLS inline off
        // #pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size_1 dim = 1
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size_1 dim = 1

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size_2 dim = 2
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size_2 dim = 2

#pragma HLS ARRAY_PARTITION variable = a_in type = cyclic factor = block_size_3 dim = 3
#pragma HLS ARRAY_PARTITION variable = b_in type = cyclic factor = block_size_3 dim = 3

        for (int i = 0; i < size_1; i += block_size_1) {
            for (int j = 0; j < size_2; j += block_size_2) {
                for (int k = 0; k < size_3; k += block_size_3) {
#pragma HLS PIPELINE
                    for (int l = 0; l < block_size_1; l++) {
                        for (int m = 0; m < block_size_2; m++) {
                            for (int n = 0; n < block_size_3; n++) {
#pragma HLS UNROLL
                                c_out[i + l][j + m][k + n] = a_in[i + l][j + m][k + n] + b_in[i + l][j + m][k + n];
                            }
                        }
                    }
                }
            }
        }
    };
};



// ElementWiseMul1DConst
template <
    typename T,
    const int size,
    const int block_size = 1>
class ElementWiseMul1DConst {
public:
    void forward(
        T x_in[size],
        T y_out[size],
        const T const_val) {
#pragma HLS inline off

#pragma HLS ARRAY_PARTITION variable = x_in type = cyclic factor = block_size dim = 1

        for (int i = 0; i < size; i += block_size) {
#pragma HLS PIPELINE
            for (int j = 0; j < block_size; j++) {
#pragma HLS UNROLL
                y_out[i + j] = x_in[i + j] * const_val;
            }
        }
    }
};

// ElementWiseMul2DConst
template <
    typename T,
    const int size_1,
    const int size_2,
    const int block_size_1 = 1,
    const int block_size_2 = 1>
class ElementWiseMul2DConst {
public:
    void forward(
        T x_in[size_1][size_2],
        T y_out[size_1][size_2],
        const T const_val) {
#pragma HLS inline off

#pragma HLS ARRAY_PARTITION variable = x_in type = cyclic factor = block_size_1 dim = 1
#pragma HLS ARRAY_PARTITION variable = x_in type = cyclic factor = block_size_2 dim = 2

        for (int i = 0; i < size_1; i += block_size_1) {
            for (int j = 0; j < size_2; j += block_size_2) {
#pragma HLS PIPELINE
                for (int k = 0; k < block_size_1; k++) {
                    for (int l = 0; l < block_size_2; l++) {
#pragma HLS UNROLL
                        y_out[i + k][j + l] = x_in[i + k][j + l] * const_val;
                    }
                }
            }
        }
    }
};

// ElementWiseMul3DConst
template <
    typename T,
    const int size_1,
    const int size_2,
    const int size_3,
    const int block_size_1 = 1,
    const int block_size_2 = 1,
    const int block_size_3 = 1>
class ElementWiseMul3DConst {
public:
    void forward(
        T x_in[size_1][size_2][size_3],
        T y_out[size_1][size_2][size_3],
        const T const_val) {
#pragma HLS inline off

#pragma HLS ARRAY_PARTITION variable = x_in type = cyclic factor = block_size_1 dim = 1
#pragma HLS ARRAY_PARTITION variable = x_in type = cyclic factor = block_size_2 dim = 2
#pragma HLS ARRAY_PARTITION variable = x_in type = cyclic factor = block_size_3 dim = 3

        for (int i = 0; i < size_1; i += block_size_1) {
            for (int j = 0; j < size_2; j += block_size_2) {
                for (int k = 0; k < size_3; k += block_size_3) {
#pragma HLS PIPELINE
                    for (int l = 0; l < block_size_1; l++) {
                        for (int m = 0; m < block_size_2; m++) {
                            for (int n = 0; n < block_size_3; n++) {
#pragma HLS UNROLL
                                y_out[i + l][j + m][k + n] = x_in[i + l][j + m][k + n] * const_val;
                            }
                        }
                    }
                }
            }
        }
    }
};

// ElementWiseAdd1DConst
template <
    typename T,
    const int size,
    const int block_size = 1>
class ElementWiseAdd1DConst {
public:
    void forward(
        T x_in[size],
        T y_out[size],
        const T const_val) {
    #pragma HLS inline off
        for (int i = 0; i < size; i += block_size) {
    #pragma HLS PIPELINE
            for (int j = 0; j < block_size; j++) {
    #pragma HLS UNROLL
                y_out[i + j] = x_in[i + j] + const_val;
            }
        }
    }
};

// ElementWiseAdd2DConst
template <
    typename T,
    const int size_1,
    const int size_2,
    const int block_size_1 = 1,
    const int block_size_2 = 1>
class ElementWiseAdd2DConst {
public:
    void forward(
        T x_in[size_1][size_2],
        T y_out[size_1][size_2],
        const T const_val) {
#pragma HLS inline off
        for (int i = 0; i < size_1; i += block_size_1) {
            for (int j = 0; j < size_2; j += block_size_2) {
#pragma HLS PIPELINE
                for (int k = 0; k < block_size_1; k++) {
                    for (int l = 0; l < block_size_2; l++) {
#pragma HLS UNROLL
                        y_out[i + k][j + l] = x_in[i + k][j + l] + const_val;
                    }
                }
            }
        }
    }
};

// ElementWiseAdd3DConst
template <
    typename T,
    const int size_1,
    const int size_2,
    const int size_3,
    const int block_size_1 = 1,
    const int block_size_2 = 1,
    const int block_size_3 = 1>
class ElementWiseAdd3DConst {
public:
    void forward(
        T x_in[size_1][size_2][size_3],
        T y_out[size_1][size_2][size_3],
        const T const_val) {
#pragma HLS inline off
        for (int i = 0; i < size_1; i += block_size_1) {
            for (int j = 0; j < size_2; j += block_size_2) {
                for (int k = 0; k < size_3; k += block_size_3) {
#pragma HLS PIPELINE
                    for (int l = 0; l < block_size_1; l++) {
                        for (int m = 0; m < block_size_2; m++) {
                            for (int n = 0; n < block_size_3; n++) {
#pragma HLS UNROLL
                                y_out[i + l][j + m][k + n] = x_in[i + l][j + m][k + n] + const_val;
                            }
                        }
                    }
                }
            }
        }
    }
};


template <
    typename T,
    const int size_0,
    const int size_1,
    const int size_2,
    const int dim,
    const int index>
class Select_3_2 {

public:
    void forward_dim_0(
        T input[size_0][size_1][size_2],
        T output[size_1][size_2]) {
#pragma HLS inline off
        for (int i = 0; i < size_1; i++) {
            for (int j = 0; j < size_2; j++) {
                output[i][j] = input[index][i][j];
            }
        }
    }

    void forward_dim_1(
        T input[size_0][size_1][size_2],
        T output[size_0][size_2]) {
#pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_2; j++) {
                output[i][j] = input[i][index][j];
            }
        }
    }

    void forward_dim_2(
        T input[size_0][size_1][size_2],
        T output[size_0][size_1]) {
#pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_1; j++) {
                output[i][j] = input[i][j][index];
            }
        }
    }
};

// select_2_1
template <
    typename T,
    const int size_0,
    const int size_1,
    const int dim,
    const int index>
class Select_2_1 {
public:
    void forward_dim_0(
        T input[size_0][size_1],
        T output[size_1]) {
#pragma HLS inline off
        for (int i = 0; i < size_1; i++) {
            output[i] = input[index][i];
        }
    }

    void forward_dim_1(
        T input[size_0][size_1],
        T output[size_0]) {
#pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            output[i] = input[i][index];
        }
    }
};

// unsqueeze_1_2
template <
    typename T,
    const int size_0,
    const int dim>
class Unsqueeze_1_2 {

public:
    void forward_dim_0(
        T input[size_0],
        T output[1][size_0]) {
#pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            output[0][i] = input[i];
        }
    }

    void forward_dim_1(
        T input[size_0],
        T output[size_0][1]) {
#pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            output[i][0] = input[i];
        }
    }

    void forward_dim_neg_2(
        T input[size_0],
        T output[1][size_0]) {
        forward_dim_0(input, output);
    }

    void foward_dim_neg_1(
        T input[size_0],
        T output[size_0][1]) {
        forward_dim_1(input, output);
    }
};

// transpose 2
template <
    typename T,
    const int size_0,
    const int size_1>
class Transpose_2 {

public:
    void forward(
        T input[size_0][size_1],
        T output[size_1][size_0]) {
#pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_1; j++) {
                output[j][i] = input[i][j];
            }
        }
    }
};

// template function to brodcast / copy 1 array to N arrays
// use variadic template to support N arrays
// all arrays are 1d arrays
// template <
//     typename T,
//     const int size_0,
//     typename... Args>
// void broadcast_1d(
//     T input[size_0],
//     T output_0[size_0],
//     Args&... args) {
// #pragma HLS inline off
//     for (int i = 0; i < size_0; i++) {
//         output_0[i] = input[i];
//     }
//     if constexpr (sizeof...(args) > 0) {
//         broadcast_1d<T, size_0>(input, args...);
//     }
// }

template <
    typename T,
    const int size_0>
class Broadcast_1d {

public:

    void broadcast_1d_1(
        T input[size_0],
        T output_0[size_0]) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            output_0[i] = input[i];
        }
    }

    void broadcast_1d_2(
        T input[size_0],
        T output_0[size_0],
        T output_1[size_0]
    ) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            output_0[i] = input[i];
            output_1[i] = input[i];
        }
    }

    void broadcast_1d_3(
        T input[size_0],
        T output_0[size_0],
        T output_1[size_0],
        T output_2[size_0]
    ) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            output_0[i] = input[i];
            output_1[i] = input[i];
            output_2[i] = input[i];
        }
    }

    void broadcast_1d_4(
        T input[size_0],
        T output_0[size_0],
        T output_1[size_0],
        T output_2[size_0],
        T output_3[size_0]
    ) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            output_0[i] = input[i];
            output_1[i] = input[i];
            output_2[i] = input[i];
            output_3[i] = input[i];
        }
    }
};

template <
    typename T,
    const int size_0,
    const int size_1>
class Broadcast_2d {

public:
    
    void broadcast_2d_1(
        T input[size_0][size_1],
        T output_0[size_0][size_1]) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_1; j++) {
                output_0[i][j] = input[i][j];
            }
        }
    }

    void broadcast_2d_2(
        T input[size_0][size_1],
        T output_0[size_0][size_1],
        T output_1[size_0][size_1]
    ) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_1; j++) {
                output_0[i][j] = input[i][j];
                output_1[i][j] = input[i][j];
            }
        }
    }

    void broadcast_2d_3(
        T input[size_0][size_1],
        T output_0[size_0][size_1],
        T output_1[size_0][size_1],
        T output_2[size_0][size_1]
    ) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_1; j++) {
                output_0[i][j] = input[i][j];
                output_1[i][j] = input[i][j];
                output_2[i][j] = input[i][j];
            }
        }
    }

    void broadcast_2d_4(
        T input[size_0][size_1],
        T output_0[size_0][size_1],
        T output_1[size_0][size_1],
        T output_2[size_0][size_1],
        T output_3[size_0][size_1]
    ) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_1; j++) {
                output_0[i][j] = input[i][j];
                output_1[i][j] = input[i][j];
                output_2[i][j] = input[i][j];
                output_3[i][j] = input[i][j];
            }
        }
    }
};

template<
    typename T,
    const int size_0,
    const int size_1,
    const int size_2>
class Broadcast_3d {

public:

    void broadcast_3d_1(
        T input[size_0][size_1][size_2],
        T output_0[size_0][size_1][size_2]) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_1; j++) {
                for (int k = 0; k < size_2; k++) {
                    output_0[i][j][k] = input[i][j][k];
                }
            }
        }
    }

    void broadcast_3d_2(
        T input[size_0][size_1][size_2],
        T output_0[size_0][size_1][size_2],
        T output_1[size_0][size_1][size_2]
    ) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_1; j++) {
                for (int k = 0; k < size_2; k++) {
                    output_0[i][j][k] = input[i][j][k];
                    output_1[i][j][k] = input[i][j][k];
                }
            }
        }
    }

    void broadcast_3d_3(
        T input[size_0][size_1][size_2],
        T output_0[size_0][size_1][size_2],
        T output_1[size_0][size_1][size_2],
        T output_2[size_0][size_1][size_2]
    ) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_1; j++) {
                for (int k = 0; k < size_2; k++) {
                    output_0[i][j][k] = input[i][j][k];
                    output_1[i][j][k] = input[i][j][k];
                    output_2[i][j][k] = input[i][j][k];
                }
            }
        }
    }

    void broadcast_3d_4(
        T input[size_0][size_1][size_2],
        T output_0[size_0][size_1][size_2],
        T output_1[size_0][size_1][size_2],
        T output_2[size_0][size_1][size_2],
        T output_3[size_0][size_1][size_2]
    ) {
    #pragma HLS inline off
        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_1; j++) {
                for (int k = 0; k < size_2; k++) {
                    output_0[i][j][k] = input[i][j][k];
                    output_1[i][j][k] = input[i][j][k];
                    output_2[i][j][k] = input[i][j][k];
                    output_3[i][j][k] = input[i][j][k];
                }
            }
        }
    }
};

template<
    typename T,
    const int size_0>
void neg_1d(
    T input[size_0],
    T output[size_0]) {
#pragma HLS inline off
    for (int i = 0; i < size_0; i++) {
        output[i] = -input[i];
    }
}

template<
    typename T,
    const int size_0,
    const int size_1>
void neg_2d(
    T input[size_0][size_1],
    T output[size_0][size_1]) {
#pragma HLS inline off
    for (int i = 0; i < size_0; i++) {
        for (int j = 0; j < size_1; j++) {
            output[i][j] = -input[i][j];
        }
    }
}

template<
    typename T,
    const int size_0,
    const int size_1,
    const int size_2>
void neg_3d(
    T input[size_0][size_1][size_2],
    T output[size_0][size_1][size_2]) {
#pragma HLS inline off
    for (int i = 0; i < size_0; i++) {
        for (int j = 0; j < size_1; j++) {
            for (int k = 0; k < size_2; k++) {
                output[i][j][k] = -input[i][j][k];
            }
        }
    }
}

template <
    typename T,
    const int size_0,
    const int size_1,
    const int size_2,
    const int block_size_0=1,
    const int block_size_1=1,
    const int block_size_2=1>
class MM{

    static_assert(size_0 > 0, "size_0 must be greater than 0");
    static_assert(size_1 > 0, "size_1 must be greater than 0");
    static_assert(size_2 > 0, "size_2 must be greater than 0");
    static_assert(block_size_0 > 0, "block_size_0 must be greater than 0");
    static_assert(block_size_1 > 0, "block_size_1 must be greater than 0");
    static_assert(block_size_2 > 0, "block_size_2 must be greater than 0");
    static_assert(size_0 % block_size_0 == 0, "size_0 must be a multiple of block_size_0");
    static_assert(size_1 % block_size_1 == 0, "size_1 must be a multiple of block_size_1");
    static_assert(size_2 % block_size_2 == 0, "size_2 must be a multiple of block_size_2");

public:
    void forward(
        T a[size_0][size_1],
        T b[size_1][size_2],
        T c[size_0][size_2]){
#pragma HLS inline off

        for (int i = 0; i < size_0; i++) {
            for (int j = 0; j < size_2; j++) {
                T sum = T(0.0);
                for (int k = 0; k < size_1; k++) {
                    sum += a[i][k] * b[k][j];
                }
                c[i][j] = sum;
            }
        }
    }
};

template <
    typename T,
    const int size_1,
    const int size_2,
    const int block_size_1 = 1,
    const int block_size_2 = 1>
class ElementWiseAdd2DCasting {
    
        static_assert(size_1 > 0, "size_1 must be greater than 0");
        static_assert(size_2 > 0, "size_2 must be greater than 0");
        static_assert(block_size_1 > 0, "block_size_1 must be greater than 0");
        static_assert(block_size_2 > 0, "block_size_2 must be greater than 0");
        static_assert(size_1 % block_size_1 == 0, "size_1 must be a multiple of block_size_1");
        static_assert(size_2 % block_size_2 == 0, "size_2 must be a multiple of block_size_2");

public:
    void forward_cast_dim_0(
        T a[size_1][size_2],
        T b[1][size_2],
        T c[size_1][size_2]) {

    #pragma HLS ARRAY_PARTITION variable = a type = cyclic factor = block_size_1 dim = 1
    #pragma HLS ARRAY_PARTITION variable = b type = cyclic factor = block_size_1 dim = 1

    #pragma HLS ARRAY_PARTITION variable = a type = cyclic factor = block_size_2 dim = 2
    #pragma HLS ARRAY_PARTITION variable = b type = cyclic factor = block_size_2 dim = 2
#pragma HLS inline off

        // for (int i = 0; i < size_1; i++) {
        //     for (int j = 0; j < size_2; j++) {
        //         c[i][j] = a[i][j] + b[0][j];
        //     }
        // }

        // tile loops
        for (int i = 0; i < size_1; i += block_size_1) {
            for (int j = 0; j < size_2; j += block_size_2) {
                #pragma HLS PIPELINE
                for (int ii = i; ii < i + block_size_1; ii++) {
                    for (int jj = j; jj < j + block_size_2; jj++) {
                        #pragma HLS UNROLL
                        c[ii][jj] = a[ii][jj] + b[0][jj];
                    }
                }
            }
        }
    }

    void forward_cast_dim_1(
        T a[size_1][size_2],
        T b[size_1][1],
        T c[size_1][size_2]) {
#pragma HLS inline off

#pragma HLS ARRAY_PARTITION variable = a type = cyclic factor = block_size_1 dim = 1
#pragma HLS ARRAY_PARTITION variable = b type = cyclic factor = block_size_1 dim = 1

#pragma HLS ARRAY_PARTITION variable = a type = cyclic factor = block_size_2 dim = 2
#pragma HLS ARRAY_PARTITION variable = b type = cyclic factor = block_size_2 dim = 2

        // tile loops
        for (int i = 0; i < size_1; i += block_size_1) {
            for (int j = 0; j < size_2; j += block_size_2) {
                #pragma HLS PIPELINE
                for (int ii = i; ii < i + block_size_1; ii++) {
                    for (int jj = j; jj < j + block_size_2; jj++) {
                        #pragma HLS UNROLL
                        c[ii][jj] = a[ii][jj] + b[ii][0];
                    }
                }
            }
        }
    }
};


// Neg1D
template <
    typename T,
    const int size_0,
    const int block_size_0 = 1>
class Neg1D {
    
    static_assert(size_0 > 0, "size_0 must be greater than 0");
    static_assert(block_size_0 > 0, "block_size_0 must be greater than 0");
    static_assert(size_0 % block_size_0 == 0, "size_0 must be a multiple of block_size_0");

public:

    void forward(
        T a[size_0],
        T b[size_0]) {
        #pragma HLS inline off
    
        for(int i = 0; i < size_0; i+= block_size_0) {
            #pragma HLS PIPELINE
            for(int ii = i; ii < i + block_size_0; ii++) {
                #pragma HLS UNROLL
                b[ii] = -a[ii];
            }
        }
    }
};

// Neg2D
template <
    typename T,
    const int size_0,
    const int size_1,
    const int block_size_0 = 1,
    const int block_size_1 = 1>
class Neg2D {
        
        static_assert(size_0 > 0, "size_0 must be greater than 0");
        static_assert(size_1 > 0, "size_1 must be greater than 0");
        static_assert(block_size_0 > 0, "block_size_0 must be greater than 0");
        static_assert(block_size_1 > 0, "block_size_1 must be greater than 0");
        static_assert(size_0 % block_size_0 == 0, "size_0 must be a multiple of block_size_0");
        static_assert(size_1 % block_size_1 == 0, "size_1 must be a multiple of block_size_1");

public:

    void forward(
        T a[size_0][size_1],
        T b[size_0][size_1]) {
        #pragma HLS inline off

        for(int i = 0; i < size_0; i+= block_size_0) {
            for(int j = 0; j < size_1; j+= block_size_1) {
                #pragma HLS PIPELINE
                for(int ii = i; ii < i + block_size_0; ii++) {
                    for(int jj = j; jj < j + block_size_1; jj++) {
                        #pragma HLS UNROLL
                        b[ii][jj] = -a[ii][jj];
                    }
                }
            }
        }
    }
};


// Neg3D
template <
    typename T,
    const int size_0,
    const int size_1,
    const int size_2,
    const int block_size_0 = 1,
    const int block_size_1 = 1,
    const int block_size_2 = 1>
class Neg3D {
        
        static_assert(size_0 > 0, "size_0 must be greater than 0");
        static_assert(size_1 > 0, "size_1 must be greater than 0");
        static_assert(size_2 > 0, "size_2 must be greater than 0");
        static_assert(block_size_0 > 0, "block_size_0 must be greater than 0");
        static_assert(block_size_1 > 0, "block_size_1 must be greater than 0");
        static_assert(block_size_2 > 0, "block_size_2 must be greater than 0");
        static_assert(size_0 % block_size_0 == 0, "size_0 must be a multiple of block_size_0");
        static_assert(size_1 % block_size_1 == 0, "size_1 must be a multiple of block_size_1");
        static_assert(size_2 % block_size_2 == 0, "size_2 must be a multiple of block_size_2");

public:
    
    void forward(
        T a[size_0][size_1][size_2],
        T b[size_0][size_1][size_2]) {
        #pragma HLS inline off

        for(int i = 0; i < size_0; i+= block_size_0) {
            for(int j = 0; j < size_1; j+= block_size_1) {
                for(int k = 0; k < size_2; k+= block_size_2) {
                    #pragma HLS PIPELINE
                    for(int ii = i; ii < i + block_size_0; ii++) {
                        for(int jj = j; jj < j + block_size_1; jj++) {
                            for(int kk = k; kk < k + block_size_2; kk++) {
                                #pragma HLS UNROLL
                                b[ii][jj][kk] = -a[ii][jj][kk];
                            }
                        }
                    }
                }
            }
        }
    }
};