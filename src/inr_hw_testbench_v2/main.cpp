#include "./main.h"

float rand_min_max(float min, float max) {
    float r = (float)rand() / (float)RAND_MAX;
    return min + r * (max - min);
}

template <int M>
void rand_array_init_1d(float x[M], float min, float max) {
    for (int i = 0; i < M; i++) {
        x[i] = rand_min_max(min, max);
    }
}

template <int M, int N>
void rand_array_init_2d(float x[M][N], float min, float max) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            x[i][j] = rand_min_max(min, max);
        }
    }
}

template <int M, int N, int O>
void rand_array_init_3d(float x[M][N][O], float min, float max) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < O; k++) {
                x[i][j][k] = rand_min_max(min, max);
            }
        }
    }
}

template <typename T, int M>
void print_array_1d(T x[M]) {

    float x_cast[M]; 
    cast_array_1d<T, float, M>(x, x_cast);

    if (M < 6) {
        std::cout << "[ ";
        for (int i = 0; i < M; i++) {
            printf(" %010.6f ", x_cast[i]);
        }
        std::cout << " ]";
    } else {
        std::cout << "[ ";
        for (int i = 0; i < 3; i++) {
            printf(" %010.6f ", x_cast[i]);
        }
        std::cout << " ... ";
        for (int i = M - 3; i < M; i++) {
            printf(" %010.6f ", x_cast[i]);
        }
        std::cout << " ]";
    }
}

template <typename T, int M, int N>
void print_array_2d(T x[M][N]) {

    float x_cast[M][N];
    cast_array_2d<T, float, M, N>(x, x_cast);

    if ( (M < 6 && N < 6) || (M < 6 && N >= 6)) {
        std::cout << "[ ";
        for (int i = 0; i < M; i++) {
            if(i > 0) {
                std::cout << "  ";
            }
            print_array_1d<float, N>(x_cast[i]);
            if (i < M - 1) {
                std::cout << std::endl;
            }
        }
        std::cout << " ]" << std::endl;
    }
    else{
        std::cout << "[ ";
        for (int i = 0; i < 3; i++) {
            if(i > 0) {
                std::cout << "  ";
            }
            print_array_1d<float, N>(x_cast[i]);
            if (i < M - 1) {
                std::cout << std::endl;
            }
        }
        std::cout << "  ." << std::endl;
        std::cout << "  ." << std::endl;
        std::cout << "  ." << std::endl;
        for (int i = M - 3; i < M; i++) {
            if(i > 0) {
                std::cout << "  ";
            }
            print_array_1d<float, N>(x_cast[i]);
            if (i < M - 1) {
                std::cout << std::endl;
            }
        }
        std::cout << " ]" << std::endl;
    }
}

bool test_load_stream() {

    const int M = 25;
    const int N = 50;
    const int O = 100;

    const int B = 7;

    float x_in_1d[M];
    float x_in_2d[M][N];
    float x_in_3d[M][N][O];

    rand_array_init_1d<M>(x_in_1d, -1, 1);
    rand_array_init_2d<M, N>(x_in_2d, -1, 1);
    rand_array_init_3d<M, N, O>(x_in_3d, -1, 1);

    float x_out_1d_gold[M];
    float x_out_2d_gold[M][N];
    float x_out_3d_gold[M][N][O];

    for (int i = 0; i < M; i++) {
        x_out_1d_gold[i] = x_in_1d[i];
        for (int j = 0; j < N; j++) {
            x_out_2d_gold[i][j] = x_in_2d[i][j];
            for (int k = 0; k < O; k++) {
                x_out_3d_gold[i][j][k] = x_in_3d[i][j][k];
            }
        }
    }

    F_TYPE x_in_1d_fixed [M];
    F_TYPE x_in_2d_fixed [M][N];
    F_TYPE x_in_3d_fixed [M][N][O];

    cast_array_1d<float, F_TYPE, M>(x_in_1d, x_in_1d_fixed);
    cast_array_2d<float, F_TYPE, M, N>(x_in_2d, x_in_2d_fixed);
    cast_array_3d<float, F_TYPE, M, N, O>(x_in_3d, x_in_3d_fixed);

    F_TYPE x_out_1d_kernel_fixed[M];
    F_TYPE x_out_2d_kernel_fixed[M][N];
    F_TYPE x_out_3d_kernel_fixed[M][N][O];

    typedef array_stream<F_TYPE, array_shape<M>, B> T_array_stream_1d;
    typedef array_stream<F_TYPE, array_shape<M, N>, B> T_array_stream_2d;
    typedef array_stream<F_TYPE, array_shape<M, N, O>, B> T_array_stream_3d;

    T_array_stream_1d x_1d_stream;
    T_array_stream_2d x_2d_stream;
    T_array_stream_3d x_3d_stream;

    array_1d_to_array_stream<F_TYPE, M, B>(x_in_1d_fixed, x_1d_stream);
    array_2d_to_array_stream<F_TYPE, M, N, B>(x_in_2d_fixed, x_2d_stream);
    array_3d_to_array_stream<F_TYPE, M, N, O, B>(x_in_3d_fixed, x_3d_stream);

    array_stream_to_array_1d<F_TYPE, M, B>(x_1d_stream, x_out_1d_kernel_fixed);
    array_stream_to_array_2d<F_TYPE, M, N, B>(x_2d_stream, x_out_2d_kernel_fixed);
    array_stream_to_array_3d<F_TYPE, M, N, O, B>(x_3d_stream, x_out_3d_kernel_fixed);

    float x_out_1d_kernel[M];
    float x_out_2d_kernel[M][N];
    float x_out_3d_kernel[M][N][O];

    cast_array_1d<F_TYPE, float, M>(x_out_1d_kernel_fixed, x_out_1d_kernel);
    cast_array_2d<F_TYPE, float, M, N>(x_out_2d_kernel_fixed, x_out_2d_kernel);
    cast_array_3d<F_TYPE, float, M, N, O>(x_out_3d_kernel_fixed, x_out_3d_kernel);

    bool pass = true;
    float eps = 1e-4;

    pass &= compare_array_1d<float, M>(x_out_1d_kernel, x_out_1d_gold, eps);
    pass &= compare_array_2d<float, M, N>(x_out_2d_kernel, x_out_2d_gold, eps);
    pass &= compare_array_3d<float, M, N, O>(x_out_3d_kernel, x_out_3d_gold, eps);

    bool empty_stream_1d = x_1d_stream.data.empty();
    bool empty_stream_2d = x_2d_stream.data.empty();
    bool empty_stream_3d = x_3d_stream.data.empty();
    bool empty_streams = empty_stream_1d && empty_stream_2d && empty_stream_3d;

    pass &= empty_streams;

    return pass;
}

bool test_copy_stream(){
    const int M = 25;
    const int N = 50;
    const int O = 100;

    const int B = 7;

    float x_in_3d[M][N][O];
    rand_array_init_3d<M, N, O>(x_in_3d, -1, 1);

    F_TYPE x_in_3d_fixed[M][N][O];
    cast_array_3d<float, F_TYPE, M, N, O>(x_in_3d, x_in_3d_fixed);

    float x_out_0_3d[M][N][O];
    float x_out_1_3d[M][N][O];
    float x_out_2_3d[M][N][O];
    float x_out_3_3d[M][N][O];

    F_TYPE x_out_0_3d_kernel_fixed[M][N][O];
    F_TYPE x_out_1_3d_kernel_fixed[M][N][O];
    F_TYPE x_out_2_3d_kernel_fixed[M][N][O];
    F_TYPE x_out_3_3d_kernel_fixed[M][N][O];

    typedef array_stream<F_TYPE, array_shape<M, N, O>, B> T_array_stream_3d;

    T_array_stream_3d x_in_3d_stream;
    T_array_stream_3d x_out_0_3d_stream;
    T_array_stream_3d x_out_1_3d_stream;
    T_array_stream_3d x_out_2_3d_stream;
    T_array_stream_3d x_out_3_3d_stream;

    array_3d_to_array_stream<F_TYPE, M, N, O, B>(x_in_3d_fixed, x_in_3d_stream);

    copy_stream<T_array_stream_3d>(x_in_3d_stream, x_out_0_3d_stream, x_out_1_3d_stream, x_out_2_3d_stream, x_out_3_3d_stream);

    array_stream_to_array_3d<F_TYPE, M, N, O, B>(x_out_0_3d_stream, x_out_0_3d_kernel_fixed);
    array_stream_to_array_3d<F_TYPE, M, N, O, B>(x_out_1_3d_stream, x_out_1_3d_kernel_fixed);
    array_stream_to_array_3d<F_TYPE, M, N, O, B>(x_out_2_3d_stream, x_out_2_3d_kernel_fixed);
    array_stream_to_array_3d<F_TYPE, M, N, O, B>(x_out_3_3d_stream, x_out_3_3d_kernel_fixed);

    cast_array_3d<F_TYPE, float, M, N, O>(x_out_0_3d_kernel_fixed, x_out_0_3d);
    cast_array_3d<F_TYPE, float, M, N, O>(x_out_1_3d_kernel_fixed, x_out_1_3d);
    cast_array_3d<F_TYPE, float, M, N, O>(x_out_2_3d_kernel_fixed, x_out_2_3d);
    cast_array_3d<F_TYPE, float, M, N, O>(x_out_3_3d_kernel_fixed, x_out_3_3d);

    bool pass = true;
    float eps = 1e-4;

    pass &= compare_array_3d<float, M, N, O>(x_out_0_3d, x_in_3d, eps);
    pass &= compare_array_3d<float, M, N, O>(x_out_1_3d, x_in_3d, eps);
    pass &= compare_array_3d<float, M, N, O>(x_out_2_3d, x_in_3d, eps);
    pass &= compare_array_3d<float, M, N, O>(x_out_3_3d, x_in_3d, eps);

    bool empty_streams = true;
    empty_streams &= x_in_3d_stream.data.empty();
    empty_streams &= x_out_0_3d_stream.data.empty();
    empty_streams &= x_out_1_3d_stream.data.empty();
    empty_streams &= x_out_2_3d_stream.data.empty();
    empty_streams &= x_out_3_3d_stream.data.empty();

    pass &= empty_streams;

    return pass;
}

bool test_elementwise_functions(){

    const int M = 25;
    const int N = 50;
    const int O = 100;
    const int B = 7;

    const float EPS = 1e-4;

    std::vector<std::string> elementwise_function_names = {
        "elementwise_add",
        "elementwise_mul",
        "elementwise_negate",
        "elementwise_square",
        "elementwise_sin",
        "elementwise_cos"
    };

    std::vector<bool> pass_flags;

    for (int i = 0; i < elementwise_function_names.size(); i++) {
        std::string elementwise_function_name = elementwise_function_names[i];

        float a_1d[M];
        float a_2d[M][N];
        float a_3d[M][N][O];

        rand_array_init_1d<M>(a_1d, -1, 1);
        rand_array_init_2d<M, N>(a_2d, -1, 1);
        rand_array_init_3d<M, N, O>(a_3d, -1, 1);

        float b_1d[M];
        float b_2d[M][N];
        float b_3d[M][N][O];

        rand_array_init_1d<M>(b_1d, -1, 1);
        rand_array_init_2d<M, N>(b_2d, -1, 1);
        rand_array_init_3d<M, N, O>(b_3d, -1, 1);

        float c_1d_gold[M];
        float c_2d_gold[M][N];
        float c_3d_gold[M][N][O];

        if (elementwise_function_name == "elementwise_add") {
            // c = a + b
            for (int i = 0; i < M; i++) {
                c_1d_gold[i] = a_1d[i] + b_1d[i];
                for (int j = 0; j < N; j++) {
                    c_2d_gold[i][j] = a_2d[i][j] + b_2d[i][j];
                    for (int k = 0; k < O; k++) {
                        c_3d_gold[i][j][k] = a_3d[i][j][k] + b_3d[i][j][k];
                    }
                }
            }
        }
        if (elementwise_function_name == "elementwise_mul") {
            // c = a * b
            for (int i = 0; i < M; i++) {
                c_1d_gold[i] = a_1d[i] * b_1d[i];
                for (int j = 0; j < N; j++) {
                    c_2d_gold[i][j] = a_2d[i][j] * b_2d[i][j];
                    for (int k = 0; k < O; k++) {
                        c_3d_gold[i][j][k] = a_3d[i][j][k] * b_3d[i][j][k];
                    }
                }
            }
        }
        if (elementwise_function_name == "elementwise_negate") {
            // c = -a
            for (int i = 0; i < M; i++) {
                c_1d_gold[i] = -a_1d[i];
                for (int j = 0; j < N; j++) {
                    c_2d_gold[i][j] = -a_2d[i][j];
                    for (int k = 0; k < O; k++) {
                        c_3d_gold[i][j][k] = -a_3d[i][j][k];
                    }
                }
            }
        }
        if (elementwise_function_name == "elementwise_square") {
            // c = a^2
            for (int i = 0; i < M; i++) {
                c_1d_gold[i] = a_1d[i] * a_1d[i];
                for (int j = 0; j < N; j++) {
                    c_2d_gold[i][j] = a_2d[i][j] * a_2d[i][j];
                    for (int k = 0; k < O; k++) {
                        c_3d_gold[i][j][k] = a_3d[i][j][k] * a_3d[i][j][k];
                    }
                }
            }
        }
        if (elementwise_function_name == "elementwise_sin") {
            // c = sin(a)
            for (int i = 0; i < M; i++) {
                c_1d_gold[i] = sin(a_1d[i]);
                for (int j = 0; j < N; j++) {
                    c_2d_gold[i][j] = sin(a_2d[i][j]);
                    for (int k = 0; k < O; k++) {
                        c_3d_gold[i][j][k] = std::sin(a_3d[i][j][k]);
                    }
                }
            }
        }
        if (elementwise_function_name == "elementwise_cos") {
            // c = cos(a)
            for (int i = 0; i < M; i++) {
                c_1d_gold[i] = cos(a_1d[i]);
                for (int j = 0; j < N; j++) {
                    c_2d_gold[i][j] = cos(a_2d[i][j]);
                    for (int k = 0; k < O; k++) {
                        c_3d_gold[i][j][k] = std::cos(a_3d[i][j][k]);
                    }
                }
            }
        }

        F_TYPE a_1d_fixed[M];
        F_TYPE a_2d_fixed[M][N];
        F_TYPE a_3d_fixed[M][N][O];

        cast_array_1d<float, F_TYPE, M>(a_1d, a_1d_fixed);
        cast_array_2d<float, F_TYPE, M, N>(a_2d, a_2d_fixed);
        cast_array_3d<float, F_TYPE, M, N, O>(a_3d, a_3d_fixed);

        F_TYPE b_1d_fixed[M];
        F_TYPE b_2d_fixed[M][N];
        F_TYPE b_3d_fixed[M][N][O];

        cast_array_1d<float, F_TYPE, M>(b_1d, b_1d_fixed);
        cast_array_2d<float, F_TYPE, M, N>(b_2d, b_2d_fixed);
        cast_array_3d<float, F_TYPE, M, N, O>(b_3d, b_3d_fixed);

        typedef array_stream<F_TYPE, array_shape<M>, B> T_array_stream_1d;
        typedef array_stream<F_TYPE, array_shape<M, N>, B> T_array_stream_2d;
        typedef array_stream<F_TYPE, array_shape<M, N, O>, B> T_array_stream_3d;

        T_array_stream_1d a_1d_stream;
        T_array_stream_2d a_2d_stream;
        T_array_stream_3d a_3d_stream;

        T_array_stream_1d b_1d_stream;
        T_array_stream_2d b_2d_stream;
        T_array_stream_3d b_3d_stream;

        T_array_stream_1d c_1d_stream;
        T_array_stream_2d c_2d_stream;
        T_array_stream_3d c_3d_stream;

        array_1d_to_array_stream<F_TYPE, M, B>(a_1d_fixed, a_1d_stream);
        array_2d_to_array_stream<F_TYPE, M, N, B>(a_2d_fixed, a_2d_stream);
        array_3d_to_array_stream<F_TYPE, M, N, O, B>(a_3d_fixed, a_3d_stream);

        array_1d_to_array_stream<F_TYPE, M, B>(b_1d_fixed, b_1d_stream);
        array_2d_to_array_stream<F_TYPE, M, N, B>(b_2d_fixed, b_2d_stream);
        array_3d_to_array_stream<F_TYPE, M, N, O, B>(b_3d_fixed, b_3d_stream);

        if (elementwise_function_name == "elementwise_add") {
            elementwise_add<T_array_stream_1d>(a_1d_stream, b_1d_stream, c_1d_stream);
            elementwise_add<T_array_stream_2d>(a_2d_stream, b_2d_stream, c_2d_stream);
            elementwise_add<T_array_stream_3d>(a_3d_stream, b_3d_stream, c_3d_stream);
        }
        if (elementwise_function_name == "elementwise_mul") {
            elementwise_mul<T_array_stream_1d>(a_1d_stream, b_1d_stream, c_1d_stream);
            elementwise_mul<T_array_stream_2d>(a_2d_stream, b_2d_stream, c_2d_stream);
            elementwise_mul<T_array_stream_3d>(a_3d_stream, b_3d_stream, c_3d_stream);
        }
        if (elementwise_function_name == "elementwise_negate") {
            elementwise_negate<T_array_stream_1d>(a_1d_stream, c_1d_stream);
            elementwise_negate<T_array_stream_2d>(a_2d_stream, c_2d_stream);
            elementwise_negate<T_array_stream_3d>(a_3d_stream, c_3d_stream);

            empty_the_stream<T_array_stream_1d>(b_1d_stream);
            empty_the_stream<T_array_stream_2d>(b_2d_stream);
            empty_the_stream<T_array_stream_3d>(b_3d_stream);
        }
        if (elementwise_function_name == "elementwise_square") {
            elementwise_square<T_array_stream_1d>(a_1d_stream, c_1d_stream);
            elementwise_square<T_array_stream_2d>(a_2d_stream, c_2d_stream);
            elementwise_square<T_array_stream_3d>(a_3d_stream,c_3d_stream);

            empty_the_stream<T_array_stream_1d>(b_1d_stream);
            empty_the_stream<T_array_stream_2d>(b_2d_stream);
            empty_the_stream<T_array_stream_3d>(b_3d_stream);
        }
        if (elementwise_function_name == "elementwise_sin") {
            elementwise_sin<T_array_stream_1d>(a_1d_stream, c_1d_stream);
            elementwise_sin<T_array_stream_2d>(a_2d_stream, c_2d_stream);
            elementwise_sin<T_array_stream_3d>(a_3d_stream, c_3d_stream);

            empty_the_stream<T_array_stream_1d>(b_1d_stream);
            empty_the_stream<T_array_stream_2d>(b_2d_stream);
            empty_the_stream<T_array_stream_3d>(b_3d_stream);
        }
        if (elementwise_function_name == "elementwise_cos") {
            elementwise_cos<T_array_stream_1d>(a_1d_stream, c_1d_stream);
            elementwise_cos<T_array_stream_2d>(a_2d_stream, c_2d_stream);
            elementwise_cos<T_array_stream_3d>(a_3d_stream, c_3d_stream);

            empty_the_stream<T_array_stream_1d>(b_1d_stream);
            empty_the_stream<T_array_stream_2d>(b_2d_stream);
            empty_the_stream<T_array_stream_3d>(b_3d_stream);
        }

        F_TYPE c_1d_kernel_fixed[M];
        F_TYPE c_2d_kernel_fixed[M][N];
        F_TYPE c_3d_kernel_fixed[M][N][O];

        array_stream_to_array_1d<F_TYPE, M, B>(c_1d_stream, c_1d_kernel_fixed);
        array_stream_to_array_2d<F_TYPE, M, N, B>(c_2d_stream, c_2d_kernel_fixed);
        array_stream_to_array_3d<F_TYPE, M, N, O, B>(c_3d_stream, c_3d_kernel_fixed);

        float c_1d_kernel_gold[M];
        float c_2d_kernel_gold[M][N];
        float c_3d_kernel_gold[M][N][O];

        cast_array_1d<F_TYPE, float, M>(c_1d_kernel_fixed, c_1d_kernel_gold);
        cast_array_2d<F_TYPE, float, M, N>(c_2d_kernel_fixed, c_2d_kernel_gold);
        cast_array_3d<F_TYPE, float, M, N, O>(c_3d_kernel_fixed, c_3d_kernel_gold);

        bool pass = true;
        float eps = 1e-4;

        pass &= compare_array_1d<float, M>(c_1d_gold, c_1d_kernel_gold, EPS);
        pass &= compare_array_2d<float, M, N>(c_2d_gold, c_2d_kernel_gold, EPS);
        pass &= compare_array_3d<float, M, N, O>(c_3d_gold, c_3d_kernel_gold, EPS);

        bool empty_streams = true;
        empty_streams &= a_1d_stream.data.empty();
        empty_streams &= a_2d_stream.data.empty();
        empty_streams &= a_3d_stream.data.empty();

        empty_streams &= b_1d_stream.data.empty();
        empty_streams &= b_2d_stream.data.empty();
        empty_streams &= b_3d_stream.data.empty();

        empty_streams &= c_1d_stream.data.empty();
        empty_streams &= c_2d_stream.data.empty();
        empty_streams &= c_3d_stream.data.empty();

        pass &= empty_streams;

        pass_flags.push_back(pass);
    }

    bool pass_all = true;
    for (int i = 0; i < pass_flags.size(); i++) {
        pass_all &= pass_flags[i];
    }

    return pass_all;
}

bool test_transpose() {
    const int M = 25;
    const int N = 50;
    const int B = 7;

    const float EPS = 1e-4;

    float x_in[M][N];
    rand_array_init_2d<M, N>(x_in, -1, 1);

    float x_out_gold[N][M];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            x_out_gold[j][i] = x_in[i][j];
        }
    }

    F_TYPE x_in_fixed[M][N];
    cast_array_2d<float, F_TYPE, M, N>(x_in, x_in_fixed);

    typedef array_stream<F_TYPE, array_shape<M, N>, B> T_array_stream_in;
    typedef array_stream<F_TYPE, array_shape<N, M>, B> T_array_stream_out;

    T_array_stream_in x_in_stream;
    T_array_stream_out x_out_stream;

    array_2d_to_array_stream<F_TYPE, M, N, B>(x_in_fixed, x_in_stream);

    transpose_2d<T_array_stream_in, T_array_stream_out>(x_in_stream, x_out_stream);

    F_TYPE x_out_kernel_fixed[N][M];
    array_stream_to_array_2d<F_TYPE, N, M, B>(x_out_stream, x_out_kernel_fixed);

    float x_out_kernel[N][M];
    cast_array_2d<F_TYPE, float, N, M>(x_out_kernel_fixed, x_out_kernel);

    bool pass = true;
    pass &= compare_array_2d<float, N, M>(x_out_gold, x_out_kernel, EPS);

    bool empty_streams = true;
    empty_streams &= x_in_stream.data.empty();
    empty_streams &= x_out_stream.data.empty();

    pass &= empty_streams;

    return pass;
}

bool test_stream_block_adapter(){
    const int M = 25;
    const int B_in = 5;
    const int B_out = 7;

    const float EPS = 1e-4;

    float x_in[M];
    rand_array_init_1d<M>(x_in, -1, 1);

    float x_out_gold[M];
    for (int i = 0; i < M; i++) {
        x_out_gold[i] = x_in[i];
    }

    F_TYPE x_in_fixed[M];
    cast_array_1d<float, F_TYPE, M>(x_in, x_in_fixed);

    typedef array_stream<F_TYPE, array_shape<M>, B_in> T_array_stream_in;
    typedef array_stream<F_TYPE, array_shape<M>, B_out> T_array_stream_out;

    T_array_stream_in x_in_stream;
    T_array_stream_out x_out_stream;

    array_1d_to_array_stream<F_TYPE, M, B_in>(x_in_fixed, x_in_stream);

    stream_block_adapter<T_array_stream_in, T_array_stream_out>(x_in_stream, x_out_stream);

    F_TYPE x_out_kernel_fixed[M];
    array_stream_to_array_1d<F_TYPE, M, B_out>(x_out_stream, x_out_kernel_fixed);

    float x_out_kernel[M];
    cast_array_1d<F_TYPE, float, M>(x_out_kernel_fixed, x_out_kernel);

    bool pass = true;
    pass &= compare_array_1d<float, M>(x_out_gold, x_out_kernel, EPS);

    bool empty_streams = true;
    empty_streams &= x_in_stream.data.empty();
    empty_streams &= x_out_stream.data.empty();
    pass &= empty_streams;

    return pass;
}

bool test_select(){
    const int M_in = 25;
    const int N_in = 50;
    const int O_in = 75;

    const int DIM = 1;
    const int INDEX = 15;

    const int B = 13;

    const int M_out = M_in;
    const int O_out = O_in;

    const float EPS = 1e-4;

    float x_in[M_in][N_in][O_in];
    rand_array_init_3d<M_in, N_in, O_in>(x_in, -1, 1);

    float x_out_gold[M_out][O_out];
    for (int i = 0; i < M_out; i++) {
        for (int j = 0; j < O_out; j++) {
            x_out_gold[i][j] = x_in[i][INDEX][j];
        }
    }

    F_TYPE x_in_fixed[M_in][N_in][O_in];
    cast_array_3d<float, F_TYPE, M_in, N_in, O_in>(x_in, x_in_fixed);

    typedef array_stream<F_TYPE, array_shape<M_in, N_in, O_in>, B> T_array_stream_in;
    typedef array_stream<F_TYPE, array_shape<M_out, O_out>, B> T_array_stream_out;

    T_array_stream_in x_in_stream;
    T_array_stream_out x_out_stream;

    array_3d_to_array_stream<F_TYPE, M_in, N_in, O_in, B>(x_in_fixed, x_in_stream);

    select<DIM, INDEX, T_array_stream_in, T_array_stream_out>(x_in_stream, x_out_stream);

    F_TYPE x_out_kernel_fixed[M_out][O_out];
    array_stream_to_array_2d<F_TYPE, M_out, O_out, B>(x_out_stream, x_out_kernel_fixed);

    float x_out_kernel[M_out][O_out];
    cast_array_2d<F_TYPE, float, M_out, O_out>(x_out_kernel_fixed, x_out_kernel);

    bool pass = true;
    pass &= compare_array_2d<float, M_out, O_out>(x_out_gold, x_out_kernel, EPS);

    bool empty_streams = true;
    empty_streams &= x_in_stream.data.empty();
    empty_streams &= x_out_stream.data.empty();
    pass &= empty_streams;

    return pass;

}

bool test_repeat_singleton_dim_2d(){

    const int B = 7;

    const float EPS = 1e-4;
    bool global_pass = true;
    
    test_dim_0:
    {
        const int M_in = 1;
        const int N_in = 3;

        const int dim = 0;
        const int num = 10;

        const int M_out = M_in * num;
        const int N_out = N_in;

        float x_in[M_in][N_in];
        float x_out_gold[M_out][N_out];

        rand_array_init_2d<M_in, N_in>(x_in, -1, 1);

        for (int i = 0; i < M_out; i++) {
            for (int j = 0; j < N_out; j++) {
                x_out_gold[i][j] = x_in[0][j];
            }
        }

        F_TYPE x_in_fixed[M_in][N_in];
        cast_array_2d<float, F_TYPE, M_in, N_in>(x_in, x_in_fixed);

        typedef array_stream<F_TYPE, array_shape<M_in, N_in>, B> T_array_stream_in;
        typedef array_stream<F_TYPE, array_shape<M_out, N_out>, B> T_array_stream_out;

        T_array_stream_in x_in_stream;
        T_array_stream_out x_out_stream;

        array_2d_to_array_stream<F_TYPE, M_in, N_in, B>(x_in_fixed, x_in_stream);

        repeat_singleton_dim_2d<dim, num>(x_in_stream, x_out_stream);

        F_TYPE x_out_kernel_fixed[M_out][N_out];
        array_stream_to_array_2d<F_TYPE, M_out, N_out, B>(x_out_stream, x_out_kernel_fixed);

        float x_out_kernel[M_out][N_out];
        cast_array_2d<F_TYPE, float, M_out, N_out>(x_out_kernel_fixed, x_out_kernel);

        bool pass = true;
        pass &= compare_array_2d<float, M_out, N_out>(x_out_gold, x_out_kernel, EPS);

        bool empty_streams = true;
        empty_streams &= x_in_stream.data.empty();
        empty_streams &= x_out_stream.data.empty();
        pass &= empty_streams;

        global_pass &= pass;
    }

    test_dim_1:
    {
        const int M_in = 3;
        const int N_in = 1;

        const int dim = 1;
        const int num = 10;

        const int M_out = M_in;
        const int N_out = N_in * num;

        float x_in[M_in][N_in];
        float x_out_gold[M_out][N_out];

        rand_array_init_2d<M_in, N_in>(x_in, -1, 1);

        for (int i = 0; i < M_out; i++) {
            for (int j = 0; j < N_out; j++) {
                x_out_gold[i][j] = x_in[i][0];
            }
        }

        F_TYPE x_in_fixed[M_in][N_in];
        cast_array_2d<float, F_TYPE, M_in, N_in>(x_in, x_in_fixed);

        typedef array_stream<F_TYPE, array_shape<M_in, N_in>, B> T_array_stream_in;
        typedef array_stream<F_TYPE, array_shape<M_out, N_out>, B> T_array_stream_out;

        T_array_stream_in x_in_stream;
        T_array_stream_out x_out_stream;

        array_2d_to_array_stream<F_TYPE, M_in, N_in, B>(x_in_fixed, x_in_stream);

        repeat_singleton_dim_2d<dim, num>(x_in_stream, x_out_stream);

        F_TYPE x_out_kernel_fixed[M_out][N_out];
        array_stream_to_array_2d<F_TYPE, M_out, N_out, B>(x_out_stream, x_out_kernel_fixed);

        float x_out_kernel[M_out][N_out];
        cast_array_2d<F_TYPE, float, M_out, N_out>(x_out_kernel_fixed, x_out_kernel);

        bool pass = true;
        pass &= compare_array_2d<float, M_out, N_out>(x_out_gold, x_out_kernel, EPS);

        bool empty_streams = true;
        empty_streams &= x_in_stream.data.empty();
        empty_streams &= x_out_stream.data.empty();
        pass &= empty_streams;

        global_pass &= pass;
    }

    return global_pass;
}

bool test_mm(){
    const int M = 25;
    const int N = 50;
    const int O = 75;

    const int B = 13;

    const float EPS = 1e-4;

    float a_in[M][N];
    float b_in[N][O];
    float c_out_gold[M][O];

    rand_array_init_2d<M, N>(a_in, -1, 1);
    rand_array_init_2d<N, O>(b_in, -1, 1);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < O; j++) {
            c_out_gold[i][j] = 0;
            for (int k = 0; k < N; k++) {
                c_out_gold[i][j] += a_in[i][k] * b_in[k][j];
            }
        }
    }

    F_TYPE a_in_fixed[M][N];
    F_TYPE b_in_fixed[N][O];
    cast_array_2d<float, F_TYPE, M, N>(a_in, a_in_fixed);
    cast_array_2d<float, F_TYPE, N, O>(b_in, b_in_fixed);

    typedef array_stream<F_TYPE, array_shape<M, N>, B> T_array_stream_a;
    typedef array_stream<F_TYPE, array_shape<N, O>, B> T_array_stream_b;

    T_array_stream_a a_in_stream;
    T_array_stream_b b_in_stream;

    array_2d_to_array_stream<F_TYPE, M, N, B>(a_in_fixed, a_in_stream);
    array_2d_to_array_stream<F_TYPE, N, O, B>(b_in_fixed, b_in_stream);

    typedef array_stream<F_TYPE, array_shape<M, O>, B> T_array_stream_c;

    T_array_stream_c c_out_stream;

    mm<T_array_stream_a, T_array_stream_b, T_array_stream_c>(a_in_stream, b_in_stream, c_out_stream);

    F_TYPE c_out_kernel_fixed[M][O];
    array_stream_to_array_2d<F_TYPE, M, O, B>(c_out_stream, c_out_kernel_fixed);

    float c_out_kernel[M][O];
    cast_array_2d<F_TYPE, float, M, O>(c_out_kernel_fixed, c_out_kernel);

    bool pass = true;
    pass &= compare_array_2d<float, M, O>(c_out_gold, c_out_kernel, EPS);

    float error_mae = compute_mae_2d<M, O>(c_out_gold, c_out_kernel);
    std::cout << "MAE: " << error_mae << std::endl;

    bool empty_streams = true;
    empty_streams &= a_in_stream.data.empty();
    empty_streams &= b_in_stream.data.empty();
    empty_streams &= c_out_stream.data.empty();
    pass &= empty_streams;

    return pass;
}

bool test_mm_v2(){

    const int M = 25;
    const int N = 50;
    const int O = 75;

    const int B = 1;

    const int P_test = 13;

    const float EPS = 1e-4;

    float a_in[M][N];
    float b_in[N][O];
    float c_out_gold[M][O];

    rand_array_init_2d<M, N>(a_in, -1, 1);
    rand_array_init_2d<N, O>(b_in, -1, 1);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < O; j++) {
            c_out_gold[i][j] = 0;
            for (int k = 0; k < N; k++) {
                c_out_gold[i][j] += a_in[i][k] * b_in[k][j];
            }
        }
    }

    F_TYPE a_in_fixed[M][N];
    F_TYPE b_in_fixed[N][O];
    cast_array_2d<float, F_TYPE, M, N>(a_in, a_in_fixed);
    cast_array_2d<float, F_TYPE, N, O>(b_in, b_in_fixed);

    typedef array_stream<F_TYPE, array_shape<M, N>, B> T_array_stream_a;
    typedef array_stream<F_TYPE, array_shape<N, O>, B> T_array_stream_b;

    T_array_stream_a a_in_stream("a_in_stream");
    T_array_stream_b b_in_stream("b_in_stream");

    array_2d_to_array_stream<F_TYPE, M, N, B>(a_in_fixed, a_in_stream);
    array_2d_to_array_stream<F_TYPE, N, O, B>(b_in_fixed, b_in_stream);

    typedef array_stream<F_TYPE, array_shape<M, O>, B> T_array_stream_c;

    T_array_stream_c c_out_stream("c_out_stream");

    mm_v2<T_array_stream_a, T_array_stream_b, T_array_stream_c, P_test>(a_in_stream, b_in_stream, c_out_stream);

    F_TYPE c_out_kernel_fixed[M][O];
    array_stream_to_array_2d<F_TYPE, M, O, B>(c_out_stream, c_out_kernel_fixed);

    float c_out_kernel[M][O];
    cast_array_2d<F_TYPE, float, M, O>(c_out_kernel_fixed, c_out_kernel);

    bool pass = true;
    pass &= compare_array_2d<float, M, O>(c_out_gold, c_out_kernel, EPS);
    
    print_array_2d<float, M, O>(c_out_gold);
    std::cout << std::endl;
    print_array_2d<float, M, O>(c_out_kernel);

    float error_mae = compute_mae_2d<M, O>(c_out_gold, c_out_kernel);
    std::cout << "MAE: " << error_mae << std::endl;

    bool empty_streams = true;
    empty_streams &= a_in_stream.data.empty();
    empty_streams &= b_in_stream.data.empty();
    empty_streams &= c_out_stream.data.empty();
    pass &= empty_streams;

    return pass;
}

bool test_mm_v3(){

    const int M = 25;
    const int N = 50;
    const int O = 75;

    const int B = 2;

    const int P_test = 13;

    const float EPS = 1e-4;

    float a_in[M][N];
    float b_in[N][O];
    float c_out_gold[M][O];

    rand_array_init_2d<M, N>(a_in, -1, 1);
    rand_array_init_2d<N, O>(b_in, -1, 1);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < O; j++) {
            c_out_gold[i][j] = 0;
            for (int k = 0; k < N; k++) {
                c_out_gold[i][j] += a_in[i][k] * b_in[k][j];
            }
        }
    }

    F_TYPE a_in_fixed[M][N];
    F_TYPE b_in_fixed[N][O];
    cast_array_2d<float, F_TYPE, M, N>(a_in, a_in_fixed);
    cast_array_2d<float, F_TYPE, N, O>(b_in, b_in_fixed);

    typedef array_stream<F_TYPE, array_shape<M, N>, B> T_array_stream_a;
    typedef array_stream<F_TYPE, array_shape<N, O>, B> T_array_stream_b;

    T_array_stream_a a_in_stream("a_in_stream");
    T_array_stream_b b_in_stream("b_in_stream");

    array_2d_to_array_stream<F_TYPE, M, N, B>(a_in_fixed, a_in_stream);
    array_2d_to_array_stream<F_TYPE, N, O, B>(b_in_fixed, b_in_stream);

    typedef array_stream<F_TYPE, array_shape<M, O>, B> T_array_stream_c;

    T_array_stream_c c_out_stream("c_out_stream");

    mm_v3<T_array_stream_a, T_array_stream_b, T_array_stream_c>(a_in_stream, b_in_stream, c_out_stream);

    F_TYPE c_out_kernel_fixed[M][O];
    array_stream_to_array_2d<F_TYPE, M, O, B>(c_out_stream, c_out_kernel_fixed);

    float c_out_kernel[M][O];
    cast_array_2d<F_TYPE, float, M, O>(c_out_kernel_fixed, c_out_kernel);

    bool pass = true;
    pass &= compare_array_2d<float, M, O>(c_out_gold, c_out_kernel, EPS);
    
    print_array_2d<float, M, O>(c_out_gold);
    std::cout << std::endl;
    print_array_2d<float, M, O>(c_out_kernel);

    float error_mae = compute_mae_2d<M, O>(c_out_gold, c_out_kernel);
    std::cout << "MAE: " << error_mae << std::endl;

    bool empty_streams = true;
    empty_streams &= a_in_stream.data.empty();
    empty_streams &= b_in_stream.data.empty();
    empty_streams &= c_out_stream.data.empty();
    pass &= empty_streams;

    return pass;
}


void test_function(bool (*test_function)(), const char *test_name, std::ofstream &log_file) {
    bool results = test_function();
    std::string test_result_message = test_name + std::string(": ") + (results ? "PASS" : "FAIL") + std::string("\n");
    printf("%s", test_result_message.c_str());
    log_file << test_result_message;
}

int main() {
    printf("##########################\n");
    printf("### inr_hw_lib_test_v2 ###\n");
    printf("##########################\n");

    // create simple log.txt that can be passed around
    std::ofstream log_file;
    log_file.open("log.txt");

    test_function(test_load_stream, "test_load_stream", log_file);
    test_function(test_copy_stream, "test_copy_stream", log_file);
    test_function(test_elementwise_functions, "test_elementwise_functions", log_file);
    test_function(test_transpose, "test_transpose", log_file);
    test_function(test_stream_block_adapter, "test_stream_block_adapter", log_file);
    test_function(test_select, "test_select", log_file);
    test_function(test_repeat_singleton_dim_2d, "test_repeat_singleton_dim_2d", log_file);
    test_function(test_mm, "test_mm", log_file);
    test_function(test_mm_v2, "test_mm_v2", log_file);
    test_function(test_mm_v3, "test_mm_v3", log_file);

    log_file.close();

    return 0;
}