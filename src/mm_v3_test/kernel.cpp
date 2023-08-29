#include "kernel.h"

void kernel_mm_v3(F_TYPE a[25][50], F_TYPE b[50][75], F_TYPE c[25][75])
{
    const int M = 25;
    const int N = 50;
    const int O = 75;

    const int B = 2;

    typedef array_stream<F_TYPE, array_shape<M, N>, B> T_array_stream_a;
    typedef array_stream<F_TYPE, array_shape<N, O>, B> T_array_stream_b;

    T_array_stream_a a_in_stream("a_in_stream");
    T_array_stream_b b_in_stream("b_in_stream");

    array_2d_to_array_stream<F_TYPE, M, N, B>(a, a_in_stream);
    array_2d_to_array_stream<F_TYPE, N, O, B>(b, b_in_stream);

    typedef array_stream<F_TYPE, array_shape<M, O>, B> T_array_stream_c;

    T_array_stream_c c_out_stream("c_out_stream");

    mm_v3<T_array_stream_a, T_array_stream_b, T_array_stream_c>(a_in_stream, b_in_stream, c_out_stream);

    array_stream_to_array_2d<F_TYPE, M, O, B>(c_out_stream, c);
}
