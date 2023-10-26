using BenchmarkTools
using CUDA

function memcopy_gp!(A, B, C, s)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
    return
end

function benchmark(; nexp=(1:3))
    for ires in 32 .* 2 .^ nexp
        nx = ny = ires
        s = 2.0
        A_d = CuArray(zeros(Float64, nx, ny))
        B_d = CuArray(ones(Float64, nx, ny))
        C_d = CuArray(rand(Float64, nx, ny))
        threads = (16, 16)
        blocks = (nx ÷ threads[1], ny ÷ threads[2])

        t_kp_gpu = @belapsed begin CUDA.@sync @cuda blocks=$blocks threads=$threads memcopy_gp!($A_d, $B_d, $C_d, $s) end

        N = 3 / 1e9 * nx * ny * sizeof(eltype(A_d))
        println("Memory copy benchmark (nx = ny = $(nx)):")
        println("  GPU Peak memory throughput (kp): $(round(N / t_kp_gpu, sigdigits=3)) GB/s")
    end
    return
end

benchmark(; nexp=(8:9))
