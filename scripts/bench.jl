using Plots
using BenchmarkTools
using CUDA

function memcopy_ap!(A, B, C, s)
    @inbounds A .= B .+ s .* C
    return
end

function memcopy_kp!(A, B, C, s)
    nx, ny = size(A)
    @inbounds Threads.@threads for iy in 1:ny
        for ix in 1:nx
            A[ix, iy] = B[ix, iy] + s * C[ix, iy]
        end
    end
    return
end

function memcopy_gp!(A, B, C, s)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    @inbounds A[ix, iy] = B[ix, iy] + s * C[ix, iy]
    return
end

function benchmark(; nexp=3)
    n_vec = []
    t_vec = []
    for ires in 32 .* 2 .^ (1:nexp)
        nx = ny = ires
        s = 2.0
        A = zeros(Float64, nx, ny)
        B = ones(Float64, nx, ny)
        C = rand(Float64, nx, ny)
        A_d = CuArray(A)
        B_d = CuArray(B)
        C_d = CuArray(C)
        threads = (16, 16)
        blocks = (nx ÷ threads[1], ny ÷ threads[2])

        # CPU
        t_ap = @belapsed memcopy_ap!($A, $B, $C, $s)
        t_kp = @belapsed memcopy_kp!($A, $B, $C, $s)

        # GPU
        t_ap_gpu = @belapsed begin CUDA.@sync memcopy_ap!($A_d, $B_d, $C_d, $s) end
        t_kp_gpu = @belapsed begin CUDA.@sync @cuda blocks=$blocks threads=$threads memcopy_gp!($A_d, $B_d, $C_d, $s) end

        N = 3 / 1e9 * nx * ny * sizeof(eltype(A))
        println("Memory copy benchmark (nx = ny = $(nx)):")
        println("  CPU Peak memory throughput (ap): $(round(N / t_ap, sigdigits=3)) GB/s")
        println("  CPU Peak memory throughput (kp): $(round(N / t_kp, sigdigits=3)) GB/s")
        println("  GPU Peak memory throughput (ap): $(round(N / t_ap_gpu, sigdigits=3)) GB/s")
        println("  GPU Peak memory throughput (kp): $(round(N / t_kp_gpu, sigdigits=3)) GB/s")
        push!(n_vec, nx)
        push!(t_vec, N / t_kp)
    end
    # p = plot(n_vec, t_vec)
    # display(p)
    # png(p, "out.png")
    return
end

benchmark(; nexp=8)
