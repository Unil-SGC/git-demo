using Plots
using BenchmarkTools
using CUDA

function memcopy_ap!(A, B, C, s)
    A .= B .+ s .* C
    return
end

function memcopy_kp!(A, B, C, s)
    nx, ny = size(A)
    for iy in 1:ny    #= Threads.@threads =#
        for ix in 1:nx
            A[ix, iy] = B[ix, iy] + s * C[ix, iy]
        end
    end
    return
end

function memcopy_gp!(A, B, C, s)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    A[ix, iy] = B[ix, iy] + s * C[ix, iy]
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
        threads = (16, 16)
        blocks = (nx ÷ threads[1], ny ÷ threads[2])

        t_ap = @belapsed memcopy_ap!($A, $B, $C, $s)
        # t_kp = @belapsed memcopy_kp!($A, $B, $C, $s)
        t_kp = @belapsed begin @cuda CUDA.@sync blocks=$blocks threads=$threads memcopy_gp!($A, $B, $C, $s) end

        N = 3 / 1e9 * nx * ny * sizeof(eltype(A))
        println("Memory copy benchmark (nx = ny = $(nx)):")
        println("  Peak memory throughput (ap): $(round(N / t_ap, sigdigits=3)) GB/s")
        println("  Peak memory throughput (kp): $(round(N / t_kp, sigdigits=3)) GB/s")
        push!(n_vec, nx)
        push!(t_vec, N / t_kp)
    end
    p = plot(n_vec, t_vec)
    display(p)
    # png(p, "out.png")
    return
end

benchmark(; nexp=2)
