# This is just a hack for "go to definition" to work in editor.
if false
    include("src/Burgers.jl")
    using .Burgers
end

using Burgers
using CairoMakie
using Lux
# using LuxCUDA
using MLUtils
using NNlib
using Optimisers
using WGLMakie
using Random
using Zygote

outdir = joinpath(@__DIR__, "output") |> mkpath

# Plot Burgers solution
let
    grid = Grid(2π, 8192)
    visc = 5e-4
    ustart = randomfield(grid, 10.0)
    u = copy(ustart)
    f = similar(u) # Force
    t = 0.0
    tstop = 0.1
    while t < tstop
        dt = 0.3 * propose_timestep(u, grid, visc)
        dt = min(dt, tstop - t) # Don't overstep
        forward_euler!(u, f, grid, visc, dt)
        t += dt
    end
    x = points(grid)
    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "u")
    lines!(ax, x, ustart; label = "Initial")
    lines!(ax, x, u; label = "Final")
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        orientation = :horizontal,
        framevisible = false,
    )
    rowgap!(fig.layout, 5)
    save("$outdir/solution.pdf", fig; backend = CairoMakie)
    fig
end

# Create Burgers dataset
grid = Grid(2π, 4096)
visc = 2e-3
data = create_data(; grid, visc, nsample = 200, ntime = 100, nsubstep = 10);

let
    isample = 1
    fig = Figure()
    x = points(grid)
    ax = Axis(fig[1, 1])
    lines!(ax, x, data[1][:, isample, 1])
    lines!(ax, x, data[1][:, isample, 100])
    fig
end

using ComponentArrays

let
    model =
        UNet(; channels = [16, 32, 64], nresidual = 2, t_embed_dim = 40, y_embed_dim = 20)
    nsample = 5
    x = randn(Float32, grid.n, 1, nsample) |> gpu_device()
    y = randn(Float32, grid.n, 1, nsample) |> gpu_device()
    t = randn(Float32, 1, 1, nsample) |> gpu_device()
    ps, st = Lux.setup(Xoshiro(0), model) |> gpu_device()
    model((x, t, y), ps, Lux.testmode(st))
    gradient(ps -> sum(abs3, first(model((x, t, y), ps, st))), ps)
end;

model = UNet(; channels = [64, 128], nresidual = 2, t_embed_dim = 64, y_embed_dim = 64)
unet = train(;
    model,
    rng = Xoshiro(0),
    nepoch = 10,
    dataloader = create_dataloader(grid, data, 200),
    opt = Adam(1.0f-3),
)

let
    dev = gpu_device()
    y, z = data
    y = reshape(y[:, 1, 30], :, 1, 1) |> f32 |> dev
    z = reshape(z[:, 1, 30], :, 1, 1) |> f32 |> dev
    x = randn!(similar(z))
    nstep = 1000
    t = fill(0f0, 1, 1, size(z, 3)) |> dev
    for i = 1:nstep
        @info i
        u = unet(x, t, y)
        @. x += 1f0 / nstep * u
        @. t += 1f0 / nstep
    end
    x *= grid.n
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, points(grid), x[:] |> cpu_device())
    lines!(ax, points(grid), z[:] |> cpu_device())
    ylims!(ax, -300, 300)
    save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    fig
end
