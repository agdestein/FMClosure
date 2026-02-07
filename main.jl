# This is just a hack for "go to definition" to work in editor.
if false
    include("src/FMClosure.jl")
    using .FMClosure
end

using FMClosure
using CairoMakie
using Lux
using CUDA, cuDNN
using MLUtils
using NNlib
using Optimisers
using WGLMakie
using Random
using Zygote

outdir = joinpath(@__DIR__, "output") |> mkpath

burgers(n, visc) = (; grid = Grid(2π, n), params = (; visc))
kdv(n) = (;
    grid = Grid(30.0, n),
    params = (;), # No params for KdV
)

# Plot solution
let
    # (; grid, params) = burgers(8192, 5e-4)
    (; grid, params) = kdv(256)
    ustart = randomfield(grid, 10.0, Xoshiro(0))
    u = copy(ustart)
    # cache = similar(u) # (forward_euler)
    cache = similar(u), similar(u), similar(u), similar(u), similar(u) # (RK4)
    t = 0.0
    tstop = 0.1
    while t < tstop
        # dt = 0.3 * propose_timestep(u, grid, visc)
        dt = 1e-3
        dt = min(dt, tstop - t) # Don't overstep
        # forward_euler!(u, cache, grid, params, dt)
        rk4!(u, cache, grid, params, dt)
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

# Create dataset
# (; grid, params) = burgers(2048, 2e-3)
(; grid, params) = kdv(256)
data = create_data(;
    grid,
    params,
    nsample = 500,
    ntime = 100,
    nsubstep = 10,
    dt = 1e-3,
    rng = Xoshiro(0),
);

# Show two successive states
let
    isample = 1
    itime = 10
    fig = Figure()
    x = points(grid)
    ax = Axis(fig[1, 1])
    lines!(ax, x, data[1][:, itime, isample])
    lines!(ax, x, data[1][:, itime+1, isample])
    save("$outdir/states.pdf", fig; backend = CairoMakie)
    fig
end

# Show one input-output pair
let
    isample = 1
    itime = 10
    fig = Figure()
    x = points(grid)
    ax = Axis(fig[1, 1])
    lines!(ax, x, data[1][:, itime, isample])
    lines!(ax, x, data[2][:, itime, isample])
    save("$outdir/io_pair.pdf", fig; backend = CairoMakie)
    fig
end

device = gpu_device()
model = UNet(;
    nspace = grid.n,
    channels = [8, 16, 32, 64],
    nresidual = 2,
    t_embed_dim = 32,
    y_embed_dim = 32,
    device,
)
model = SimpleNet(;
    nspace = grid.n,
    nchannel = 16,
    nresidual = 2,
    t_embed_dim = 32,
    y_embed_dim = 32,
    device,
)
ps, st = train(;
    model,
    rng = Xoshiro(0),
    nepoch = 10,
    dataloader = create_dataloader(grid, data, 300, Xoshiro(0)),
    opt = AdamW(1.0f-3),
    device,
    # params = (ps, st),
);
unet = (x, t, y) -> first(model((x, t, y), ps, Lux.testmode(st)))

# Plot one prediction
let
    isample = 1
    itime = 10
    y, z = data
    y = reshape(y[:, itime, isample], :, 1, 1) |> f32 |> device
    z = reshape(z[:, itime, isample], :, 1, 1) |> f32 |> device
    x = randn!(similar(z))
    nstep = 100
    t = fill(0.0f0, 1, 1, size(z, 3)) |> device
    h = 1.0f0 / nstep
    for i = 1:nstep
        @info i
        u = unet(x, t, y)
        @. x += h * u
        @. t += h
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    input = y[:] |> cpu_device()
    target = z[:] |> cpu_device()
    prediction = x[:] |> cpu_device()
    # lines!(ax, points(grid), input; label = "Input")
    lines!(ax, points(grid), target; label = "Target")
    lines!(ax, points(grid), prediction; label = "Prediction")
    # lines!(ax, points(grid), input + target; label = "Target")
    # lines!(ax, points(grid), input + prediction; label = "Prediction")
    axislegend(ax)
    save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    fig
end

# Plug FM model back into physical time stepping loop
let
    isample = 1
    inputs, _ = data
    ntime = 10
    y = reshape(inputs[:, 1, isample], :, 1, 1) |> f32 |> device
    x = similar(y) |> device
    t = fill(0.0f0, 1, 1, 1) |> device
    nsubstep = 4
    h = 1.0f0 / nsubstep
    for itime = 1:ntime # Physical time stepping
        @show itime
        fill!(t, 0)
        randn!(x) # Random initial conditions
        for isub = 1:nsubstep # Pseudo-time stepping
            u = unet(x, t, y)
            @. x += h * u
            @. t += h
        end
        @. y += x # x is the physical step
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    input = inputs[:, 1, isample]
    target = inputs[:, ntime+1, isample]
    prediction = y[:] |> cpu_device()
    # lines!(ax, points(grid), input; label = "Input")
    lines!(ax, points(grid), target; label = "Target")
    lines!(ax, points(grid), prediction; label = "Prediction")
    axislegend(ax)
    save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    fig
end
