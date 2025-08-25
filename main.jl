# This is just a hack for "go to definition" to work in editor.
if false
    include("src/FMClosure.jl")
    using .FMClosure
end

using FMClosure
using CairoMakie
using Lux
using MLUtils
using NNlib
using Optimisers
using WGLMakie
using Random
using Zygote

outdir = joinpath(@__DIR__, "output") |> mkpath

burgers(n, visc) = (; grid = Grid(2π, 8192), params = (; visc))
kdv(n) = (;
    grid = Grid(30.0, n),
    params = (;), # No params for KdV
)

# Plot solution
let
    # (; grid, params) = burgers(8192, 5e-4)
    (; grid, params) = kdv(256)
    ustart = randomfield(grid, 10.0)
    u = copy(ustart)
    # cache = similar(u) # (forward_euler)
    cache = similar(u), similar(u), similar(u), similar(u), similar(u) # (RK4)
    t = 0.0
    tstop = 0.1
    while t < tstop
        @show t
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
dt = 1e-3
data = create_data(; grid, params, nsample = 10, ntime = 100, nsubstep = 10, dt);

# Show two successive states
let
    isample = 1
    itime = 10
    fig = Figure()
    x = points(grid)
    ax = Axis(fig[1, 1])
    lines!(ax, x, data[1][:, itime, isample])
    lines!(ax, x, data[1][:, itime+1, isample])
    fig |> display
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
    fig |> display
end

model = UNet(; channels = [16, 32, 64], nresidual = 2, t_embed_dim = 40, y_embed_dim = 20)
unet = train(;
    model,
    rng = Xoshiro(0),
    nepoch = 30,
    dataloader = create_dataloader(grid, data, 50),
    opt = AdamW(1.0f-3),
)

let
    isample = 1
    itime = 1
    y, z = data
    y = reshape(y[:, itime, isample], :, 1, 1) |> f32
    z = reshape(z[:, itime, isample], :, 1, 1) |> f32
    x = randn!(similar(z))
    nstep = 10
    t = fill(0.0f0, 1, 1, size(z, 3))
    for i = 1:nstep
        @info i
        u = unet(x, t, y)
        @. x += 1 / nstep * u
        @. t += 1 / nstep
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    # lines!(ax, points(grid), y[:]; label = "Input")
    lines!(ax, points(grid), y[:] + z[:]; label = "Target")
    lines!(ax, points(grid), y[:] + x[:]; label = "Prediction")
    # lines!(ax, points(grid), z[:]; label = "Target")
    # lines!(ax, points(grid), x[:]; label = "Prediction")
    axislegend(ax)
    save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    fig
end
