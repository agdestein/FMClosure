"Grid of domain length `l` with `n` points."
struct Grid{T}
    l::T
    n::Int
end

# Call `g(i)` to shift an index `i` periodically 
@inline (g::Grid)(i) = mod1(i, g.n)

"Get grid spacing."
dx(g::Grid) = g.l / g.n

"""
The left point is always at zero.
The (`n + 1`-th) right point is at `l`, but it is not included since it
is periodically redundant.
"""
points(g::Grid) = range(0, g.l, g.n + 1)[1:(end-1)]

"Call `f(args..., i)` for all grid indices `i`."
apply!(f, g::Grid, args) =
    for i = 1:g.n
        f(args..., i)
    end

# "Burgers equation right hand side."
# @inline function force!(f, u, g::Grid, (; visc), i)
#     h = dx(g)
#     a = -(u[i] + u[i-1|>g])^2 / 8 + visc * (u[i] - u[i-1|>g]) / h
#     b = -(u[i+1|>g] + u[i])^2 / 8 + visc * (u[i+1|>g] - u[i]) / h
#     f[i] = (b - a) / h
# end

"Korteweg-de Vries equation right hand side."
@inline function force!(f, u, g::Grid, _, i)
    h = dx(g)
    a = (u[i] + u[i-1|>g])^2 / 4
    b = (u[i+1|>g] + u[i])^2 / 4
    # b = u[i+1|>g]^2 / 2
    # a = u[i-1|>g]^2 / 2
    f[i] = 3 * (b - a) / h - (u[i+2|>g] / 2 - u[i+1|>g] + u[i-1|>g] - u[i-2|>g] / 2) / h^3
end

function forward_euler!(u, f, grid, visc, dt)
    apply!(force!, grid, (f, u, grid, visc))
    @. u += dt * f
end

function rk4!(u, cache, grid, visc, dt)
    v, k1, k2, k3, k4 = cache
    apply!(force!, grid, (k1, u, grid, visc))
    @. v = u + dt / 2 * k1
    apply!(force!, grid, (k2, v, grid, visc))
    @. v = u + dt / 2 * k2
    apply!(force!, grid, (k3, v, grid, visc))
    @. v = u + dt * k3
    apply!(force!, grid, (k4, v, grid, visc))
    @. u += dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
end

propose_timestep(u, g::Grid, visc) = min(dx(g) / maximum(abs, u), dx(g)^2 / visc)

function randomfield(g::Grid, kpeak, rng)
    amp = sqrt(4 / kpeak / 3 / sqrt(π))
    k = 0:div(g.n, 2)
    c = @. amp * (k / kpeak)^2 * exp(-(k / kpeak)^2 / 2 + 2π * im * rand(rng))
    irfft(c * g.n, g.n)
end

function create_data(; grid, params, nsample, nsubstep, ntime, dt, rng)
    inputs = zeros(grid.n, ntime, nsample)
    outputs = similar(inputs)
    adaptive = isnothing(dt)
    for isample = 1:nsample
        @show isample
        u = randomfield(grid, 10.0, rng)
        cache = similar(u), similar(u), similar(u), similar(u), similar(u)
        for itime = 1:ntime
            # @show (isample, itime)
            inputs[:, itime, isample] = u
            for isubstep = 1:nsubstep
                # forward_euler!(u, cache, grid, params, dt)
                rk4!(u, cache, grid, params, dt)
            end
            outputs[:, itime, isample] = u
        end
    end
    @. outputs -= inputs # Let the difference be the target
    inputs, outputs
end
