"Grid of domain length `l` with `n` points."
struct Grid{T}
    l::T
    n::Int
end

# Call `g(i)` to shift an index `i` periodically 
@inline (g::Grid)(i) = mod1(i, g.n)

"Get grid spacing."
dx(g::Grid) = g.l / g.n

points(g::Grid) = range(dx(g) / 2, g.l - dx(g) / 2, g.n)

"Call `f(args..., i)` for all grid indices `i`."
apply!(f, g::Grid, args) =
    for i = 1:g.n
        f(args..., i)
    end

@inline function force!(f, u, g::Grid, visc, i)
    h = dx(g)
    a = -(u[i] + u[i-1|>g])^2 / 8 + visc * (u[i] - u[i-1|>g]) / h
    b = -(u[i+1|>g] + u[i])^2 / 8 + visc * (u[i+1|>g] - u[i]) / h
    f[i] = (b - a) / h
end

function forward_euler!(u, f, grid, visc, dt)
    apply!(force!, grid, (f, u, grid, visc))
    @. u += dt * f
end

propose_timestep(u, g::Grid, visc) = min(dx(g) / maximum(abs, u), dx(g)^2 / visc)

function randomfield(g::Grid, kpeak)
    amp = sqrt(4 / kpeak / 3 / sqrt(π))
    k = 0:div(g.n, 2)
    c = @. amp * (k / kpeak)^2 * exp(-(k / kpeak)^2 / 2 + 2π * im * rand())
    irfft(c * g.n, g.n)
end

function create_data(; grid, visc, nsample, nsubstep, ntime, cfl = 0.3)
    inputs = zeros(grid.n, nsample, ntime)
    outputs = similar(inputs)
    for isample = 1:nsample
        @show isample
        u = randomfield(grid, 10.0)
        f = similar(u)
        for itime = 1:ntime
            # @show (isample, itime)
            # Skip first iter to get initial pair
            itime == 1 || for _ = 1:nsubstep
                dt = 0.3 * propose_timestep(u, grid, visc)
                forward_euler!(u, f, grid, visc, dt)
            end
            apply!(force!, grid, (f, u, grid, visc))
            inputs[:, isample, itime] = u
            outputs[:, isample, itime] = f
        end
    end
    inputs, outputs
end
