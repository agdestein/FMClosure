silu(x) = @. x / (1 + exp(-x))

"Conv with periodic padding (`pad` on each side)."
CircularConv(args...; pad, kwargs...) = Chain(
    WrappedFunction(x -> NNlib.pad_circular(x, pad; dims = 1)),
    Conv(args...; kwargs...),
)

function FourierEncoder(dim)
    @assert dim % 2 == 0
    half_dim = div(dim, 2)
    weights = randn(Float32, 1, half_dim) |> gpu_device()
    @compact(; weights) do t
        freqs = @. 2 * t * weights
        sin_embed = @. sqrt(2.0f0) * sinpi(freqs)
        cos_embed = @. sqrt(2.0f0) * cospi(freqs)
        output = hcat(sin_embed, cos_embed)
        @return output
    end
end

ResidualLayer(n, nt, ny) =
    @compact(;
        block1 = Chain(silu, BatchNorm(n), CircularConv((3,), n => n; pad = 1)),
        block2 = Chain(silu, BatchNorm(n), CircularConv((3,), n => n; pad = 1)),
        time_adapter = Chain(
            ReshapeLayer((nt,)),
            Dense(nt => nt, silu),
            Dense(nt => n),
            ReshapeLayer((1, n)),
        ),
        y_adapter = Chain(
            CircularConv((3,), ny => ny, silu; pad = 1),
            CircularConv((3,), ny => n; pad = 1),
        ),
    ) do (x, t_embed, y_embed)
        res = copy(x)

        # Initial conv block
        x = block1(x)

        # Add time embedding
        t_embed = time_adapter(t_embed)
        x = x .+ t_embed

        # Add y embedding (conditional embedding)
        y_embed = y_adapter(y_embed)
        x = x .+ y_embed

        # Second conv block
        x = block2(x)

        # Add back residual
        x = x .+ res

        @return x
    end

Encoder(nin, nout, nresidual, nt, ny) =
    @compact(;
        res_blocks = fill(ResidualLayer(nin, nt, ny), nresidual),
        downsample = CircularConv((3,), nin => nout; stride = 2, pad = 1),
    ) do (x, t_embed, y_embed)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        x = downsample(x)
        @return x
    end

Midcoder(nchannel, nresidual, nt, ny) =
    @compact(;
        res_blocks = fill(ResidualLayer(nchannel, nt, ny), nresidual),
    ) do (x, t_embed, y_embed)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        @return x
    end

Decoder(nin, nout, nresidual, nt, ny) =
    @compact(;
        upsample = Chain(Upsample(2, :bilinear), CircularConv((3,), nin => nout; pad = 1)),
        res_blocks = fill(ResidualLayer(nout, nt, ny), nresidual),
    ) do (x, t_embed, y_embed)
        x = upsample(x)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        @return x
    end

UNet(; channels, nresidual, t_embed_dim, y_embed_dim) =
    @compact(;
        init_conv = Chain(
            CircularConv((3,), 1 => channels[1]; pad = 1),
            BatchNorm(channels[1]),
            silu,
        ),
        time_embedder = FourierEncoder(t_embed_dim),
        y_embedders = map(
            i -> CircularConv((3,), 1 => y_embed_dim; stride = 2^(i-1), pad = 1),
            1:length(channels),
        ),
        encoders = map(
            i -> Encoder(channels[i], channels[i+1], nresidual, t_embed_dim, y_embed_dim),
            1:(length(channels)-1),
        ),
        decoders = map(
            i -> Decoder(channels[i], channels[i-1], nresidual, t_embed_dim, y_embed_dim),
            length(channels):-1:2,
        ),
        midcoder = Midcoder(channels[end], nresidual, t_embed_dim, y_embed_dim),
        final_conv = CircularConv((3,), channels[1] => 1; pad = 1),
    ) do (x, t, y)
        # Embed t and y
        t_embed = time_embedder(t)
        # y_embed = y_embedder(y)

        # Initial convolution
        x = init_conv(x)

        residuals = ()
        y_embeds = ()

        # Encoders
        for (encoder, y_embedder) in zip(encoders, y_embedders)
            y_embed = y_embedder(y)
            x = encoder((x, t_embed, y_embed))
            residuals = residuals..., copy(x)
            y_embeds = y_embeds..., y_embed
        end

        # Midcoder
        y_embed = y_embedders[end](y)
        x = midcoder((x, t_embed, y_embed))

        # Decoders
        for decoder in decoders
            y_embeds..., y_embed = y_embeds
            residuals..., res = residuals
            x = x + res
            x = decoder((x, t_embed, y_embed))
        end

        # Final convolution
        x = final_conv(x)

        @return x
    end

function create_dataloader(grid, data, batchsize)
    y, z = data
    y, z = reshape(y, grid.n, 1, :), reshape(z, grid.n, 1, :)
    y, z = (y, z) |> f32
    # z ./= grid.n
    DataLoader((y, z); batchsize, shuffle = true, partial = false)
end

"""
Train an flow-matching ODE to predict next state (`z`) from current state (`y`).
The ODE has Gaussian initial contitions `x0` and evolve via `dx = model(x, t, y) dt`
from time 0 to 1.
The target trajectory `x` is a linear interpolation between `x0` and `z`.
"""
function train(; model, rng, nepoch, dataloader, opt)
    device = gpu_device()
    ps, st = Lux.setup(rng, model) |> device
    train_state = Training.TrainState(model, ps, st, opt)
    loss = MSELoss()
    for iepoch = 1:nepoch, (ibatch, batch) in enumerate(dataloader)
        y, z = batch |> device
        x0 = randn!(similar(z)) # Gaussian initial conditions
        t = rand!(similar(z, 1, 1, size(z, ndims(z)))) # Pseudo-times
        x = @. t * z + (1 - t) * x0 # Linear interpolation
        u = @. z - x0 # Linear conditional vector field
        _, l, _, train_state =
            Training.single_train_step!(AutoZygote(), loss, ((x, t, y), u), train_state)
        ibatch % 1 == 0 && @info "iepoch = $iepoch, ibatch = $ibatch, loss = $l"
    end
    ps_freeze = train_state.parameters
    st_freeze = train_state.states
    (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))
end
