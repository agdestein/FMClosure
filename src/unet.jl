silu(x) = @. x / (1 + exp(-x))

"Conv with periodic padding (`pad` on each side)."
CircularConv(args...; pad, kwargs...) = Chain(
    WrappedFunction(x -> NNlib.pad_circular(x, pad; dims = 1)),
    Conv(args...; kwargs...),
)

"""
Upsample periodic field by a factor of 2.
The grids contain `n + 1` and `2n + 1` points, respectively.
The left and right boundary points overlap periodically, and
so the value of the input field in the right point is not
included in the input `x`.
"""
CircularUpsample() =
    WrappedFunction() do x
        n = size(x, 1)
        x = pad_circular(x, (0, 1); dims = 1) # Add redundant right point
        x = upsample_linear(x; size = 2 * n + 1)
        selectdim(x, 1, 1:(2*n)) # Remove redundant right point
    end

function FourierEncoder(dim, device)
    @assert dim % 2 == 0
    half_dim = div(dim, 2)
    weights = randn(Float32, 1, half_dim)
    @compact(; weights) do t
        freqs = @. 2 * t * weights
        sin_embed = @. sqrt(2.0f0) * sinpi(freqs)
        cos_embed = @. sqrt(2.0f0) * cospi(freqs)
        output = hcat(sin_embed, cos_embed)
        @return output
    end
end

ResidualLayer(nspace, nchan, nt, ny) =
    @compact(;
        block1 = Chain(
            gelu,
            LayerNorm((nspace, nchan)),
            CircularConv((3,), nchan => nchan; pad = 1),
        ),
        block2 = Chain(
            gelu,
            LayerNorm((nspace, nchan)),
            CircularConv((3,), nchan => nchan; pad = 1),
        ),
        time_adapter = Chain(
            ReshapeLayer((nt,)),
            Dense(nt => nt, gelu),
            Dense(nt => nchan),
            ReshapeLayer((1, nchan)),
        ),
        y_adapter = Chain(
            CircularConv((3,), ny => ny, gelu; pad = 1),
            CircularConv((3,), ny => nchan; pad = 1),
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

Encoder(nspace, nin, nout, nresidual, nt, ny) =
    @compact(;
        res_blocks = fill(ResidualLayer(nspace, nin, nt, ny), nresidual),
        downsample = CircularConv((3,), nin => nout; stride = 2, pad = 1),
    ) do (x, t_embed, y_embed)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        x = downsample(x)
        @return x
    end

Midcoder(nspace, nchannel, nresidual, nt, ny) =
    @compact(;
        res_blocks = fill(ResidualLayer(nspace, nchannel, nt, ny), nresidual),
    ) do (x, t_embed, y_embed)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        @return x
    end

Decoder(nspace, nin, nout, nresidual, nt, ny) =
    @compact(;
        upsample = Chain(CircularUpsample(), CircularConv((3,), nin => nout; pad = 1)),
        res_blocks = fill(ResidualLayer(nspace, nout, nt, ny), nresidual),
    ) do (x, t_embed, y_embed)
        x = upsample(x)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        @return x
    end

UNet(; nspace, channels, nresidual, t_embed_dim, y_embed_dim, device) =
    @compact(;
        init_conv = Chain(
            CircularConv((3,), 1 => channels[1]; pad = 1),
            LayerNorm((nspace, channels[1])),
            gelu,
        ),
        time_embedder = FourierEncoder(t_embed_dim, device),
        y_embedders = map(
            i -> CircularConv((3,), 1 => y_embed_dim; stride = 2^(i - 1), pad = 1),
            1:length(channels),
        ),
        encoders = map(
            i -> Encoder(
                div(nspace, 2^(i - 1)),
                channels[i],
                channels[i+1],
                nresidual,
                t_embed_dim,
                y_embed_dim,
            ),
            1:(length(channels)-1),
        ),
        decoders = map(
            i -> Decoder(
                div(nspace, 2^(i - 2)),
                channels[i],
                channels[i-1],
                nresidual,
                t_embed_dim,
                y_embed_dim,
            ),
            length(channels):-1:2,
        ),
        midcoder = Midcoder(
            div(nspace, 2^(length(channels) - 1)),
            channels[end],
            nresidual,
            t_embed_dim,
            y_embed_dim,
        ),
        final_conv = CircularConv((3,), channels[1] => 1; pad = 1, use_bias = false),
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

function create_dataloader(grid, data, batchsize, rng)
    y, z = data
    y, z = reshape(y, grid.n, 1, :), reshape(z, grid.n, 1, :)
    y, z = (y, z) |> f32
    # z ./= grid.n
    DataLoader((y, z); batchsize, shuffle = true, partial = false, rng)
end

"""
Train an flow-matching ODE to predict next state (`z`) from current state (`y`).
The ODE has Gaussian initial contitions `x0` and evolve via `dx = model(x, t, y) dt`
from time 0 to 1.
The target trajectory `x` is a linear interpolation between `x0` and `z`.
"""
function train(; model, rng, nepoch, dataloader, opt, device)
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
