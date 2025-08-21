module Burgers

using FFTW
using Lux
using MLUtils
using NNlib
using Random

include("discretization.jl")
include("unet.jl")

export Grid, points, force!, forward_euler!, propose_timestep, randomfield, create_data
export UNet, create_dataloader, train

end # module Burgers
