module RNN2D

using Flux: Flux, onehot, softmax
using NNlib: elu
using Zygote: Zygote, @ignore

using BSON: @save, @load

using Random
using LinearAlgebra

using StatsBase
using Measurements

using DelimitedFiles

export save_model, load_model
export apply_model, sample_model, probability
export make_batches, format_data, format_input_data, generate_space, format_sites, null_vector
export magnetization, abs_staggered_magnetization, rydberg_square_energy_density, swap, to_pm1, to_01, flip_spin
export stats_dict
export RNNCell2D

include("architecture.jl")
include("utils.jl")
include("model_applications.jl")
include("observables/spin_utils.jl")
include("observables/measure.jl")

end # module