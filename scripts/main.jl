using DrWatson: savename

using RNN2D

using Flux

using Random
using LinearAlgebra
using Statistics

using DelimitedFiles
using JSON

using ArgParse

SCRATCH_PATH = ""

function continue_training(path, RNN_ID)
    checkpoints = filter(readdir(path)) do s
        endswith(s, ".bson") && contains(s, "chk") && contains(s, RNN_ID)
    end
    if isempty(checkpoints)
        starting_chk = 1
        rnn_path = nothing
    else
        chks = []
        for chk in checkpoints
            chknum = Int(split(split(chk, "=")[end], ".")[1])
            observable_file = RNN_ID * "_chk=$(chknum)_observables.json"
            if isfile(observable_file)
                append!(chks, chknum)
            end
        end

        last_chk = maximum(chks)
        starting_chk = last_chk + 1
        rnn_path = path * "model_chk=$(last_chk).bson"
    end

    return starting_chk, rnn_path
end    

function train(parsed_args)
    # system parameters
    L = parsed_args["L"]
    δ = parsed_args["delta"]
    R_b = parsed_args["R"]
    A = collect(1:(nspins ÷ 2)) # for SWAP

    # hyperparameters
    lr = parsed_args["lr"]
    batch_size = parsed_args["batchsize"]
    num_samples = parsed_args["n"]
    width = parsed_args["w"]

    # measurements and saving
    epochs = parsed_args["epochs"]
    chk = parsed_args["chk"]
    period = parsed_args["period"]
    verbose = parsed_args["verbose"]

    test_size = parsed_args["test_size"]
    # this is also where the RNN is going to be saved
    path = joinpath(SCRATCH_PATH, "L=$L", "delta=$δ")

    println("Running 2DRNN(L=$L, R_b=$R_b, Ω=1, δ=$δ).")

    labels = 0:1
    nspins, num_batches, batched_data, test_data = format_input_data(path, batch_size, labels; test_size=test_size)
    RNN_ID = "2DRNN_L=$(L)_delta=$(delta)_Rb=$(Rb)_width=$width"
    savepath = joinpath(path, RNN_ID)

    starting_chk, rnn_path = continue_training(path, RNN_ID)
    if (rnn_path == nothing) && (starting_chk == 1)
        m = Chain(
            RNNCell2D(length(labels), width), 
            Flux.Dense(width, length(labels))
        )
    else
        m = load_model(rnn_path)
    end
    
    function loss(x)
        mxs = apply_model(m, x, L, L, labels)
        sum(Flux.logitcrossentropy.(mxs, x))
    end

    opt = Flux.ADAM(lr)
    # optimizer that Mohamed uses in DOI: 10.1103/PhysRevResearch.2.023358
    #opt = Flux.Optimiser(ADAM(0.005), InvDecay(0.0002))
    
    vals = ["mean", "std_error", "variance"]
    observables = ["energy_densities", "abs_stag_mags", "swaps"]
    observable_dict = Dict{String, Dict{String, Vector{Float64}}}()

    for c in starting_chk:chk
    
        # initialize things we need to save during training
        epoch_nums = collect(((c-1)*epochs + period):period:(epochs*c))
        for obs in observables
            observable_dict[obs] = Dict(k => [] for k in vals)
        end
        
        for e in 1:epochs
            batched_data = batched_data[shuffle(1:num_batches), :]
            for batch in batched_data
                Flux.train!(loss, Flux.params(m), [batch], opt)
            end
    
            if e % period == 0

                samples_info = sample_model(m, num_samples, L, L, labels)
                energy_density = rydberg_square_energy_density(m, samples_info, labels, Rb, delta)
                abs_stag_mag = abs_staggered_magnetization(samples_info)
                SWAP = swap(m, samples_info, labels, A)

                if verbose
                    println("Epoch: $e")
                    @show energy_density
                    @show abs_stag_mag
                    @show SWAP
                    println()
                end
    
                for obs in observables
                    
                    # TODO: streamline this
                    if obs == "energy_densities"
                        for k in vals
                            append!(observable_dict[obs][k], energy_density[k])
                        end
                    end
                    if obs == "abs_stag_mags"
                        for k in vals
                            append!(observable_dict[obs][k], abs_stag_mag[k])
                        end
                    end
                    if obs == "swaps"
                        for k in vals
                            append!(observable_dict[obs][k], SWAP[k])
                        end
                    end
                    
                end
                
            end
        end
        
        save_model(savepath * "_model_chk=$(c)_.bson", m)
        
        if c == 1
            # create a single dictionary to save epoch_data and all observables
            contents = Dict()
            for obs in observables
                contents[obs] = Dict{String, Vector{Float64}}()
                for k in vals
                    contents[obs][k] = observable_dict[obs][k]
                end
            end
            
            contents["epochs"] = epoch_nums
            
        elseif c > 1
            previous_file_name = savepath * "_chk=$(c-1)_observables.json"
            contents = JSON.parse(open(f->read(f, String), previous_file_name))
            
            for (obs, _) in contents
                if obs == "epochs"
                    append!(contents["epochs"], epoch_nums)
                else
                    for k in vals
                        append!(contents[obs][k], observable_dict[obs][k])
                    end
                end
            end
        end
        
        observable_file = savepath * "_chk=$c" * "_observables.json"
    
        # save data
        open(observable_file, "w") do f
            write(f, JSON.json(contents))
        end
        
        # remove old observable file, but keep the previous one just in case
        old_observable_file_name = savepath * "_chk=$(c-2)" * "_observables.json"
        if isfile(old_observable_file_name)
            rm(old_observable_file_name)
        end
            
        # remove old model file, but keep the previous one just in case
        old_model_file_name = savepath * "_model_chk=$(c-2)" * "_.bson"
        if isfile(old_model_file_name)
            rm(old_model_file_name)
        end
        
    end
end

###############################################################################

s = ArgParseSettings()

@add_arg_table! s begin
    "train"
        help = "Train a 2D rNN on Rydberg QMC data"
        action = :command
end

@add_arg_table! s["train"] begin
    "L"
        help = "Dimensions of square lattice in x direction."
        required = true
        arg_type = Int

    "--delta"
        help = "Strength of the detuning"
        arg_type = Float64
        default = 1.0
    "--radius", "-R"
        help = "Rydberg blockade radius (in units of the lattice spacing). Controls the strength of the interaction."
        arg_type = Float64
        default = 1.2

    "--lr"
        help = "Learning rate for ADAM optimization."
        arg_type = Float64
        default = 0.001
    "--batchsize"
        help = "Batch size for stochastic optimization."
        arg_type = Int64
        default = 100
    "-n", "--measurements"
        help = "number of samples to draw from 2DRNN to calculate observables at every training step."
        arg_type = Int64
        default = 100_000
    "-w", "--width"
        help = "Hidden layer size."
        arg_type = Int64
        required = true

    "--epochs"
        help = "Number of iterations through the entire input data set per checkpoint."
        arg_type = Int64
        default = 2000
    "--chk"
        help = "Number of checkpoints to make. Similar to the number of batches in a MC simulation."
    "--period"
        help = "Number of epochs wherein observables will be calculated."
        arg_type = Int64
        default = 50
    "--verbose"
        help = "Print observable values every period."
        action = :store_true
    
    "--testsize"
        help = "Number of samples to remove from input dataset to use as test data."
        arg_type = Int64
        default = 0
end

parsed_args = parse_args(ARGS, s)

if parsed_args["%COMMAND%"] == "train"
    @time train(parsed_args["train"])
end