function sample_model(m, num_samples::Int, Lx::Int, Ly::Int, labels::UnitRange{Int64})
    # initialize chain to zero config
    N = Lx * Ly
    samples = zeros(Int, num_samples, N)
    prob_samples = ones(Float64, num_samples)
    width = length(m[1].b) # how long each hidden vector is

    # initial model inputs are zero states
    # Zygote needs to ignore the format_sites function
    v0 = null_vector(num_samples, labels)
    h0 = zeros(Float32, width, num_samples)
    
    benched_hiddens = [h0 for _ in 1:Lx] # store hidden units here
    benched_xs = [v0 for _ in 1:Lx]
    for n in 1:N
        row = div(n-1, Lx) + 1
        column = rem(n-1, Lx) + 1

        hh = (column == 1) ? h0 : benched_hiddens[column-1]
        hv = benched_hiddens[column]
        xh = (column == 1) ? v0 : benched_xs[column-1]
        xv = benched_xs[column]

        h_n, out = m[1](hh, hv, xh, xv)
        benched_hiddens[column] = h_n
        # softmax layer 
        # this gives p(spin n = 1 | previous n-1 spins) 
        probs = softmax(m[2](h_n))
        #d = Bernoulli.(probs)
        #x = collect(rand.(d))
        rs = rand(num_samples)
        # the second index is needed because we need p(spin_n = 1), not p(spin_n = 0)
        result = Int.(probs[2,:] .> rs)
        samples[:,n] = result

        result = format_sites(result, labels)
        benched_xs[column] = result

        prob_samples .*= dropdims(sum(dot.(probs, result), dims=1); dims=1)
    end

    return prob_samples, samples
end

function probability(m, v::Vector{Vector{Matrix{Float32}}}, Lx::Int, Ly::Int, labels::UnitRange{Int64})
    N = size(v[1])[1]
    num_samples = size(v[1][1])[2]
    prob = ones(Float64, num_samples) # aggregate probabilities here

    width = length(m[1].b) # how long each hidden vector is

    # initial model inputs are zero states
    v0 = null_vector(num_samples, labels)
    h0 = zeros(Float32, width, num_samples)
    benched_hiddens = [h0 for _ in 1:Lx] # store hidden units here

    for n in 1:N
        row = div(n-1, Lx) + 1
        column = rem(n-1, Lx) + 1

        hh = (column == 1) ? h0 : benched_hiddens[column-1]
        hv = benched_hiddens[column]

        xh = (column == 1) ? v0 : v[1][n-1]
        xv = (row == 1) ? v0 : v[1][n-Lx]

        h_n, out = m[1](hh, hv, xh, xv)
        benched_hiddens[column] = h_n
        # softmax layer 
        # this gives p(spin n = 1 | previous n-1 spins) 
        x̂_n = softmax(m[2](h_n))

        prob .*= dropdims(sum(dot.(x̂_n, v[1][n]), dims=1); dims=1)
    end

    return prob
end

function apply_model(m, v::Vector{Matrix{Float32}}, Lx::Int, Ly::Int, labels::UnitRange{Int64})
    N = size(v, 1)
    num_samples = size(v[1], 2)

    width = length(m[1].b) # how long each hidden vector is

    # initial model inputs are zero states
    # Zygote needs to ignore the format_sites function
    v0 = @ignore null_vector(num_samples, labels)
    h0 = zeros(Float32, width, num_samples)
    
    # array mutation will happen when things are modified inside benched_hiddens and mxs
    # Zygote cannot deal with array mutation currently
    # Zygote.Buffer is a band-aid to the issue
    benched_hiddens = Zygote.Buffer([h0 for _ in 1:Lx], false) # store hidden units here
    mxs = Zygote.Buffer([zeros(Float32, length(labels), num_samples) for _ in 1:N], false)
    for n in 1:N
        row = div(n-1, Lx) + 1
        column = rem(n-1, Lx) + 1

        hh = (column == 1) ? h0 : benched_hiddens[column-1]
        hv = benched_hiddens[column]
        xh = (column == 1) ? v0 : v[n-1]
        xv = (row == 1) ? v0 : v[n-Lx]

        h_n, out = m[1](hh, hv, xh, xv)
        benched_hiddens[column] = h_n
        mxs[n] += m[2](h_n)
    end

    #copy(mxs) is currently required so that Zygote can take gradients of this function
    return copy(mxs)
end

function save_model(path::String, model)
    @save path model
end

function load_model(path::String)
    # automatically calls the loaded model "model"
    @load path model
    return model
end