function make_batches(data, num_batches::Int, batch_size::Int, num_labels::Int, Neff::Int)
    batched_data = Array{Array{Float32,2},1}[]
    for n in 1:num_batches
        batch = Array{Float32,2}[]
        rows::Array{Int,1} = (collect(((n-1)*batch_size+1):(n*batch_size)))
        for j in 1:Neff
            site = zeros(Int, num_labels, batch_size)
            for (idx, i) in enumerate(rows)
                for l in 1:num_labels
                    site[l,idx] = data[i,j][l]
                end            
            end
            push!(batch, site)
        end
        push!(batched_data, batch)
    end
    return batched_data
end

function format_data(data, labels::UnitRange{Int}; batch_size=nothing)
    # distinguish a single sample from multiple samples
    if length(size(data)) == 2
        data_size, N = size(data)
    else
        data_size, N = 1, size(data)[1]
    end

    # onehot the data
    data = map(ch -> onehot(ch, labels), data) # onehot the training data

    # now divide the data into batches
    num_batches = isnothing(batch_size) ? 1 : data_size ÷ batch_size
    batch_size = isnothing(batch_size) ? data_size : batch_size

    batched_data = make_batches(data, num_batches, batch_size, length(labels), N)

    # strip away redundant dims
    batched_data = isnothing(batch_size) ? batched_data[1] : batched_data

    return batched_data
end


function format_input_data(data_path::String, batch_size::Int, labels::UnitRange{Int}; psi_path=nothing, test_size=nothing, test_path=nothing, pad=false)
    data = Int.(readdlm(data_path))
    N = size(data)[2]

    # calculate test_size
    if test_size == nothing
        # automatically halve the input data set if test_size isn't specified
        test_size = size(data)[1] ÷ 2
        train_size = test_size
    else
        train_size = size(data)[1] - test_size
    end

    train_data = data[begin:train_size, :]
    test_data = data[(train_size+1):end, :]

    num_batches = train_size ÷ batch_size

    if pad
        Neff = N + 1
        # put a zero-state at the beginning of each dataset.
        # In this way, we automatically initialize the RNN sequence
        # N+1 (+1 from the added zero-state) sites now!
        train_tmp = zeros(Int, train_size, N+1) # placeholder
        test_tmp = zeros(Int, test_size, N+1) # placeholder

        for i in 1:train_size
            train_tmp[i,2:(N+1)] = train_data[i,:]
        end

        for i in 1:test_size
            test_tmp[i,2:(N+1)] = test_data[i,:]
        end

        train_data = train_tmp    
        test_data = test_tmp
    else
        Neff = N
    end   
    
    train_data = map(ch -> onehot(ch, labels), train_data) # onehot the training data
    test_data = map(ch -> onehot(ch, labels), test_data) # onehot the training data
    
    # make batches out of the training data
    batched_data = make_batches(train_data, num_batches, batch_size, length(labels), Neff)

    # no need to make batches out of the test data. 
    # batch_size = 1 used
    test_data = make_batches(test_data, 1, test_size, length(labels), Neff)

    if !isnothing(psi_path)
        psi = readdlm(psi_path)[:,1]
        return N, num_batches, batched_data, psi, test_data
    else
        return N, num_batches, batched_data, test_data
    end
end

function generate_space(N::Int)
    # generate the hilbert space
    space = zeros(Int, 2^N, N)
    for i in 0:(2^N-1)
        x = digits(i, base=2, pad=N) |> reverse
        for j in 1:N
            space[i+1,j] = x[j]
        end
    end
    return space
end

function format_sites(sites, labels::UnitRange{Int64})
    sites = map(ch -> onehot(ch, labels), sites)
    sites = make_batches(sites, 1, size(sites)[1], length(labels), 1)[1][1]
    return sites
end 

function null_vector(num_samples::Int, labels::UnitRange{Int64})
    return zeros(Float32, length(labels), num_samples)
end