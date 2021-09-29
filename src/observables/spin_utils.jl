# flip spins, change to +/- 1 notation, etc

function to_pm1(samples)
    return (2 .* samples) .- 1
end

function to_01(samples)
    return (samples .+ 1) ./ 2
end

function flip_spin(idx, samples)
    samples[:, idx] .⊻= 1
    return samples
end