function stats_dict(f::Function, x::Vector{Float64})
    y = f.(x)
    μ, stdev = mean_and_std(y)
    σ = stdev / sqrt(length(x))
    _, var = mean_and_var(x)
    #μ = mean(y)
    #σ = std(y; mean=μ) / sqrt(length(x))
    dict = Dict("mean" => μ, "std_error" => σ, "variance" => var)
    return dict
end
stats_dict(x::Vector{Float64}) = stats_dict(identity, x)

function magnetization(samples::Matrix{Int})
    samples = to_pm1(samples)
    nspins = size(samples, 2)
    mags = sum(samples, dims = 2) / nspins
    return stats_dict(mags)
end

function rydberg_potential(Rb, i, j, Lx)
    xi, yi = div(i-1, Lx) + 1, rem(i-1, Lx) + 1
    xj, yj = div(j-1, Lx) + 1, rem(j-1, Lx) + 1

    dist = sqrt( (xi - xj)^2. + (yi - yj)^2. )
    return (Rb / dist)^6.
end

function rydberg_square_energy_density(m, samples_info, labels, Rb, delta)
    prob_samples, samples = samples_info
    psi_coeffs = sqrt.(prob_samples)

    num_samples, nspins = size(samples)
    L = Int(sqrt(nspins))

    energies = zeros(Float64, num_samples)
    for i in 1:(nspins-1)
        # delta term
        energies .-= delta .* samples[:, i]

        # Rabi term, Omega = 1 is the convention
        flipped_psi_coeff = sqrt.(probability(m, format_data(flip_spin(i, samples), labels), L, L, labels))
        energies .-= 0.5 .* (flipped_psi_coeff ./ psi_coeffs)

        # flip spin back
        samples = flip_spin(i, samples)

        # potential term
        for j in (i+1):nspins
            energies .+= rydberg_potential(Rb, i, j, L) .* samples[:, i] .* samples[:, j]
        end
    end

    # last terms for delta and Rabi were missed in the sum over i in 1:(nspins-1)
    energies .-= delta .* samples[:, nspins]

    flipped_psi_coeff = sqrt.(probability(m, format_data(flip_spin(nspins, samples), labels), L, L, labels))
    # flip spin back
    samples = flip_spin(nspins, samples)

    energies .-= 0.5 .* (flipped_psi_coeff ./ psi_coeffs)
    return stats_dict(energies / nspins)
end

function _swap(s1, s2, A)
    _s = copy(s1[:, A])
    s1[:, A] = s2[:, A]
    s2[:, A] = _s
    return s1, s2
end

function swap(m, samples_info, labels::UnitRange{Int64}, A::Vector{Int64})
    #  each sample is independent, we perform a swap against
    #  the next sample in the batch (looping around to the first if we've
    #  reached the end of the batch).
    prob_samples1, samples1 = samples_info
    psi_coeffs1 = sqrt.(prob_samples1)
    L = Int(sqrt(size(samples1, 2)))

    samples2 = circshift(samples1, 1)
    psi_coeffs2 = circshift(psi_coeffs1, 1)
    samples1_, samples2_ = _swap(copy(samples1), copy(samples2), A)

    psi_coeffs1_ = sqrt.(probability(m, format_data(samples1_, labels), L, L, labels))
    psi_coeffs2_ = sqrt.(probability(m, format_data(samples2_, labels), L, L, labels))

    weight1 = psi_coeffs1_ ./ psi_coeffs1
    weight2 = psi_coeffs2_ ./ psi_coeffs2

    return stats_dict(weight1 .* weight2)
end

function abs_staggered_magnetization(samples_info)
    _, samples = samples_info
    num_samples, nspins = size(samples)
    L = Int(sqrt(nspins))
    stag_mag = zeros(Float64, num_samples)

    for n in 1:nspins
        i = div(n-1, L) + 1
        j = rem(n-1, L) + 1
        stag_mag += ((-1)^(i + j)) .* (samples[:, n] .- 0.5)
    end

    stag_mag = abs.(stag_mag / nspins)
    return stats_dict(stag_mag)
end