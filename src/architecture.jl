struct RNNCell2D{F,A,V,S}
    σ::F
    Wih::A # horizontal inputs
    Wiv::A # vertical inputs
    Whh::A # hortizontal hiddens
    Whv::A # vertical hiddens
    b::V
    state0::S
end
  
RNNCell2D(in::Integer, out::Integer, σ=elu; init=Flux.glorot_uniform, initb=zeros, init_state=zeros) = 
RNNCell2D(σ, init(out, in), init(out, in), init(out, out), init(out, out), initb(Float32, out), init_state(Float32, out,1))

function (m::RNNCell2D{F,A,V,<:AbstractMatrix{T}})(hh, hv, xh::Union{AbstractVecOrMat{T},Flux.OneHotArray}, xv::Union{AbstractVecOrMat{T},Flux.OneHotArray}) where {F,A,V,T}
    σ, Wih, Wiv, Whh, Whv, b = m.σ, m.Wih, m.Wiv, m.Whh, m.Whv, m.b
    h = σ.(Wih*xh .+ Wiv*xv .+ Whh*hh .+ Whv*hv .+ b)
    sz = size(xh)
    return h, reshape(h, :, sz[2:end]...)
end

Flux.@functor RNNCell2D

function Base.show(io::IO, l::RNNCell2D)
    print(io, "RNNCell2D(", size(l.Wih, 2), ", ", size(l.Wih, 1))
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end