import Base.Threads.@threads

"""
    Correlation is an N x N matrix

Columns are the Stokes parameters, rows are correlations
corrs::Matrix{Correlation}
"""
struct Correlation
    #  triples are correlation index, and model and Jacobian coefficients
    index::Integer
    coeff::AbstractFloat
end

Correlation(corr) = Correlation(corr[1], corr[2])

"""
   Stokes parameter
"""
mutable struct Stokes
    corrs::Matrix{Correlation}
    model::Union{Vector,Nothing}
    function Stokes(corrs, model)
        cors  = map(c->Correlation(c),corrs)
        length(size(cors)) == 1 ? new(reshape(cors, size(cors)[1],1),nothing) : new(cors, nothing)
    end
end

Stokes(cors) = Stokes(cors, nothing)

num_params(stokes::Stokes) = size(stokes.corrs)[2]
num_chans(stokes::Stokes)  = size(stokes.corrs)[1]

function concat_cdata(stokes::Stokes, data::Array{<:Complex, 3})
    println("cdata, 3")
    vcat([corr.coeff.*vec(data[corr.index,:,:]) for corr in stokes.corrs]...)
end

function concat_rdata(stokes::Stokes, data::Array{<:Complex, 2})
    println("rdata, 2")
    vcat([vcat(corr.coeff.*real(data[:,corr.index]), corr.coeff.*imag(data[:,corr.index]))
          for corr in stokes.corrs]...)
end

function concat_rdata(stokes::Stokes, data::Array{<:Complex, 3})
    hcat([hcat(corr.coeff.*real(data[corr.index,:,:]),
               corr.coeff.*imag(data[corr.index,:,:]))
          for corr in stokes.corrs]...)
end

function concat_cdata(stokes::Stokes, data::Array{<:Complex, 5})
    vcat([corr.coeff.*vec(data[corr.index,:,:,:,:]) for corr in stokes.corrs]...)
end

function concat_rdata(stokes::Stokes, data::Array{<:Complex, 5})
    vcat([vcat(corr.coeff.*real(vec(data[corr.index,:,:,:,:])),
               corr.coeff.*imag(vec(data[corr.index,:,:,:,:])))
          for corr in stokes.corrs]...)
end

function concat_cweights(stokes::Stokes, weights::Array{<:AbstractFloat,3})
    vcat([vec(weights[corr.index,:,:]) for corr in stokes.corrs]...)
end

function concat_rweights(stokes::Stokes, weights::Array{<:AbstractFloat,2})
    vcat([repeat(weights[:,corr.index], 2) for corr in stokes.corrs]...)
end

function concat_rweights(stokes::Stokes, weights::Array{<:AbstractFloat,3})
    hcat([repeat(weights[corr.index,:,:], 1, 2) for corr in stokes.corrs]...)
end

function concat_cweights(stokes::Stokes, weights::Array{<:AbstractFloat,5})
    vcat([vec(weights[corr.index,:,:,:,:]) for corr in stokes.corrs]...)
end

function concat_rweights(stokes::Stokes, weights::Array{<:AbstractFloat,5})
    vcat([repeat(vec(weights[corr.index,:,:,:,:]), 2) for corr in stokes.corrs]...)
end

function (stokes::Stokes)(phasor::Vector{T2} where {T2<:Complex})
    ncor = length(phasor)
    nmod = length(stokes.corrs)*2*ncor
    if isnothing(stokes.model) || length(stokes.model) != nmod
        stokes.model = Array{Float32}(undef, nmod)
    end
    # for (j, ndx) in enumerate(CartesianIndices(stokes.corrs))
    Threads.@threads for j=1:length(CartesianIndices(stokes.corrs))
        n = (j-1)*2*ncor
        @inbounds stokes.model[n+1:n+ncor]        .= real(phasor)
        @inbounds stokes.model[n+ncor+1:n+2*ncor] .= imag(phasor)
    end
    stokes.model
end

function (stokes::Stokes)(phasor::Matrix{T2} where {T2<:Complex})
    ncor = size(phasor)[1]
    nmod = length(stokes.corrs)*2*ncor
    if isnothing(stokes.model) || length(stokes.model) != nmod
        stokes.model = Array{Float32}(undef, nmod)
    end
    for (j, ndx) in enumerate(CartesianIndices(stokes.corrs))
        n = (j-1)*2*ncor
        @inbounds stokes.model[n+1:n+ncor]        .= real(phasor[:,ndx[2]])
        @inbounds stokes.model[n+ncor+1:n+2*ncor] .= imag(phasor[:,ndx[2]])
    end
    stokes.model
end

#  Correlation definitions

#=  MS defined Stokes type enumertion
 0 => undefined
 1 =>  I,  2 =>  Q,  3 =>  U,  4 =>  V,
 5 => RR,  6 => RL,  7 => LR,  8 => LL,
 9 => XX, 10 => XY, 11 => YX, 12 => YY,
13 => RX, 14 => RY, 15 => LX, 16 => LY,
17 => XR, 18 => XL, 19 => YR, 20 => YL,
21 => PP, 22 => PQ, 23 => QP, 24 => QQ,
25 => RCircular,    26 => LCircular,
27 => Linear,       28 => Ptotal,
29 => Plinear,      30 => PFtotal,
31 => PFlinear,     32 => Pangle
=#

#=  FITs defined Stokes type enumeration
 1 =>  I,  2 =>  Q,  3 =>  U,  4 =>  V,
-1 => RR, -2 => LL, -3 => RL, -4 => LR,
-5 => XX, -6 => YY, -7 => XY, -8 => YX,
 6 => PFlinear,      7 => Pangle
 100+ => MS defined Stokes type
=#

#  Proper implementation needs patch to PyCall to efficiently
#  input string arrays (sp. the FEED Table POLARIZATION_TYPE keyword).

#  * only circular feeds are currently defined

RR   = Stokes([(1,  1)])
LL   = Stokes([(4,  1)])
II   = Stokes([(1,  1);
               (4,  1)])
QQ   = Stokes([(2,  1);
               (3,  1)])
UU   = Stokes([(3, -1);
               (2,  1)])
VV   = Stokes([(1,  1);
               (4, -1)])
RRLL = Stokes([(1,  1)  (4,  1)])
IV   = Stokes([(1,  1)  (1,  1);
               (4,  1)  (4, -1)])
QU   = Stokes([(2,  1)  (3, -1);
               (3,  1)  (2,  1)])
IQUV = Stokes([(1,  1)  (2,  1)  (3, -1)  (1,  1);
               (4,  1)  (3,  1)  (2,  1)  (4, -1)])
