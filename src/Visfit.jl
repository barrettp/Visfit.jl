module Visfit

# using Printf
# using CUDA
# using JuMP, Ipopt
# using Convex, SCS
using PyCall, Statistics, StatsBase
using LsqFit # , CMPFit, Optim

import Base.Threads.@threads

# CUDA.allowscalar(false)

ENV["JULIA_DEBUG"] = "CUDA"

const sec2rad = pi/180.0/3600.0

struct MSet
    antenna::Array
    frequency::Array
    spw::Array
    time::Array
    uvw::Array
    weight::Array
    data::Array
end

function read(vis::String, scan::Int;
              weightcol::String="WEIGHT", datacol::String="CORRECTED_DATA")

    ENV["CASAPATH"] = "/home/paul/casapy/ linux"

    py"""
    import sys
    from casac import casac
    ms = casac.ms()
    tb = casac.table()

    sys.path += ["/data/JVLA/17B-144/EF_Eri-A-5.4.1/"]
    """

    tb_open, tb_close = py"tb.open", py"tb.close"
    tb_open(vis*"/ANTENNA")
    antenna = pycall(py"tb.getcol", Array{String}, "NAME")
    tb_close()
    
    tb_open(vis*"/SPECTRAL_WINDOW")
    frequency = pycall(py"tb.getcol", Array{Float32,2}, "CHAN_FREQ")
    tb_close()

    selectcol = ("uvw", lowercase(weightcol), "data_desc_id", "time", lowercase(datacol))
    selecttyp = (Array{Float64,2}, Array{Float32,2}, Array{Int32,1}, Array{Float64,1}, Array{Complex{Float32},3})

    ms_open, ms_select, ms_getdata, ms_close = py"ms.open", py"ms.select", py"ms.getdata", py"ms.close"
    ms_open(vis)
    ms_select(Dict([("scan_number", scan)]))
    uvw  = pycall(ms_getdata, PyDict{String,selecttyp[1]}, selectcol[1])[selectcol[1]]
    wght = pycall(ms_getdata, PyDict{String,selecttyp[2]}, selectcol[2])[selectcol[2]]
    spwd = pycall(ms_getdata, PyDict{String,selecttyp[3]}, selectcol[3])[selectcol[3]]
    time = pycall(ms_getdata, PyDict{String,selecttyp[4]}, selectcol[4])[selectcol[4]]
    data = pycall(ms_getdata, PyDict{String,selecttyp[5]}, selectcol[5])[selectcol[5]]
    ms_close()

    spwd = unique(spwd).+1
    time = unique(time)

    Npol, Nchn = size(data)[1:2]
    Ncor = div(length(antenna)*(length(antenna)-1), 2)
    Nexp = length(time)
    Nspw = length(spwd)
    #=
    println(size(antenna), "  ", size(frequency), "  ", size(time), "  ", size(spwd), "  ",
            size(reshape(uvw, (3, Ncor, Nexp, Nspw))), "  ",
            size(repeat(reshape(wght, (Npol, 1, Ncor, Nexp, Nspw)), 1,Nchn,1,1,1)), "  ",
            size(reshape(data, (Npol, Nchn, Ncor, Nexp, Nspw))))
    =#
    MSet(antenna, frequency, spwd, time, reshape(uvw, (3, Ncor, Nexp, Nspw)),
         repeat(reshape(wght, (Npol, 1, Ncor, Nexp, Nspw)), 1,Nchn,1,1,1),
         reshape(data, (Npol, Nchn, Ncor, Nexp, Nspw)))
end

function setWeights(mset::MSet)

    Npol, Nchn, Ncor, Nexp, Nspw = size(mset.data)

    weight = deepcopy(mset.weight)
    for p in 1:Npol
        for (j::UInt32, s::UInt32) in collect(enumerate(mset.spw))
            datam = Array{Float64}(undef,Nchn)
            @threads for k in 1:Nchn
                I = findall(!iszero, mset.weight[p,k,:,:,j])
                datam[k] = mean(abs.(mset.data[p,k,:,:,j][I]))
            end
            datad = abs.(datam .- median(datam))
            datae = median(datad)
            weight[p,:,:,:,j] .*= map(x -> x<10. ? 1. : 0.,
                                      datae != 0. ? datad./datae :
                                      fill(100.0,size(datad)))
        end
    end
    #=
    for j in 1:length(mset.spw)
        println(j, "  ",
                join([all(iszero, weight[1,c,:,:,j]) ? "$c " : "" for c in 1:Nchn]), ",  ",
                join([all(iszero, weight[2,c,:,:,j]) ? "$c " : "" for c in 1:Nchn]), ",  ",
                join([all(iszero, weight[3,c,:,:,j]) ? "$c " : "" for c in 1:Nchn]), ",  ",
                join([all(iszero, weight[4,c,:,:,j]) ? "$c " : "" for c in 1:Nchn]))
    end
    =#
    MSet(mset.antenna, mset.frequency, mset.spw, mset.time, mset.uvw, weight, mset.data)
end

function filter(mset::MSet, spw=1:typemax(UInt32), time=[0.,Inf])

    s = max(first(mset.spw),first(spw)):min(last(mset.spw),last(spw))
    j = first(s)-first(mset.spw)+1:last(s)-first(mset.spw)+1
    #=
    println(size(mset.antenna), "  ", size(mset.frequency[:,s]), "  ", size(mset.time), "  ",
            size(mset.spw), "  ", size(mset.uvw[:,:,:,j]), "  ", size(mset.weight[:,:,:,:,j]), "  ",
            size(mset.data[:,:,:,:,j]))
    =#
    MSet(mset.antenna, view(mset.frequency, :,s), view(mset.spw, j), mset.time,
         view(mset.uvw, :,:,:,j), view(mset.weight, :,:,:,:,j), view(mset.data, :,:,:,:,j))
end

function delta(Npars)
    function _delta(x, p)
        p1::Float64, p2::Float64 = p[1]*sec2rad, p[2]*sec2rad
        f  = x[1,:].*p1 .+ x[2,:].*p2 .+ x[3,:].*(sqrt(1.0 - p1*p1 - p2*p2) - 1.0)
        c, s = cos.(f), sin.(f)
        # A  = vcat(p[3].*c, p[3].*s, p[3].*c, p[3].*s)
        A  = vcat(p[3].*c, p[3].*s)
    end
end

function deltaIn(A::Array{Float64}, x, p)
    p1, p2 = p[1]*sec2rad, p[2]*sec2rad
    f  = x[1,:].*p1 .+ x[2,:].*p2 .+ x[3,:].*(sqrt(1.0 - p1*p1 - p2*p2) - 1.0)
    c, s = cos.(f), sin.(f)
    A .= vcat(p[3].*c, p[3].*s, p[3].*c, p[3].*s)
end

function deltaJacIn(J::Array{Float64,2}, x, p)
    p1, p2 = p[1]*sec2rad, p[2]*sec2rad
    f = x[1,:].*p1 .+ x[2,:].*p2 .+ x[3,:].*(sqrt(1.0 - p1*p1 - p2*p2) - 1.0)
    c, s = cos.(f), sin.(f)
    x13 = x[1,:] .- x[3,:].*(p1/sqrt(1.0 - p1*p1 - p2*p2))
    x23 = x[2,:] .- x[3,:].*(p2/sqrt(1.0 - p1*p1 - p2*p2))
    J[:,1] .= vcat(-p[3].*s.*x13, p[3].*c.*x13, -p[3].*s.*x13, p[3].*c.*x13)
    J[:,2] .= vcat(-p[3].*s.*x23, p[3].*c.*x23, -p[3].*s.*x23, p[3].*c.*x23)
    J[:,3] .= vcat(c, s, c, s)
    J
end
    
function optimize(mset::MSet, model::Symbol, start::Array{Float64}, stokes::String; kwargs...)

    Npol, Nchn, Ncor, Nexp, Nspw = size(mset.data)
    Ndat = prod(size(mset.data)[2:5])

    xProduct = Dict(
        [("I",  [(1, 1.), (4, 1.)]),
         ("Q",  [(2, 1.), (3, 1.)]),
         ("U",  [(2, 1.), (3,-1.)]),
         ("V",  [(1, 1.), (4,-1.)]),
         ("RR", [(1, 1.)]),
         ("LL", [(4, 1.)])])
    println(stokes, "  ", xProduct[stokes])

    freq = reshape(mset.frequency, 1,Nchn,1,1,Nspw)
    uvw  = reshape(mset.uvw, 3,1,Ncor,Nexp,Nspw)
    lambda = 2.0*pi/299792458.0 * reshape(freq.*uvw, 3,Ndat)
    cdata = vcat([vcat(w.*reshape(real(mset.data[p,:,:,:,:]),Ndat),
                       w.*reshape(imag(mset.data[p,:,:,:,:]),Ndat))
                  for (p, w)=xProduct[stokes]]...)
    wdata = vcat([vcat(reshape(mset.weight[p,:,:,:,:],Ndat),
                       reshape(mset.weight[p,:,:,:,:],Ndat))
                  for (p, w)=xProduct[stokes]]...)
    println(size(cdata), size(wdata))

    if haskey(kwargs, :inplace) && kwargs[:inplace] == true
        result = curve_fit(deltaIn, deltaJacIn, lambda, cdata, wdata, start; kwargs...)
    else
        result = curve_fit(delta, lambda, cdata, wdata, start; kwargs...)
    end        
    result
end

function visfit(vis::String, scan::Int, start::Array{Float64}=[0, 0, 0],
                stokes="I", spw=1:typemax(UInt32), time::Array{Float64}=[0, Inf];
                kwargs...)

    scandata = read(vis, scan)
    gooddata = setWeights(scandata)
    uvdata = filter(gooddata, spw)

    result = optimize(uvdata, :delta, start, stokes; kwargs...)
    println(dof(result)*4/length(uvdata.data), "  ", coef(result), "  ",
            stderror(result))
    result
end

end
