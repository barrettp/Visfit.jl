module Visfit

export visfit, nu, defaultmodel, source1, delta,
    RR, LL, II, QQ, UU, VV, RRLL, IV, QU, IQUV,
    seconds, minutes, hours,
    channels, hertz, scans, windows,
    +, -, *, /, value, unit

# using CUDA
using LinearAlgebra, Statistics, StatsBase
using LsqFit # Nonconvex
using MeasurementSet, AstroTime, Unitful
import MeasurementSet as MS

include("stokes.jl")
include("models.jl")
# include("iterators.jl")

# CUDA.allowscalar(false)
# ENV["JULIA_DEBUG"] = "CUDA"

const LAMBDAC = 2.0*pi/299792458.0

nu = nothing

source1 = delta()
defaultmodel(u, v) = source1(u, v[1:2]).*v[3]

# defaultmodel = (u, v) -> source1(u, v[1:2]).*v[3].*(nu./mean(nu)).^v[4]
# defaultmodel = (u, v) -> source1(u, v[1:2]).* [v[3].*(nu./mean(nu)).^v[4] v[5].*(nu./mean(nu)).^v[6]]

#=
function funcmod(var)
end

function ChainRulesCore.rrule(::typeof(funcmod), var::AbstractVector)
    valu = funcmod(var)
    grad = ForwardDiff.gradient(funcmod, var)
    function funcmod_pullback(vbar)
        return NoTangent(), Δ * grad
    end
    return valu, funcmod_pullback
end

function clenshawsum(α, β, a, ϕ, N)
    b = zeros(N+3)
    for k=N+1:-1:1
        b[k] = a[k] + α[k]*b[k+1] + β[k+1]*b[k+2]
    end
    ϕ[1]*a[1] + ϕ[2]*b[2] + β[2]*ϕ[1]*b[3]
end

function setweights(mset::MSet)

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
    return MSet(mset.antenna, mset.frequency, mset.spw, mset.field, mset.scan,
                mset.time, mset.uvw, weight, mset.data)
end
=#

function subscan_exposures(scan, exposure)
    # start, stop = scan[1].time[1]seconds, scan[end].time[end]seconds
    start, stop = scan[1].time[1], scan[end].time[end]
    # println("($start, $stop), $exposure")
    if exposure == nothing
        # length, step = stop-start+1seconds, stop-start+1seconds
        length, step = stop-start+1, stop-start+1
        exposures = [(t, t+length-1) for t in range(start, stop; step=step)]
    # elseif typeof(exposure) <: AstroPeriod
    elseif typeof(exposure) <: Real
        length, step = exposure, exposure
        exposures = [(t, t+length-1) for t in range(start, stop; step=step)]
    elseif typeof(exposure) <: StepRangeLen
        start  = exposure.ref
        stop   = exposure.ref + exposure.len
        step   = exposure.step
        length = exposure.len
        exposures = [(t, t+length-1) for t in range(start, stop; step=step)]
    elseif typeof(exposure) <: NamedTuple && haskey(exposure, :integration) && exposure[:integration] == true
        times  = unique(scan[1].time)
        start  = haskey(exposure, :start)  ? exposure[:start]  : 1
        stop   = haskey(exposure, :stop)   ? exposure[:stop]   : Base.length(times)
        step   = haskey(exposure, :step)   ? exposure[:step]   : 1
        length = haskey(exposure, :length) ? exposure[:length] : 1
        exposures = [(t-1, t+1) for t in times[start:step:stop]]
    elseif typeof(exposure) <: NamedTuple
        start  = haskey(exposure, :start)  ? exposure[:start]  : start
        stop   = haskey(exposure, :stop)   ? exposure[:stop]   : stop
        step   = haskey(exposure, :step)   ? exposure[:step]   : start - start + 1
        length = haskey(exposure, :length) ? exposure[:length] : step
        exposures = [(t, t+length-1) for t in range(start, stop; step=step)]
    end
    # exposures = [(t, t+length-1seconds) for t in range(start, stop; step=step)]
    # exposures = [(t, t+length-1) for t in range(start, stop; step=step)]
    return exposures
end

function subscan_windows(scan, window)
    start, stop = scan.spwd[1], scan.spwd[end]
    # println("($start, $stop), $window")
    if window == nothing
        length, step = stop-start+1, stop-start+1
    elseif typeof(window) <: Number
        length, step = window, window
    elseif typeof(window) <: UnitRange
        start  = hasproperty(window, :start) ? window.start : start
        stop   = hasproperty(window, :stop)  ? window.stop  : stop
        step   = hasproperty(window, :step)  ? window.step  : stop - start + 1
        length = step
    elseif typeof(window) <: StepRange
        start  = hasproperty(window, :start) ? window.start : start
        stop   = hasproperty(window, :stop)  ? window.stop  : stop
        step   = hasproperty(window, :step)  ? window.step  : stop - start + 1
        length = step
    elseif typeof(window) <: NamedTuple
        start  = haskey(window, :start)   ? window[:start]  : start
        stop   = haskey(window, :stop)    ? window[:stop]   : stop
        step   = haskey(window, :step)    ? window[:step]   : stop - start + 1
        length = haskey(window, :length)  ? window[:length] : step
    end
    windows = [(t, t+length-1) for t in range(start, stop; step=step)]
    # println(windows)
    return windows
end

#=
function optimize(model::Function, stokes::Stokes, mset::MSet, start, sumchans=1:typemax(Int32); kwargs...)

    prime  = mset.primary

    spwin  = unique(prime["DATA_DESC_ID"])

    msflag = .!primary["FLAG_ROW"] .& Base.filter(w -> (w+1) in spwin, prime["DATA_DESC_ID"])

    msdata = ("CORRECTED_DATA" in keys(prime) ? prime["CORRECTED_DATA"] ?
              prime["DATA"])[:,:,msflag]
    mswght = prime["WEIGHT"][:,msflag]
    msuvw  = prime["UVW"][:,msflag]

    # Npol, Nchn, Ncor, Nexp, Nspw = size(mset.data)
    # println("$Npol  $Nchn  $Ncor  $Nexp  $Nspw")
    Npolr, Nchan, Ncorr = size(msdata)
    # Ndat = prod(size(mset.data)[2:5])

    freq  = windows(mset).chanfreq[]
    # uvw   = LAMBDAC.* reshape(freq, 1,Nchn,1,1,Nspw) .* reshape(mset.uvw, 3,1,Ncor,Nexp,Nspw)
    uvw   = LAMBDAC .* reshape(freq, 1, Nchn, Ncorr) .* reshape(msuvw, 3, 1, Ncorr)
    

    #  1) optionally average spectral window channels

    ## global nu = reshape(ones(Ncor*Nexp).*mean(freq, dims=1), :)
    ## freq  = reshape(ones(Ncor,Nexp,Nspw) .* reshape(mean(freq, dims=1), 1,1,Nspw),
    ##                 prod(size(mset.data)[3:5]))

    #  2) handle missing data

    data  = sum(mset.data.*mset.weight, dims=2)./sum(mset.weight, dims=2)
    data[isnan.(data)] .= 0
    
    wght  = mean(mset.weight, dims=2)
    cdata = concat_rdata(stokes, data)
    wdata = concat_rweights(stokes, wght)

    #  3) ensure correct size of start (and limits) vector
    #  4) enable Float32 and Float64

    #=
    funcmod = (v) -> sum( wdata.*(stokes(model(uvw, v)) - cdata).^2 )

    println(typeof( funcmod([0., 0., 0., 0.]) ))
    obj  = Model( funcmod )
    addvar!(obj, lower, upper)
    options = MMAOptions(store_trace=true)
    result = Nonconvex.optimize(obj, GCMMA(), start, options=options)
    =#
    
    result = curve_fit((u, v) -> model(u, v) |> stokes, uvw, cdata, wdata, start;
                       lambda=5, lambda_increase=3, lambda_decrease=1/5, kwargs...)

    return result
end
=#
function optimize(model::Function, stokes::Stokes, input::NamedTuple, start, sumchans=1:typemax(Int32); kwargs...)

    #=
    spec  = unique(input.spec)

    flag = .!input.flag .& Base.filter(w -> (w+1) in spwin, input.spec)

    uvw, data, wght = input.uvw[:,flag], input.data[:,:,flag], input.weight[:,flag]
    freq  = hcat([freq[s] for s=input.spec[flag]]...)
    Npol, Nchn, Ncor = size(data)
    uvw   = LAMBDAC .* reshape(freq, 1, Nchn, Ncorr) .* reshape(uvw, 3, 1, Ncorr)
    =#

    #  1) optionally average spectral window channels

    ## global nu = reshape(ones(Ncor*Nexp).*mean(freq, dims=1), :)
    ## freq  = reshape(ones(Ncor,Nexp,Nspw) .* reshape(mean(freq, dims=1), 1,1,Nspw),
    ##                 prod(size(mset.data)[3:5]))

    #  2) handle missing data

    # data  = sum(mset.data.*mset.weight, dims=2)./sum(mset.weight, dims=2)
    # data[isnan.(data)] .= 0
    
    # wght  = mean(mset.weight, dims=2)
    uvw   = input.uvw
    println("$(size(uvw)),  $(size(input.data)),  $(size(input.weight))")
    cdata = concat_rdata(stokes, input.data)
    wdata = concat_rweights(stokes, input.weight)
    println("$(size(uvw)),  $(size(cdata)),  $(size(wdata))")

    #  3) ensure correct size of start (and limits) vector
    #  4) enable Float32 and Float64

    #=
    funcmod = (v) -> sum( wdata.*(stokes(model(uvw, v)) - cdata).^2 )

    println(typeof( funcmod([0., 0., 0., 0.]) ))
    obj  = Model( funcmod )
    addvar!(obj, lower, upper)
    options = MMAOptions(store_trace=true)
    result = Nonconvex.optimize(obj, GCMMA(), start, options=options)
    =#
    
    result = curve_fit((u, v) -> model(u, v) |> stokes, uvw, cdata, wdata, start;
                       lambda=5, lambda_increase=3, lambda_decrease=1/5, kwargs...)

    return result
end
#=
function visfit(msname::String; iterators=nothing, model=defaultmodel, stokes=II, start=[0., 0., 0.001], kwargs...)

    #           start=[0., 0., 0.001], spws=nothing, times=nothing, sumchans=1:typemax(Int32); kwargs...)

    #  Check number of variables
    #  Check phasor shape
    
    results = []
    sbscans = readscans(msname)
    
    #  Iterate over models, exposure (fields:scans), frequencies (windows:channels), correlations (baselines:stokes)
    #  Model, Exposure/Field/Scan, Frequency/Window/Channel, Correlation/Stokes/Baseline

    if isnothing(iterators)
        iterators = (scan=[s.scan for s in sbscans],)
    elseif typeof(iterators) == Scan
        iterators = (scan=iterators,)
    end
    println(iterators)

    # scans = Base.filter(s->s.scan == scan, readscans(msname))
    # subscan_exposures(scans, times)
    # subscan_windoes(scans[1], spws)
    
    for iter in Iterators.product(iterators)
        println(iter)
        scandata = readdata(msname, iter)
        gooddata = setweights(scandata)
        uvdata = MS.filter(gooddata, spw)

        result = optimize(model, stokes, uvdata, start, sumchans; kwargs...)

        J = result.jacobian
        stderr = sqrt.(abs.(diag(pinv(J'*J))))
        println(mean(uvdata.time), "  ", mean(uvdata.frequency)*1e-9, "  ", coef(result), "  ", stderr./8)
        push!(results, result)
    end

    return results
end
=#

#=
  * Implement single loop iterator
  * Implement channel summing
  * Separate iteration and optimization
=#

function visfit(msname::String; scan=scan, model=defaultmodel, stokes=II, start=[0., 0., 0.001],
                spws=nothing, times=nothing, sumchans=1:typemax(Int32),
                datacol="CORRECTED_DATA", weightcol="WEIGHT", kwargs...)

    #  Check number of variables
    #  Check phasor shape
    
    #  Single loop iteratation:
    #      models, exposure (fields:scans), frequencies (windows:channels),
    #      correlations (baselines:stokes)
    #  Model, Exposure/Field/Scan, Frequency/Window/Channel, Correlation/Stokes/Baseline
    #  Separate iteration and optimization
    #  Channel summing
    #  Half-precision data & weights
    #  Global optimization
    #  Closure rules
    #  Outlier flagging
    

    println("stokes: $(size(stokes.corrs))")
    results = []
    mset = MSet(msname)
    scans = Base.filter(s->s.index in scan, MS.scans(mset))
    # println("scan: $scans")
    for time in subscan_exposures(scans, times)
    # for time in times
        # println("time: $time")
        scandata = select(mset, scan, time, weightcol=weightcol, datacol=datacol)
        # println("scandata: $(size(scandata))")

        before, flaglen = count(scandata.primary["FLAG"]), length(scandata.primary["FLAG"])
        # flagrfi!(scandata) # , Flaggers.SumThreshold(9.1), Flaggers.rayleigh_scaling, freqsen=0.5)
        after  = count(scandata.primary["FLAG"])
        # println("$flaglen,  $before,  $after")
        gooddata = scandata

        for spw in subscan_windows(scans[1], spws)
            # uvdata = MS.filter(gooddata, spw)
            prime = gooddata.primary
            freqs = windows(mset)

            flagrow = .!prime["FLAG_ROW"] .& Base.map(w -> w in spw[1]:spw[2], prime["DATA_DESC_ID"])
            # println("""$(spw[1]:spw[2]),  $(count(.!prime["FLAG_ROW"])),  $(count(flagrow))""")
            # println("""$(size(prime["UVW"])),  $(size(prime["CORRECTED_DATA"])),  $(size(prime["WEIGHT"]))""")

            #  Permutate data for computational efficiency by putting correlations in proximity.
            flag = permutedims(prime["FLAG"][:,:,flagrow], (3, 2, 1))
            uvw  = permutedims(prime["UVW"][:,flagrow], (2, 1))
            data = permutedims(prime[datacol][:,:,flagrow], (3, 2, 1))
            wght = permutedims(prime[weightcol][:,flagrow], (2, 1))
            ncor, nchn, npol = size(data)

            # println("$(size(uvw)),  $(size(data)),  $(size(wght))")
            if true # sumchans
                tdata = convert(Array{Union{eltype(data), Missing}, ndims(data)}, data)
                # println("$(size(tdata)),  $(size(reshape(wght, ncor, 1, npol)))")
                tdata[flag] .= missing
                # println("$(typeof(tdata)),   $(size(tdata)),  $(count(ismissing, tdata))")
                rdata, rflag = zeros(eltype(data), ncor, npol), zeros(Bool, ncor, npol)
                for k=1:npol
                    for j=1:ncor
                        rdata[j,k] = mean(skipmissing(tdata[j,:,k]))
                        rflag[j,k] = !isfinite(rdata[j,k]) ? true : false
                    end
                end
                # println(findall(x->x==true, rflag))
                rdata[rflag] .= 0
                data = rdata
                wght[rflag] .= 0
                # println("$(typeof(data)),   $(size(data)),  $(count(ismissing, data)),  $(count(!isfinite, data))")

                # freq  = permutedims([mean(freqs[s].chanfreq) for s=prime["DATA_DESC_ID"][flagrow]], (2, 1))
                freq  = [mean(freqs[s].chanfreq) for s=prime["DATA_DESC_ID"][flagrow]]
                # println(size(freq))
                uvw   = reshape(LAMBDAC .* reshape(freq, ncor, 1) .* reshape(uvw, ncor, 3), :, 3)
                data, wght = reshape(data, :, npol), reshape(wght, :, npol)
            else
                wght  = repeat(reshape(wght, ncor, 1, npol), 1, nchn, 1)
                wght[flag] .= 0.

                freq  = permutedims(hcat([freqs[s].chanfreq for s=prime["DATA_DESC_ID"][flagrow]]...), (2, 1))
                uvw   = reshape(LAMBDAC .* reshape(freq, ncor, nchn, 1) .* reshape(uvw, ncor, 1, 3), :, 3)
                data, wght = reshape(data, :, npol), reshape(wght, :, npol)
            end
            println("$(size(uvw)),  $(size(data)),  $(size(wght))")
            # println("$(count(!isfinite, uvw)),  $(count(!isfinite, data)),  $(count(!isfinite, wght))")

            cdata = concat_rdata(stokes, data)
            wdata = concat_rweights(stokes, wght)
            println("$(size(uvw)),  $(size(cdata)),  $(size(wdata))")
            flux(u, v) = model(u, v) |> stokes

            # println(cdata[1:10])
 
            result = curve_fit(flux, uvw, cdata, wdata, start;
                               lambda=5, lambda_increase=3, lambda_decrease=1/5, kwargs...)

            # result = optimize(model, stokes, input, start, sumchans; kwargs...)

            J = result.jacobian
            stderr = sqrt.(abs.(diag(pinv(J'*J))))
            mfreq = mean(vcat([freqs[s].chanfreq for s=unique(prime["DATA_DESC_ID"][flagrow])]...))

            println(mean(prime["TIME"]), "  ", mfreq*1e-9u"GHz", "   ", # round(mfreq*1e-9, digits=4), "  ",
                    round.(coef(result)[1:2], digits=3), "  ", round.(stderr[1:2], digits=3), "  ",
                    convert.(Int, round.(coef(result)[3:end].*1e6)), "  ",
                    convert.(Int, round.(stderr[3:end]*1e6)))

            push!(results, (time=mean(prime["TIME"]), freq=mfreq, param=result.param, error=stderr))
        end
    end
    results
end

end
