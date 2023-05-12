module Obsfit

using Visfit
using Formatting, Glob
using Statistics, StatsBase
using LsqFit # Nonconvex
using AstroTime
using LinearAlgebra
using MeasurementSet
import MeasurementSet as MS

include("Models/shape.jl")
include("Models/delta.jl")
include("stokes.jl")

const LAMBDAC = 2.0*pi/299792458.0
const MJD = :modified_julian

mjd(date) = value(modified_julian(date))

function hms2deg(ra)
    h, m, s = split(ra, limit=3)
    15*(abs(parse(Float64,h)) + parse(Float64,m)/60 + parse(Float64,s)/3600)
end

function dms2deg(dec)
    d, m, s = split(dec, limit=3)
    (contains(d, "-") ? -1 : 1)*(abs(parse(Float64,d)) + parse(Float64,m)/60 + parse(Float64,s)/3600)
end

function deg2hms(ra)
    hra = (ra < 0 ? ra+360 : ra)/15
    format("{1:02d} {2:02d} {3:06.3f}", trunc(Int, hra)%24, trunc(Int, 60*hra)%60, 3600*hra%60)
end

function deg2dms(dec)
    ddec = abs(dec)
    format("{1:+03d} {2:02d} {3:05.2f}", sign(dec)*trunc(Int, ddec)%360, trunc(Int, 60*ddec)%60, 3600*ddec%60)
end

function rad2hms(ra)
    hra = 180/pi*(ra < 0 ? ra+2*pi : ra)/15
    format("{1:02d} {2:02d} {3:06.3f}", trunc(Int, hra)%24, trunc(Int, 60*hra%60), 3600*hra%60)
end

function rad2dms(dec)
    ddec = abs(180/pi*dec)
    format("{1:+03d} {2:02d} {3:05.2f}", sign(dec)*trunc(Int, ddec)%360, trunc(Int, 60*ddec%60), 3600*ddec%60)
end

distance(p1, p2) = sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)

function obsfit(blocks=1:1000, output=nothing, sbdirs="CXK_???-6.2.1", maxIter=200)

    rootdir  = splitdir(pwd())[end]
    smtr, prop = split(rootdir, "-")
    println(rootdir)

    mcvs = [split(strip(r), ",") for r=readlines("MCVs-gaia-edr3.csv")[1:end]]
    mcvs = [[m[1], m[2], m[9], m[15], m[16], m[22], m[23]] for m in mcvs]

    results = Dict()

    for sbdir=sort(glob(sbdirs))[blocks]
        cd(sbdir)
        if output != nothing outfile = open(replace(output, "sbdir"=>sbdir), "a") end

        msname = glob(rootdir*"*.ms")[1]
        println(sbdir, "  ", msname, "\n")
        mset = MSet(msname)
        if output != nothing println(outfile, sbdir, "  ", msname, "\n") end
        sblck, vers = split(sbdir, "-")
        sbbnd, sbnum = split(sblck, "_")
       
        targs = Base.filter(s->contains(s.name, "MCV"), MS.sources(mset))
        scans = Base.filter(s->any([s.fieldid==t.sourceid for t=targs]), MS.scans(mset))

        specw = MS.windows(mset)
        chans = Dict{String,Array{Int32}}()
        [haskey(chans,b) ? push!(chans[b],parse(Int32,s)+1) : chans[b]=[parse(Int32,s)+1]
         for (b,z,s)=[split(split(r.name,"_")[2], "#") for r=specw]]
        bands = [Dict(:name=>k, :spw=>extrema(sort(v))) for (k,v)=chans] # needs fix
        
        for scanrec=scans
            scan = scanrec.index
            spwd = extrema(scanrec.spwd)
            spw0 = "$(spwd[1]-1)~$(spwd[2]-1)"
            spw  = "$(spwd[1])~$(spwd[2])"
            band = Base.filter(b->"$(b[:spw][1])~$(b[:spw][2])"==spw, bands)[1]
            targ = Base.filter(t->t.sourceid==scanrec.fieldid, targs)[1]
            sblk = sbbnd == "CXKQ" ? replace(replace(sbnum,"B"=>"D"),"A"=>"C") : sbnum
            mcv  = sort(map(m -> (distance(180/pi.*targ.direction,
                                           (hms2deg(m[4]), dms2deg(m[5]))), m),
                            Base.filter(m -> m[3] == "20"*smtr*"-"*sblk &&
                                        contains(m[7], spw0), mcvs)))[1][2]

            offset = 3600 .*(hms2deg(mcv[4]) - 180/pi*targ.direction[1],
                             dms2deg(mcv[5]) - 180/pi*targ.direction[2])

            println(join([scan, targ.sourceid, replace(targ.name, " "=>"_"),
                          band[:name], spw, "     ", 
                          rad2hms(targ.direction[1]), rad2dms(targ.direction[2]),
                          mcv[4], mcv[5],
                          format("{1:6.2f}", offset[1]), format("{1:6.2f}", offset[2]),
                          mcv[6]], "  "))
            if output != nothing
                println(outfile, join([scan, targ.sourceid, replace(targ.name, " "=>"_"),
                                       band[:name], spw, "     ", 
                                       rad2hms(targ.direction[1]), rad2dms(targ.direction[2]),
                                       mcv[4], mcv[5],
                                       format("{1:6.2f}", offset[1]), format("{1:6.2f}", offset[2]),
                                       mcv[6]], "  "))
            end

            gooddata = MS.select(mset, scan)
            prime = gooddata.primary
            # nlog2 = floor(Int, log2(length(unique(prime["TIME"]))))
            # goodstats = flagrfi!(gooddata, SumThreshold(nlog2, 1))
            # println(count(Base.filter(x->x[1]!=x[2], goodstats)))
            # gooddata = MS.setweights(scandata)
            # cspw = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16]
            # uvdata   = band[:name] == "C" ? MS.filterwin(gooddata, cspw) : MS.filter(gooddata, band[:spw])
            # global nu = uvdata.frequency
            
            freqs = windows(mset)

            flagrow = .!prime["FLAG_ROW"] .& Base.map(w -> w in spwd[1]:spwd[2], prime["DATA_DESC_ID"])
            # println("""$(spwd[1]:spwd[2]),  $(count(.!prime["FLAG_ROW"])),  $(count(flagrow))""")
            # println("""$(size(prime["UVW"])),  $(size(prime["CORRECTED_DATA"])),  $(size(prime["WEIGHT"]))""")

            #  Permutate data for computational efficiency by putting correlations in proximity.
            flag = permutedims(prime["FLAG"][:,:,flagrow], (3, 2, 1))
            uvw  = permutedims(prime["UVW"][:,flagrow], (2, 1))
            data = permutedims(prime["CORRECTED_DATA"][:,:,flagrow], (3, 2, 1))
            wght = permutedims(prime["WEIGHT"][:,flagrow], (2, 1))
            ncor, nchn, npol = size(data)

            # println("$(size(uvw)),  $(size(data)),  $(size(wght))")
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

            # println("$(size(uvw)),  $(size(data)),  $(size(wght))")
            # println("$(count(!isfinite, uvw)),  $(count(!isfinite, data)),  $(count(!isfinite, wght))")

            source1 = delta()
            function model1(u, v)
                # nus = nu./mean(nu)
                # source1(u, v[1:2]).* [v[3].*nus.^v[4] v[5].*nus.^v[6] v[7].*nus.^v[8] v[9].*nus.^v[10]]
                source1(u, v[1:2]).* [v[3] v[4] v[5] v[6]]
            end
            
            function model2(u, v)
                # nus = nu./mean(nu)
                # source1(u, v[1:2]).* [v[3].*nus.^v[4] v[5].*nus.^v[6]]
                source1(u, v[1:2]).* [v[3] v[4]]
            end
            
            psf1 = 2*parse(Float64, mcv[6])
            # start1 = [   0.,    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]
            # lower1 = [-psf1, -psf1, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf]
            # upper1 = [ psf1,  psf1,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf]
            start1 = [   0.,    0., 0.001, 0.001, 0.001, 0.001]
            lower1 = [-psf1, -psf1,  -Inf,  -Inf,  -Inf,  -Inf]
            upper1 = [ psf1,  psf1,   Inf,   Inf,   Inf,   Inf]
            # result1 = Visfit.optimize(model1, Visfit.IQUV, uvdata, start1; lower=lower1, upper=upper1,
            #                           maxIter=maxIter, show_trace=false)
            stokes1 = IQUV
            cdata = concat_rdata(stokes1, data)
            wdata = concat_rweights(stokes1, wght)

            result1 = curve_fit((u, v) -> model1(u, v) |> stokes1, uvw, cdata, wdata, start1;
                                lower=lower1, upper=upper1, maxIter=maxIter,
                                lambda=5, lambda_increase=3, lambda_decrease=1/5)


            psf2 = 2*parse(Float64, mcv[6])
            start2 = [   0.,    0., 0.001, 0.001]
            lower2 = [-psf2, -psf2,  -Inf,  -Inf]
            upper2 = [ psf2,  psf2,   Inf,   Inf]

            # result2 = Visfit.optimize(model2, Visfit.RRLL, uvdata, start2; lower=lower2, upper=upper2,
            #                           maxIter=maxIter, show_trace=false)
            stokes2 = RRLL
            cdata = concat_rdata(stokes2, data)
            wdata = concat_rweights(stokes2, wght)

            result2 = curve_fit((u, v) -> model2(u, v) |> stokes2, uvw, cdata, wdata, start2;
                                lower=lower2, upper=upper2, maxIter=maxIter,
                                lambda=5, lambda_increase=3, lambda_decrease=1/5)

            
            J = result1.jacobian
            stderr = sqrt.(abs.(diag(pinv(J'*J))))

            fld, nam = targ.sourceid, targ.name
            tra   = deg2hms(hms2deg(mcv[4]) + coef(result1)[1]/3600)
            dtra  = format("{1:.3f}", stderr[1]/15/8)
            tdec  = deg2dms(dms2deg(mcv[5]) + coef(result1)[2]/3600)
            dtdec = format("{1:.2f}", stderr[2]/8)
            I,   Q,  U,  V = convert.(Int, round.(1e6.*coef(result1)[3:6]))
            dI, dQ, dU, dV = convert.(Int, round.(1e6.*stderr[3:6]./8))
            
            println("$scan  $fld  $nam  $(band[:name])  $spw  IQUV:  $tra ± $dtra,  $tdec ± $dtdec,  $I ± $dI,  $Q ± $dQ,  $U ± $dU,  $V ± $dV")
            if output != nothing
                println(outfile, "$scan  $fld  $nam  $(band[:name])  $spw  IQUV:  $tra ± $dtra,  $tdec ± $dtdec,  $I ± $dI,  $Q ± $dQ,  $U ± $dU,  $V ± $dV")
            end

            J = result2.jacobian
            stderr = sqrt.(abs.(diag(pinv(J'*J))))

            fld, nam = targ.sourceid, targ.name
            tra   = deg2hms(hms2deg(mcv[4]) + coef(result2)[1]/3600)
            dtra  = format("{1:.3f}", stderr[1]/15/8)
            tdec  = deg2dms(dms2deg(mcv[5]) + coef(result2)[2]/3600)
            dtdec = format("{1:.2f}", stderr[2]/8)
            RR,   LL = convert.(Int, round.(1e6.*coef(result2)[3:4]))
            dRR, dLL = convert.(Int, round.(1e6.*stderr[3:4]./8))
            println("$scan  $fld  $nam  $(band[:name])  $spw  RRLL:  $tra ± $dtra,  $tdec ± $dtdec,  $RR ± $dRR,  $LL ± $dLL\n")
            if output != nothing
                println(outfile, "$scan  $fld  $nam  $(band[:name])  $spw  RRLL:  $tra ± $dtra,  $tdec ± $dtdec,  $RR ± $dRR,  $LL ± $dLL\n")
            end

        end

        if output != nothing close(outfile) end
        cd("..")
    end
end

end
