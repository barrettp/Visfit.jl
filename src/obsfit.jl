module Obsfit

function hms2deg(ra)
    h, m, s = split(ra, limit=2)
    15*(parse(Float64,h) + parse(Float64,m)/60 + parse(Float64,s)/3600)
end

function dms2deg(dec)
    d, m, s = split(dec, limit=2)
    parse(Float64,h) + parse(Float64,m)/60 + parse(Float64,s)/3600
end

bandpat  = r"\s+(?<id>\d+)\s+EVLA_(?<name>\w)#"
fieldpat = r"\s+(?<field>\d+)\s+NONE\s+(?<name>MCV\s+J[-+0-9]+)\s+(?<ra>[0-9:.]+)\s+(?<dec>[-+0-9.]+)"
rootdir  = splitdir(pwd())[end]
smtr, prop = split(rootdir, "-")
println(rootdir)

mcvs = [split(strip(r), ",") for r=readlines("MCVs-gaia-edr3.csv")[1:end]]
println(mcvs)
mcvs = [[m[0], m[1], m[8], m[14], m[15], m[21], m[22]] for m in mcvs]
println(mcvs)

results = Dict()

end
