# using ChainRulesCore, ForwardDiff
using Statistics: mean

import Base.Threads.@threads

const SEC2RAD = pi/180/3600

function delta!(shape::Shape, uvw, var)
    ncor, nthr = size(uvw)[1], Threads.nthreads()
    step, remain = divrem(ncor, nthr)
    if isnothing(shape.phasor) || length(shape.phasor) != ncor
        shape.phasor = complex.(zeros(Float32, ncor))
    end

    dRA, dDec = SEC2RAD.*(var[1:2] .+ shape.center)
    w   = sqrt(1.0 - dRA*dRA - dDec*dDec)
    if isnothing(shape.values) || any(abs.(var.-shape.values) .> maximum(eps.(var)))
        shape.values = var

        Threads.@threads for thr = 1:nthr
            j1, j2 = (thr-1)*step+1, thr*step+(thr == nthr ? remain : 0)
            u1, u2, u3 = (view(uvw, :, j)[j1:j2] for j=1:3)
            @inbounds shape.phasor[j1:j2] .= tophasor.(dRA.*u1 .+ dDec.*u2 .+ (w-1).*u3)
        end
    end
    shape.phasor
end

delta(center=[0., 0.]) = Shape([Variable("ΔRA", 0.), Variable("ΔDec", 0.)], delta!, center)

#=
function delta_gradient(uvw, var)
    Nchn, Nobs, Nspw = size(uvw)[2], prod(size(uvw)[3:4]), size(uvw)[5]

    dRA, dDec = SEC2RAD.*(var[1:2] .+ shape.center)
    w   = sqrt(1.0 - dRA*dRA - dDec*dDec)

    @threads for spw = 1:Nspw
        u1, u2, u3 = (reshape(view(uvw, j,:,:,:spw), Nchn,Nobs) for j=1:3)
        # j1, j2 = (spw-1)*Nobs+1, spw*Nobs
        phi = dRA.*u1 .+ dDec.*u2 .+ (w-1).*u3
        u113, u223 = u1 .- dRA/w*u3, u2 .- dDec/w*u3
        du1 = reshape(mean(tophasor(pi/2+phi).*u113, dims=1), Nobs)
        du2 = reshape(mean(tophasor(pi/2+phi).*u223, dims=1), Nobs)
    end
end

function deltacomp!(stokes, xdata, mbuf, ubuf, offset=[0, 0])

    ctype, dtype = Complex{Float64}, Float64
    Nchn, Ndat = size(xdata)[2], prod(size(xdata)[3:5])
    Mdat = prod(size(xdata)[2:5])

    # println("Entering deltacomp!")
    if has_cuda() && false
        gmod = unsafe_wrap(CuArray{ctype}, convert(CUDA.CuPtr{ctype}, mbuf), Mdat)
        cmod = unsafe_wrap(Array{ctype}, convert(Ptr{ctype}, mbuf), Mdat)
        # gdlr = unsafe_wrap(CuArray{dtype}, convert(CUDA.CuPtr{dtype}, mbuf), Mdat)
        # cdlr = unsafe_wrap(Array{dtype}, convert(Ptr{dtype}, mbuf), Mdat)
        # gdli = unsafe_wrap(CuArray{dtype}, convert(CUDA.CuPtr{dtype}, mbuf), Mdat)
        # cdli = unsafe_wrap(Array{dtype}, convert(Ptr{dtype}, mbuf), Mdat)
        guvw = unsafe_wrap(CuArray{dtype}, convert(CUDA.CuPtr{dtype}, ubuf), (3, Mdat))
        cuvw = unsafe_wrap(Array{dtype}, convert(Ptr{dtype}, ubuf), (3, Mdat))
        @inbounds cuvw .= reshape(xdata, 3, Mdat)
    else
        phas = zeros(size(xdata)[2:5])
    end
    gi, gr = zeros(Ndat), zeros(Ndat)

    function _deltacomp!(F::AbstractArray, x, p)

        # println("Entering _deltacomp!")
        p1, p2 = SEC2RAD.*(p[1:2] .+ offset)

        if has_cuda() && false
            Nthr = 256
            Ntot = Mdat == Int(floor(Mdat/Nthr)*Nthr) ? Mdat : Int((floor(Mdat/Nthr)+1)*Nthr)
            pars = CuArray([p1, p2])
            println(Nthr, "  ", div(Ntot,Nthr), "  ", size(cmod), "  ", size(cuvw))
            @cuda threads=Nthr blocks=div(Ntot,Nthr) deltaGPU!(gmod, pars, guvw)
            # @cuda threads=Nthr blocks=div(Ntot,Nthr) deltaGPU!(gdlr, gdli, pars, guvw)
            synchronize()
            @inbounds gr .= reshape(mean(real.(reshape(cmod, Nchn, Ndat)), dims=1), Ndat)
            @inbounds gi .= reshape(mean(imag.(reshape(cmod, Nchn, Ndat)), dims=1), Ndat)
            # @inbounds gr .= reshape(mean(reshape(cdlr, Nchn, Ndat), dims=1), Ndat)
            # @inbounds gi .= reshape(mean(reshape(cdli, Nchn, Ndat), dims=1), Ndat)
        else
            w   = sqrt(1.0 - p1*p1 - p2*p2)
            Nchn, Nobs, Nspw = size(x)[2], prod(size(x)[3:4]), size(x)[5]
            @threads for spw = 1:Nspw
                x1, x2, x3 = view(x,1,:,:,:,spw), view(x,2,:,:,:,spw), view(x,3,:,:,:,spw)
                j1, j2 = (spw-1)*Nobs+1, spw*Nobs
                @inbounds phas[:,:,:,spw] .= p1.*x1 .+ p2.*x2 .+ (w - 1).*x3
                @inbounds gr[j1:j2] .= reshape(mean(cos.(reshape(phas[:,:,:,spw], Nchn, Nobs)), dims=1), Nobs)
                @inbounds gi[j1:j2] .= reshape(mean(sin.(reshape(phas[:,:,:,spw], Nchn, Nobs)), dims=1), Nobs)
            end
        end
        
        if stokes == "IQUV"
            P = [p[3], p[3], p[3], p[3], p[4], p[4], p[4], p[4],
                 p[5], p[5], p[5], p[5], p[6], p[6], p[6], p[6]]
            A = [gr, gi, gr, gi, gr, gi, gr, gi, gr, gi, gr, gi, gr, gi, gr, gi]
        elseif stokes in ["IV", "QU"]
            P = [p[3], p[3], p[3], p[3], p[4], p[4], p[4], p[4]]
            A = [gr, gi, gr, gi, gr, gi, gr, gi]
        elseif stokes == "RRLL"
            P, A = [p[3], p[3], p[4], p[4]], [gr, gi, gr, gi]
        elseif stokes in ["RR", "LL"] 
            P, A = [p[3], p[3]], [gr, gi]
        else
            P, A = [p[3], p[3], p[3], p[3]], [gr, gi, gr, gi]
        end
        for j=1:length(P) @inbounds F[(j-1)*Ndat+1:j*Ndat] .= P[j].*A[j] end
        return
    end
end

function deltaJacInm(stokes, dsize, offset=[0,0])
    Ndat = prod(dsize[2:5])
    sc   = Array{Tuple{Float64,Float64},4}(undef, dsize[2:5])
    sc  .= [(0, 0)]
    x113 = zeros(Float64,dsize[2:5])
    x223 = zeros(Float64,dsize[2:5])
    Mdat = div(Ndat,64)
    zro  = zeros(Mdat)
    dnr, dni  = zeros(Mdat), zeros(Mdat)
    d1r, d1i  = zeros(Mdat), zeros(Mdat)
    d2r, d2i  = zeros(Mdat), zeros(Mdat)

    function _deltaJacInm(J::Array{Float64,2}, x, p)
        p1, p2 = SEC2RAD.*(p[1:2] .+ offset)
        w     = sqrt(1.0 - p1*p1 - p2*p2)
        x1, x2, x3 = x[1,:,:,:,:], x[2,:,:,:,:], x[3,:,:,:,:]
        @. sc = sincos(p1*x1 + p2*x2 + (w - 1)*x3)
        @. x113, x223 = x1 - p1/w*x3, x2 - p2/w*x3
        d1r  .= reshape(mean(-first.(sc).*x113, dims=1), Mdat)
        d1i  .= reshape(mean(  last.(sc).*x113, dims=1), Mdat)
        d2r  .= reshape(mean(-first.(sc).*x223, dims=1), Mdat)
        d2i  .= reshape(mean(  last.(sc).*x223, dims=1), Mdat)
        dnr  .= reshape(mean(  last.(sc), dims=1), Mdat)
        dni  .= reshape(mean( first.(sc), dims=1), Mdat)

        if stokes == "IQUV"
            P = reshape([p[3], p[3], p[3], p[3], p[4], p[4], p[4], p[4],
                         p[5], p[5], p[5], p[5], p[6], p[6], p[6], p[6],
                         p[3], p[3], p[3], p[3], p[4], p[4], p[4], p[4],
                         p[5], p[5], p[5], p[5], p[6], p[6], p[6], p[6],
                            1,    1,    1,    1,    1,    1,    1,    1,
                            1,    1,    1,    1,    1,    1,    1,    1,
                            1,    1,    1,    1,    1,    1,    1,    1,
                            1,    1,    1,    1,    1,    1,    1,    1,
                            1,    1,    1,    1,    1,    1,    1,    1,
                            1,    1,    1,    1,    1,    1,    1,    1,
                            1,    1,    1,    1,    1,    1,    1,    1,
                            1,    1,    1,    1,    1,    1,    1,    1], 16, 6)
            A = reshape([ d1r,  d1i,  d1r,  d1i,  d1r,  d1i,  d1r,  d1i,
                          d1r,  d1i,  d1r,  d1i,  d1r,  d1i,  d1r,  d1i,
                          d2r,  d2i,  d2r,  d2i,  d2r,  d2i,  d2r,  d2i,
                          d2r,  d2i,  d2r,  d2i,  d2r,  d2i,  d2r,  d2i,
                          dnr,  dni,  dnr,  dni,  zro,  zro,  zro,  zro,
                          zro,  zro,  zro,  zro,  zro,  zro,  zro,  zro,
                          zro,  zro,  zro,  zro,  dnr,  dni,  dnr,  dni,
                          zro,  zro,  zro,  zro,  zro,  zro,  zro,  zro,
                          zro,  zro,  zro,  zro,  zro,  zro,  zro,  zro,
                          zro,  zro,  zro,  zro,  dnr,  dni,  dnr,  dni], 16, 6)
        elseif stokes in ["IV", "QU"]
            P = reshape([p[3], p[3], p[3], p[3], p[4], p[4], p[4], p[4],
                         p[3], p[3], p[3], p[3], p[4], p[4], p[4], p[4],
                            1,    1,    1,    1,    1,    1,    1,    1,
                            1,    1,    1,    1,    1,    1,    1,    1], 8, 4)
            A = reshape([ d1r,  d1i,  d1r,  d1i,  d1r,  d1i,  d1r,  d1i,
                          d2r,  d2i,  d2r,  d2i,  d2r,  d2i,  d2r,  d2i,
                          dnr,  dni,  dnr,  dni,  zro,  zro,  zro,  zro,
                          zro,  zro,  zro,  zro,  dnr,  dni,  dnr,  dni], 8, 4)
        elseif stokes == "RRLL"
            P = reshape([p[3], p[3], p[4], p[4], p[3], p[3], p[4], p[4],
                            1,    1,    1,    1,    1,    1,    1,    1], 4, 4)
            A = reshape([ d1r,  d1i,  d1r,  d1i,  d2r,  d2i,  d2r,  d2i,
                          dnr,  dni,  zro,  zro,  zro,  zro,  dnr,  dni], 4, 4)
        elseif stokes in ["RR", "LL"]
            P = reshape([p[3], p[3], p[3], p[3],    1,    1], 2, 3)
            A = reshape([ d1r,  d1i,  d2r,  d2i,  dnr,  dni], 2, 3)
        else
            P = reshape([p[3], p[3], p[3], p[3], p[3], p[3], p[3], p[3],
                            1,    1,    1,    1], 4, 3)
            A = reshape([ d1r,  d1i,  d1r,  d1i,  d2r,  d2i,  d2r,  d2i,
                          dnr,  dni,  dnr,  dni], 4, 3)
        end
        for j=1:length(p)
            for k=1:length(P[:,j])
                J[(k-1)*Mdat+1:k*Mdat,j] .= P[k,j].*A[k,j]
            end
        end
        return
    end
end

function deltaDderInm(stokes, x, dsize, offset=[0,0])
    Ndat = prod(dsize[2:5])
    sc   = Array{Tuple{Float64,Float64},4}(undef, dsize[2:5])
    sc  .= [(0, 0)]
    p113, p123, p223 = zeros(dsize[2:5]), zeros(dsize[2:5]), zeros(dsize[2:5])
    x113, x123 = zeros(dsize[2:5]), zeros(dsize[2:5])
    x213, x223 = zeros(dsize[2:5]), zeros(dsize[2:5])
    Mdat = div(Ndat,64)
    h11r, h11i = zeros(Mdat), zeros(Mdat)
    h12r, h12i = zeros(Mdat), zeros(Mdat)
    h13r, h13i = zeros(Mdat), zeros(Mdat)
    h22r, h22i = zeros(Mdat), zeros(Mdat)
    h23r, h23i = zeros(Mdat), zeros(Mdat)

    function _deltaDderInm(dder, p, v)
        p1, p2 = SEC2RAD.*(p[1:2] .+ offset)
        v1, v2 = v[1:2]
        w     = sqrt(1.0 - p1*p1 - p2*p2)
        sc   .= sincos.(p1.*x[1,:,:,:,:] .+ p2.*x[2,:,:,:,:] .+
                        (w - 1).*x[3,:,:,:,:])
        p113 .= (w^2 + p1^2)/w^3 .* x[3,:,:,:,:]
        p123 .=        p1*p2/w^3 .* x[3,:,:,:,:]
        p223 .= (w^2 + p2^2)/w^3 .* x[3,:,:,:,:]
        x113 .= x[1,:,:,:,:] .- p1/w.*x[3,:,:,:,:]
        x123 .= x[1,:,:,:,:] .- p2/w.*x[3,:,:,:,:]
        x213 .= x[2,:,:,:,:] .- p1/w.*x[3,:,:,:,:]
        x223 .= x[2,:,:,:,:] .- p2/w.*x[3,:,:,:,:]
        
        if stokes == "IQUV"
            #=
            J[:,1] .= vcat(d1r, d1i, d1r, d1i, d1r, d1i, d1r, d1i,
                           d1r, d1i, d1r, d1i, d1r, d1i, d1r, d1i)
            J[:,2] .= vcat(d2r, d2i, d2r, d2i, d2r, d2i, d2r, d2i,
                           d2r, d2i, d2r, d2i, d2r, d2i, d2r, d2i)
            J[:,3] .= vcat(  r,   i,   r,   i,   r,   i,   r,   i)
            J[:,4] .= vcat(  r,   i,   r,   i,   r,   i,   r,   i)
            J[:,5] .= vcat(  r,   i,   r,   i,   r,   i,   r,   i)
            J[:,6] .= vcat(  r,   i,   r,   i,   r,   i,   r,   i)
            =#
        elseif stokes in ["IV", "QU"]
            #=
            J[:,1] .= vcat(d1r, d1i, d1r, d1i, d1r, d1i, d1r, d1i)
            J[:,2] .= vcat(d2r, d2i, d2r, d2i, d2r, d2i, d2r, d2i)
            J[:,3] .= vcat(  r,   i,   r,   i)
            J[:,4] .= vcat(  r,   i,   r,   i)
            =#
        elseif stokes == "RRLL"
            h11r .= reshape(mean(-p[3].*( p113.*first.(sc) .+ x113.^2 .* last.(sc)) .+
                                 -p[4].*( p113.*first.(sc) .+ x113.^2 .* last.(sc)), dims=1), Mdat)
            h11i .= reshape(mean(-p[3].*( p113.* last.(sc) .+ x113.^2 .*first.(sc)) .+
                                 -p[4].*( p113.* last.(sc) .+ x113.^2 .*first.(sc)), dims=1), Mdat)
            h12r .= reshape(mean(-p[3].*(-p123.*first.(sc) .+ x113.*x223.* last.(sc)) .+
                                 -p[4].*(-p123.*first.(sc) .+ x113.*x223.* last.(sc)), dims=1), Mdat)
            h12i .= reshape(mean(-p[3].*( p123.* last.(sc) .+ x113.*x223.*first.(sc)) .+
                                 -p[4].*( p123.* last.(sc) .+ x113.*x223.*first.(sc)), dims=1), Mdat)
            h13r .= reshape(mean(-x113.*first.(sc), dims=1), Mdat)
            h13i .= reshape(mean( x113.* last.(sc), dims=1), Mdat)
            h14r  = h13r
            h14i  = h13i
            h22r .= reshape(mean(-p[3].*( p223.*first.(sc) .+ x223.^2 .* last.(sc)) .+
                                 -p[4].*( p223.*first.(sc) .+ x223.^2 .* last.(sc)), dims=1), Mdat)
            h22i .= reshape(mean(-p[3].*( p223.* last.(sc) .+ x223.^2 .*first.(sc)) .+
                                 -p[4].*( p223.* last.(sc) .+ x223.^2 .*first.(sc)), dims=1), Mdat)
            h23r .= reshape(mean(-x223.*first.(sc), dims=1), Mdat)
            h23i .= reshape(mean( x223.* last.(sc), dims=1), Mdat)
            h24r  = h23r
            h24i  = h23i
            ndxr, ndxi = 1:Mdat, Mdat+1:2*Mdat
            dder[ndxr] .= (h11r.*v1^2 .+ h12r.*2*v1*v2 .+ h13r.*2*v1*v[3] .+ h14r.*2*v1*v[4] .+
                           .+ h22r.*v2^2 .+ h23r.*2*v2*v[3] .+ h24r.*2*v2*v[4])
            dder[ndxi] .= (h11i.*v1^2 .+ h12i.*2*v1*v2 .+ h13i.*2*v1*v[3] .+ h14i.*2*v1*v[4] .+
                           .+ h22i.*v2^2 .+ h23i.*2*v2*v[3] .+ h24i.*2*v2*v[4])
        elseif stokes in ["RR", "LL"]
            #=
            J[:,1] .= vcat(d1r, d1i)
            J[:,2] .= vcat(d2r, d2i)
            J[:,3] .= vcat(  r,   i)
            =#
        else
            #=
            J[:,1] .= vcat(d1r, d1i, d1r, d1i)
            J[:,2] .= vcat(d2r, d2i, d2r, d2i)
            J[:,3] .= vcat(  r,   i,   r,   i)
            =#
        end
        return
    end
end

function deltaIQUVRes!(stokes, xdata, cdata, wdata, offset=[0, 0])
    #  Return residual
    N  = size(xdata)[2]
    f  = Array{Float64,1}(undef,N)
    sc = Array{Tuple{Float64,Float64},1}(undef,N)
    s  = Array{Float64,1}(undef,N)
    c  = Array{Float64,1}(undef,N)
    println(N)

    function _deltaIQUVRes(r, p)
        p1::Float64, p2::Float64 = SEC2RAD.*[p[1] + offset[1], p[2] + offset[2]]
        f  .= xdata[1,:].*p1 .+ xdata[2,:].*p2 .+ xdata[3,:].*(sqrt(1.0 - p1*p1 - p2*p2) - 1.0)
        sc .= sincos.(f)
        s  .= first.(sc)
        c  .= last.(sc)
        if stokes == "IQUV"
            # A .= vcat(p[3].*c, p[3].*s, p[3].*c, p[3].*s, p[4].*c, p[4].*s, p[4].*c, p[4].*s,
            #           p[5].*c, p[5].*s, p[5].*c, p[5].*s, p[6].*c, p[6].*s, p[6].*c, p[6].*s)
            mdata = vcat(c, s, c, s)
            r[1] = 0
            r[2] = 0
            r[3] = sum( ((cdata[     1: 4*N] - p[3].*mdata)/wdata[     1: 4*N])^2 )
            r[4] = sum( ((cdata[ 4*N+1: 8*N] - p[4].*mdata)/wdata[ 4*N+1: 8*N])^2 )
            r[5] = sum( ((cdata[ 8*N+1:12*N] - p[5].*mdata)/wdata[ 8*N+1:12*N])^2 )
            r[6] = sum( ((cdata[12*N+1:16*N] - p[6].*mdata)/wdata[12*N+1:16*N])^2 )
        elseif stokes in ["IV", "QU", "RRLL"]
            A .= vcat(p[3].*c, p[3].*s,  p[4].*c,  p[4].*s)
        elseif stokes in ["RR", "LL"] 
            # A .= vcat(p[3].*c, p[3].*s)
            mdata = vcat(c, s)
            res  = sum(wdata.*(cdata .- p[3].*mdata).^2)
            r[1] = res
            r[2] = res
            r[3] = res
        else
            A .= vcat(p[3].*c, p[3].*s,  p[3].*c,  p[3].*s)
        end
        # return A
        return
    end
end

#  GPU version

#=
function deltaGPU!(model, p, x)
    Nchn, Ndat = size(x)[2:3]
    j = (blockIdx().x - 1)*blockDim().x + threadIdx().x

    if j <= Ndat
        vx  = view(x,1:3,1:Nchn,j)
        sum = 0+im*0
        @cuda dynamic=true threads=div(Nchn,2) blocks=1 sum_channels!(sum, p, vx)
        CUDA.device_synchronize()

        if j == 1
            model[j] = sum
        end
    end
    return nothing
end

function sum_channels!(sum, p, x)

    #  blockdim().x <= Nchn/2
    
    Nchn = size(x)[2]
    temp = @cuStaticSharedMem(Complex{Float64}, 512)

    j, s = threadIdx().x, blockDim().x

    w  = CUDA.sqrt(1.0 - p[1]*p[1] - p[2]*p[2])
    if j+s <= Nchn
        phas1 = p[1]*x[1,j, ] + p[2]*x[2,j, ] + (w - 1)*x[3,j, ]
        phas2 = p[1]*x[1,j+s] + p[2]*x[2,j+s] + (w - 1)*x[3,j+s]
        temp[j] = (CUDA.cos(phas1) + CUDA.cos(phas2)) +
            im*(CUDA.sin(phas1) + CUDA.sin(phas2))
    else
        phas1 = p[1]*x[1,j] + p[2]*x[2,j] + (w - 1)*x[3,j]
        temp[j] = CUDA.cos(phas1) + im*CUDA.sin(phas1)
    end
    sync_threads()
    s >>= 1
    
    while s > 0
        temp[j] += temp[j+s]
        if s > 32
            sync_threads()
        end
        s >>= 1
    end
    if j == 1
        sum = temp[1]/Nchn
    end
    return nothing
end
=#

function deltaGPU!(model, p, x)
    Mdat = size(x)[2]
    j = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    if j == 1 @cuprintln(size(x)[1], "  ", size(x)[2], "  ", Mdat) end
    if j <= Mdat
        w    = CUDA.sqrt(1.0 - p[1]*p[1] - p[2]*p[2])
        phas = p[1]*x[1,j] + p[2]*x[2,j] + (w - 1)*x[3,j]
        model[j] = CUDA.cos(phas) + im*CUDA.sin(phas)
        # r[j] = CUDA.cos(phas)
        # i[j] = CUDA.sin(phas)
    end
    return nothing
end
=#
