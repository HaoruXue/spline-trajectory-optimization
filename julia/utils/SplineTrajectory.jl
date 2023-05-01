using LinearAlgebra
using StaticArrays
import BSplineKit as BK
using PyCall
using QuadGK

struct SplineTrajectory
    spl_x::BK.Spline
    spl_y::BK.Spline

    function SplineTrajectory(X::AbstractVector{T}, Y::AbstractVector{T}, k::Int) where {T}
        length(X) == length(Y) || 
            throw(DimensionMismatch("X and Y coordinates must have the same length."))
        # N = length(X)
        # t = collect(T, range(0.0, step = 1.0 / N, stop = 1.0))[begin:end-1]
        # intp(x, y) = BK.interpolate(x, y, BK.BSplineOrder(k), BK.Periodic(1.0))

        X_closed = [X; X[1]]
        Y_closed = [Y; Y[1]]
        intp = pyimport("scipy.interpolate")
        tck, u = intp.splprep((X_closed, Y_closed), s=10, per=true, k=k)
        coeff_before = k ÷ 2
        coeff_after = k - coeff_before
        B = BK.PeriodicBSplineBasis(BK.BSplineOrder(k), tck[1][k+1:end-k])
        x = BK.Spline(B, tck[2][1][coeff_before+1:end-coeff_after])
        y = BK.Spline(B, tck[2][2][coeff_before+1:end-coeff_after])

        new(x, y)
    end
end

function eval_traj_section(traj_s::SplineTrajectory, t_min::Float64, t_max::Float64)::Float64
    spl_x = BK.Derivative(1) * traj_s.spl_x
    spl_y = BK.Derivative(1) * traj_s.spl_y
    function integrate_length(t::Float64)
        sqrt(spl_x(t) ^ 2 + spl_y(t) ^ 2)
    end
    integral, _ = quadgk(integrate_length, t_min, t_max)
    return integral
end

function eval_traj_length(traj_s::SplineTrajectory)::Float64
    return eval_traj_section(traj_s, 0.0, 1.0)
end

function traj_ev(spl::SplineTrajectory, t::AbstractVector{T}, op = BK.Derivative(0)) where {T}
    N = length(t)
    x = zeros(T, N)
    y = zeros(T, N)
    spl_x = op * spl.spl_x
    spl_y = op * spl.spl_y
    for i=1:N
        ti = t[i]
        x[i] = spl_x(ti)
        y[i] = spl_y(ti)
    end
    return x, y
end
