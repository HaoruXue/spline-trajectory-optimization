include("SplineTrajectory.jl")
using Luxor

TRAJ_X = 1
TRAJ_Y = 2
TRAJ_Z = 3
TRAJ_YAW = 4
TRAJ_SPEED = 5
TRAJ_CURVATURE = 6
TRAJ_DIST_TO_SF_BWD = 7
TRAJ_DIST_TO_SF_FWD = 8
TRAJ_REGION = 9
TRAJ_LEFT_BOUND_X = 10
TRAJ_LEFT_BOUND_Y = 11
TRAJ_RIGHT_BOUND_X = 12
TRAJ_RIGHT_BOUND_Y = 13
TRAJ_BANK = 14
TRAJ_LON_ACC = 15
TRAJ_LAT_ACC = 16
TRAJ_TIME = 17
TRAJ_IDX = 18
TRAJ_ITERATION_FLAG = 19

struct DiscreteTrajectory
    name::String
    N::UInt
    length::Float64
    origin_lat::Float64
    origin_lon::Float64
    origin_alt::Float64
    data::Matrix{Float64}
end

function discretize_trajectory(traj_s::SplineTrajectory, interval::Float64)
    traj_length = eval_traj_length(traj_s)
    N = Int(traj_length ÷ interval)
    ts = [range(0.0, 1.0, N + 1);]
    ts = ts[begin:end-1]

    x, y = traj_ev(traj_s, ts)
    dx, dy = traj_ev(traj_s, ts, BK.Derivative(1))
    d2x, d2y = traj_ev(traj_s, ts, BK.Derivative(2))
    yaw = atan.(dy, dx)
    curvature = (dx .* d2y .- dy .* d2x) ./ sqrt.((dx .^ 2 .+ dy .^ 2) .^ 3)

    traj_d = DiscreteTrajectory("untitled", N, traj_length, 0.0, 0.0, 0.0, zeros(Float64, N, 19))
    traj_d.data[:, TRAJ_X] .= x
    traj_d.data[:, TRAJ_Y] .= y
    traj_d.data[:, TRAJ_YAW] .= yaw
    traj_d.data[:, TRAJ_CURVATURE] .= 1.0 ./ abs.(curvature)

    for i = 2:N
        traj_d.data[i, TRAJ_DIST_TO_SF_BWD] = traj_d.data[i-1, TRAJ_DIST_TO_SF_BWD] + eval_traj_section(traj_s, ts[i-1], ts[i])
    end
    traj_d.data[:, TRAJ_DIST_TO_SF_FWD] = traj_length .- traj_d.data[:, TRAJ_DIST_TO_SF_BWD]

    return traj_d
end

function set_trajectory_metadata!(
    traj_d::DiscreteTrajectory,
    name::String=nothing,
    origin_lat::Float64=nothing,
    origin_lon::Float64=nothing,
    origin_alt::Float64=nothing
)

    isnothing(name) || (traj_d.name = name)
    isnothing(origin_lat) || (trajd_.origin_lat = origin_lat)
    isnothing(origin_lon) || (trajd_.origin_lon = origin_lon)
    isnothing(origin_alt) || (trajd_.origin_alt = origin_alt)
end

function set_trajectory_bounds(
    traj_d::DiscreteTrajectory,
    left::DiscreteTrajectory,
    right::DiscreteTrajectory,
    max_dist::Float64=100.0
)
    left_points = [Point(left.data[i, TRAJ_X], left.data[i, TRAJ_Y]) for i in 1:left.N]
    right_points = [Point(right.data[i, TRAJ_X], right.data[i, TRAJ_Y]) for i in 1:right.N]
    left_poly = poly(left_points, close=true)
    right_poly = poly(right_points, close=true)

    function find_closest_intersect(pt, intersections::Vector{Point})
        if !isempty(intersections)
            int_pt = intersections[1]
            int_dist = distance(pt, int_pt)
            for (i, int_pt_i) in enumerate(intersections)
                dist = distance(pt, int_pt_i)
                if dist < int_dist
                    int_pt = int_pt_i
                    int_dist = dist
                end
            end
            return int_pt
        end
        return pt
    end

    for i in 1:traj_d.N
        yaw, x, y = traj_d.data[i, TRAJ_YAW], traj_d.data[i, TRAJ_X], traj_d.data[i, TRAJ_Y]
        pt = Point(x, y)
        yaw_normal = yaw + π / 2
        max_normal = Point(x + max_dist * cos(yaw_normal), y + max_dist * sin(yaw_normal))
        min_normal = Point(x + max_dist * cos(yaw_normal + π), y + max_dist * sin(yaw_normal + π))
        left_ints = intersectlinepoly(max_normal, min_normal, left_poly)
        right_ints = intersectlinepoly(max_normal, min_normal, right_poly)

        left_int = find_closest_intersect(pt, left_ints)
        right_int = find_closest_intersect(pt, right_ints)

        traj_d.data[i, TRAJ_LEFT_BOUND_X] = left_int.x
        traj_d.data[i, TRAJ_LEFT_BOUND_Y] = left_int.y
        traj_d.data[i, TRAJ_RIGHT_BOUND_X] = right_int.x
        traj_d.data[i, TRAJ_RIGHT_BOUND_Y] = right_int.y
    end
end