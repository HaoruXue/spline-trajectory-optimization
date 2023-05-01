using CSV
using DataFrames

function load_raw_trajectory(f::String)
    df = DataFrame(CSV.File(f))
    return df.x, df.y, df.z
end