# Shared framework tables, keyed-directory discovery and group building for
# the Lorenz plot scripts; data files live at data/<package>/<os>_<gpu>/<name>.

# display name => (subdirectory, filename prefix)
# Note: MPGOS data files are stored under `CPP/` in the repo's `data/` folder.
const FRAMEWORKS = [
    ("Julia", "Julia", "Julia"),
    ("MPGOS", "CPP", "MPGOS"),
    ("JAX", "JAX", "Jax"),
    ("PYTORCH", "PYTORCH", "Torch"),
    ("CUBIE", "CUBIE", "Cubie"),
    ("CUBIE_MLIR", "CUBIE_MLIR", "Cubie_mlir"),
    ("MYOKIT CUDA", "MYOKIT_CUDA", "Myokit_cuda"),
]

# color/marker choices per framework
const COLORS = Dict("Julia"=>:Green, "MPGOS"=>:Orange, "JAX"=>:Red,
    "PYTORCH"=>:DarkRed, "CUBIE"=>:Blue, "CUBIE_MLIR"=>:Purple,
    "MYOKIT CUDA"=>:Black)
const MARKERS = Dict("Julia"=>:circle, "MPGOS"=>:utriangle, "JAX"=>:diamond,
    "PYTORCH"=>:xcross, "CUBIE"=>:star5, "CUBIE_MLIR"=>:hexagon,
    "MYOKIT CUDA"=>:rect)

# Call f(display, key, os, gpu, path, capture) for every keyed file matching
# each framework's prefix + suffix_pattern (a regex source with one capture).
function eachkeyedfile(f, base_path, suffix_pattern)
    for (display, dir, prefix) in FRAMEWORKS
        dpath = joinpath(base_path, dir)
        isdir(dpath) || continue
        pat = Regex("^" * prefix * suffix_pattern * "\$")
        for key in sort(readdir(dpath))
            kpath = joinpath(dpath, key)
            isdir(kpath) || continue
            parts = split(key, '_')
            length(parts) == 2 || continue
            os, gpu = String(parts[1]), String(parts[2])
            for fname in sort(readdir(kpath))
                m = match(pat, fname)
                m === nothing && continue
                f(display, String(key), os, gpu, joinpath(kpath, fname),
                  String(m.captures[1]))
            end
        end
    end
end

# (label, subset) groups per key, os, gpu and "all"; most specific first,
# and a group repeating an earlier group's key set is dropped.
function build_groups(series::Vector{T}) where {T}
    groups = Tuple{String, Vector{T}}[]
    seen = Set{Set{String}}()
    function add_group!(label, sel)
        isempty(sel) && return
        ks = Set(s.key for s in sel)
        ks in seen && return
        push!(seen, ks)
        push!(groups, (label, sel))
    end
    for key in sort(unique(s.key for s in series))
        add_group!(key, filter(s -> s.key == key, series))
    end
    for os in sort(unique(s.os for s in series))
        add_group!(os, filter(s -> s.os == os, series))
    end
    for gpu in sort(unique(s.gpu for s in series))
        add_group!(gpu, filter(s -> s.gpu == gpu, series))
    end
    add_group!("all", series)
    return groups
end
