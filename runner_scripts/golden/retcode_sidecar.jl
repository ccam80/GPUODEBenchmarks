# Retcode sidecar for golden generators: <golden>_retcodes.csv lists the unconverged rows.

using SciMLBase: successful_retcode

"Write `<outfile minus .csv>_retcodes.csv` naming each failed row, or delete it when all solves converged."
function write_retcode_sidecar(outfile, codes)
    sidecar = replace(outfile, r"\.csv$" => "_retcodes.csv")
    failed = [i for i in eachindex(codes) if !successful_retcode(codes[i])]
    if isempty(failed)
        rm(sidecar; force = true)
        @info "All $(length(codes)) solves converged for $(basename(outfile))"
        return
    end
    open(sidecar, "w") do io
        println(io, "row,retcode")
        for i in failed
            println(io, "$(i),$(codes[i])")
        end
    end
    @warn "$(length(failed)) of $(length(codes)) solves did not converge for " *
          "$(basename(outfile)); rows listed in $(basename(sidecar))"
end
