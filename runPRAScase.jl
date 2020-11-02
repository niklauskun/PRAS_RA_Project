using PRAS, FileIO, JLD, DelimitedFiles
foldername = "testPRAS10.30" # whatever you named the folde
casename = "VRE0.5_wind_2012base100%_8760.pras"
casename2 = "VRE0.1_wind_2012base100%_8760_addgulfsolar.pras"

path = joinpath(homedir(), "Desktop", foldername, "PRAS_files", casename)
path2 = joinpath(homedir(), "Desktop", foldername, "PRAS_files", casename2)

function run_path_model(input_path, casename, foldername, samples)
    model = SystemModel(input_path)
    results = assess(SequentialMonteCarlo(samples=samples), Network(), model)
    dump(results)
    region_lole_list = [results.regionloles[i].val for i in keys(results.regionloles)]
    region_eue_list = [results.regioneues[i].val for i in keys(results.regioneues)]
    period_lolp_list = [results.periodlolps[i].val for i in keys(results.periodlolps)]
    period_eue_list = [results.periodeues[i].val for i in keys(results.periodeues)]
    region_period_eues_list = [results.regionalperiodeues[i].val for i in keys(results.regionalperiodeues)]
    region_period_lolps_list = [results.regionalperiodlolps[i].val for i in keys(results.regionalperiodlolps)]
    flows_list = [results.flows[i].val for i in keys(results.flows)]
    utilizations_list = [results.utilizations[i].val for i in keys(results.utilizations)]

    # write desired outputs to csvs, using path-based naming convention
    cd(joinpath(homedir(), "Desktop", foldername, "results"))
    case_str = casename[1:findlast(isequal('.'), casename) - 1]
    writedlm(string(case_str, "_", "regionlole.csv"), region_lole_list)
    writedlm(string(case_str, "_", "regioneue.csv"), region_eue_list)
    writedlm(string(case_str, "_", "periodlolp.csv"), period_lolp_list)
    writedlm(string(case_str, "_", "periodeue.csv"), period_eue_list)
    writedlm(string(case_str, "_", "regionperiodeue.csv"), region_period_eues_list, ",")
    writedlm(string(case_str, "_", "regionperiodlolp.csv"), region_period_lolps_list, ",")
    writedlm(string(case_str, "_", "flows.csv"), flows_list, ",")
    writedlm(string(case_str, "_", "utilizations.csv"), utilizations_list, ",")
end

function run_path_elcc(input_path, input_path2, capacity, zone_str, samples)
    m = SystemModel(input_path)
    m2 = SystemModel(input_path2)
    min_elcc, max_elcc = assess(ELCC{EUE}(capacity, zone_str), SequentialMonteCarlo(samples=samples), Network(), m, m2)
    return min_elcc, max_elcc
end

function run_path_efc(input_path, input_path2, capacity, zone_str, samples)
    m = SystemModel(input_path)
    m2 = SystemModel(input_path2)
    min_efc, max_efc = assess(EFC{EUE}(capacity, zone_str), SequentialMonteCarlo(samples=samples), Network(), m, m2)
    return min_efc, max_efc
end

## RUN FUNCTIONS ONCE YOU HAVE LOADED THEM
# be careful with number of samples - the choice really affects runtime (though also more samples reduces error in results)
run_path_model(path,casename,foldername, 10000)
run_path_elcc(path,path2,100,"26",200000)
run_path_efc(path,path2,100,"26",200000)


# Old code that was run line-by-line, can be used for checking
mysystemmodel = SystemModel(path)
mysystemmodel2 = SystemModel(path2)

results = assess(SequentialMonteCarlo(samples=10000), Network(), mysystemmodel)

# be careful with number of samples - the choice really affects runtime (though also more samples reduces error in EFC/ELCC calcs)
min_efc, max_efc = assess(EFC{EUE}(100, "26"), SequentialMonteCarlo(samples=100_000), SpatioTemporal(), mysystemmodel, mysystemmodel2)
min_elcc, max_elcc = assess(ELCC{EUE}(100, "26"), SequentialMonteCarlo(samples=1000), Minimal(), mysystemmodel, mysystemmodel2)