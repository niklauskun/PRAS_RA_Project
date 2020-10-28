using PRAS, FileIO, JLD, DelimitedFiles
foldername = "testPRAS10.27" # whatever you named the folder
casename = "VRE0.5_wind_2040NRELHihgModerate_8760.pras"
casename2 = "VRE0.5_wind_2040NRELHihgModerate_8760_addgulfwind.pras"

path = joinpath(homedir(), "Desktop", foldername, "PRAS_files", casename)
path2 = joinpath(homedir(), "Desktop", foldername, "PRAS_files", casename)

function run_path_model(input_path, casename, foldername, samples)
    model = SystemModel(input_path)
    results = assess(SequentialMonteCarlo(samples=samples), SpatioTemporal(), model)
    dump(results)
    region_lole_list = [results.regionloles[i].val for i in keys(results.regionloles)]
    region_eue_list = [results.regioneues[i].val for i in keys(results.regioneues)]
    period_lolp_list = [results.periodlolps[i].val for i in keys(results.periodlolps)]
    period_eue_list = [results.periodeues[i].val for i in keys(results.periodeues)]
    region_period_eues_list = [results.regionalperiodeues[i].val for i in keys(results.regionalperiodeues)]
    region_period_lolps_list = [results.regionalperiodlolps[i].val for i in keys(results.regionalperiodlolps)]

    # write desired outputs to csvs, using path-based naming convention
    cd(joinpath(homedir(), "Desktop", foldername, "results"))
    case_str = casename[1:findlast(isequal('.'), casename) - 1]
    writedlm(string(case_str, "_", "regionlole.csv"), region_lole_list)
    writedlm(string(case_str, "_", "regioneue.csv"), region_eue_list)
    writedlm(string(case_str, "_", "periodlolp.csv"), period_lolp_list)
    writedlm(string(case_str, "_", "periodeue.csv"), period_eue_list)
    writedlm(string(case_str, "_", "regionperiodeue.csv"), region_period_eues_list, ",")
    writedlm(string(case_str, "_", "regionperiodlolp.csv"), region_period_lolps_list, ",")
end

# be careful with number of samples - the choice really affects runtime (though also more samples reduces error in results)
run_path_model(path,casename,foldername, 1000)

mysystemmodel = SystemModel(path)
mysystemmodel2 = SystemModel(path2)

# be careful with number of samples - the choice really affects runtime (though also more samples reduces error in EFC/ELCC calcs)
min_efc, max_efc = assess(EFC{EUE}(100, "26"), SequentialMonteCarlo(samples=100_000), SpatioTemporal(), mysystemmodel, mysystemmodel2)
min_elcc, max_elcc = assess(ELCC{EUE}(100, "26"), SequentialMonteCarlo(samples=10_000), Minimal(), mysystemmodel, mysystemmodel2)