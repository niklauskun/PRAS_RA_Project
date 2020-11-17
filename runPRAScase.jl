using PRAS, FileIO, JLD, DelimitedFiles, DataFrames, CSV, XLSX, TickTock
foldername = "testPRAS11.12" # whatever you named the folde
casename = "VRE0.2_wind_2012base100%_8760_50%tx_18%IRM_nostorage_addgulfsolar.pras"
casename2 = "VRE0.2_wind_2012base100%_8760_50%tx_18%IRM_nostorage_addgulfsolar.pras"
casename3 = "VRE0.2_wind_2012base100%_8760_25%tx_18%IRM_nostorage_addgulfsolar.pras"
casename4 = "VRE0.2_wind_2012base100%_8760_25%tx_18%IRM_nostorage_addgulfsolar.pras"

path = joinpath(homedir(), "Desktop", foldername, "PRAS_files", casename)
path2 = joinpath(homedir(), "Desktop", foldername, "PRAS_files", casename2)

path3 = joinpath(homedir(), "Desktop", foldername, "PRAS_files", casename3)
path4 = joinpath(homedir(), "Desktop", foldername, "PRAS_files", casename4)

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

function run_path_model_convolution(input_path, casename, foldername, samples)
    model = SystemModel(input_path)
    results = assess(Convolution(), Temporal(), model)
    dump(results)
    period_lolp_list = [results.lolps[i].val for i in keys(results.lolps)]
    period_eue_list = [results.eues[i].val for i in keys(results.eues)]

    # write desired outputs to csvs, using path-based naming convention
    cd(joinpath(homedir(), "Desktop", foldername, "results"))
    case_str = string("CONVOLUTION_", casename[1:findlast(isequal('.'), casename) - 1])
    writedlm(string(case_str, "_", "periodlolp.csv"), period_lolp_list)
    writedlm(string(case_str, "_", "periodeue.csv"), period_eue_list)
end

function run_path_elcc(input_path, input_path2, capacity, zone_str, samples)
    m = SystemModel(input_path)
    m2 = SystemModel(input_path2)
    min_elcc, max_elcc = assess(ELCC{EUE}(capacity, zone_str), SequentialMonteCarlo(samples=samples), Network(), m, m2)
    return min_elcc, max_elcc
end

function run_model_elcc(m, m2, capacity, zone_str, samples, pval)
    min_elcc, max_elcc = assess(ELCC{EUE}(capacity, zone_str, p_value=pval), SequentialMonteCarlo(samples=samples), Network(), m, m2)
    return min_elcc, max_elcc
end

function run_model_elcc_convolution(m, m2, capacity, zone_str)
    min_elcc, max_elcc = assess(ELCC{EUE}(capacity, zone_str), Convolution(), Temporal(), m, m2)
    return min_elcc, max_elcc
end

function run_path_elcc_minimal(input_path, input_path2, capacity, zone_str, samples, pval)
    m = SystemModel(input_path)
    m2 = SystemModel(input_path2)
    min_elcc, max_elcc = assess(ELCC{EUE}(capacity, zone_str, p_value=pval), SequentialMonteCarlo(samples=samples), Minimal(), m, m2)
    return min_elcc, max_elcc
end

function run_path_efc(input_path, input_path2, capacity, zone_str, samples)
    m = SystemModel(input_path)
    m2 = SystemModel(input_path2)
    min_efc, max_efc = assess(EFC{EUE}(capacity, zone_str), SequentialMonteCarlo(samples=samples), Network(), m, m2)
    return min_efc, max_efc
end

function augment_system_storage(original_system, zone, storage_duration, MW)
    oldzonename = string(zone) # look up info about existing similar generator
    newzonename = string(oldzonename, "_2")
    idx = findfirst(x -> x == oldzonename, original_system.storages.names) # first only
    # grab old values from similar resource in-place, with capacity re-scaled to augmented resource capacity
    category = original_system.storages.categories[idx]
    charge_capacity = original_system.storages.charge_capacity[idx,:]
    discharge_capacity = original_system.storages.discharge_capacity[idx,:]
    energy_capacity = original_system.storages.energy_capacity[idx,:]
    new_charge_capacity = MW
    new_discharge_capacity = MW
    new_energy_capacity = MW * storage_duration # duration should be int
    charge_efficiency = original_system.storages.charge_efficiency[idx,:]
    discharge_efficiency = original_system.storages.discharge_efficiency[idx,:]
    carryover_efficiency = original_system.storages.carryover_efficiency[idx,:]
    lambda = original_system.storages.λ[idx,:]
    mu = original_system.storages.μ[idx,:]

    # then add all these things by replacing the final, and return the augmented SystemModel
    replace_idx = findfirst(x -> x == "z_generic", original_system.storages.names)# size(original_system.generators.names)[1]
    original_system.storages.names[replace_idx] = newzonename
    original_system.storages.categories[replace_idx] = category
    original_system.storages.charge_capacity[replace_idx,:] .= new_charge_capacity
    original_system.storages.discharge_capacity[replace_idx,:] .= new_discharge_capacity
    original_system.storages.energy_capacity[replace_idx,:] .= new_energy_capacity
    original_system.storages.charge_efficiency[replace_idx,:] = charge_efficiency'
    original_system.storages.discharge_efficiency[replace_idx,:] = discharge_efficiency'
    original_system.storages.carryover_efficiency[replace_idx,:] = carryover_efficiency'
    original_system.storages.λ[replace_idx,:] = lambda'
    original_system.generators.μ[replace_idx,:] = mu'
    return original_system
end

function augment_system_generator(original_system, zone, resource_type, MW)
    oldzonename = string(zone, resource_type)
    newzonename = string(oldzonename, "_2")
    idx = findfirst(x -> x == oldzonename, original_system.generators.names) # first only
    # grab old values from similar resource in-place, with capacity re-scaled to augmented resource capacity
    category = original_system.generators.categories[idx]
    capacity = original_system.generators.capacity[idx,:]
    new_capacity = round.(Int64, MW / maximum(capacity) * capacity) # rescales profile and rounds to nearest Int
    lambda = original_system.generators.λ[idx,:]
    mu = original_system.generators.μ[idx,:]
    # then add all these things by replacing the final, and return the augmented SystemModel
    replace_idx = findfirst(x -> x == "z_generic", original_system.generators.names)# size(original_system.generators.names)[1]
    original_system.generators.names[replace_idx] = newzonename
    original_system.generators.categories[replace_idx]
    original_system.generators.capacity[replace_idx,:] = new_capacity'
    original_system.generators.λ[replace_idx,:] = lambda'
    original_system.generators.μ[replace_idx,:] = mu'
    return original_system
end

function ELCC_wrapper_storage(casename, sys_path, aug_path, samples, pval, capacity, duration)
    basesystem = SystemModel(sys_path)
    # run the base system to determine regional EUE, LOLE
    results = assess(SequentialMonteCarlo(samples=samples), Network(), basesystem)
    # dump(results) #I think not needed
    region_lole_list = [results.regionloles[i].val for i in keys(results.regionloles)] # alphabetical by default so no worries
    region_eue_list = [results.regioneues[i].val for i in keys(results.regioneues)] # alphabetical by default so no worries
    # create DF to store results
    df = DataFrame(resourcename=String[], capacity=Int[], energy=Int[], pval=Float64[], samples=Int[], minelcc=Int[], maxelcc=Int[], ZoneEUE=Float64[], ZoneLOLE=Float64[])
    miso_array = XLSX.readdata(joinpath(homedir(), "Desktop", foldername, "NREL-Seams Model (MISO).xlsx"), "Mapping", "A2:C23")
    zones = string.(miso_array[:,3]) 
    zone_nums = string.(miso_array[:,1])
    tick()
    for (i, zone) in enumerate(zones)
        println(zone, " ", zone_nums[i])
        println("running storage add in: ", zone)
        # results.regioneues[zone].val #may be superior to idx
        ZoneLOLE = region_lole_list[i]
        ZoneEUE = region_eue_list[i]
        augsystem = SystemModel(aug_path)
        augmodel = augment_system_storage(augsystem, zone, duration, capacity)
        println("case model loaded, running ELCC...")
        min_elcc, max_elcc = run_model_elcc(basesystem, augmodel, capacity, zone_nums[i], samples, pval)
        println("...case ELCC run, storing data")
        push!(df, [string(zone, duration, "hour"),capacity,capacity * duration,pval,samples,min_elcc,max_elcc,ZoneEUE,ZoneLOLE]) # write results into dataframe
        laptimer()
    end
    # write df to csv
    case_str = string("storageELCC_", casename[1:findlast(isequal('.'), casename) - 1], ".csv") # naming convention for storing data
    cd(joinpath(homedir(), "Desktop", foldername, "results"))
    CSV.write(case_str, df)
    tock()
    return df
end

function ELCC_wrapper_solar(casename, sys_path, aug_path, samples, pval, capacity)
    # iterates ELCC calls for system resource augmentation
    basesystem = SystemModel(sys_path)
    # run the base system to determine regional EUE, LOLE
    results = assess(SequentialMonteCarlo(samples=samples), Network(), basesystem)
    region_lole_list = [results.regionloles[i].val for i in keys(results.regionloles)] # alphabetical by default so no worries
    region_eue_list = [results.regioneues[i].val for i in keys(results.regioneues)] # alphabetical by default so no worries
    df = DataFrame(resourcename=String[], capacity=Int[], pval=Float64[], samples=Int[], minelcc=Int[], maxelcc=Int[], zoneEUE=Float64[], zoneLOLE=Float64[])
    techs = ["UtilitySolar"] # "UtilityWind","DistributedSolar"s
    miso_array = XLSX.readdata(joinpath(homedir(), "Desktop", foldername, "NREL-Seams Model (MISO).xlsx"), "Mapping", "A2:C23")
    zones = string.(miso_array[:,3]) 
    zone_nums = string.(miso_array[:,1])
    tick()
    for resource in techs
        for (i, zone) in enumerate(zones)
            println(zone, " ", zone_nums[i])
            println("running resource: ", resource, " ", zone)
            ZoneLOLE = region_lole_list[i]
            ZoneEUE = region_eue_list[i]
            augsystem = SystemModel(aug_path)
            augmodel = augment_system_generator(augsystem, zone, resource, capacity)
            println("case model loaded, running ELCC...")
            min_elcc, max_elcc = run_model_elcc(basesystem, augmodel, capacity, zone_nums[i], samples, pval)
            println("...case ELCC run, storing data")
            push!(df, [string(zone, resource),capacity,pval,samples,min_elcc,max_elcc,ZoneEUE,ZoneLOLE]) # write results into dataframe
            laptimer()
        end
    end
    # write df to csv
    case_str = string("solarELCC_", casename[1:findlast(isequal('.'), casename) - 1], ".csv") # naming convention for storing data
    cd(joinpath(homedir(), "Desktop", foldername, "results"))
    CSV.write(case_str, df)
    tock()
    return df
end

function ELCC_wrapper_wind(casename, sys_path, aug_path, samples, pval, capacity)
    # iterates ELCC calls for system resource augmentation
    basesystem = SystemModel(sys_path)
    # run the base system to determine regional EUE, LOLE
    results = assess(SequentialMonteCarlo(samples=samples), Network(), basesystem)
    region_lole_list = [results.regionloles[i].val for i in keys(results.regionloles)] # alphabetical by default so no worries
    region_eue_list = [results.regioneues[i].val for i in keys(results.regioneues)] # alphabetical by default so no worries
    df = DataFrame(resourcename=String[], capacity=Int[], pval=Float64[], samples=Int[], minelcc=Int[], maxelcc=Int[], zoneEUE=Float64[], zoneLOLE=Float64[])
    techs = ["UtilityWind"] # "UtilityWind","DistributedSolar"s
    miso_array = XLSX.readdata(joinpath(homedir(), "Desktop", foldername, "NREL-Seams Model (MISO).xlsx"), "Mapping", "A2:C23")
    zones = string.(miso_array[:,3]) 
    zone_nums = string.(miso_array[:,1])
    tick()
    for resource in techs
        for (i, zone) in enumerate(zones)
            println(zone, " ", zone_nums[i])
            println("running resource: ", resource, " ", zone)
            ZoneLOLE = region_lole_list[i]
            ZoneEUE = region_eue_list[i]
            augsystem = SystemModel(aug_path)
            augmodel = augment_system_generator(augsystem, zone, resource, capacity)
            println("case model loaded, running ELCC...")
            min_elcc, max_elcc = run_model_elcc(basesystem, augmodel, capacity, zone_nums[i], samples, pval)
            println("...case ELCC run, storing data")
            push!(df, [string(zone, resource),capacity,pval,samples,min_elcc,max_elcc,ZoneEUE,ZoneLOLE]) # write results into dataframe
            laptimer()
        end
    end
    # write df to csv
    case_str = string("windELCC_", casename[1:findlast(isequal('.'), casename) - 1], ".csv") # naming convention for storing data
    cd(joinpath(homedir(), "Desktop", foldername, "results"))
    CSV.write(case_str, df)
    tock()
    return df
end

# wrapped ELCC runs
# these take a very long time if you're not careful
ELCC_wrapper_storage(casename,path,path2,2500,.2,500,6)
ELCC_wrapper_storage(casename3,path3,path4,2500,.2,500,6)

ELCC_wrapper_solar(casename,path,path2,2500,.2,500)
ELCC_wrapper_solar(casename3,path3,path4,2500,.2,500)

ELCC_wrapper_wind(casename,path,path2,2500,.2,500)
ELCC_wrapper_wind(casename3,path3,path4,2500,.2,500)

# run and create results (EUE, LOLE, etc.) for a single case
run_path_model(path4,casename4,foldername, 10000)
run_path_model(path2,casename2,foldername, 10000)

## RUN FUNCTIONS ONCE YOU HAVE LOADED THEM
# be careful with number of samples - the choice really affects runtime (though also more samples reduces error in results)
run_path_elcc_minimal(path,path2,100,"26",500, .2)
run_path_elcc(path,path2,100,"26",5000)
run_path_model_convolution(path2,casename2,foldername, 1000)

# sandbox elcc single-run cases
# be careful with number of samples - the choice really affects runtime (though also more samples reduces error in EFC/ELCC calcs)
min_efc, max_efc = assess(EFC{EUE}(100, "26"), SequentialMonteCarlo(samples=100_000), SpatioTemporal(), mysystemmodel, mysystemmodel2)
min_elcc, max_elcc = assess(ELCC{EUE}(100, "26"), SequentialMonteCarlo(samples=1000), Minimal(), mysystemmodel, mysystemmodel2)