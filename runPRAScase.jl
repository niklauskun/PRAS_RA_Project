

path = "C:/Users/llavin/Desktop/testPRAS10.20/PRAS_files/VRE0.5_wind_2040NRELHihgModerate_2208summer.pras"
path2 = "C:/Users/llavin/Desktop/testPRAS10.20/PRAS_files/VRE0.5_wind_2040NRELHihgModerate_2208summer_addgulfwind.pras"
# path3 = "C:/Users/Luke/Desktop/testPRAS10.20/PRAS_files/VRE0.5_wind_base_168_addnipswind.pras"
using PRAS, FileIO, JLD, DelimitedFiles
mysystemmodel = SystemModel(path)
mysystemmodel2 = SystemModel(path2)
mysystemmodel3 = SystemModel(path3)
myresults = assess(SequentialMonteCarlo(samples=100_000), SpatioTemporal(), mysystemmodel)
myresults2 = assess(SequentialMonteCarlo(samples=100_000), SpatioTemporal(), mysystemmodel2)
dump(myresults)
dump(myresults2)

min_efc, max_efc = assess(EFC{EUE}(1000, "26"), SequentialMonteCarlo(samples=100_000), SpatioTemporal(), mysystemmodel, mysystemmodel2)

min_elcc, max_elcc = assess(ELCC{EUE}(100, "26"), SequentialMonteCarlo(samples=100_000), Minimal(), mysystemmodel, mysystemmodel2)

# list the results
region_lole_list = [myresults.regionloles[i].val for i in keys(myresults.regionloles)]
region_eue_list = [myresults.regioneues[i].val for i in keys(myresults.regioneues)]
period_lolp_list = [myresults.periodlolps[i].val for i in keys(myresults.periodlolps)]
period_eue_list = [myresults.periodeues[i].val for i in keys(myresults.periodeues)]
region_period_eues_list = [myresults.regionalperiodeues[i].val for i in keys(myresults.regionalperiodeues)]
region_period_lolps_list = [myresults.regionalperiodlolps[i].val for i in keys(myresults.regionalperiodlolps)]

# write desired outputs to csvs
writedlm("regionlole.csv",region_lole_list)
writedlm("regioneue.csv",region_eue_list)
writedlm("periodlolp.csv",period_lolp_list)
writedlm("periodeue.csv",period_eue_list)
writedlm("regionperiodeue.csv",region_period_eues_list,",")
writedlm("regionperiodlolps.csv",region_period_lolps_list,",")