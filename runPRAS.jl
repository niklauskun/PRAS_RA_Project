# ;cd C:/Users/llavin/Desktop/PRAS
# ]activate . #activate environment

# user defined inputs

path = "C:/Users/llavin/Desktop/PRAS/novgwind.pras"
pathvg = "C:/Users/llavin/Desktop/PRAS/vg.pras"
pathhighvg = "C:/Users/llavin/Desktop/PRAS/high_vg.pras"

# new path
path = "C:/Users/llavin/Desktop/PRAS_RA_MISO_930/testfile.pras"

# run model
using PRAS, FileIO, JLD, DelimitedFiles
mysystemmodel = SystemModel(path)
w_vg = SystemModel(pathvg)
w_high_vg = SystemModel(pathhighvg)
# myresults = assess(Convolution(),Minimal(),mysystemmodel)
myresults = assess(SequentialMonteCarlo(samples=100_000), SpatioTemporal(), mysystemmodel)
myresults_vg = assess(SequentialMonteCarlo(samples=100_000), SpatioTemporal(), w_vg)
myresults_highvg = assess(SequentialMonteCarlo(samples=100_000), SpatioTemporal(), w_high_vg)
# collect desired outputs
dump(myresults)
dump(myresults_vg)
dump(myresults_highvg)
# collect(keys(myresults.regionloles))

region_lole_list = [myresults.regionloles[i].val for i in keys(myresults.regionloles)]
region_eue_list = [myresults.regioneues[i].val for i in keys(myresults.regioneues)]
period_lolp_list = [myresults.periodlolps[i].val for i in keys(myresults.periodlolps)]
period_eue_list = [myresults.periodeues[i].val for i in keys(myresults.periodeues)]
region_period_eues_list = [myresults.regionalperiodeues[i].val for i in keys(myresults.regionalperiodeues)]
region_period_lolps_list = [myresults.regionalperiodlolps[i].val for i in keys(myresults.regionalperiodlolps)]

# change folder here so it goes to case folder

# write desired outputs to csvs
writedlm("regionlole.csv",region_lole_list)
writedlm("regioneue.csv",region_eue_list)
writedlm("periodlolp.csv",period_lolp_list)
writedlm("periodeue.csv",period_eue_list)
writedlm("regionperiodeue.csv",region_period_eues_list,",")
writedlm("regionperiodlolps.csv",region_period_lolps_list,",")

# do also for the vg files
region_lole_vg_list = [myresults_vg.regionloles[i].val for i in keys(myresults_vg.regionloles)]
region_eue_vg_list = [myresults_vg.regioneues[i].val for i in keys(myresults_vg.regioneues)]
period_eue_vg_list = [myresults_vg.periodeues[i].val for i in keys(myresults_vg.periodeues)]
region_period_eues_vg_list = [myresults_vg.regionalperiodeues[i].val for i in keys(myresults_vg.regionalperiodeues)]

writedlm("regionlolevg.csv",region_lole_vg_list)
writedlm("regioneuevg.csv",region_eue_vg_list)
writedlm("periodeuevg.csv",period_eue_vg_list)
writedlm("regionperiodeuevg.csv",region_period_eues_vg_list,",")

# do also for the high vg files
region_lole_highvg_list = [myresults_highvg.regionloles[i].val for i in keys(myresults_highvg.regionloles)]
region_eue_highvg_list = [myresults_highvg.regioneues[i].val for i in keys(myresults_highvg.regioneues)]
period_eue_highvg_list = [myresults_highvg.periodeues[i].val for i in keys(myresults_highvg.periodeues)]
region_period_eues_highvg_list = [myresults_highvg.regionalperiodeues[i].val for i in keys(myresults_highvg.regionalperiodeues)]

writedlm("regionlolehighvg.csv",region_lole_highvg_list)
writedlm("regioneuehighvg.csv",region_eue_highvg_list)
writedlm("periodeuehighvg.csv",period_eue_highvg_list)
writedlm("regionperiodeuehighvg.csv",region_period_eues_highvg_list,",") 