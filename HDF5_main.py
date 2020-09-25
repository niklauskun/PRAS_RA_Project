
import numpy as np
import pandas as pd
import h5py
import json
import os
os.chdir('C:/Users/llavin/Desktop/PRAS')

from HDF5_utils import load_seams,clean_tx,clean_gen,add_gen,create_gen_failure_recovery_cols,HDF5Case

#data loads
#pickle these eventually to speed up code
seams_transmission_df = load_seams('Transmission', wb="NREL-Seams Model (MISO).xlsx")
seams_generation_df = load_seams('Generation', wb="NREL-Seams Model (MISO).xlsx")
seams_load_df = load_seams('Load', wb="NREL-Seams Model (MISO).xlsx")
seams_mapping_df = load_seams('Mapping', wb="NREL-Seams Model (MISO).xlsx")

#double load at just the one bus
#print(seams_load_df.MEC.values)
#seams_load_df.MEC = 2*seams_load_df.MEC.values
#print(seams_load_df.MEC.values)
#additional gens to throw in
additional_gen = ['Solar1','Solar','MEC_33',33,100,'MISO-9',0.0,0.0,0.0,0.0]
additional_gen2 = ['Wind1','Wind','MEC_33',33,250,'MISO-9',0.0,0.0,0.0,0.0]
additional_gen3 = ['Wind2','Wind','CBPC-NIPCO_7',7,250,'MISO-9',0.0,0.0,0.0,0.0]

#and select which to use
gens_to_add = [additional_gen2]

#need also shape for solar,wind
solar_shape = [0.5,0.5,0.5,0.5,0.5,0.5,.05,.1,.2,.4,.6,.8,.9,.8,.6,.4,.3,.2,.05,0.5,0.5,0.5,0.5,0.5]*365
wind_shape = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]*365
#wind_shape = [.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5]*365
#print(len(wind_shape))
#clean tx
retain_cols = ['Line','From','To','FW','BW','Area From','Area To']
seams_transmission_df = clean_tx(seams_transmission_df,retain_cols)
#clean gen
seams_generation_df = clean_gen(seams_generation_df,seams_mapping_df)
#seams_generation_df = add_gen(seams_generation_df,additional_gen)
for g_list in gens_to_add:
        seams_generation_df = add_gen(seams_generation_df,g_list)
seams_generation_df = create_gen_failure_recovery_cols(seams_generation_df)


#define case metadata
vgbool = True
metadata = {'pras_dataversion':'v0.5.0',
        'start_timestamp':'2012-01-01T00:00:00-05:00',
        'timestep_count':24,
                'timestep_length':1,
               'timestep_unit':'h',
               'power_unit':'MW',
               'energy_unit':'MWh'}


#create and export case
case = HDF5Case(seams_transmission_df,seams_generation_df,seams_load_df,seams_mapping_df,
metadata['timestep_count'],solar_shape,wind_shape,include_vg=vgbool)
case.create_all()
case.write_HDF5('perfect_vg_test_allzones.pras',**metadata)