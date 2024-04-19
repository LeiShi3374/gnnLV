import os
import numpy as np

datapath = "emulationResults/beamData/dge_2_2_300_5.0e-05_40_128_2_753"
truedisp_file = f"{datapath}/trueDisplacement.npy"
preddisp_file = f"{datapath}/predDisplacement.npy"
refdisp_file = f"{datapath}/referenceCoords.npy"

truedisp = np.load(truedisp_file)

print()



