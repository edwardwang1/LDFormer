import os
import numpy as np
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import random


PTV_DIR = "Embeddings/PTV"
OARS_Dir = "Embeddings/OARs"
DOSE_DIR = "Embeddings/Dose"
IDE_DIR = "Embeddings/IDE"
PATIENT_DATA_PATH = "PatientDataFile.csv"
CONTEXT_SAVE_DIR = "Embeddings/Contexts"

DOSE_DICT = {18: 1,
            20: 2,
            24: 3,
            25: 4,
            28: 5,
            30: 6,
            34: 7,
            35: 8,
            40: 9,
            45: 10,
            48: 11,
            50: 12,
            54: 13,
            55: 14,
            60: 15,
            }

PTV_EMB_LEN = 14
OARS_EMB_LEN = 96
DOSE_EMB_LEN = 384
IDE_EMB_LEN = 96
CONTEXT_LEN = 646

assert CONTEXT_LEN == OARS_EMB_LEN + PTV_EMB_LEN * 5 + DOSE_EMB_LEN + IDE_EMB_LEN


for modal in ["Training", "Validation", "Testing"]:
    os.makedirs(os.path.join(CONTEXT_SAVE_DIR, modal), exist_ok=True)

train_list_path = "Data/train_IDs.txt"
val_list_path = "Data/val_IDs.txt"
test_list_path = "Data/test_IDs.txt"

with open(train_list_path) as f:
    training_patients = [line.rstrip('\n') for line in f]

with open(val_list_path) as f:
    validation_patients = [line.rstrip('\n') for line in f]

with open(test_list_path) as f:
    test_patients = [line.rstrip('\n') for line in f]

df = pd.read_csv(PATIENT_DATA_PATH)
for i in tqdm(range(df.shape[0])):
    patientID = df.iloc[i]["Patient"]
    doses = df.iloc[i]["Dose"].split(",")
    fracs = df.iloc[i]["Fraction"].split(",")
    ptv_ids = df.iloc[i]["PTVs"].split(",")

    if patientID in training_patients:
        split = "Training"
    elif patientID in validation_patients:
        split = "Validation"
    elif patientID in test_patients:
        split = "Testing"
    else:
        print("Patient not found in any split")

    oar_ind = list(np.loadtxt(os.path.join(OARS_Dir, split, patientID + ".txt"), delimiter=",").astype(int))
    ide_ind = list(np.loadtxt(os.path.join(IDE_DIR, split, patientID + ".txt"), delimiter=",").astype(int))
    dose_ind = list(np.loadtxt(os.path.join(DOSE_DIR, split, patientID + ".txt"), delimiter=",").astype(int))

    if split == "Training":
        l = 0
        ptv_indices = list(range(len(ptv_ids)))
        for ptv_order in itertools.permutations(ptv_indices):
            context = [0] * CONTEXT_LEN

            assert OARS_EMB_LEN == len(oar_ind)
            assert DOSE_EMB_LEN == len(dose_ind)
            assert IDE_EMB_LEN == len(ide_ind)

            context[:OARS_EMB_LEN] = oar_ind
            context[-DOSE_EMB_LEN:] = dose_ind
            context[OARS_EMB_LEN:OARS_EMB_LEN + IDE_EMB_LEN] = ide_ind


            curr_index = 0
            for j in ptv_order:
                ptv_ind = list(np.loadtxt(os.path.join(PTV_DIR, split, patientID + "_" + ptv_ids[j] + ".txt"), delimiter=",").astype(int))
                ptv_ind.append(DOSE_DICT[int(doses[j])])
                ptv_ind.append(fracs[j])
                context[OARS_EMB_LEN + IDE_EMB_LEN + curr_index * PTV_EMB_LEN: OARS_EMB_LEN + IDE_EMB_LEN + (curr_index + 1) * PTV_EMB_LEN] = ptv_ind
                assert PTV_EMB_LEN == len(ptv_ind)
                curr_index += 1

            assert len(context) == CONTEXT_LEN
            #Only perform write 50% of time if len(ptv_ids) >=5
            if len(ptv_ids) >= 5:
                if np.random.rand() > 0.5:
                    with open(os.path.join(CONTEXT_SAVE_DIR, split, patientID + "_" + "_".join([ptv_ids[m] for m in ptv_order]) + ".txt"), "w") as f:
                        for item in context:
                            f.write("%s\n" % item)
            else:
                with open(os.path.join(CONTEXT_SAVE_DIR, split, patientID + "_" + "_".join([ptv_ids[m] for m in ptv_order]) + ".txt"), "w") as f:
                    for item in context:
                        f.write("%s\n" % item)

    else:
        context = [0] * CONTEXT_LEN

        assert OARS_EMB_LEN == len(oar_ind)
        assert DOSE_EMB_LEN == len(dose_ind)
        assert IDE_EMB_LEN == len(ide_ind)

        context[:OARS_EMB_LEN] = oar_ind
        context[-DOSE_EMB_LEN:] = dose_ind
        context[OARS_EMB_LEN:OARS_EMB_LEN + IDE_EMB_LEN] = ide_ind

        for j in range(len(ptv_ids)):
            ptv_ind = list(np.loadtxt(os.path.join(PTV_DIR, split, patientID + "_" + ptv_ids[j] + ".txt"), delimiter=",").astype(int))
            ptv_ind.append(DOSE_DICT[int(doses[j])])
            ptv_ind.append(fracs[j])
            context[OARS_EMB_LEN + IDE_EMB_LEN + j * PTV_EMB_LEN: OARS_EMB_LEN + IDE_EMB_LEN + (j + 1) * PTV_EMB_LEN] = ptv_ind
            assert PTV_EMB_LEN == len(ptv_ind)

        assert len(context) == CONTEXT_LEN
        with open(os.path.join(CONTEXT_SAVE_DIR, split, patientID + "_" + "_".join([ptv_ids[m] for m in range(len(ptv_ids))]) + ".txt"), "w") as f:
            for item in context:
                f.write("%s\n" % item)



