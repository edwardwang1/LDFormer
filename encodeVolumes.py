import numpy as np
import torch
import vqvae
import vqvae2D
import os
from config import load_config
from matplotlib import pyplot as plt
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

def saveImg2D(v1, save_path):
    if v1.ndim == 3:
        v1 = np.argmax(v1, axis=0)[:, :]

    f, axarr = plt.subplots(1, 1)
    axarr.imshow(v1)
    axarr.text(0.95, 0.95, f'max: {np.max(v1):.2f}\nmin: {np.min(v1):.2f}', 
             verticalalignment='top', horizontalalignment='right', 
             transform=axarr.transAxes, color='black', fontsize=12, weight='bold')
    axarr.set_axis_off()
    
    f.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(f)
    return f

def saveImg3D(v1, save_path, indexing="max"):
    #get indicies of max value of v1

    if v1.ndim == 4:
        v1 = np.argmax(v1, axis=0)[:, :, :]

    if indexing == "middle":
        ax_loc = v1.shape[0] // 2
        sag_loc = v1.shape[1] // 2
        cor_loc = v1.shape[2] // 2
    else:
        index_of_max = np.where(v1 == np.max(v1))
        ax_loc = index_of_max[0][0]
        sag_loc = index_of_max[1][0]
        cor_loc = index_of_max[2][0]

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(v1[ax_loc, :, :])
    axarr[0].set_axis_off()
    axarr[1].imshow(v1[:, sag_loc, :])
    axarr[1].set_axis_off()
    axarr[2].imshow(v1[:, :, cor_loc])
    axarr[2].set_axis_off()

    f.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.1)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(f)
    return f

def encode(model, volume_path, save_path, oars=False, to_save=True):
    data = np.load(volume_path)

    if oars:
        data = np.eye(8)[data].transpose(2, 0, 1).astype(np.float32) #should be 8 normally
        

    if oars:
        quantized, indicies = model.encode_to_c(torch.from_numpy(data).unsqueeze(0).float().type(torch.FloatTensor).cuda())
    else:
        quantized, indicies = model.encode_to_c(torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().cuda())

    #convert indices to int
    indicies = indicies.cpu().detach().numpy()
    indicies = indicies.astype(int).flatten()
    
    # print(np.min(indicies), np.max(indicies))

    indicies += 1 #incrementing so that 0 refers to the padding index

    #print(indicies.shape)

    # Save the indicies
    if to_save:
        #print(save_path)
        np.savetxt(save_path, indicies, fmt='%i', delimiter=",")


def encode2D(model, volume_path, save_path, oars=False, to_save=True):
    data = np.load(volume_path)

    if oars:
        data = np.eye(8)[data].transpose(2, 0, 1).astype(np.float32) #should be 8 normally
        

    if oars:
        quantized, indicies = model.encode_to_c(torch.from_numpy(data).unsqueeze(0).float().type(torch.FloatTensor).cuda())
    else:
        quantized, indicies = model.encode_to_c(torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float().cuda())

    #convert indices to int
    indicies = indicies.cpu().detach().numpy()
    indicies = indicies.astype(int).flatten()
    
    indicies += 1

    # Save the indicies
    if to_save:
        #print(save_path)
        np.savetxt(save_path, indicies, fmt='%i', delimiter=",")

def decode(model, encoding_input_path, save_path, oars=False):
    indicies = np.loadtxt(encoding_input_path, delimiter=",").reshape(-1, 1)
    indicies -= 1 #decrementing to be consistent with the encoding
    indicies_torch = torch.tensor(indicies, dtype=torch.int64).cuda()

#    print(model._vq_vae._input_shape)
    
    x_recon = model.decode_from_c(indicies_torch)

    x_recon_numpy = x_recon.squeeze(0).squeeze(0).cpu().detach().numpy()

    if oars:
        indexing = "middle"
    else:
        indexing = "max"
    saveImg3D(x_recon_numpy, save_path, indexing)
    # print(x_recon_numpy)

    return x_recon_numpy

def decode2D(model, encoding_input_path, save_path, oars=False):
    indicies = np.loadtxt(encoding_input_path, delimiter=",").reshape(-1, 1)
    indicies -= 1 #decrementing to be consistent with the encoding
    indicies_torch = torch.tensor(indicies, dtype=torch.int64).cuda()

#    print(model._vq_vae._input_shape)
    
    x_recon = model.decode_from_c(indicies_torch)

    x_recon_numpy = x_recon.squeeze(0).squeeze(0).cpu().detach().numpy()

    saveImg2D(x_recon_numpy, save_path)

    return x_recon_numpy

def runEncode(volume_type, volume_dir, save_dir, weight_dir_dict, config_dict):
    if volume_type == "OARs":
        oars = True
    else:
        oars = False
    #
    # if volume_type == "IDE":
    #     volume_type = "Dose"
    weight_dir = weight_dir_dict[volume_type]
    model_config = config_dict[volume_type]
    model = vqvae.VQVAE(model_config)
    model.load_state_dict(torch.load(weight_dir, map_location=torch.device('cuda')))
    model.eval()
    model.to('cuda')
    files = os.listdir(volume_dir)
    for file in tqdm(files):
        encode(model, os.path.join(volume_dir, file), os.path.join(save_dir, file[:-4] + ".txt"), oars)

def runEncode2D(volume_type, volume_dir, save_dir, weight_dir_dict, config_dict):
    if volume_type == "OARs":
        oars = True
    else:
        oars = False
    #
    # if volume_type == "IDE":
    #     volume_type = "Dose"
    weight_dir = weight_dir_dict[volume_type]
    model_config = config_dict[volume_type]
    model = vqvae2D.VQVAE(model_config)
    model.load_state_dict(torch.load(weight_dir, map_location=torch.device('cuda')))
    model.eval()
    model.to('cuda')
    files = os.listdir(volume_dir)
    for file in tqdm(files):
        encode2D(model, os.path.join(volume_dir, file), os.path.join(save_dir, file[:-4] + ".txt"), oars)
        

def initialize(model, volume_dir, oars=False):
    files = os.listdir(volume_dir)
    for file in files:
        encode(model, os.path.join(volume_dir, file), "temp.txt", oars, to_save=False)
        break

def runDecode(volume_type, encoding_input_dir, volume_dir, img_save_dir, weight_dir_dict, config_dict, original_dose_dir=None):
    if volume_type == "OARs":
        oars = True
    else:
        oars = False

    # if volume_type == "IDE":
    #     volume_type = "Dose"
    
    weight_dir = weight_dir_dict[volume_type]
    model_config = config_dict[volume_type]
    model = vqvae.VQVAE(model_config)
    model.load_state_dict(torch.load(weight_dir, map_location=torch.device('cuda')))
    model.eval()
    model.to('cuda')
    initialize(model, volume_dir, oars)
    files = os.listdir(encoding_input_dir)
    for file in tqdm(files):
        #if np.random.rand() < 0.05:
        if True:
            #print(img_save_dir, file[:-4] + ".png")
            x_recon = decode(model, os.path.join(encoding_input_dir, file), os.path.join(img_save_dir, file[:-4] + ".png"), oars)
        # if volume_type == "Dose" and original_dose_dir is not None:
        #     original_dose = np.load(os.path.join(original_dose_dir, file[:-4] + ".npy"))
        #     np.save(os.path.join(save_dir, "EncodedDoseNumpy", file[:-4] + ".npy"), np.stack([original_dose, x_recon], axis=0))
    #     x_recon_numpy = x_recon.squeeze(0).squeeze(0).cpu().detach().numpy()
    #     #print(x_recon_numpy.shape)
    #     np.save(os.path.join(save_dir, file[:-4] + ".npy"), x_recon.cpu().detach().numpy())

def runDecode2D(volume_type, encoding_input_dir, volume_dir, img_save_dir, weight_dir_dict, config_dict, original_dose_dir=None):
    if volume_type == "OARs":
        oars = True
    else:
        oars = False

    # if volume_type == "IDE":
    #     volume_type = "Dose"
    
    weight_dir = weight_dir_dict[volume_type]
    model_config = config_dict[volume_type]
    model = vqvae2D.VQVAE(model_config)
    model.load_state_dict(torch.load(weight_dir, map_location=torch.device('cuda')))
    model.eval()
    model.to('cuda')
    initialize(model, volume_dir, oars)
    files = os.listdir(encoding_input_dir)
    for file in tqdm(files):
        if np.random.rand() < 0.01:
            #print(img_save_dir, file[:-4] + ".png")
            x_recon = decode2D(model, os.path.join(encoding_input_dir, file), os.path.join(img_save_dir, file[:-4] + ".png"), oars)
        # if volume_type == "Dose" and original_dose_dir is not None:
        #     original_dose = np.load(os.path.join(original_dose_dir, file[:-4] + ".npy"))
        #     np.save(os.path.join(save_dir, "EncodedDoseNumpy", file[:-4] + ".npy"), np.stack([original_dose, x_recon], axis=0))
    #     x_recon_numpy = x_recon.squeeze(0).squeeze(0).cpu().detach().numpy()
    #     #print(x_recon_numpy.shape)
    #     np.save(os.path.join(save_dir, file[:-4] + ".npy"), x_recon.cpu().detach().numpy())

if __name__ == "__main__":
    ptv_dir = "Data/PTVsOnly5Lesions/"
    oar_dir = "Data/2DOARsOnly5Lesions/"
    dose_dir = "Data/2DDoseOnly5Lesions/"
    ide_dir = "Data/2DIDEOnly5Lesions/"

    with open('Configs/MICCAI2024_config.json', 'r') as f:
        model_config_dict = json.load(f)

    model_config_dose = vqvae2D.VQVAEConfig(**model_config_dict['dose'])
    model_config_ide =  vqvae2D.VQVAEConfig(**model_config_dict['ide'])
    model_config_oars =  vqvae2D.VQVAEConfig(**model_config_dict['oars'])
    model_config_ptv = vqvae.VQVAEConfig(**model_config_dict['ptv'])
    #model_config_ptv = None

    config_dict = {"Dose": model_config_dose,
                   "IDE": model_config_ide,
                    "OARs": model_config_oars,
                    "PTV": model_config_ptv,}

    weight_dir_dict = model_config_dict['weight_dir_dict']

    # For transformer

    save_dir = "Embeddings/"
    img_save_dir = "EmbeddingsImages/"
 
    # Make directories
    for modality in ["PTV", "OARs", "Dose", "IDE"]:
        for split in ["Training", "Testing", "Validation"]:
            os.makedirs(os.path.join(save_dir, "2D" + modality, split), exist_ok=True)
            os.makedirs(os.path.join(img_save_dir, "2D" + modality, split), exist_ok=True)

    for split in ["Validation", "Training", "Testing"]:
        for modality, volume_dir in {#"PTV": ptv_dir,
                           "OARs": oar_dir,
                           "Dose": dose_dir,
                                     "IDE": ide_dir,
                                     }.items():
            emb_save_dir = os.path.join(save_dir, "2D" + modality, split)
            if modality == "PTV":
                runEncode(modality, os.path.join(volume_dir, split), emb_save_dir, weight_dir_dict, config_dict)
                runDecode(modality, emb_save_dir, os.path.join(volume_dir, split),
                            os.path.join(img_save_dir, "2D" + modality, split), weight_dir_dict, config_dict)
            else:
                runEncode2D(modality, os.path.join(volume_dir, split), emb_save_dir, weight_dir_dict, config_dict)
                if modality == "Dose" and split == "Validation":
                    runDecode2D(modality, emb_save_dir, os.path.join(volume_dir, split), os.path.join(img_save_dir, "2D" + modality, split), weight_dir_dict, config_dict, original_dose_dir=os.path.join(dose_dir, split))
                else:
                    runDecode2D(modality, emb_save_dir, os.path.join(volume_dir, split),
                            os.path.join(img_save_dir, "2D" + modality, split), weight_dir_dict, config_dict)
