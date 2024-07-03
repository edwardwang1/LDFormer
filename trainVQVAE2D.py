from torch.utils.data import DataLoader
from torch import nn
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Any, List
import random
from monai.transforms import Rotate
from loss import GradientLoss, IsodoseLoss


from vqvae2D import VQVAE, VQVAEConfig
from dataset import Volumes
from config import load_config


@dataclass
class TrainVQVAEConfig:
    ptv_dir_train: str = "PTVs_train"
    oar_dir_train: str = "OARs_train"
    dose_dir_train: str = "Doses_train"
    ptv_dir_val: str = "PTVs_val"
    oar_dir_val: str = "OARs_val"
    dose_dir_val: str = "Doses_val"
    ptv_dir_test: str = "PTVs_test"
    oar_dir_test: str = "OARs_test"
    dose_dir_test: str = "Doses_test"
    batch_size: int = 16
    epochs: int = 300
    test_interval: int = 20
    learning_rate: float = 3e-4
    log_save_dir: str = "logs"
    img_save_dir: str = "imgs"
    weight_save_dir: str = "weights"
    vq_target: str = "PTV"
    bce_loss_weight: float = 1.0
    cce_loss_weight: List[float] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1, 1])
    recon_loss_coeff: int = 1
    use_val_instead_of_test = True
    weight_path: str = ""

    
class FlipLeftRightUpDown:
    def __init__(self, p=0.5):
        self.p = p

    # def __call__(self, x):
    #     if random.random() < self.p:
    #         x = torch.flip(x, dims=[0])
    #     if random.random() < self.p:
    #         x = torch.flip(x, dims=[1])
    #     return x
        
    def __call__(self, x):
        rand = random.random()
        if rand < 0.25:
            x = torch.flip(x, dims=[0, 1])
        elif rand >= 0.25 and rand < 0.5:
            x = torch.flip(x, dims=[0])
        elif rand >= 0.5 and rand < 0.75:
            x = torch.flip(x, dims=[1])
        return x

def saveImg(v1, v2, save_path):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(v1)
    axarr[0].text(0.95, 0.95, f'max: {np.max(v1):.2f}\nmin: {np.min(v1):.2f}', 
             verticalalignment='top', horizontalalignment='right', 
             transform=axarr[0].transAxes, color='black', fontsize=12, weight='bold')
    axarr[0].set_axis_off()
    axarr[1].imshow(v2)
    axarr[1].text(0.95, 0.95, f'max: {np.max(v2):.2f}\nmin: {np.min(v2):.2f}', 
             verticalalignment='top', horizontalalignment='right', 
             transform=axarr[1].transAxes, color='black', fontsize=12, weight='bold')
    axarr[1].set_axis_off()
    

    f.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    plt.savefig(save_path, bbox_inches='tight')
    #plt.close(f)
    return f

def train(exp_name_base, exp_name, model_config, train_config):
    #Create transform to augment training data
    flip_transform = FlipLeftRightUpDown(p=0.5) #RotateAndFlip(p=0.5)

    # Define your loss function and directory based on target
    if train_config.vq_target == "PTV":
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_config.bce_loss_weight]).to(device))
        criterion2 = nn.MSELoss().to(device)
        train_dataset = Volumes(directory=train_config.ptv_dir_train, transform=flip_transform)
        if train_config.use_val_instead_of_test:
            val_dataset = Volumes(directory=train_config.ptv_dir_val)
        else:
            test_dataset = Volumes(directory=train_config.ptv_dir_test)
    elif train_config.vq_target == "OAR":
        criterion = nn.CrossEntropyLoss(weight = torch.tensor(train_config.cce_loss_weight).type(torch.FloatTensor).to(device)) #weight = torch.tensor(train_config.cce_loss_weight).type(torch.FloatTensor).to(device)
        train_dataset = Volumes(directory=train_config.oar_dir_train, transform=flip_transform)
        if train_config.use_val_instead_of_test:
            val_dataset = Volumes(directory=train_config.oar_dir_val)
        else:
            test_dataset = Volumes(directory=train_config.oar_dir_test)
    elif train_config.vq_target == "Dose":
        criterion = nn.MSELoss().to(device)
        criterion2 = IsodoseLoss().to(device)
        train_dataset = Volumes(directory=train_config.dose_dir_train, transform=flip_transform)
        if train_config.use_val_instead_of_test:
            val_dataset = Volumes(directory=train_config.dose_dir_val)
        else:
            test_dataset = Volumes(directory=train_config.dose_dir_test)
    else:
        print("ERROR: Unknown vq_target")
        return
    
    # Create DataLoader objects for the train and test sets
    batch_size = train_config.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    if train_config.use_val_instead_of_test:
        test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Create an instance of your model
    model = VQVAE(config=model_config).to(device)
    if len(train_config.weight_path) > 1:
        print("Loading weights from " + train_config.weight_path)
        model.load_state_dict(torch.load(train_config.weight_path, map_location=torch.device(device)))

    # Define your optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=0.01)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    # Run your training / validation loops
    epochs = train_config.epochs
    test_interval = train_config.test_interval

    log_path = os.path.join(train_config.log_save_dir, exp_name_base, exp_name)
    img_path = os.path.join(train_config.img_save_dir, exp_name_base, exp_name)
    weight_path = os.path.join(train_config.weight_save_dir, exp_name_base, exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    writer = SummaryWriter(log_path)
    epoch_loop = tqdm.tqdm(range(epochs + 1))

    data_variance = train_config.recon_loss_coeff
    # for data in train_dataloader:
    #     data_variance = np.var(data.numpy())
    #     break

    loss = 0
    loss_test = 0
    test_iter = 0
    train_iter = 0

    for epoch in epoch_loop:
        train_iter = 0
        sum_train_loss = 0
        sum_train_vq_loss = 0
        sum_train_recon_loss = 0
        sum_train_perplexity = 0
        for batch_idx, volumes in enumerate(train_dataloader):
            model.train()
            if train_config.vq_target == "OAR":
                volumes_np = volumes.numpy()
                data = np.eye(model_config.in_channels)[volumes_np].transpose(0, 3, 1, 2).astype(np.float32)
                data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
                oar_target = torch.from_numpy(volumes_np).type(torch.LongTensor).to(device)
            else:
                data = volumes.unsqueeze(1).type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            vq_loss, data_recon, perplexity = model(data)
            if train_config.vq_target == "OAR":
                recon_error = criterion(data_recon, oar_target) / data_variance #For OARs, for CCE, the target array should not be one hot encoded
            else:
                recon_error = criterion(data_recon, data) / data_variance
            if train_config.vq_target == "PTV":
                recon_error = recon_error + criterion2(data_recon, data) / data_variance #For PTV, use the BCE and MSE to massage loss
            if train_config.vq_target == "Dose":
                #recon_error = recon_error + criterion2(data_recon, data) / data_variance #For dose, use the isodose loss as the additional loss
                 pass
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()
            train_iter += 1
            sum_train_loss += loss
            sum_train_vq_loss += vq_loss
            sum_train_recon_loss += recon_error
            sum_train_perplexity += perplexity
            if batch_idx == 0 and epoch % test_interval == 0:
                train_figs = []
                if train_config.vq_target == "OAR":
                    for l in range(5):
                        train_figs.append(saveImg(
                            np.argmax(data.cpu().detach().numpy(), axis=1)[l, :, :],
                            np.argmax(data_recon.cpu().detach().numpy(), axis=1)[l, :, :],
                            os.path.join(img_path, f"train_{epoch}_{l + 1}.png")
                        ))
                else:
                    for l in range(5):
                        train_figs.append(saveImg(data[l, :, :].squeeze(0).cpu().detach().numpy(), data_recon[l, :, :].squeeze(0).cpu().detach().numpy(), os.path.join(img_path, f"train_{epoch}_{l + 1}.png")))

        if epoch % 5 == 0:
            #print(f'Epoch: {epoch} Train loss: {sum_train_loss/train_iter} Train vq loss: {sum_train_vq_loss/train_iter} Train recon loss: {sum_train_recon_loss/train_iter}')
            print(f'Epoch: {epoch} Train loss: {sum_train_loss/train_iter:.5f} Train vq loss: {sum_train_vq_loss/train_iter:.5f} Train recon loss: {sum_train_recon_loss/train_iter:.5f}')
            writer.add_scalar('train_loss', sum_train_loss/train_iter, epoch)
            writer.add_scalar('train_vq_loss', sum_train_vq_loss/train_iter, epoch)
            writer.add_scalar('train_recon_loss', sum_train_recon_loss/train_iter, epoch)
            writer.add_scalar('train_perplexity', sum_train_perplexity/train_iter, epoch)

        if epoch % test_interval == 0:
            test_iter = 0
            sum_test_loss = 0
            sum_test_vq_loss = 0
            sum_test_recon_loss = 0
            sum_test_perplexity = 0 
            
            model.eval()
            with torch.no_grad():
                for batch_idx_test, volumes_test in enumerate(test_dataloader):
                    if train_config.vq_target == "OAR":
                        volumes_np = volumes_test.numpy()
                        data_test = np.eye(model_config.in_channels)[volumes_np].transpose(0, 3, 1, 2).astype(np.float32)
                        data_test = torch.from_numpy(data_test).type(torch.FloatTensor).to(device)
                        oar_target_test = torch.from_numpy(volumes_np).type(torch.LongTensor).to(device)
                    else:
                        data_test = volumes_test.unsqueeze(1).type(torch.FloatTensor).to(device)

                    vq_loss_test, data_recon_test, perplexity_test = model(data_test)
                    if train_config.vq_target == "OAR":
                        recon_error_test = criterion(data_recon_test, oar_target_test) / data_variance #For OARs, for CCE, the target array should not be one hot encoded
                    else:
                        recon_error_test = criterion(data_recon_test, data_test)  / data_variance     
                    if train_config.vq_target == "PTV":
                        recon_error_test = recon_error_test + + criterion2(data_recon_test, data_test) / data_variance
                    
                    loss_test = recon_error_test + vq_loss_test
                    test_iter +=1
                    sum_test_loss += loss_test
                    sum_test_vq_loss += vq_loss_test
                    sum_test_recon_loss += recon_error_test
                    sum_test_perplexity += perplexity_test 
                    if batch_idx_test == 0:
                        # save weight
                        if epoch != 0:
                            #pass
                            torch.save(model.state_dict(), os.path.join(weight_path, "model_" + str(epoch) + ".pth"))
                        test_figs = []
                        if train_config.vq_target == "OAR":
                            for l in range(5):
                                test_figs.append(saveImg(np.argmax(data_test.cpu().detach().numpy(), axis=1)[0, :, :],
                                         np.argmax(data_recon_test.cpu().detach().numpy(), axis=1)[0, :, :],
                                         os.path.join(img_path, f"test_{epoch}_{l + 1}.png")))

                        else:
                            for l in range(5):
                                test_figs.append(saveImg(data_test[l, 0, :, :].cpu().detach().numpy(), data_recon_test[l, 0, :, :].cpu().detach().numpy(), os.path.join(img_path, f"test_{epoch}_{l + 1}.png")))

            
                writer.add_scalar('test_loss', sum_test_loss/test_iter, epoch)
                writer.add_scalar('test_vq_loss', sum_test_vq_loss/test_iter, epoch)
                writer.add_scalar('test_recon_loss', sum_test_recon_loss/test_iter, epoch)
                writer.add_scalar('test_perplexity', sum_test_perplexity/test_iter, epoch)

                for l in range(5):
                    writer.add_figure(f"Images from Training Set {l + 1}", train_figs[l], epoch)
                    plt.close(train_figs[l])
                    writer.add_figure(f"Images from Testing Set {l + 1}", test_figs[l], epoch)
                    plt.close(test_figs[l])

        model._vq_vae.random_restart()
        model._vq_vae.reset_usage()
        #scheduler.step()
        
    #Save numpy arrays of the last predictions and targets
    #np.save(os.path.join(img_path, "train_target.npy"), np.stack([data_test[0, 0, :, :, :].cpu().detach().numpy(), data_recon_test[0, 0, :, :, :].cpu().detach().numpy()]))

    writer.add_hparams(
        {"num_hiddens": num_hiddens, "num_residual_hiddens": num_residual_hiddens, "num_residual_layers": num_residual_layers, "num_downsample_layers": num_downsample_layers, 
         "embedding_dim": embedding_dim, "batch_size": batch_size, "num_embeddings": num_embeddings, "commitment_cost": commitment_cost, "decay": decay,
         "class_weight": train_config.bce_loss_weight, "recon_loss_coeff": train_config.recon_loss_coeff},
        {"hparam/last_loss_total_test": sum_test_loss/test_iter, "hparam/last_loss_vq_test": sum_test_vq_loss/test_iter,
         "hparam/last_loss_recon_test": sum_test_recon_loss/test_iter},
        run_name=log_path)  # <- see here
    writer.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config("vqvae2DConf.yml")
    ptv_dir_train = config.PTV_DIR_TRAIN
    oar_dir_train = config.OAR_DIR_TRAIN
    dose_dir_train = config.DOSE_DIR_TRAIN
    ptv_dir_val = config.PTV_DIR_VAL
    oar_dir_val = config.OAR_DIR_VAL
    dose_dir_val = config.DOSE_DIR_VAL
    ptv_dir_test = config.PTV_DIR_TEST
    oar_dir_test = config.OAR_DIR_TEST
    dose_dir_test = config.DOSE_DIR_TEST
    save_dir = config.SAVE_DIR
    extra_conv = config.EXTRA_CONV
    in_channels = config.IN_CHANNELS

    log_save_dir = os.path.join(save_dir, "Logs")
    img_save_dir = os.path.join(save_dir, "Images")
    weight_save_dir = os.path.join(save_dir, "Weights")

    vq_target= config.VQ_TARGET

    # if config.VQ_TARGET == "PTV":
    #     in_channels = 1
    #     extra_conv = True
    # elif config.VQ_TARGET == "OAR":
    #     in_channels = 8
    #     extra_conv = True
    # elif config.VQ_TARGET == "Dose":
    #     in_channels = 1
    #     extra_conv = False

    exp_name_base = config.EXP_NAME
    bce_loss_classweight = config.VQ_LOSS_BINARYCLASSWEIGHT
    cce_loss_classweight = config.VQ_LOSS_MULTICLASSWEIGHT
    
    i = 0
    for lr in config.VQ_LR:
        for num_hiddens in config.VQ_NUM_HIDDENS:
            for num_residual_hiddens in config.VQ_NUM_RESIDUAL_HIDDENS:
                for num_residual_layers in config.VQ_NUM_RESIDUAL_LAYERS:
                    for num_downsample_layers in config.VQ_NUM_DOWNSAMPLE_LAYERS:
                        for embedding_dim in config.VQ_EMBEDDING_DIM:
                            for num_embeddings in config.VQ_NUM_EMBEDDINGS:
                                for commitment_cost in config.VQ_COMMITMENT_COST:
                                    for decay in config.VQ_DECAY:
                                        for recon_loss_coeff in config.VQ_RECON_LOSS_COEFF:
                                            i += 1
                                            train_config = TrainVQVAEConfig(batch_size=config.VQ_BATCH_SIZE, epochs=config.VQ_NUM_EPOCHS,
                                                                    learning_rate=float(lr), test_interval=config.VQ_TEST_INTERVAL, vq_target=vq_target,
                                                                    log_save_dir=log_save_dir, img_save_dir=img_save_dir, weight_save_dir=weight_save_dir,
                                                                    bce_loss_weight=bce_loss_classweight,
                                                                    recon_loss_coeff=recon_loss_coeff, 
                                                                    ptv_dir_train=ptv_dir_train, oar_dir_train=oar_dir_train, dose_dir_train=dose_dir_train,
                                                                    ptv_dir_val=ptv_dir_val, oar_dir_val=oar_dir_val, dose_dir_val=dose_dir_val,
                                                                    ptv_dir_test=ptv_dir_test, oar_dir_test=oar_dir_test, dose_dir_test=dose_dir_test,
                                                                    cce_loss_weight=cce_loss_classweight,
                                                                    weight_path=config.PRETRAINED_WEIGHT_PATH)
                                            model_config = VQVAEConfig(in_channels=in_channels, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
                                                                    num_residual_hiddens=num_residual_hiddens, num_embeddings=num_embeddings,
                                                                    embedding_dim=embedding_dim, num_downsample_layers=num_downsample_layers,
                                                                    commitment_cost=commitment_cost, decay=decay, extra_conv=extra_conv)
                                            train(exp_name_base, f"Run{str(i)}", model_config, train_config)