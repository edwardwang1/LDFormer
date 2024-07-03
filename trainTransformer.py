
from torch.utils.data import random_split
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
from typing import List

from dataset import Context, ContextFromDirectory
from config import load_config
from nanoGPT import GPT, GPTConfig
from vqvae import VQVAE, VQVAEConfig
import json
import math
import torch.nn.functional as F

@dataclass
class TrainTransformerConfig:
    context_dir_train: str = ""
    context_dir_test: str = ""
    context_dir_validation: str = ""
    batch_size: int = 16
    epochs: int = 400
    test_interval: int = 50
    learning_rate: float = 3e-4
    log_save_dir: str = "logs"
    img_save_dir: str = "imgs"
    weight_save_dir: str = "weights"
    logit_save_dir: str = "logits"
    block_size: int = 256
    num_target_tokens: int = 384
    lr_warmup_iters: int = 1000
    decay_lr: bool = False
    weight_path: str = ""
    model_config_dict_path: str = ""

def decode(model, indices_np, device="cpu"):
    indices = indices_np.copy()
    indices -= 1 #decrementing to be consistent with the encoding
    indices_torch = torch.tensor(indices, dtype=torch.int64).to(device)
    indices_torch = torch.clamp(indices_torch, min=0)

    x_recon = model.decode_from_c(indices_torch)
    
    x_recon_numpy = x_recon.squeeze(0).squeeze(0).cpu().detach().numpy()

    return x_recon_numpy

def saveImg(v1, v2, save_path, indexing="max"):
    if indexing == "middle":
        ax_loc = v1.shape[0] // 2
        sag_loc = v1.shape[1] // 2
        cor_loc = v1.shape[2] // 2
    else:
        index_of_max = np.where(v1 == np.max(v1))
        ax_loc = int(np.mean(index_of_max[0]))
        sag_loc = int(np.mean(index_of_max[1]))
        cor_loc = int(np.mean(index_of_max[2]))

    f, axarr = plt.subplots(3, 2)
    axarr[0, 0].imshow(v1[ax_loc, :, :])
    axarr[0, 0].set_axis_off()
    axarr[1, 0].imshow(v1[:, sag_loc, :])
    axarr[1, 0].set_axis_off()
    axarr[2, 0].imshow(v1[:, :, cor_loc])
    axarr[2, 0].set_axis_off()

    axarr[0, 1].imshow(v2[ax_loc, :, :])
    axarr[0, 1].set_axis_off()
    axarr[1, 1].imshow(v2[:, sag_loc, :])
    axarr[1, 1].set_axis_off()
    axarr[2, 1].imshow(v2[:, :, cor_loc])
    axarr[2, 1].set_axis_off()

    f.tight_layout(pad=0.4, w_pad=-8, h_pad=0.1)
    plt.savefig(save_path, bbox_inches='tight')
    #plt.close(f)
    return f

def saveLatentImg(latent1, latent2, save_path, max=10, min=0):
    #latent 1 and latent 2 are the the same size, 1 dimensional vectors
    #reshape latent 1 and 2 to be of shape (4, -1)
    latent1 = latent1.reshape(8, -1)
    latent2 = latent2.reshape(8, -1)

    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(latent1, vmin=min, vmax=max)
    axarr[0].set_axis_off()
    axarr[1].imshow(latent2, vmin=min, vmax=max)
    axarr[1].set_axis_off()

    f.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    return f


def getQuantizedVectors(vqvae, encoding_indices):
    encoding_indices = encoding_indices.clone()
    encoding_indices = encoding_indices - 1
    encoding_indices = torch.clamp(encoding_indices, min=0)
    encodings = torch.zeros(encoding_indices.shape[0], vqvae._vq_vae._num_embeddings, device=encoding_indices.device)
    encodings.scatter_(1, encoding_indices, 1)

    # Quantize and unflatten
    quantized = torch.matmul(encodings, vqvae._vq_vae._embedding.weight).view(vqvae._vq_vae._input_shape).permute(0, 4, 1, 2, 3).contiguous()

    return quantized

def saveLatentImgVectors(model, latent1, latent2, save_path, device="cuda"):
    true_quantized = getQuantizedVectors(model, torch.from_numpy(latent1.reshape(-1, 1)).long().to(device))
    recon_quantized = getQuantizedVectors(model, torch.from_numpy(latent2.reshape(-1, 1)).long().to(device))

    MSE_loss = F.mse_loss(true_quantized, recon_quantized).item()

    true_quantized = true_quantized.detach().cpu().numpy().reshape(16, -1)
    recon_quantized = recon_quantized.detach().cpu().numpy().reshape(16, -1)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(true_quantized, aspect='auto')
    ax[0].set_axis_off()
    ax[1].imshow(recon_quantized, aspect='auto')
    ax[1].set_axis_off()
    fig.tight_layout(pad=0.4, w_pad=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    return fig, MSE_loss

def get_lr(it, warmup_iters, max_iters, learning_rate):
    lr_decay_iters = max_iters
    min_lr = learning_rate / 10
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters + 1)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train(exp_name_base, exp_name, transformer_config, train_config):
    with open(train_config.model_config_dict_path, 'r') as f:
        model_config_dict = json.load(f)

    model_config_dose = VQVAEConfig(**model_config_dict['dose'])
    weight_dir_dict = model_config_dict['weight_dir_dict']
    device = "cuda"

    vqvqe = VQVAE(model_config_dose)
    vqvqe.load_state_dict(torch.load(weight_dir_dict['Dose'], map_location=torch.device(device)))
    vqvqe.eval()
    vqvqe.to(device)

    block_size = train_config.block_size

    train_dataset = ContextFromDirectory(directory=train_config.context_dir_train)
    valid_dataset = ContextFromDirectory(directory=train_config.context_dir_validation)
    #test_dataset = Context(directory=train_config.context_dir_test)

    # Create DataLoader objects for the train and test sets
    batch_size = train_config.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    learning_rate = train_config.learning_rate
    max_iters = train_config.epochs

    eval_interval = train_config.test_interval # keep frequent because we'll overfit

    #Train loop
    model = GPT(transformer_config)
    model.to(device)
    if len(train_config.weight_path) > 1:
        model.load_state_dict(torch.load(train_config.weight_path, map_location=torch.device(device)))
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    beta1 = 0.9
    beta2 = 0.95
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, betas=(beta1, beta2), device_type=device)

    num_examples = 5
    
    #Create dirs for saving
    log_path = os.path.join(train_config.log_save_dir, exp_name_base, exp_name)
    img_path = os.path.join(train_config.img_save_dir, exp_name_base, exp_name)
    weight_path = os.path.join(train_config.weight_save_dir, exp_name_base, exp_name)
    logit_path = os.path.join(train_config.logit_save_dir, exp_name_base, exp_name)
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if not os.path.exists(logit_path):
        os.makedirs(logit_path)

    writer = SummaryWriter(log_path)

    epoch_loop = tqdm.tqdm(range(max_iters + 1))
    for epoch in epoch_loop:
        #Only decay learning rate if flag is set
        if train_config.decay_lr:
            lr = get_lr(epoch, train_config.lr_warmup_iters, max_iters, learning_rate=learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        sum_train_loss = 0
        sum_train_loss_v2 = 0
        #model.train()
        train_iter = 0
        train_loss_iter = 0
        for batch_idx, contexts in enumerate(train_dataloader):
            model.train()
            x = contexts[:, :-1].to(device)
            y = contexts[:, 1:].to(device)        

            """
            The line
            bias = bias.masked_fill(x.unsqueeze(1).unsqueeze(1) == 0, 0)
            replaces all elements of the triangular mask with 0 if the PTV value is 0, corresponding to no PTV
            
            Verify with this
            a = torch.tensor([[1, 1, 4, 5, 0, 0, 0 , 3, 4, 2], [1, 2, 9, 0, 0, 0, 0 , 6, 4, 3], [2, 4, 0, 0, 0, 0, 0 , 9, 2, 2], [1, 8, 4, 5, 2, 7, 0 , 2, 4, 1]])
            b = torch.tril(torch.ones(4, 10, 10)).view(4, 1, 10, 10)
            c = b.masked_fill(a.unsqueeze(1).unsqueeze(1) == 0, 0)

            print(a)
            print(b)
            print(c)
            """
            bias = torch.tril(torch.ones(x.shape[0], block_size, block_size)).view(x.shape[0], 1, block_size, block_size).to(device)
            bias = bias.masked_fill(x.unsqueeze(1).unsqueeze(1) == 0, 0)

            logits, loss = model(x, bias, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_iter += 1
            sum_train_loss += loss.detach().item()

            if epoch % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    if batch_idx == 0:
                        #Saving Logits and Images for training
                        input_token_train = contexts[:, :-train_config.num_target_tokens].to(device)
                        y_train = contexts[:, -train_config.num_target_tokens:].to(device)

                        bias_train = torch.tril(torch.ones(input_token_train.shape[0], input_token_train.shape[1], input_token_train.shape[1])).view(input_token_train.shape[0], 1, input_token_train.shape[1], input_token_train.shape[1]).to(device)
                        bias_train = bias_train.masked_fill(input_token_train.unsqueeze(1).unsqueeze(1) == 0, 0)

                        predictions_train, train_loss_v2 = model.generate(input_token_train, train_config.num_target_tokens, bias_train, y_train, top_k=1)

                        #train_loss_v2 = 1 - torch.sum(predictions_train[:, -train_config.num_target_tokens:] == y_train).float() / torch.numel(y_train)


                        train_fig_list = []
                        train_fig_latent_list = []
                        train_fig_quantized_latent_list = []
                        train_recon_mse_losses = []
                        train_recon_latent_mse_losses = []
                        for i in range(num_examples):
                            pred_indices_train = predictions_train.cpu().detach().numpy()[i, -train_config.num_target_tokens:]
                            pred_indices_train = pred_indices_train.astype(int).flatten()
                            real_indices_train = y_train.detach().cpu().numpy()[i].flatten()                    

                            combined = np.hstack((real_indices_train.reshape(-1, 1), pred_indices_train.reshape(-1, 1)))
                            np.savetxt(os.path.join(logit_path, "train_logits_epoch_" + str(epoch) + "_" + str(i) + ".txt"), combined, fmt='%i', delimiter=",")
                            x_recon_fake = decode(vqvqe, pred_indices_train.reshape(train_config.num_target_tokens, 1), device)
                            x_recon_real = decode(vqvqe, real_indices_train.reshape(train_config.num_target_tokens, 1), device)

                            train_recon_mse_losses.append(F.mse_loss(torch.tensor(x_recon_fake).to(device), torch.tensor(x_recon_real).to(device)).item())

                            train_fig_list.append(saveImg(x_recon_real, x_recon_fake, os.path.join(img_path, "train_" + str(epoch) + "_" + str(i) + ".png")))
                            train_fig_latent_list.append(saveLatentImg(pred_indices_train, real_indices_train, os.path.join(img_path, "train_latent_" + str(epoch) + "_" + str(i) + ".png"), max=transformer_config.vocab_size, min=0))
                            train_fig_quantized_latent, train_recon_latent_mse_loss = saveLatentImgVectors(vqvqe, pred_indices_train, real_indices_train, os.path.join(img_path, "train_quantized_latent_" + str(epoch) + "_" + str(i) + ".png"), device)
                            train_fig_quantized_latent_list.append(train_fig_quantized_latent)
                            train_recon_latent_mse_losses.append(train_recon_latent_mse_loss)

                        sum_train_loss_v2 += train_loss_v2
                        train_loss_iter += 1
                        train_recon_mse_loss = np.mean(train_recon_mse_losses)
                        train_recon_latent_mse_loss = np.mean(train_recon_latent_mse_losses)

        if epoch % 10 == 0:
            writer.add_scalar('train_loss', sum_train_loss/train_iter, epoch)

        if epoch % eval_interval == 0:    
            test_loss_iter = 0
            sum_test_loss_v2 = 0
            sum_test_loss = 0
            test_iter = 0
            model.eval()
            with torch.no_grad():
                for batch_idx_test, contexts_test in enumerate(valid_dataloader):
                    model.eval()
                    x_test = contexts_test[:, :-1].to(device)
                    y_t = contexts_test[:, 1:].to(device)

                    bias_t = torch.tril(torch.ones(x_test.shape[0], block_size, block_size)).view(x_test.shape[0], 1, block_size, block_size).to(device)
                    bias_t = bias_t.masked_fill(x_test.unsqueeze(1).unsqueeze(1) == 0, 0)

                    logits_test, test_loss = model(x_test, bias_t, y_t)
                    sum_test_loss += test_loss.detach().item()

                    test_iter += 1

                    if batch_idx_test == 0:
                        input_token_test = contexts_test[:, :-train_config.num_target_tokens].to(device)
                        y_test = contexts_test[:, -train_config.num_target_tokens:].to(device)

                        bias_test = torch.tril(torch.ones(input_token_test.shape[0], input_token_test.shape[1], input_token_test.shape[1]).view(input_token_test.shape[0], 1, input_token_test.shape[1], input_token_test.shape[1])).to(device)
                        bias_test = bias_test.masked_fill(input_token_test.unsqueeze(1).unsqueeze(1) == 0, 0)

                        predictions_test, test_loss_v2 = model.generate(input_token_test, train_config.num_target_tokens, bias_test, y_test, top_k=1)

                        #test_loss = 1 - torch.sum(predictions_test[:, -train_config.num_target_tokens:] == y_test).float() / torch.numel(y_test)
                    

                        #Saving weights
                        if epoch != 0:
                            #pass
                            torch.save(model.state_dict(), os.path.join(weight_path, "model_" + str(epoch) + ".pth"))
                        #Saving first element of batch
                        test_fig_list = []
                        test_fig_latent_list = []
                        test_fig_quantized_latent_list = []
                        test_recon_latent_mse_losses = []
                        test_recon_mse_losses = []
                        for i in range(num_examples):
                            pred_indices_test = predictions_test.cpu().detach().numpy()[i, -train_config.num_target_tokens:]
                            pred_indices_test = pred_indices_test.astype(int).flatten()
                            real_indices_test = y_test.detach().cpu().numpy()[i].flatten()
                            combined = np.hstack((real_indices_test.reshape(-1, 1), pred_indices_test.reshape(-1, 1)))
                            np.savetxt(os.path.join(logit_path, "test_logits_epoch_" + str(epoch) + "_" + str(i) + ".txt"), combined, fmt='%i', delimiter=",")
                            x_recon_fake = decode(vqvqe, pred_indices_test.reshape(train_config.num_target_tokens, 1), device)
                            x_recon_real = decode(vqvqe, real_indices_test.reshape(train_config.num_target_tokens, 1), device)
                            test_recon_mse_losses.append(F.mse_loss(torch.tensor(x_recon_fake).to(device), torch.tensor(x_recon_real).to(device)).item())
                            test_fig_list.append(saveImg(x_recon_real, x_recon_fake, os.path.join(img_path, "test_" + str(epoch) + "_" + str(i) + ".png")))
                            test_fig_latent_list.append(saveLatentImg(pred_indices_test, real_indices_test, os.path.join(img_path, "test_latent_" + str(epoch) + "_" + str(i) + ".png"), max=transformer_config.vocab_size, min=0))
                            test_fig_quantized_latent, test_recon_latent_mse_loss = saveLatentImgVectors(vqvqe, pred_indices_test, real_indices_test, os.path.join(img_path, "test_quantized_latent_" + str(epoch) + "_" + str(i) + ".png"), device)
                            test_fig_quantized_latent_list.append(test_fig_quantized_latent)
                            test_recon_latent_mse_losses.append(test_recon_latent_mse_loss)

                        test_loss_iter += 1
                        sum_test_loss_v2 += test_loss_v2
                        test_recon_mse_loss = np.mean(test_recon_mse_losses)
                        test_recon_latent_mse_loss = np.mean(test_recon_latent_mse_losses)


            writer.add_scalar('test_loss', sum_test_loss/test_iter, epoch)

    
            writer.add_scalar('train_loss_v2', sum_train_loss_v2/train_loss_iter, epoch)
            writer.add_scalar('train_recon_mse_loss', train_recon_mse_loss, epoch)
            writer.add_scalar('train_quant_latent_mse_loss', train_recon_latent_mse_loss, epoch)
            for i in range(num_examples):
                writer.add_figure("Images from Training Set " + str(i), train_fig_list[i], epoch)
                plt.close(train_fig_list[i])
                writer.add_figure("Latent from Training Set " + str(i), train_fig_latent_list[i], epoch)
                plt.close(train_fig_latent_list[i])
                writer.add_figure("Quantized Latent from Training Set " + str(i), train_fig_quantized_latent_list[i], epoch)
                plt.close(train_fig_quantized_latent_list[i])

            writer.add_scalar('test_loss_v2', sum_test_loss_v2/test_loss_iter, epoch)
            writer.add_scalar('test_recon_mse_loss', test_recon_mse_loss, epoch)
            writer.add_scalar('test_quant_latent_mse_loss', test_recon_latent_mse_loss, epoch)
            for i in range(num_examples):
                writer.add_figure("Images from Testing Set " + str(i), test_fig_list[i], epoch)
                plt.close(test_fig_list[i])
                writer.add_figure("Latent from Testing Set " + str(i), test_fig_latent_list[i], epoch)
                plt.close(test_fig_latent_list[i])
                writer.add_figure("Quantized Latent from Testing Set " + str(i), test_fig_quantized_latent_list[i], epoch)
                plt.close(test_fig_quantized_latent_list[i])
    
    writer.add_hparams(
        {"n_layers": transformer_config.n_layer, "n_heads": transformer_config.n_head, "n_embds": transformer_config.n_embd, 
         "dropout": transformer_config.dropout, "bias": transformer_config.bias,
            "lr": learning_rate, "lr_warmup_iters": train_config.lr_warmup_iters,
            "batch_size": train_config.batch_size,
         },
        {"hparam/last_loss_total_test": sum_test_loss_v2/test_loss_iter},
        run_name=log_path)  # <- see here
    writer.close()       


if __name__ == '__main__':
    config = load_config("transformerConf.yml")
    context_dir_train = config.CONTEXT_DIR_TRAIN
    context_dir_test = config.CONTEXT_DIR_TEST
    context_dir_validation = config.CONTEXT_DIR_VALIDATION
    num_epochs = config.NUM_EPOCHS
    save_dir = config.SAVE_DIR

    logits_save_dir = os.path.join(save_dir, "Logits")
    log_save_dir = os.path.join(save_dir, "Logs")
    img_save_dir = os.path.join(save_dir, "Images")
    weight_save_dir = os.path.join(save_dir, "Weights")

    exp_name_base = config.EXP_NAME

    i = 0
    for learning_rate in config.LEARNING_RATE:
        for n_layer in config.N_LAYERS:
            for n_head in config.N_HEADS:
                for n_embd in config.N_EMBD:
                    for dropout in config.DROPOUT:
                        for bias in config.BIAS:
                            for warmup_iters in config.LR_WARMUP_ITERS:
                                for batch_size in config.BATCH_SIZE:
                                    for decay_lr in config.DECAY_LR:
                                        i += 1
                                        transformer_config = GPTConfig(n_layer = n_layer, n_head = n_head,
                                                                    n_embd = n_embd, dropout = dropout, bias=bias,
                                                                    vocab_size=config.VOCAB_SIZE, block_size=config.BLOCK_SIZE,
                                                                    num_target_tokens=config.NUM_TARGET_TOKENS
                                                                    )
                                        train_config = TrainTransformerConfig(context_dir_train=context_dir_train,
                                                                            context_dir_test=context_dir_test,
                                                                            context_dir_validation=context_dir_validation,
                                                                                batch_size= batch_size, epochs=num_epochs,
                                                                                test_interval=config.TEST_INTERVAL,
                                                                                learning_rate=float(learning_rate), log_save_dir=log_save_dir,
                                                                                img_save_dir=img_save_dir, weight_save_dir=weight_save_dir,
                                                                                logit_save_dir=logits_save_dir, block_size=config.BLOCK_SIZE,
                                                                                num_target_tokens=config.NUM_TARGET_TOKENS,
                                                                                lr_warmup_iters=warmup_iters,
                                                                                weight_path=config.PRETRAINED_WEIGHT_PATH,
                                                                                decay_lr=decay_lr,
                                                                                model_config_dict_path=config.MODEL_CONFIG_DICT_PATH)
                                        print(f"Run{str(i)}", transformer_config)
                                        train(exp_name_base, f"Run{str(i)}", transformer_config, train_config)
                            