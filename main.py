# main.py
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import time
import numpy as np
import random
import json

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam

import monai
from tqdm import tqdm
from sklearn import metrics
from statistics import mean
import matplotlib.pyplot as plt
import wandb

import src.utils.conutils as utils 
from src.data.dataloader import nucleiSegDataset
from src.models.unet3plus.unet3p import UNet_3Plus
from src.utils.combined_loss import NucleiInstanceLoss
from src.utils.post_process import postprocess_batch
from src.utils.evaluation import evaluate_instance_segmentation

torch.cuda.set_device(0)


def setSeeds():
    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.set_grad_enabled(True)

def wandb_init(config, run_name):
    # Initialize wandb
    print(config.other.wandb)
    config_dict = OmegaConf.to_container(config)
    wandb.init(project=config.other.wandb, config=config_dict, name=run_name)

def make_preNecessities(exp_dir):
    # make logging directory
    
    dirs_ = [exp_dir + "/models/", exp_dir + "/plots/"]
    for dir_ in dirs_:
        os.makedirs(dir_, exist_ok=True)

def save_validation_images(segment_type, epoch, gt, predicted, exp_dir):

    # save 1 image
    if segment_type == "semantic":
        gt = utils.get_binary_image(gt)
        predicted = utils.get_binary_image(predicted)

        # normalize gt and predicted
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
        predicted = (predicted - np.min(predicted)) / (np.max(predicted) - np.min(predicted))
    else:
        gt = gt.squeeze().numpy()
        predicted = predicted


    os.makedirs(exp_dir + "/step/", exist_ok=True)
    plt.imsave(exp_dir + f"/step/{epoch}_gt.png", gt)
    plt.imsave(exp_dir + f"/step/{epoch}_pred.png", predicted)



def run_epoch(seg_type, model, dataloader, optimizer, criterion, epoch, num_epochs, device,exp_dir, mode='train'):
    
    if mode == 'train':
        model.train()
    else:
        model.eval()

    if seg_type == "semantic":
        epoch_losses = []
        epoch_accuracies = []
        epoch_mIoUs = []
    else:
        epoch_losses = []
        epoch_mIoUs = []
        epoch_dice = []
        epoch_hd95 = []
        epoch_precision = []
        epoch_recall = []
        epoch_f1 = []
        epoch_aji = []

    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
        if seg_type == "semantic":
            output = model(batch['image'].to(device))
            stk_gt = batch['mask'].squeeze().cpu().numpy()
            stk_out = output.squeeze().cpu().detach().numpy()

            #print(stk_gt.shape, stk_out.shape)
            loss = criterion(output, batch['mask'].to(device))

            acc = metrics.accuracy_score(stk_gt.flatten(), (stk_out > 0.5).flatten())
            mIoU = metrics.jaccard_score(stk_gt.flatten(), (stk_out > 0.5).flatten(), average='weighted')

        else:
            output, quant_loss_image, quant_loss_lnn = model(batch['image'].to(device))
            loss, other_losses = criterion(output, batch['mask'].to(device), quant_loss_image, quant_loss_lnn)
            try:
                quant_losses = {'quant_loss_image': quant_loss_image.detach().cpu().numpy()[0], 'quant_loss_lnn': quant_loss_lnn.detach().cpu().numpy()[0]}
            except:
                quant_losses = {'quant_loss_image': np.array([0]), 'quant_loss_lnn': np.array([0])}
            final_instance_masks = postprocess_batch(output)
            eval_metrics = evaluate_instance_segmentation(final_instance_masks, batch['mask'].numpy())
        
        
        

        if mode == 'train':
            #loss.requires_grad = True
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if mode == 'val' and i ==0:
            if seg_type == "semantic":
                save_validation_images(seg_type, epoch, stk_gt[0], stk_out[0], exp_dir)
            else:
                save_validation_images(seg_type, epoch, batch['mask'][0], final_instance_masks[0], exp_dir)

        if seg_type == "semantic":
            epoch_losses.append(loss.item())
            epoch_accuracies.append(acc)
            epoch_mIoUs.append(mIoU)
        else:
            epoch_losses.append(loss.item())
            # {'Dice', 'HD95', IoU', 'Precision', 'Recall', 'F1', 'AJI}
            epoch_dice.append(eval_metrics['Dice'])
            epoch_hd95.append(eval_metrics['HD95'])
            epoch_mIoUs.append(eval_metrics['IoU'])
            epoch_precision.append(eval_metrics['Precision'])
            epoch_recall.append(eval_metrics['Recall'])
            epoch_f1.append(eval_metrics['F1'])
            epoch_aji.append(eval_metrics['AJI'])

    if seg_type == "semantic":
        mean_loss = mean(epoch_losses)
        mean_accuracy = mean(epoch_accuracies)
        mean_mIoU = mean(epoch_mIoUs)
        return (mean_loss, mean_mIoU, mean_accuracy)
    else:
        epoch_losses = [float(x) for x in epoch_losses]
        epoch_mIoUs  = [float(x) for x in epoch_mIoUs]
        epoch_dice   = [float(x) for x in epoch_dice]
        epoch_hd95   = [float(x) for x in epoch_hd95]
        epoch_precision = [float(x) for x in epoch_precision]
        epoch_recall = [float(x) for x in epoch_recall]
        epoch_f1 = [float(x) for x in epoch_f1]
        epoch_aji = [float(x) for x in epoch_aji]

        mean_loss = mean(epoch_losses)
        mean_mIoU = mean(epoch_mIoUs)
        mean_dice = mean(epoch_dice)
        mean_hd95 = mean(epoch_hd95)
        mean_precision = mean(epoch_precision)
        mean_recall = mean(epoch_recall)
        mean_f1 = mean(epoch_f1)
        mean_aji = mean(epoch_aji)
        return {"loss": mean_loss, 
                "quant_losses": quant_losses,
                "other_losses": other_losses,
                "mIoU": mean_mIoU, 
                "Dice": mean_dice, 
                "HD95": mean_hd95, 
                "Precision": mean_precision, 
                "Recall": mean_recall, 
                "F1": mean_f1, 
                "AJI": mean_aji}



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # check log printing
    if config.other.log_print:
        console_handler = utils.log_printer()
        logging.getLogger().addHandler(console_handler)

    # Obtain a module-level logger
    logger = logging.getLogger(__name__)

    print(f"Selected Dataset - {config.dataset.name}, see the config/log file for more details.")
    # log the configuration
    logger.info("Experiment Configuration:\n%s", OmegaConf.to_yaml(config))  
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    exp_dir = hydra_cfg['runtime']['output_dir']
    print(f"Experiment output directory: {exp_dir}")

    # set seeds
    setSeeds()
    start_time = time.time()

    # make necessary directories
    make_preNecessities(exp_dir)
    logger.info("Making necessary directories for the experiment")

    # initialize wandb
    if config.other.wandb:
        run_name = exp_dir.split("/")[-2] + "_" + exp_dir.split("/")[-1]
        wandb_init(config, run_name)
        logger.info("Initialized wandb for logging")

    # saving config file
    logger.info("Saving the configuration file")
    with open(exp_dir + "/config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # set cuda device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')
    logger.info(f"Using {device} device")

    # load model
    model = UNet_3Plus(config, exp_dir)
    model.to(device)
    print(f"Model loaded to {device}")
    logger.info(f"Model loaded to {device}")

    # model segmentation type
    seg_type = config.model.segmentation_type

    # load dataloader
    train_ds = nucleiSegDataset(config, 'train')
    val_ds = nucleiSegDataset(config, 'val')

    train_dataloader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)

    print(f"Train and Validation dataloaders loaded")
    logger.info(f"Train and Validation dataloaders loaded")

    # load optimizer and loss function
    optimizer = Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=0)
    logger.info(f"Loading Optimizer: Set to Adam with lr={config.training.learning_rate} and weight_decay=0")

    if seg_type =="semantic":
        criterion = monai.losses.DiceFocalLoss()
        logger.info(f"Loading Loss Function: Set to DiceFocalLoss")
    else:
        criterion = NucleiInstanceLoss()
        logger.info(f"Loading Loss Function: Set to NuceiInstanceLoss")

    num_epochs = config.training.epochs
    logger.info(f"Training set for {num_epochs} epochs")

    # train stats
    logger.info("Initializing Training Stats")
    train_losses = []
    train_accuracies = []
    val_losses = []
    best_val_accuracy = 0
    val_accuracies = []
    train_mIoUs = []
    val_mIoUs = []

    print("Starting Training")
    logger.info("Starting Training\n")

    for epoch in range(num_epochs):
        
        # train
        if seg_type == "semantic":
            train_loss, train_mIoU, train_accuracy = run_epoch(seg_type, model, train_dataloader, optimizer, criterion, epoch, num_epochs, device, exp_dir, 'train')
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_mIoUs.append(train_mIoU)

            print('train_loss:', train_loss)
            print('train_accuracy:', train_accuracy)
            print('train_mIoU:', train_mIoU)

            logger.info(f"Epoch {epoch} - Training Loss: {train_loss}, Training Accuracy: {train_accuracy}, Training mIoU: {train_mIoU}")
        else:
            eval_metrics = run_epoch(seg_type, model, train_dataloader, optimizer, criterion, epoch, num_epochs, device, exp_dir, 'train')
            train_losses.append(eval_metrics['loss'])
            train_mIoUs.append(eval_metrics['mIoU'])
            train_accuracies.append(eval_metrics['Dice'])

            print('train_loss:', eval_metrics['loss'])
            print('train_accuracy:', eval_metrics['Dice'])
            print('train_mIoU:', eval_metrics['mIoU'])

        # log train stats
        
        if seg_type == "semantic":
            if config.other.wandb:
                wandb.log({"train_loss": train_loss, 
                    "train_mIoU": train_mIoU, 
                    "train_accuracy": train_accuracy, 
                    })
        else:
            print_metrics = {
                "train_loss": eval_metrics['loss'],
                "train_mIoU": eval_metrics['mIoU'],
                "train_dice": eval_metrics['Dice'],
                "train_hd95": eval_metrics['HD95'],
                "train_precision": eval_metrics['Precision'],
                "train_recall": eval_metrics['Recall'],
                "train_f1": eval_metrics['F1'],
                "train_aji": eval_metrics['AJI'],
                # other losses: (quant_loss, lovasz_loss, seed_loss, smooth_loss)
                "mean_quant_loss": eval_metrics['other_losses'][0].detach().cpu().item(),
                "quant_loss_image": eval_metrics['quant_losses']['quant_loss_image'].tolist(),
                "quant_loss_lnn": eval_metrics['quant_losses']['quant_loss_lnn'].tolist(),
                "lovasz_loss": eval_metrics['other_losses'][1].detach().cpu().item(),
                "seed_loss": eval_metrics['other_losses'][2].detach().cpu().item(),
                "smooth_loss": eval_metrics['other_losses'][3].detach().cpu().item()
            }
            if config.other.wandb:
                wandb.log(print_metrics)
            # log to file
            utils.save_metrics(exp_dir, "train", epoch, print_metrics)



        # validate
        if seg_type == "semantic":
            val_loss, val_mIoU, val_accuracy = run_epoch(seg_type, model, val_dataloader, optimizer, criterion, epoch, num_epochs, device, exp_dir, 'val')
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_mIoUs.append(val_mIoU)

            print('val_loss:', val_loss)
            print('val_accuracy:', val_accuracy)
            print('val_mIoU:', val_mIoU)

            logger.info(f"Epoch {epoch} - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation mIoU: {val_mIoU}")
        else:
            eval_metrics = run_epoch(seg_type, model, val_dataloader, optimizer, criterion, epoch, num_epochs, device, exp_dir, 'val')
            val_losses.append(eval_metrics['loss'])
            val_mIoUs.append(eval_metrics['mIoU'])
            val_accuracy = eval_metrics['Dice']
            val_accuracies.append(val_accuracy)

            print('val_loss:', eval_metrics['loss'])
            print('val_accuracy:', val_accuracy)
            print('val_mIoU:', eval_metrics['mIoU'])

        # log val stats
        if seg_type == "semantic":
            if config.other.wandb:
                wandb.log({"val_loss": val_loss,
                        "val_mIoU": val_mIoU,
                        "val_accuracy": val_accuracy})
        else:
            print_metrics = {
                "val_loss": eval_metrics['loss'],
                "val_mIoU": eval_metrics['mIoU'],
                "val_dice": eval_metrics['Dice'],
                "val_hd95": eval_metrics['HD95'],
                "val_precision": eval_metrics['Precision'],
                "val_recall": eval_metrics['Recall'],
                "val_f1": eval_metrics['F1'],
                "val_aji": eval_metrics['AJI'],
                # other losses: (quant_loss, lovasz_loss, seed_loss, smooth_loss)
                "mean_quant_loss": eval_metrics['other_losses'][0].detach().cpu().item(),
                "quant_loss_image": eval_metrics['quant_losses']['quant_loss_image'].tolist(),
                "quant_loss_lnn": eval_metrics['quant_losses']['quant_loss_lnn'].tolist(),
                "lovasz_loss": eval_metrics['other_losses'][1].detach().cpu().item(),
                "seed_loss": eval_metrics['other_losses'][2].detach().cpu().item(),
                "smooth_loss": eval_metrics['other_losses'][3].detach().cpu().item()
            }
            if config.other.wandb:
                wandb.log(print_metrics)
            # log to file
            utils.save_metrics(exp_dir, "val", epoch, print_metrics)
            
        
        # save interval model
        save_interval = 10
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, exp_dir + "/models/best_model.pt")
        
        if epoch % save_interval == 0:
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, exp_dir + f"/models/model_{epoch}.pt")

    print("Training Completed")
    logger.info("Training Completed")
    print(f"Total time taken: {time.time() - start_time}")
    logger.info(f"Total time taken: {time.time() - start_time}")
    print("Best Validation Accuracy: ", best_val_accuracy)
    logger.info(f"Best Validation Accuracy: {best_val_accuracy}")

    # log to wandb and finish
    if config.other.wandb:
        wandb.log({"total_time": time.time() - start_time})
        wandb.log({"best_val_accuracy": best_val_accuracy})
        wandb.finish()





    
    
    

if __name__ == "__main__":
    main()
