# train jepa model with kdd99dataset, write a training process and evaluation process.

from datasets.utils import KDDCup99Dataset
from jepa import TEncoder, Predictors, apply_mask, TJEPA
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
import random
from torch.optim import AdamW
from scheduler import WarmupCosineSchedule, CosineWDSchedule
from attention import PositionalEncoding
from easydict import EasyDict
from tqdm import tqdm
import logging
import copy


# 设置各种随机种子包括torch，numpy，python等，方便复现结果
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)

def main_process():
    config = yaml.load(open('./transformers/jepa_config.yaml'), Loader=yaml.FullLoader)
    config = EasyDict(config)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('Start training jepa model with kdd99dataset')
    ema = config.train_config.ema
    
    # model
    src_encoder = TEncoder(config.model_config.encoder_config)
    pred = Predictors(config.model_config.pred_config)
    targ_encoder = copy.deepcopy(src_encoder)
    for p in targ_encoder.parameters():
        p.requires_grad = False
    jepa_config = {
        'd_model': config.model_config.encoder_config.d_model,
        'seq_len': config.train_config.seq_len,
        'src_encoder': src_encoder,
        'pred_layer': pred,
        'tag_encoder': targ_encoder,
        'mask_len': config.train_config.mask_len,
        'cate_feature_num': config.model_config.encoder_config.cate_feature_num
    }
    jepa_config = EasyDict(jepa_config)
    tjepa = TJEPA(jepa_config).to(config.train_config.device)
    # dataset
    train_dataset = KDDCup99Dataset(config.train_config.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=config.train_config.batch_size, shuffle=True, num_workers=config.train_config.num_workers)
    valid_dataset = KDDCup99Dataset(config.train_config.valid_data_path)
    valid_loader = DataLoader(valid_dataset, batch_size=config.train_config.batch_size, shuffle=False, num_workers=config.train_config.num_workers)
    test_dataset = KDDCup99Dataset(config.train_config.test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=config.train_config.batch_size, shuffle=False, num_workers=config.train_config.num_workers)
    # optimizer and scheduler
    model_params = list(src_encoder.parameters()) + list(pred.parameters()) + list(targ_encoder.parameters())
    optimizer = AdamW(model_params, lr=config.train_config.learning_rate, weight_decay=config.train_config.weight_decay)
    scheduler = WarmupCosineSchedule(optimizer, int(config.train_config.warmup_steps*len(train_loader)), config.train_config.start_lr, config.train_config.learning_rate, T_max=(1.25*config.train_config.num_epochs*len(train_loader)), final_lr=config.train_config.final_lr)
    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(1.25*config.train_config.num_epochs*len(train_loader))
                          for i in range(int(1.25*config.train_config.num_epochs*len(train_loader))+1))
    best_valid_loss = float('inf')
    early_stop_cnt = 0
    for epoch in tqdm(range(config.train_config.num_epochs)):
        
        train_epoch(tjepa, optimizer, scheduler, momentum_scheduler, train_loader, logger, config)
        valid_loss = valid_epoch(tjepa, valid_loader, config)
        early_stop_cnt += 1
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(tjepa.src_encoder.state_dict(), config.train_config.save_path + '/src_encoder.pth')
            torch.save(tjepa.pred_layer.state_dict(), config.train_config.save_path + '/pred.pth')
            torch.save(tjepa.tag_encoder.state_dict(), config.train_config.save_path + '/targ_encoder.pth')
            early_stop_cnt = 0
        if config.train_config.early_stop_patience < early_stop_cnt:
            break
        logger.info(f'Epoch: {epoch+1},  Valid Loss: {valid_loss:.6f}, Best Valid Loss: {best_valid_loss:.6f}')

def train_epoch(tjepa, optimizer, scheduler, momentum_scheduler, train_loader, logger,config):
    tjepa.train()
    total_loss = 0
    mask_len = config.train_config.mask_len
    for i, batch in enumerate(tqdm(train_loader)):
        scheduler.step()
        data, label = batch
        data = data.to(config.train_config.device)
        _, loss = tjepa(data)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % config.train_config.log_interval == 0:
            logger.info(f'Train  Step: {i+1}, Loss: {loss.item():.6f}, lr: {optimizer.param_groups[0]["lr"]:.6f}')
        # Step 2. update target encoder
     # Step 3. momentum update of target encoder
    with torch.no_grad():
        m = next(momentum_scheduler)
        for param_q, param_k in zip(tjepa.src_encoder.parameters(), tjepa.tag_encoder.parameters()):
            param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

def valid_epoch(tjepa, valid_loader, config):
    tjepa.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_loader)):
            data, label = batch
            data = data.to(config.train_config.device)
            _, loss = tjepa(data)
            total_loss += loss.item()
    return total_loss / len(valid_loader)


if   __name__ == '__main__':
    main_process()