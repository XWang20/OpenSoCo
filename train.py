import torch,os
import bmtrain as bmp
from model_center.model import Roberta, RobertaConfig
from model_center.dataset import MMapIndexedDataset, DistributedMMapIndexedDataset, DistributedDataLoader
from model_center.utils import print_inspect
from dataset import BertDataset
from torch.utils.tensorboard import SummaryWriter
import time
from arguments import get_args
from scale_model import scale_roberta_model
import os

import wandb

def get_file_path(root_dir):
    p = []
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if name[0] != '.':
                p.append(os.path.join(root, name))
    return p

def get_last_step(args, current_step):
    p = get_file_path(os.path.join(args.save, 'checkpoints'))
    last_step = 0
    for filename in p:
        step = int(filename.split('/')[-1].split('.')[-2].split('-')[-1])
        if step > last_step and step != current_step:
            last_step = step
    return last_step

def get_model(args):
    config = RobertaConfig.from_json_file(args.model_config)
    assert isinstance(config, RobertaConfig)
    model = Roberta(config)
    if (args.load != None) and (get_last_step(args, 0) == 0):
        bmp.print_rank(f"Loading from checkpoint {args.load}...")
        bmp.load(model, args.load)
    else:
        bmp.print_rank(f"Loading from checkpoint-{args.start_step}.pt...")
        bmp.load(model, os.path.join(args.save, "checkpoints", f"checkpoint-{args.start_step}.pt"))
    print_inspect(model, "*")
    return model

def get_optimizer(args, model):
    optimizer = bmp.optim.AdamOffloadOptimizer(model.parameters(), 
                                                lr = 5e-4,
                                                betas = (0.9, 0.98),
                                                weight_decay=args.weight_decay)
    if args.load is not None:
        if os.path.exists(os.path.join(args.save, 'optimizers', "optimizer.rank-%d.opt" % 0)):
            states = torch.load(
                os.path.join(args.save, 'optimizers', "optimizer.rank-%d.opt" % (bmp.rank())))
            # del states['state'] # 不加载state
            # optimizer_state = optimizer.state_dict()
            # optimizer_state.update(states)
            # optimizer.load_state_dict(optimizer_state)
            optimizer.load_state_dict(states)
            for name, param in optimizer.state_dict().items():
                print(name, param)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == 'linear':
        lr_scheduler = bmp.lr_scheduler.Linear(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step)
    else:
        lr_scheduler = bmp.lr_scheduler.Noam(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step) 
    return lr_scheduler

def get_optim_manager(args, optimizer, lr_scheduler):
    optim_manager = bmp.optim.OptimManager(loss_scale = args.loss_scale)
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    return optim_manager

def setup_model_and_optimizer(args):
    # get the model
    model = get_model(args)
    bmp.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    optim_manager = get_optim_manager(args, optimizer, lr_scheduler)
    bmp.synchronize()
    # get the memory usage
    bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmp.synchronize()
    return model, optimizer, lr_scheduler, optim_manager

def get_train_dataset(args):
    print(bmp.rank(), bmp.world_size())
    input_ids_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'input_ids', bmp.rank(), bmp.world_size())
    lm_pos_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'lm_pos', bmp.rank(), bmp.world_size())
    masked_labels_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'masked_labels', bmp.rank(), bmp.world_size())
    length_list_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'length_list', bmp.rank(), bmp.world_size())
    bert_dataset = BertDataset(input_ids_dataset, lm_pos_dataset, masked_labels_dataset, length_list_dataset)

    return bert_dataset

def get_valid_dataset(dataset_path):
    input_ids_dataset = MMapIndexedDataset(os.path.join(dataset_path,'valid', 'input_ids'))
    lm_pos_dataset = MMapIndexedDataset(os.path.join(dataset_path, 'valid','lm_pos'))
    masked_labels_dataset = MMapIndexedDataset(os.path.join(dataset_path, 'valid','masked_labels'))
    length_list_dataset = MMapIndexedDataset(os.path.join(dataset_path, 'valid','length_list'))
    bert_dataset = BertDataset(input_ids_dataset, lm_pos_dataset, masked_labels_dataset, length_list_dataset)

    return bert_dataset

def valid(model, dev_dataloader, loss_func, step, writer):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for data in dev_dataloader:
            input_ids, attention_mask, labels = data
            input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, return_logits=True)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
            global_loss = bmp.sum_loss(loss).item()
            valid_loss += global_loss

        if bmp.rank() == 0:
            writer.add_scalar("Loss/dev", valid_loss, step)
            wandb.log({"loss/dev": valid_loss}, step=step)

        bmp.print_rank(
                        "{} | Iter: {:6d} | valid  loss: {:.4f}".format(
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            step,
                            valid_loss / len(dev_dataloader)
                        )
                    )
    model.train()

def batch_iter(args, dataset):
    # 中文模型
    # st = args.start_step * args.batch_size
    # st = 16500*128*2+38500*256   #需要手动计算
    # switch to max_length=256, origin st = 347500, current st = 364000. so pass data st = 364000-347500
    # 8卡换到4卡, pass st *= 2
    # 中途停了一次，current st=369500, max_length=256, st+=(369500-364000)*256
    # 遇到nan了，要跳过一些数据继续训，current st=392500, max_length=256, st+=(392500-364000)*256=28500*256, 再跳过一截数据，假设多跳过1w step的数据，st+=38500*256
    # 英文模型
    # st = 0  # 从第一个数据开始训练
    st = 2500
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    while True:
        input_ids, attention_mask, labels = dataset[st]
        st += 1
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

        if len(input_ids_list) > args.batch_size:
            yield {
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "labels": torch.stack(labels_list)
            }
            input_ids_list = []
            attention_mask_list = []
            labels_list = []


def scale_down_model(scale, model, args):
    bmp.print_rank(f"Nan loss inspected. Now scaling down the model with factor 10.0...\nBefore Scaling:")
    print_inspect(model, "*")
    new_dict = scale_roberta_model(scale, model, config_file = args.model_config)
    bmp.print_rank(f"After Scaling:")
    print_inspect(model, "*") 
    model.load_state_dict(new_dict)
    return model


def pretrain(args, model, optimizer, lr_scheduler, optim_manager, train_dataset, dev_dataloader):
    loss_func = bmp.loss.FusedCrossEntropy(ignore_index=-100)

    start_step = args.start_step
    log_loss = 0
    os.makedirs(os.path.join(args.save, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'optimizers'), exist_ok=True)
    if bmp.rank() == 0:
        writer = SummaryWriter(os.path.join(args.save, 'tensorborads'))
    else:
        writer = None
    valid(model, dev_dataloader, loss_func, start_step // args.gradient_accumulate, writer)
    for step, data in enumerate(batch_iter(args, train_dataset)):
        if (start_step + step + 1) % args.gradient_accumulate == 1:
            optim_manager.zero_grad() # when not doing
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        labels = data['labels'].cuda()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, return_logits=True)
        loss = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        global_loss = bmp.sum_loss(loss).item()
        log_loss += global_loss
        loss = loss / args.gradient_accumulate
        optim_manager.backward(loss)
        if (start_step + step + 1) % args.gradient_accumulate == 0:
            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type = 2)
            try:
                optim_manager.step()
            except:
                # for name, param in model.state_dict().items():
                #     print(name, param)
                #     print("grad=", param.grad)
                exit(0)
            if torch.isnan(grad_norm):
                model = scale_down_model(scale = 10.0, model = model, args = args)
        if (start_step + step + 1) % args.log_iters == 0:
            bmp.print_rank(
                    "{} | Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f}".format(
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        (step + 1 + start_step) // args.gradient_accumulate,
                        log_loss / args.log_iters,
                        lr_scheduler.current_lr,
                        optim_manager.loss_scale,
                        grad_norm
                    )
                )
            log_loss = 0

        if (start_step + step + 1) % args.valid_iters == 0:
            valid(model, dev_dataloader, loss_func, start_step + step + 1, writer)

        if bmp.rank() == 0:
            wandb.log({"loss/train": global_loss, 
                       "grad_norm": grad_norm,
                       "loss_scale": optim_manager.loss_scale,
                       "learning_rate": lr_scheduler.current_lr}, step=step+start_step+1)
            
            writer.add_scalar("Loss/train", global_loss, step + start_step + 1)

        if args.save != None and (step + start_step + 1) % args.save_iters == 0:
            bmp.save(model, os.path.join(args.save, 'checkpoints', "checkpoint-%d.pt" % (step + start_step + 1)))
            # save optimizer
            torch.save(optimizer.state_dict(),
                os.path.join(args.save, 'optimizers', "optimizer.rank-%d.opt" % (bmp.rank())))           
            bmp.print_rank(f"Saving checkpoint at {(step + start_step + 1) } step.")
        
def init_wandb(args):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="opensoco",
        name="en-a100-2",
        
        # track hyperparameters and run metadata
        config={
            "batch_size": args.batch_size,
            "start_step": args.start_step,
            "grad_clipping": args.clip_grad,
            "gradient_accumulate": args.gradient_accumulate,
            "learning_rate": args.lr
        }
    )

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    bmp.print_rank("Init bmp distributed.")
    bmp.init_distributed(seed=args.seed, zero_level=2)
    
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def main():
    args = initialize()
    last_step = get_last_step(args, args.start_step)
    if last_step > args.start_step:
        args.start_step = last_step
    args.start_step = 360000
    print(args)
    init_wandb(args)
    model, optimizer, lr_scheduler, optim_manager = setup_model_and_optimizer(args)
    train_dataset = get_train_dataset(args)
    valid_dataset = get_valid_dataset(args.test_dataset)
    dev_dataloader = DistributedDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    pretrain(args, model, optimizer, lr_scheduler, optim_manager, train_dataset, dev_dataloader)

if __name__ == '__main__':
    main()
