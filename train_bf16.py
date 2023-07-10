import torch,os
import bmtrain as bmp
from model_center.model import Roberta, RobertaConfig
from model_center.dataset import MMapIndexedDataset, DistributedMMapIndexedDataset, DistributedDataLoader
from dataset import BertDataset
import time
import datetime 
from arguments import get_args
from scale_model import scale_roberta_model

from typing import Optional
from bmtrain.loss.cross_entropy import OpFusedCrossEntropy, OpFusedCrossEntropyInplace

class FusedCrossEntropy(torch.nn.Module):
    r"""This criterion computes the cross entropy loss between input and target.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.
    `input` has to be a Tensor of size :math:`(minibatch, C)`.

    The `target` that this criterion expects should contain either:

    - Class indices in the range :math:`[0, C-1]` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

      Note that this case is equivalent to the combination of :class:`~torch.nn.LogSoftmax` and
      :class:`~torch.nn.NLLLoss`.

    - Probabilities for each class; useful when labels beyond a single class per minibatch item
      are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
      :attr:`reduction` set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\exp(\sum_{i=1}^C x_{n,i})} y_{n,c}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

    .. note::
        The performance of this criterion is generally better when `target` contains class
        indices, as this allows for optimized computation. Consider providing `target` as
        class probabilities only when a single class label per minibatch item is too restrictive.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`.
        - Target: If containing class indices, shape :math:`(N)` where each value is
          :math:`0 \leq \text{targets}[i] \leq C-1`. If containing class probabilities,
          same shape as the input.
        - Output: If :attr:`reduction` is ``'none'``, shape :math:`(N)`.
          Otherwise, scalar.

    Examples::

        >>> # Example of target with class indices
        >>> loss_func = bmt.loss.FusedCrossEntropy()
        >>> input = torch.randn(32, 100).half()
        >>> target = torch.randint(0, 100, (32,)).long()
        >>> loss = loss_func(input, target)
        >>> loss.backward()
    """
    def __init__(self,
                 weight: Optional[torch.Tensor] = None,
                 ignore_index: int = -100,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0, # TODO not supported yet
                 inplace: bool = False,
                ) -> None:
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace = inplace

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            ret = OpFusedCrossEntropyInplace.apply(input, target.int(), self.ignore_index) # return float tensor
        else:
            ret = OpFusedCrossEntropy.apply(input, target.int(), self.ignore_index) # return float tensor

        if self.weight is not None:
            if self.weight.dim() != 1 or self.weight.size(0) != input.size(1):
                raise ValueError("weight should be a 1D tensor of size C");
            w = self.weight[torch.where(target==self.ignore_index, 0, target)].float()
            w[target==self.ignore_index] = 0
        else:
            w = (target != self.ignore_index).int()

        ret = w * ret

        if self.reduction == "none":
            return ret
        elif self.reduction == "sum":
            return ret.sum()
        elif self.reduction == "mean":
            return ret.sum() / w.sum().float()


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
    config.dtype=torch.bfloat16
    assert isinstance(config, RobertaConfig)
    model = Roberta(config)

    # make checkpoint dir
    # os.system(f"hdfs dfs -mkdir {os.path.join(args.hdfs_save, 'checkpoints')}")
    os.makedirs(os.path.join(args.save, 'checkpoints'), exist_ok=True)

    if (args.load != None) and (args.start_step == 0):
        bmp.print_rank(f"Loading from checkpoint {args.load}...")
        bmp.load(model, args.load)
    else:
        bmp.print_rank(f"Loading from checkpoint-{args.start_step}.pt...")
        ckpt_path = os.path.join(args.save, "checkpoints", f"checkpoint-{args.start_step}.pt")
        bmp.load(model, ckpt_path)

    for name, param in model.named_parameters():
        if torch.isnan(param).sum() > 0:
            bmp.print_rank(f"NaN values found in parameter {name}. Aborting training.")
            exit(0)
    
    model = model.to(torch.bfloat16)
    return model

def get_optimizer(args, model):
    # change to bf16
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = 1e-5,
                                 betas = (0.9, 0.95),
                                 weight_decay=args.weight_decay)

    # # fp16
    # optimizer = bmp.optim.AdamOffloadOptimizer(model.parameters(), 
    #                                             lr = args.lr,
    #                                             betas = (0.9, 0.95),
    #                                             weight_decay=args.weight_decay)
    
    # if args.save is not None:
    #     bmp.print_rank("Loading the optimizer...")
        
    #     # # if use the momentum, load optimizer
    #     # states = torch.load(
    #     #     os.path.join(args.save, 'checkpoints', "checkpoint.rank-%d.opt" % (bmp.rank())))
        
    #     # # if use the momentum, load the "state" in the optimizer state_dict
    #     # optimizer.load_state_dict(states)
        
    #     # if dont use the momentum, delete the "state" in the optimizer state_dict
    #     # states = torch.load(
    #     #     os.path.join(args.save, 'checkpoints', "checkpoint.rank-%d.opt" % 0))
    #     states = torch.load(
    #         os.path.join(args.save, 'checkpoints', "optimizer.rank-%d.opt" % 0))

    #     del states['state']
    #     optimizer_state = optimizer.state_dict()
    #     # optimizer_state["param_groups"][0]["lr"] = optimizer_state["param_groups"][0]["lr"]*0.5
    #     optimizer_state.update(states)
    #     optimizer.load_state_dict(optimizer_state)

    #     for name, param in optimizer.state_dict().items():
    #         if name == "param_groups":
    #             bmp.print_rank(name, param)
                
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
    elif args.lr_decay_style == 'cosine':
        lr_scheduler = bmp.lr_scheduler.Cosine(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step) 
    return lr_scheduler

def lower_learning_rate(args, model, lr_scheduler, scale_factor):

    current_lr = lr_scheduler.current_lr

    optimizer = bmp.optim.AdamOffloadOptimizer(model.parameters(), 
                                                lr = current_lr*scale_factor,
                                                betas = (0.9, 0.95),
                                                weight_decay=args.weight_decay)

    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == 'linear':
        lr_scheduler = bmp.lr_scheduler.Linear(optimizer, 
                                         start_lr = current_lr*scale_factor,
                                         warmup_iter = 0, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = 0)
    elif args.lr_decay_style == 'cosine':
        lr_scheduler = bmp.lr_scheduler.Cosine(optimizer, 
                                         start_lr = current_lr*scale_factor,
                                         warmup_iter = 0, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = 0) 
    return optimizer, lr_scheduler

def get_optim_manager(args, optimizer, lr_scheduler):
    optim_manager = bmp.optim.OptimManager(loss_scale = args.loss_scale, loss_scale_steps=256)
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    return optim_manager

def setup_model_and_optimizer(args):
    # get the model
    model = get_model(args)
    bmp.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmp.synchronize()
    # get the memory usage
    bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmp.synchronize()
    return model, optimizer, lr_scheduler

def get_train_dataset(args):
    bmp.print_rank(f"load dataset from path {args.input_dataset}")
    # wait_time=10*bmp.rank()
    # time.sleep(wait_time)
    print(bmp.rank(), bmp.world_size())
    
    input_ids_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'input_ids', bmp.rank(), bmp.world_size())
    lm_pos_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'lm_pos', bmp.rank(), bmp.world_size())
    masked_labels_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'masked_labels', bmp.rank(), bmp.world_size())
    length_list_dataset = DistributedMMapIndexedDataset(args.input_dataset, 'length_list', bmp.rank(), bmp.world_size())
    print(bmp.rank(), "finish load train DistributedMMapIndexedDataset")
    bert_dataset = BertDataset(input_ids_dataset, lm_pos_dataset, masked_labels_dataset, length_list_dataset)
    print(bmp.rank(), "finish load train bert_dataset")

    return bert_dataset

def get_valid_dataset(dataset_path):
    input_ids_dataset = MMapIndexedDataset(os.path.join(dataset_path,'valid', 'input_ids'))
    lm_pos_dataset = MMapIndexedDataset(os.path.join(dataset_path, 'valid','lm_pos'))
    masked_labels_dataset = MMapIndexedDataset(os.path.join(dataset_path, 'valid','masked_labels'))
    length_list_dataset = MMapIndexedDataset(os.path.join(dataset_path, 'valid','length_list'))
    print(bmp.rank(), "finish load val DistributedMMapIndexedDataset")
    bert_dataset = BertDataset(input_ids_dataset, lm_pos_dataset, masked_labels_dataset, length_list_dataset)
    print(bmp.rank(), "finish load val bert_dataset")

    return bert_dataset

def valid(args, model, dev_dataloader, step, writer):
    loss_func = FusedCrossEntropy(ignore_index=-100)
    bmp.print_rank("start valid! ")
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for step, data in enumerate(dev_dataloader):
            input_ids, attention_mask, labels = data
            input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
            print(f"valid batch size: {input_ids.size()} | rank: {bmp.rank()}")
            logits = model(input_ids=input_ids, attention_mask=attention_mask, return_logits=True)
            print(f"valid logits size: {logits.size()} logits dtype: {logits.dtype} loss dtype: {labels.dtype} | rank: {bmp.rank()}")
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
            print(f"rank: {bmp.rank()} | step: {step} | loss: {loss}")
            global_loss = bmp.sum_loss(loss).item()
            valid_loss += global_loss

        if bmp.rank() == 0:
            if args.report_to == "tensorboard":
                writer.add_scalar("loss/dev", valid_loss/len(dev_dataloader), step)
            elif args.report_to == "wandb":
                wandb.log({"loss/dev": valid_loss/len(dev_dataloader)}, step=step)

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
    st = (args.start_step + 61000 - 357500) * args.batch_size
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
    bmp.print_rank(f"Now scaling down the model with factor 10.0...")
    new_dict = scale_roberta_model(scale, model, config_file = args.model_config)

    model.load_state_dict(new_dict)
    for name, param in model.named_parameters():
        if torch.isnan(param).sum() > 0:
            if args.report_to == "wandb" and bmp.rank() == 0:
                wandb.alert(
                    title="NaN values found in model.",
                    text=f"find nan values in parameter {name} after scaling model. stop training.",
                    level=wandb.AlertLevel.ERROR
                )
            bmp.print_rank(f"NaN values found in parameter {name}. ")
            exit(0)
    return model

def pretrain(args, model, optimizer, lr_scheduler, train_dataset, dev_dataloader):
    loss_func = bmp.loss.FusedCrossEntropy(ignore_index=-100)

    start_step = args.start_step
    skip_step = 0
    log_loss = 0
    if args.report_to == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        
        bmp.print_rank("start init tensorboard")

        # report training log to or tensorboard
        if bmp.rank() == 0:
            # 获取当前时间并格式化为字符串  
            now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

            # 创建目录  
            tensorboard_dir = os.path.join(args.save, 'tensorboard', str(args.start_step), now)
            os.makedirs(tensorboard_dir)
            bmp.print_rank("init tensorboard_dir: ", tensorboard_dir)

            writer = SummaryWriter(tensorboard_dir)
        else:
            writer = None
    
    # evaluate model before training
    valid(args, model, dev_dataloader, start_step, writer)

    for step, data in enumerate(batch_iter(args, train_dataset)):
        if (start_step + step + 1) % args.gradient_accumulate == 1:
            optimizer.zero_grad() # when not doing
        input_ids = data['input_ids'].cuda()
        attention_mask = data['attention_mask'].cuda()
        labels = data['labels'].cuda()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, return_logits=True)
        loss = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        global_loss = bmp.sum_loss(loss).item()
        log_loss += global_loss
        loss = loss / args.gradient_accumulate
        # loss = optimizer.loss_scale(loss)
        loss.backward()

        # step optimizer
        if (start_step + step + 1) % args.gradient_accumulate == 0:
            grad_norm = bmp.optim.clip_grad_norm(optimizer.param_groups, max_norm= args.clip_grad, scale = optimizer.scale, norm_type = 2)
            bmp.optim_step(optimizer, lr_scheduler)

            # update the training state to the integrations
            if bmp.rank() == 0:
                if args.report_to == "wandb":
                    wandb.log({"loss/train": global_loss, 
                            "grad_norm": grad_norm,
                            # "loss_scale": optim_manager.loss_scale,
                            "learning_rate": lr_scheduler.current_lr}, step=step+start_step+1)
                elif args.report_to == "tensorboard":
                    writer.add_scalar("loss/train", global_loss, step + start_step + 1)
                    writer.add_scalar("grad_norm", grad_norm, step + start_step + 1)
                    # writer.add_scalar("loss_scale", optim_manager.loss_scale, step + start_step + 1)
                    writer.add_scalar("learning_rate", lr_scheduler.current_lr, step + start_step + 1)

        # log the training state to console
        if (start_step + step + 1) % args.log_iters == 0:
            bmp.print_rank(
                    "{} | Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f}".format(
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        step + 1 + start_step,
                        log_loss / args.log_iters,
                        lr_scheduler.current_lr,
                        # optim_manager.loss_scale,
                        grad_norm
                    )
                )
            log_loss = 0

        if (start_step + step + 1) % args.valid_iters == 0:
            valid(args, model, dev_dataloader, start_step + step + 1, writer)

        if args.save != None and (step + start_step + 1) % args.save_iters == 0:

            # save checkpoint
            model_path = os.path.join(args.save, 'checkpoints', "checkpoint-%d.pt" % (step + start_step + 1))
            bmp.save(model, os.path.join(args.save, model_path))
            
            # 但此处先写空文件
            model_status = f'{model_path.split(".pt")[0]}.success'
            bmp.print_rank(f"saving status into: {model_status}")
            with open(model_status, "w") as fout:
                pass

            # save optimizer
            optimizer_path = os.path.join("checkpoints", "checkpoint.rank-%d.opt" % (bmp.rank()))
            torch.save(optimizer.state_dict(), os.path.join(args.save, optimizer_path))

            bmp.print_rank(f"Saving checkpoint at {(step + start_step + 1) } step.")

def init_wandb(args):
    # start a wandb run to track this script
    if bmp.rank() == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="opensoco",
            name="en-real-107-run-7",
            notes="start training from 337500 step. try skip nan grad.",
            
            # track hyperparameters and run metadata
            config={
                "batch_size": args.batch_size,
                "start_step": args.start_step,
                "grad_clipping": args.clip_grad,
                "gradient_accumulate": args.gradient_accumulate,
                "learning_rate": args.lr,
                "adam_betas": 0.98,
                "gpu": 4,
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

    # # get last checkpoint step
    # last_step = get_last_step(args, args.start_step)
    # if last_step > args.start_step:
    #     args.start_step = last_step

    args.start_step = 407000

    # init wandb and tensorboard
    if args.report_to == "wandb":
        import wandb
        init_wandb(args)
    
    bmp.print_rank(args)

    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    train_dataset = get_train_dataset(args)
    valid_dataset = get_valid_dataset(args.test_dataset)
    dev_dataloader = DistributedDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    bmp.print_rank("finish loading dataset.")
    bmp.synchronize()

    pretrain(args, model, optimizer, lr_scheduler, train_dataset, dev_dataloader)

if __name__ == '__main__':
    main()
