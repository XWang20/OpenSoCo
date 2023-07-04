import os
import json
import datetime
import time
import logging
import argparse
import numpy as np

import torch
from transformers import AutoTokenizer, BertTokenizer, DebertaV2ForSequenceClassification,MegatronBertForSequenceClassification
from model import BertModel, RobertaModel
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from jaccard_similarity_score import jaccard_similarity_score

import bmtrain as bmt

from model_center.model import BertConfig,RobertaConfig
from model_center.dataset import DistributedDataLoader

from prepare_dataset import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--metric_for_best_model", type=str)
    parser.add_argument("--data_process_method", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--eval_strategy", type=str, default="step")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--train_sample_num", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--do_train", type=int, default=1)
    parser.add_argument("--output_dir", type=str, help="output directory.")
    parser.add_argument("--delete_checkpoint", type=bool)
    parser.add_argument("--adapter", type=bool)
    parser.add_argument("--problem_type", type=str, default="single_label_classification")

    return parser.parse_args()

args=get_args()

# 修正模型名称
if args.model_name == "chinese-roberta-wwm-ext-base":
    args.model_name = "chinese-roberta-wwm-ext"

if os.path.exists(os.path.join(args.output_dir, 'test_results.json')):
    exit()

bmt.init_distributed(seed=args.seed)

if args.model_name == "bertweet-base":
    PADDING_LEN = 128
else:
    PADDING_LEN = 512

log_iter = 10
epochs = args.epoch
if args.adapter:
    batch_size = 64
else:
    batch_size = 32
warm_up_ratio = 0.01

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

BEST_MODEL_PATH = os.path.join(output_dir, 'model.pt')
TEST_RESULT_PATH = os.path.join(output_dir, 'test_results.json')
VALIDATION_RESULT_PATH = os.path.join(output_dir, 'validation_result.json')

logger = logging.getLogger(f'{args.dataset_name}_{args.model_name}_{args.seed}_{args.learning_rate}')
logger.setLevel(level=logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info(str(args))
logger.info("Prepare dataset...")
logger.info(f"local rank:{bmt.rank()}, world size:{bmt.world_size()}")

dataset_process = {
    "txt": get_dataset,
    "multi_label": get_multi_label_dataset,
    "single_label": get_single_label_dataset,
    "stance": get_stance_dataset,
    "emergent": get_emergent_dataset,
    "two_sentence": get_two_sentence_dataset,
    "get_NEP_label_dataset": get_NEP_label_dataset
}

do_preprocess = True
dataset_dir = args.dataset_path

if not os.path.exists(os.path.join(args.dataset_path, 'train.csv')) or args.train_sample_num == 0:
    args.do_train = False
    train_texts, train_labels = dataset_process[args.data_process_method](os.path.join(args.dataset_path, 'test.csv'), do_preprocess = do_preprocess)
    val_texts, val_labels = dataset_process[args.data_process_method](os.path.join(args.dataset_path, 'validation.csv'), do_preprocess = do_preprocess)
    test_texts, test_labels = dataset_process[args.data_process_method](os.path.join(args.dataset_path, 'test.csv'), do_preprocess = do_preprocess)
else:
    train_texts, train_labels = dataset_process[args.data_process_method](os.path.join(args.dataset_path, 'train.csv'), do_preprocess = do_preprocess, train_sample_num=args.train_sample_num)
    val_texts, val_labels = dataset_process[args.data_process_method](os.path.join(args.dataset_path, 'validation.csv'), do_preprocess = do_preprocess)
    test_texts, test_labels = dataset_process[args.data_process_method](os.path.join(args.dataset_path, 'test.csv'), do_preprocess = do_preprocess)

if args.train_sample_num !=1 and len(train_texts) < args.train_sample_num:
    os.remove(output_dir)
    exit()

logger.info(f"train, val and test size is {len(train_labels)}, {len(val_labels)}, {len(test_labels)}")

if isinstance(test_labels[0], list):
    args.problem_type = "multi_label_classification"
    label_num = len(test_labels[0])
    logger.info(f"train, val and test label is {len(train_labels[0])}, {len(val_labels[0])}, {len(test_labels[0])}")
else:
    args.problem_type = "single_label_classification"
    label_num = len(set(test_labels))
    assert set(train_labels) == set(val_labels) == set(test_labels)
    logger.info(f"train, val and test label is {set(train_labels)}, {set(val_labels)}, {set(test_labels)}")

logger.info(f"{train_texts[0]}, {train_labels[0]}")

def get_model(args, model_path, label_num):
    if args.model_name == "deberta-xxlarge":
        model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v2-xxlarge", torch_dtype=torch.float16, num_labels=label_num, problem_type=args.problem_type)
    elif args.model_name == "erlangshen":
        model = MegatronBertForSequenceClassification.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B", num_labels=label_num, problem_type=args.problem_type)
    elif args.model_type.lower() == "bert":
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        model = BertModel(config, model_path=model_path, label_num=label_num)
    else:
        config = RobertaConfig.from_json_file("./config/deberta_prenorm.json")
        model_path = os.getenv("CHECKPOINT")
        platform_config_path = os.getenv("PLATFORM_CONFIG_PATH")
        model_path = os.path.join(json.load(open(platform_config_path, "r", encoding="utf-8"))["model_map"]["wx_lm"], args.model_name)
        bmt.print_rank("loading from model_path: {}".format(model_path))
        model = RobertaModel(config, model_path=model_path, label_num=label_num)

    model = bmt.BMTrainModelWrapper(model)
    
    if args.adapter:
        from opendelta import AdapterModel
        
        if args.model_name == "deberta-xxlarge":
            delta_model = AdapterModel(backbone_model=model, modified_modules=['output'], backend='bmt')
            delta_model.freeze_module(exclude=["deltas", "classifier", "pooler"], set_state_dict=True)
        elif args.model_name == "erlangshen":
            delta_model = AdapterModel(backbone_model=model, modified_modules=['ln'], backend='bmt')
            delta_model.freeze_module(exclude=["deltas", "classifier"], set_state_dict=True)
        delta_model.log()

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path), strict=False)
    return model

def get_tokenizer(args, model_path):

    if args.model_name == "bertweet-base":
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
    elif "prenorm" in args.model_name or "checkpoint" in args.model_name:
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        tokenizer_obj = Tokenizer.from_file("./config/tokenizer.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
        tokenizer.pad_token = '<pad>'
        tokenizer.eos_token = '</s>'
        tokenizer.unk_token = '<unk>'
        tokenizer.mask_token = '<mask>'
        tokenizer.sep_token = '</s>'
        tokenizer.cls_token = '<s>'
        tokenizer.bos_token = '<s>'
    elif args.model_name == "deberta-xxlarge":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")
    elif args.model_name == "erlangshen":
        tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return tokenizer

tokenizer = get_tokenizer(args, args.model_path)
logger.info(tokenizer)

tokens_train = tokenizer.batch_encode_plus(
    train_texts,
    max_length = PADDING_LEN,
    padding='max_length',
    truncation=True,
    add_special_tokens = True

)

tokens_val = tokenizer.batch_encode_plus(
    val_texts,
    max_length = PADDING_LEN,
    padding='max_length',
    truncation=True,
    add_special_tokens = True
)

tokens_test = tokenizer.batch_encode_plus(
    test_texts,
    max_length = PADDING_LEN,
    padding='max_length',
    truncation=True,
    add_special_tokens = True
)

train_data = TensorDataset(torch.tensor(tokens_train['input_ids']), \
    torch.tensor(tokens_train['attention_mask']), \
    torch.tensor(train_labels))
val_data = TensorDataset(torch.tensor(tokens_val['input_ids']), \
    torch.tensor(tokens_val['attention_mask']), \
    torch.tensor(val_labels))
test_data = TensorDataset(torch.tensor(tokens_test['input_ids']), \
    torch.tensor(tokens_test['attention_mask']), \
    torch.tensor(test_labels))

if bmt.rank == 0:
    logger.info(train_data[0])

train_dataloader = DistributedDataLoader(train_data, batch_size = batch_size, shuffle = True)
val_dataloader = DistributedDataLoader(val_data, batch_size = batch_size, shuffle = False)
test_dataloader = DistributedDataLoader(test_data, batch_size = batch_size, shuffle = False)

# optimizer and lr-scheduler
total_step = (len(train_dataloader)) * epochs
step_per_epoch = len(train_dataloader)
early_stopping = 0
best_valid_result = 0
st_epoch = 0
st_time = time.time()

model = get_model(args, args.model_path, label_num)

if args.do_train:

    optimizer = bmt.optim.AdamOptimizer(model.parameters(),lr = args.learning_rate, weight_decay=1e-2, betas = (0.9, 0.999))

    lr_scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                            start_lr = args.learning_rate,
                                            warmup_iter = warm_up_ratio*total_step,
                                            end_iter = total_step)
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

def compute_metrics(y_labels, y_preds):
    if args.problem_type == "multi_label_classification":
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(y_preds))
        # use threshold to turn probs into integer predictions
        preds = np.zeros(probs.shape)
        preds[np.where(probs >= 0.5)] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(y_labels, preds, average='macro')
        accuracy = jaccard_similarity_score(y_labels, preds)
    else:
        accuracy = accuracy_score(y_labels, y_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_labels, y_preds, average='macro')

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1}

def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        pd = [] # prediction
        gt = [] # ground_truth
        for data in dataloader:
            input_ids, attention_mask, labels = (i.cuda() for i in data)
            if args.model_name == "deberta-xxlarge" or args.model_name == "erlangshen":
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if args.problem_type == "single_label_classification":
                logits = logits.argmax(dim=-1)

            pd.extend(logits.cpu().tolist())
            gt.extend(labels.cpu().tolist())

        # gather results from all distributed processes
        pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
        gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()

        # calculate metric
        result = compute_metrics(gt, pd)
        return result

if args.do_train:
    # train
    optim_manager = bmt.optim.OptimManager(loss_scale=2**20)
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    stop = False

    for epoch in range(st_epoch, epochs):
        if bmt.rank == 0:
            logger.info("Epoch {} begin...".format(epoch + 1))
        model.train()
        for step, data in enumerate(train_dataloader):
            optim_manager.zero_grad()
            input_ids, attention_mask, labels = (i.cuda() for i in data)
            if args.model_name == "deberta-xxlarge" or args.model_name == "erlangshen":
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

            global_loss = bmt.sum_loss(loss).item()
            optim_manager.backward(loss)

            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_norm = 1.0, norm_type = 2)
            optim_manager.step()
            
            if torch.torch.isnan(loss):
                if bmt.rank == 0:
                    logger.info(f"epoch: [{epoch+1}/{epochs}] | step: [{step+epoch*step_per_epoch+1}/{total_step}] | loss: {global_loss:.2f} | lr: {lr_scheduler.current_lr:.2f} | grad_norm: {grad_norm:.2f}")
                    logger.info(f"loss is nan, stop...")
                stop = True

            if step % log_iter == 0 and bmt.rank == 0:
                elapsed_time = time.time() - st_time
                etc_time = (elapsed_time)/(step+epoch*step_per_epoch+1)*total_step - elapsed_time
                logger.info(f"epoch: [{epoch+1}/{epochs}] | step: [{step+epoch*step_per_epoch+1}/{total_step}] | etc: [{str(datetime.timedelta(seconds=etc_time))}] | loss: {global_loss:.2f} | lr: {lr_scheduler.current_lr:.2f} | grad_norm: {grad_norm:.2f}")

            # evaluate
            if (step+1 == step_per_epoch) or (args.eval_strategy == "step" and (step+1) % 150 == 0):
                result = evaluate(model, val_dataloader)
                early_stopping += 1

                if result[args.metric_for_best_model] > best_valid_result:
                    best_valid_result = result[args.metric_for_best_model]
                    if bmt.rank() == 0:
                        logger.info("saving the new best model...\n") # save checkpoint
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    early_stopping = 0

                # save result and state
                if bmt.rank() == 0:
                    logger.info(f"validation result: {result}\n")
                validation_state = {"epoch": epoch, "step": step, "validation_result": result[args.metric_for_best_model], "best_validation_result": best_valid_result, "early_stopping": early_stopping}
                
                f = open(VALIDATION_RESULT_PATH, "a")
                f.write(json.dumps(validation_state)+"\n")
                f.close()

            # early stop
            if (early_stopping > 20) or (step+epoch*step_per_epoch+1) > 100000:
                if bmt.rank() == 0:
                    logger.info(f"{args.metric_for_best_model} have not rising for 5 evaluation or steps > 100,000. Stopping..")
                stop = True

            if stop:
                break
        if stop:
            break

    # load delta weights
    model.load_state_dict(torch.load(BEST_MODEL_PATH), strict=False)

logger.info("Checking performance...\n")
result = evaluate(model, test_dataloader)

if bmt.rank() == 0:
    logger.info(f"test result: {result}")
    f = open(TEST_RESULT_PATH, "a")
    f.write(json.dumps({"validation result": best_valid_result, "test result": result}))
    f.close()

    if args.delete_checkpoint:
        for path in [BEST_MODEL_PATH]:
            os.remove(path)
