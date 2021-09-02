#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT pretraining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing
import re

import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.distributed as dist

from bert4rec.dataset import DataReader
from bert4rec.bert4rec import BertModel, BertConfig

from utils.args import ArgumentGroup, print_arguments, check_cuda

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("bert_config_path",      str,  "./config/bert_config_ml-1m_256.json",  "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint",       str,  None,                         "Init checkpoint to resume training from.")
model_g.add_arg("last_steps",            int,  0,                            "Last steps for optimizer scheduler.")
model_g.add_arg("checkpoints",           str,  "checkpoints",                "Path to save checkpoints.")
model_g.add_arg("weight_sharing",        bool, True,                         "If set, share weights between word embedding and masked lm.")
model_g.add_arg("generate_neg_sample",   bool, True,                         "If set, randomly generate negtive samples by positive samples.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch",             int,    100,     "Number of epoches for training.")
train_g.add_arg("learning_rate",     float,  0.0001,  "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",      float,  0.01,    "Weight decay rate for L2 regularizer.")
train_g.add_arg("num_train_steps",   int,    800000, "Total steps to perform pretraining.")
train_g.add_arg("warmup_steps",      int,    1000,    "Total steps to perform warmup when pretraining.")
train_g.add_arg("save_steps",        int,    10000,   "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps",  int,    1000,    "The steps interval to evaluate model performance.")
train_g.add_arg("use_fp16",          bool,   False,   "Whether to use fp16 mixed precision training.")
train_g.add_arg("loss_scaling",      float,  1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

log_g = ArgumentGroup(parser,     "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")
log_g.add_arg("verbose",             bool,   False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_name",            str,  "ml-20m",       "Path to training data.")
data_g.add_arg("data_dir",            str,  "./data/train/",       "Path to training data.")
data_g.add_arg("validation_set_dir",  str,  "./data/validation/",  "Path to validation data.")
data_g.add_arg("test_set_dir",        str,  None,                  "Path to test data.")
data_g.add_arg("vocab_path",          str,  "./config/vocab.txt",  "Vocabulary path.")
data_g.add_arg("max_seq_len",         int,  200,                   "Tokens' number of the longest seqence allowed.")
data_g.add_arg("batch_size",          int,  8192,
               "The total number of examples in one batch for training, see also --in_tokens.")
data_g.add_arg("in_tokens",           bool, False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("is_distributed",               bool,   False,  "If set, then start distributed training.")
run_type_g.add_arg("use_cuda",                     bool,   True,   "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor",            bool,   False,  "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,      "Ihe iteration intervals to clean up temporary variables.")
run_type_g.add_arg("do_test",                      bool,   False,  "Whether to perform evaluation on test data set.")

args = parser.parse_args()
# yapf: enable.


def train(args):
    print("pretraining start")

    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    BertRec = BertModel(config=bert_config,
                        weight_sharing=args.weight_sharing,
                        use_fp16=args.use_fp16)
    #for name, param in BertRec.named_parameters():
    #    print(name)
    

    data_path = args.data_dir  #args.test_set_dir if args.do_test else args.validation_set_dir
    train_dataset = DataReader( 
                                data_path=data_path,
                                batch_size=args.batch_size,
                                max_len=args.max_seq_len,
                                num=2237830  #339430  #
                                )
   
    train_loader = train_dataset.get_samples()   #paddle.io.DataLoader(dataset=train_dataset, batch_size=None)

    val_dataset = DataReader( 
                                data_path=args.validation_set_dir,
                                batch_size=args.batch_size,
                                max_len=args.max_seq_len,
                                num=138493  #6040   #
                                )
    #train_dataset = MyDataset(16*16)
    val_loader = val_dataset.get_samples()    #paddle.io.DataLoader(dataset=val_dataset, batch_size=None)
    print("-----------data reader finished----------------")
    #time.sleep(10)
    if args.init_checkpoint:
        layer_state_dict = paddle.load(args.init_checkpoint)
        BertRec.set_state_dict(layer_state_dict, use_structured_name=True)
    
    epochs = args.epoch

    def write_data_txt(src_ids, pos_ids, input_mask, mask_pos, mask_label):
        np.savetxt("src_ids.txt", src_ids.numpy(), fmt ='%d') 
        np.savetxt("pos_ids.txt", pos_ids.numpy(), fmt ='%d')
        np.savetxt("input_mask.txt", input_mask.numpy(), fmt ='%d')
        np.savetxt("mask_pos.txt", mask_pos.numpy(), fmt ='%d')
        np.savetxt("mask_label.txt", mask_label.numpy(), fmt ='%d')
    total_steps = args.last_steps
    #optim = paddle.optimizer.Adam(learning_rate=scheduler2, parameters=mymodel.parameters())
    # 用Adam作为优化函数
    def apply_decay_param(param_name):
        #print(param_name)
        for r in ["layer_norm", "b_0"]:
            if re.search(r,param_name) is not None:
                return False
        return True

    for epoch in range(epochs):
        BertRec.train()
        if total_steps < args.warmup_steps:
            scheduler = paddle.optimizer.lr.LinearWarmup(
                learning_rate=args.learning_rate, 
                warmup_steps=args.warmup_steps, 
                start_lr=0, 
                end_lr=args.learning_rate, 
                last_epoch  = total_steps,
                verbose=False)
        else:
            scheduler = paddle.optimizer.lr.PolynomialDecay(
                learning_rate=args.learning_rate, 
                decay_steps=args.num_train_steps, 
                end_lr=0, 
                last_epoch=total_steps-args.warmup_steps,
                verbose=False)
        optim = paddle.optimizer.AdamW(
                                    learning_rate=scheduler, 
                                    weight_decay=args.weight_decay, 
                                    apply_decay_param_fun=apply_decay_param, 
                                    grad_clip=nn.ClipGradByGlobalNorm(clip_norm=5.0),
                                    parameters=BertRec.parameters()
                                    ) 
        #acc_calc = paddle.metric.Accuracy()
        total_loss = 0
        total_hr = 0
        batch_id = 0
        for batch_id, data in enumerate(train_loader()):   #for data in train_loader():      #for batch_id, data in enumerate(train_loader()):
            #print("-----------1 batch data finished----------------")
            #time.sleep(10)
            src_ids, pos_ids, input_mask,  mask_pos, mask_label = data
            src_ids = paddle.to_tensor(src_ids, dtype='int32')
            pos_ids = paddle.to_tensor(pos_ids, dtype='int32')
            input_mask = paddle.to_tensor(input_mask, dtype='int32')
            mask_pos = paddle.to_tensor(mask_pos, dtype='int32')
            mask_label = paddle.to_tensor(mask_label, dtype='int64')
            #write_data_txt(src_ids, pos_ids, paddle.reshape(input_mask,shape=[-1,1]), mask_pos, mask_label)
            sent_ids = paddle.zeros(shape=[args.batch_size, args.max_seq_len], dtype='int32')

            fc_out = BertRec(src_ids, pos_ids, sent_ids, input_mask, mask_pos)
            mask_lm_loss, lm_softmax = nn.functional.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label, return_softmax=True)
            mean_mask_lm_loss = paddle.mean(mask_lm_loss)
            
            hr_10 = paddle.metric.accuracy(lm_softmax, mask_label, k=10)
            total_hr += hr_10.numpy()
            
            # 总得loss为两部分loss的和
            loss =  mean_mask_lm_loss
            total_loss += loss.numpy()
            #print("-----------1 loop finished----------------")
            #time.sleep(10)

            loss.backward()
            if batch_id % 100 == 0:
                print("total steps: {}, epoch: {}, batch_id: {}, loss is: {}, HR@10 is: {}".format(total_steps, epoch, batch_id, loss.numpy(), hr_10.numpy()))
                #print(mask_lm_loss)
            optim.step()
            optim.clear_grad()
            scheduler.step()
            total_steps += 1
            if total_steps % 4000 == 0:
                layer_state_dict = BertRec.state_dict()
                opt_stat_dict = optim.state_dict()
                paddle.save(layer_state_dict, "/home/aistudio/output/bert_"+args.data_name+".pdparams")
                paddle.save(opt_stat_dict, "/home/aistudio/output/bert_"+args.data_name+".pdopt")
                print("save parmas in ./output")
             
        print("epoch: {}, aver loss is: {}, HR@10 is: {}".format(epoch, total_loss/(1+batch_id), total_hr/(batch_id+1)))    
        
        
        total_hr_eval = 0
        total_loss = 0
        BertRec.eval()
        for batch_id, data in enumerate(val_loader()):
            src_ids, pos_ids, input_mask,  mask_pos, mask_label = data
            src_ids = paddle.to_tensor(src_ids, dtype='int32')
            pos_ids = paddle.to_tensor(pos_ids, dtype='int32')
            input_mask = paddle.to_tensor(input_mask, dtype='int32')
            mask_pos = paddle.to_tensor(mask_pos, dtype='int32')
            mask_label = paddle.to_tensor(mask_label, dtype='int64')
            sent_ids = paddle.zeros(shape=[args.batch_size, args.max_seq_len], dtype='int32')
            fc_out = BertRec(src_ids, pos_ids, sent_ids, input_mask, mask_pos)
                
            mask_lm_loss, lm_softmax = nn.functional.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label, return_softmax=True)
            mean_mask_lm_loss = paddle.mean(mask_lm_loss)
            loss = mean_mask_lm_loss
            total_loss += loss.numpy()
            hr_10 = paddle.metric.accuracy(lm_softmax, mask_label, k=10)
            total_hr_eval += hr_10.numpy()

        print("[Eval] Avr loss is {}, avr HR@10 is: {}".format(total_loss/(1+batch_id), total_hr_eval/(batch_id+1)))
        

def calculate_top_k_accuracy(logits, targets, k=2):
    values, indices = paddle.topk(logits, k=k, sorted=True)
    y = paddle.reshape(targets, [-1, 1])
    correct = (y == indices).astype('float32') * 1.  # 对比预测的K个值中是否包含有正确标签中的结果
    top_k_accuracy = paddle.mean(correct) * k  # 计算最后的准确率
    return top_k_accuracy


if __name__ == '__main__':
    print_arguments(args)
    check_cuda(args.use_cuda)
    if args.do_test:
        test(args)
    else:
        train(args)
