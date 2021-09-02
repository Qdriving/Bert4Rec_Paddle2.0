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
"""BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import numpy as np
#import paddle.fluid as fluid
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from bert4rec.transformer_encoder20 import encoder, pre_post_process_layer, pre_post_no_a_layer


class BertConfig(object):
    """ 根据config_path来读取网络的配置 """
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing bert model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class BertModel(nn.Layer):
    def __init__(self,
                 config,
                 weight_sharing=True,
                 use_fp16=False):
        super(BertModel, self).__init__()
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._sent_types = config['type_vocab_size']
        hidden_act = config['hidden_act']
        if hidden_act == "gelu":
            self._hidden_act = nn.GELU()
        else:
            self._hidden_act = nn.ReLU()
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._weight_sharing = weight_sharing

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._dtype = "float16" if use_fp16 else "float32"

        # Initialize all weigths by truncated normal initializer, and all biases 
        # will be initialized by constant zero by default.
        self._param_initializer = nn.initializer.TruncatedNormal(
            std=config['initializer_range'])

        self.word_emb = nn.Embedding(num_embeddings=self._voc_size, 
                                   embedding_dim=self._emb_size,
                                   name=self._word_emb_name, 
                                   weight_attr=paddle.ParamAttr(
                                       initializer=self._param_initializer),
                                   sparse=False)
        #self.position_emb = nn.Embedding(num_embeddings=self._max_position_seq_len,
        #                                 embedding_dim=self._emb_size,
        #                                 weight_attr=paddle.ParamAttr(
        #                                     name=self._pos_emb_name, 
        #                                     initializer=self._param_initializer),
        #                                 sparse=False)
        self.position_emb_out = self.create_parameter(shape=[self._max_position_seq_len, self._emb_size],
                                                      dtype=self._dtype,
                                                      attr=paddle.ParamAttr(
                                                            name="Position_Embedding.weight",
                                                            initializer=self._param_initializer),
                                                      is_bias=True)
                                    
        self.sent_emb = nn.Embedding(num_embeddings=self._sent_types,
                                         embedding_dim=self._emb_size,
                                         weight_attr=paddle.ParamAttr(
                                             name=self._sent_emb_name, 
                                             initializer=self._param_initializer),
                                         sparse=False)

        self.enc_pre_process_layer = pre_post_no_a_layer('nd', self._prepostprocess_dropout, self._emb_size, name='pre_encoder')
        self._enc_out_layer = encoder(
                                n_layer=self._n_layer,
                                n_head=self._n_head,
                                d_key=self._emb_size // self._n_head,
                                d_value=self._emb_size // self._n_head,
                                d_model=self._emb_size,
                                d_inner_hid=self._emb_size * 4,   #self._emb_size * 4,
                                prepostprocess_dropout=self._prepostprocess_dropout,
                                attention_dropout=self._attention_dropout,
                                relu_dropout=0,
                                hidden_act=self._hidden_act,
                                preprocess_cmd="",
                                postprocess_cmd="dan",
                                param_initializer=self._param_initializer,
                                name='encoder')

        self.mask_trans_feat = nn.Linear(in_features=self._emb_size, 
                                         out_features=self._emb_size, 
                                         weight_attr=paddle.ParamAttr(
                                             name="mask_lm_trans_fc.w_0", 
                                             initializer=self._param_initializer),
                                         bias_attr=paddle.ParamAttr(name='mask_lm_trans_fc.b_0'))
        self.mask_trans_act = self._hidden_act

        self.mask_post_process_layer = pre_post_no_a_layer('n', None, self._emb_size, name='mask_lm_trans')

        if self._weight_sharing:
            self.mask_lm_out_bias = self.create_parameter(
                                                            shape=[self._voc_size],
                                                            dtype=self._dtype,
                                                            attr=paddle.ParamAttr(
                                                                name="mask_lm_out_fc.b_0",
                                                                initializer=paddle.nn.initializer.Constant(value=0.0)),
                                                            is_bias=True)
        else:
            self.fc_out = nn.Linear(
                                in_features=self._emb_size, 
                                out_features=self._voc_size, 
                                weight_attr=paddle.ParamAttr(
                                    name="mask_lm_out_fc.w_0", 
                                    initializer=self._param_initializer),
                                bias_attr=paddle.ParamAttr(
                                    name="mask_lm_out_fc.b_0",
                                    initializer=paddle.nn.initializer.Constant(value=0.0))
                                )

    def forward(self, src_ids, position_ids, sent_ids, input_mask, mask_pos):
        # padding id in vocabulary must be set to 0
        # 模型中的三种embedding
        emb_out = self.word_emb(src_ids)
        position_embs_out = paddle.expand(self.position_emb_out, shape=[position_ids.shape[0], self._max_position_seq_len, self._emb_size])  #self.position_emb(position_ids)
        #print(emb_out.shape, position_emb_out.shape)
        emb_out = emb_out + position_embs_out
        sent_emb_out = self.sent_emb(sent_ids)
        emb_out = emb_out + sent_emb_out

        # 接下来是transformer的encoder部分(随机的mask，构造任务，过encoder等)
        emb_out = self.enc_pre_process_layer(emb_out)

        if self._dtype == "float16":
            input_mask = paddle.cast(x=input_mask, dtype=self._dtype)
        else:
            input_mask = paddle.cast(x=input_mask, dtype='float32')
        ######### follow #########
        #to_mask = paddle.reshape(x=input_mask, shape=[input_mask.shape[0], 1, input_mask.shape[1]])
        #broadcast_ones = paddle.ones(shape=[input_mask.shape[0], input_mask[1], 1], dtype='float32')
        #self_attn_mask = broadcast_ones * to_mask
        ######################
        self_attn_mask = paddle.matmul(
            x=input_mask, y=input_mask, transpose_y=True)

        self_attn_mask = paddle.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = paddle.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        self._enc_out = self._enc_out_layer(enc_input=emb_out, attn_bias=n_head_self_attn_mask)

        mask_pos = paddle.cast(x=mask_pos, dtype='int32')

        # extract the first token feature in each sentence
        #next_sent_feat = self.get_pooled_output()
        reshaped_emb_out = paddle.reshape(
            x=self._enc_out, shape=[-1, self._emb_size])
        # extract masked tokens' feature
        mask_feat = paddle.gather(x=reshaped_emb_out, index=mask_pos, axis=0)

        # transform: fc
        mask_trans_feat_out = self.mask_trans_feat(mask_feat)
        mask_trans_feat_out = self.mask_trans_act(mask_trans_feat_out)
        # transform: layer norm 
        mask_trans_feat_out = self.mask_post_process_layer(out=mask_trans_feat_out)

        for name, param in self.named_parameters():
            if name == "word_emb.weight":
                y_tensor = param
                break

        if self._weight_sharing:
            fc_out = paddle.matmul(
                x=mask_trans_feat_out,
                y=y_tensor,
                transpose_y=True)

            fc_out += self.mask_lm_out_bias

        else:
            fc_out = self.fc_out(mask_trans_feat_out)

        return fc_out

    def get_sequence_output(self):
        return self._enc_out
