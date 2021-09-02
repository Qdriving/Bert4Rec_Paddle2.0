import os, math

import paddle
import paddle.nn as nn 
import paddle.static as static


class multi_head_attention(nn.Layer):
    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 dropout_rate=0.,
                 cache=None,
                 param_initializer=None,
                 name='multi_head_att'):
        super(multi_head_attention, self).__init__()
        self.q_linear = nn.Linear(in_features=d_model, 
                                  out_features=d_key * n_head, 
                                  weight_attr=paddle.ParamAttr(
                                      name=name + '_query_fc.w_0',
                                      initializer=param_initializer),
                                  bias_attr=name + '_query_fc.b_0')
        self.k_linear = nn.Linear(in_features=d_model, 
                                  out_features=d_key * n_head,
                                  weight_attr=paddle.ParamAttr(
                                      name=name + '_key_fc.w_0',
                                      initializer=param_initializer),
                                  bias_attr=name + '_key_fc.b_0')
        self.v_linear = nn.Linear(in_features=d_model, 
                                  out_features=d_value * n_head,
                                  weight_attr=paddle.ParamAttr(
                                      name=name + '_value_fc.w_0',
                                      initializer=param_initializer),
                                  bias_attr=name + '_value_fc.b_0')

        self.out_linear = nn.Linear(in_features=d_key * n_head, 
                                  out_features=d_model, 
                                  weight_attr=paddle.ParamAttr(
                                                              name=name + '_output_fc.w_0',
                                                              initializer=param_initializer),
                                  bias_attr=name + '_output_fc.b_0')
        self.n_head = n_head
        
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.cache = cache
        self.dropout_rate = dropout_rate

    def forward(self, queries, keys, values, attn_bias):

        def __split_heads(x, n_head):
            """
            Reshape the last dimension of inpunt tensor x so that it becomes two
            dimensions and then transpose. Specifically, input a tensor with shape
            [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
            with shape [bs, n_head, max_sequence_length, hidden_dim].
            """
            hidden_size = x.shape[-1]
            # The value 0 in shape attr means copying the corresponding dimension
            # size of the input as the output dimension size.
            reshaped = paddle.reshape(
                x=x, shape=[0, 0, n_head, hidden_size // n_head])

            # permuate the dimensions into:
            # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
            return paddle.transpose(x=reshaped, perm=[0, 2, 1, 3])

        def __combine_heads(x):
            """
            Transpose and then reshape the last two dimensions of inpunt tensor x
            so that it becomes one dimension, which is reverse to __split_heads.
            """
            if len(x.shape) == 3: return x
            if len(x.shape) != 4:
                raise ValueError("Input(x) should be a 4-D Tensor.")

            trans_x = paddle.transpose(x, perm=[0, 2, 1, 3])
            # The value 0 in shape attr means copying the corresponding dimension
            # size of the input as the output dimension size.
            return paddle.reshape(
                x=trans_x,
                shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]])

        def scaled_dot_product_attention(is_training, q, k, v, attn_bias, d_key, dropout_rate):
            """
            Scaled Dot-Product Attention
            """
            #scaled_q = paddle.scale(x=q, scale=d_key**-0.5)
            #product = paddle.matmul(x=scaled_q, y=k, transpose_y=True)

            attention_scores = paddle.matmul(x=q, y=k, transpose_y=True)
            product = paddle.multiply(attention_scores,
                                   paddle.to_tensor(1.0 / math.sqrt(float(d_key)), dtype='float32')
                                   )

            if attn_bias is not None:
                product += attn_bias
            weights = nn.functional.softmax(product)
            if dropout_rate:
                weights = nn.functional.dropout(
                    weights,
                    p=dropout_rate,
                    mode="upscale_in_train",
                    training=is_training)
            out = paddle.matmul(weights, v)
            return out

        keys = queries if keys is None else keys
        values = keys if values is None else values
        
        if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
            raise ValueError("Inputs: quries, keys and values should all be 3-D tensors.")

        q = self.q_linear(queries)
        k = self.k_linear(keys)
        v = self.v_linear(values)

        if self.cache is not None:  # use cache and concat time steps
            # Since the inplace reshape in __split_heads changes the shape of k and
            # v, which is the cache input for next time step, reshape the cache
            # input from the previous time step first.
            k = self.cache["k"] = paddle.concat(
                [paddle.reshape(
                    self.cache["k"], shape=[0, 0, self.d_model]), k], axis=1)
            v = self.cache["v"] = paddle.concat(
                [paddle.reshape(
                    self.cache["v"], shape=[0, 0, self.d_model]), v], axis=1)

        q = __split_heads(q, self.n_head)
        k = __split_heads(k, self.n_head)
        v = __split_heads(v, self.n_head)

        ctx_multiheads = scaled_dot_product_attention(self.training, q, k, v, attn_bias, self.d_key,
                                                      self.dropout_rate)

        out = __combine_heads(ctx_multiheads)

        # Project back to the model size.
        proj_out = self.out_linear(out)

        return proj_out


class pre_post_process_layer(nn.Layer):
    def __init__(self,
                 process_cmd,
                 dropout_rate = 0.,
                 norm_shape=768,
                 name=''):
        super(pre_post_process_layer, self).__init__()
        self.process_cmd = process_cmd
        self.name = name
        for cmd in process_cmd:
            if cmd == "d":
                self.dropout_rate = dropout_rate
                self.dropout = nn.Dropout(p=dropout_rate, mode="upscale_in_train")
            elif cmd == "n":
                self.LayerNormal = nn.LayerNorm(norm_shape, 
                                            epsilon=1e-05,
                                            weight_attr=paddle.ParamAttr(
                                                                        name=self.name + '_layer_norm_scale',
                                                                        initializer=nn.initializer.Constant(1.)), 
                                            bias_attr=paddle.ParamAttr(
                                                                        name=self.name + '_layer_norm_bias',
                                                                        initializer=nn.initializer.Constant(0.))
                                            )

    def forward(self, out, prev_out=None):
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                if prev_out is not None:
                    out = out + prev_out
            elif cmd == "n":  # add layer normalization
                out_dtype = out.dtype
                if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
                    out = paddle.cast(x=out, dtype="float32")
                out = self.LayerNormal(out) 
                if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
                    out = paddle.cast(x=out, dtype="float16")
            elif cmd == "d":  # add dropout
                if self.dropout_rate:
                    out = self.dropout(out)
        return out

class pre_post_no_a_layer(nn.Layer):
    def __init__(self,
                 process_cmd,
                 dropout_rate = 0.,
                 norm_shape=768,
                 name=''):
        super(pre_post_no_a_layer, self).__init__()
        self.process_cmd = process_cmd
        self.name = name
        for cmd in process_cmd:
            if cmd == "d":
                self.dropout_rate = dropout_rate
                self.dropout = nn.Dropout(p=dropout_rate, mode="upscale_in_train")
            elif cmd == "n":
                self.LayerNormal = nn.LayerNorm(norm_shape, 
                                            epsilon=1e-05,
                                            weight_attr=paddle.ParamAttr(
                                                                        name=self.name + '_layer_norm_scale',
                                                                        initializer=nn.initializer.Constant(1.)), 
                                            bias_attr=paddle.ParamAttr(
                                                                        name=self.name + '_layer_norm_bias',
                                                                        initializer=nn.initializer.Constant(0.))
                                            )

    def forward(self, out):
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                raise ValueError("[CMD]: Error cmd or layer.") 
            elif cmd == "n":  # add layer normalization
                out_dtype = out.dtype
                if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
                    out = paddle.cast(x=out, dtype="float32")
                out = self.LayerNormal(out) 
                if out_dtype == paddle.fluid.core.VarDesc.VarType.FP16:
                    out = paddle.cast(x=out, dtype="float16")
            elif cmd == "d":  # add dropout
                if self.dropout_rate:
                    out = self.dropout(out)
        return out

        
class positionwise_feed_layer(nn.Layer):
    def __init__(self,
                 d_inner_hid,
                 d_hid,
                 dropout_rate,
                 hidden_act,
                 param_initializer=None,
                 name='ffn'):
         super(positionwise_feed_layer, self).__init__()

         self.dropout_rate = dropout_rate
         self.dropout = nn.Dropout(p=dropout_rate, mode="upscale_in_train")
         
         self.hidden_linear = nn.Linear(in_features=d_hid,
                                        out_features=d_inner_hid,
                                        weight_attr=paddle.ParamAttr(
                                                        name=name + '_fc_0.w_0',
                                                        initializer=param_initializer),
                                        bias_attr=name + '_fc_0.b_0')
         self.hidden_act = hidden_act
         self.out_linear = nn.Linear(in_features=d_inner_hid,
                                     out_features=d_hid,
                                     weight_attr=paddle.ParamAttr(
                                                        name=name + '_fc_1.w_0',
                                                        initializer=param_initializer),
                                     bias_attr=name + '_fc_1.b_0')

    def forward(self, x):
        hidden = self.hidden_linear(x)
        hidden = self.hidden_act(hidden)
        #if self.dropout_rate:
        #    hidden = dropout(hidden)
        out = self.out_linear(hidden)
        return out


class encoder_layer(nn.Layer):
    def __init__( self,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name=''):
       
         super(encoder_layer, self).__init__()
         self.pre_process_layer = pre_post_no_a_layer(preprocess_cmd, attention_dropout, norm_shape=d_model, name=name+ '_pre_att')
         self.multi_head_attn = multi_head_attention(
                                                    d_key,
                                                    d_value,
                                                    d_model,
                                                    n_head,
                                                    attention_dropout,
                                                    param_initializer=param_initializer,
                                                    name=name + '_multi_head_att')
         
         self.post_process_layer = pre_post_process_layer(postprocess_cmd, attention_dropout, norm_shape=d_model, name=name+'_post_att')
         self.pre_ffn_layer = pre_post_no_a_layer(preprocess_cmd, attention_dropout, norm_shape=d_model, name=name+ '_pre_ffn')
         self.positionwise_feed_layer = positionwise_feed_layer(d_inner_hid,
                                                              d_model,
                                                              relu_dropout,
                                                              hidden_act,
                                                              param_initializer,
                                                              name=name+'_ffn')
         self.post_ffn_layer = pre_post_process_layer(postprocess_cmd, attention_dropout, norm_shape=d_model, name=name+'_post_ffn')

    def forward(self, enc_input, attn_bias):
        #multi_input = self.pre_process_layer(enc_input)
        multi_output = self.multi_head_attn(queries=enc_input, keys=None, values=None, attn_bias=attn_bias)
        attn_output = self.post_process_layer(prev_out=enc_input, out=multi_output)
        #attn_output = self.pre_ffn_layer(attn_output)
        ffd_output = self.positionwise_feed_layer(attn_output)
        out = self.post_ffn_layer(prev_out=attn_output, out=ffd_output)

        return out


class encoder(nn.Layer):
    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 hidden_act,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 param_initializer=None,
                 name=''):
        super(encoder, self).__init__()
        self.encoder_layer = nn.LayerList([encoder_layer(
                                             n_head,
                                             d_key,
                                             d_value,
                                             d_model,
                                             d_inner_hid,
                                             prepostprocess_dropout,
                                             attention_dropout,
                                             relu_dropout,
                                             hidden_act,
                                             preprocess_cmd,
                                             postprocess_cmd,
                                             param_initializer,
                                             name + '_layer_' + str(i))
                                             for i in range(n_layer)])
        self.pre_process_layer = pre_post_no_a_layer(preprocess_cmd, prepostprocess_dropout, norm_shape=d_model, name="post_encoder")


    def forward(self, enc_input, attn_bias):
        for enc in self.encoder_layer:
            enc_output = enc(enc_input, attn_bias)
            enc_input = enc_output

        #enc_output =  self.pre_process_layer(out=enc_output)
        return enc_output
