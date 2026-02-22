def forward(self, inputs: torch.Tensor): # 进ResidualAttentionBlock的参数inputs：clip模型输入的是Tensor；mmrlpp输入的是列表
        # 定义层类型前缀，方便日志区分 当text_layer传入的参数为false（默认），当前的ResidualAttentionBlock是视觉编码器层
        layer_type = "Text" if self.text_layer else "Vision"

        # 构建clip时候，ResidualAttentionBlock处理的是张量
        if self.model == "CLIP":
            x = inputs
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
    

        elif self.model == "MMRLpp":
            x = inputs[0] # token embedding序列 （列表的第一个元素）因为已经进入了transformer所以 tkoen embedding的形状已经是（L,B,D）
            compound_rep_tokens = inputs[1] # 要插入的n_rep_tokens个表示token列表（列表的第二个元素）
            counter = inputs[2] # 当前正在处理第几层（在mmrlpp.py的文本编码器里初始化为0）
            # 每次只要插入表示token，counter+1

            beta = self.beta

            if len(compound_rep_tokens) > 0:  # 检查compound_rep_tokens这个列表是否非空（至少有一层需要插入表示token）

                # 视觉编码器分支
                if not self.text_layer:
                    # 判断当前ResidualAttentionBlock层是否需要执行插入表示token的操作
                    if self.layer in self.rep_tokens_layers: # layer初始化为0+1=1，因此当ResidualAttentionBlock层数为1-5层时不执行以下代码
                    # 因此1-5层的counter一直为0

                        # 当layer=5+1=6时，执行以下代码，此时，counter=0

                        # 视觉编码器的PRC实现
                        # 准备好要插入的表示token
                        visual_context = compound_rep_tokens[counter] # 取出当前层的表示token。compound_rep_tokens：(5, 512)

                        # 将表示token按照batch维度扩展，并且通过permute重组维度：(Batch, 5, 512) -> (5, Batch, 512)
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2)# 此时x的第二维代表batch


                        # PRC逻辑 使用前一层的部分比例的 rep_tokens

                        # 第一次插入表示token
                        if self.layer == self.rep_tokens_layers[0]: # 如果当前层是表示令牌层列表的第一个（即第6层）这里是再做一次层数的判断
                            # x的结构是: [ClassToken, Patch1, Patch2, ...]

                            prefix = x[:1, :, :] # 取出 Class Token (第0个)
                            suffix = x[1:, :, :] # 取出后面的所有 Patch Tokens
                            # 第一次插入表示令牌，直接使用当前表示令牌
                            x = torch.cat([prefix, visual_context, suffix], dim=0) # 在token长度维度（第一个维度）
                            # x的结构: [ClassToken, rep_tokens, Patches...]

                        else:  # 后续层插入表示token（第7,8,9,10,11,12层），使用PRC融合
                            # x的结构: [ClassToken, old rep_tokens, Patches...]
                            prefix = x[:1, :, :]

                            # 从x里提取上一层插入的rep_tokens
                            rep_tokens_prelayer = x[1:1 + self.n_rep_tokens, :, :] # 如果插入5个，就是取[1 : 1+5]
                            suffix = x[1 + self.n_rep_tokens:, :, :] # 剩下的 Patches embedding

                            # PRC
                            x = torch.cat([prefix, beta * visual_context + (1 - beta) * rep_tokens_prelayer, suffix], dim=0)

                        counter += 1

                # 文本编码器分支
                else:
                    # 判断当前ResidualAttentionBlock层是否需要执行插入表示token的操作
                    if self.layer in self.rep_tokens_layers: # 当layer在前5层时候不运行以下代码

                        # 文本编码器的PRC实现
                        # 此时ResidualAttentionBlock生成6次，layer=6时，执行以下代码
                        # 准备好要插入的表示token
                        textual_context = compound_rep_tokens[counter] # 获取compound_rep_tokens这个列表里的第一份rep_tokens数据
                        # 将rep_tokens按照当前输入的批次进行扩展并且进行维度变换
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2)

                        #插入表示token
                        if self.layer == self.rep_tokens_layers[0]:
                            prefix = x[:1, :, :]
                            suffix = x[1:, :, :]
                            x = torch.cat([prefix, textual_context, suffix], dim=0)
                        else:
                            prefix = x[:1, :, :]
                            rep_tokens_prelayer = x[1:1 + self.n_rep_tokens, :, :]
                            #rep_tokens_prelayer = 0
                            suffix = x[1 + self.n_rep_tokens:, :, :]       
                            x = torch.cat([prefix, beta * textual_context + (1 - beta) * rep_tokens_prelayer, suffix], dim=0)

                        counter += 1

                    # 处理 Attention Mask (因为序列变长了，Mask也要变大) 原始的attn_mask 是一个 77 x 77 的矩阵，现在插入5个token后矩阵要变为82 x 82
                    if self.layer >= self.rep_tokens_layers[0]:
                        width = x.shape[0] # 当前token序列的长度（x的第一个维度）
                        self.attn_mask = torch.empty(width, width)
                        self.attn_mask.fill_(float("-inf"))
                        self.attn_mask.triu_(1)  # zero out the lower diagonal

            #-----Adapter 插入位置-----
            if self.use_adapter:
                # x 的形状是 [Seq_Len, Batch, Dim]
                # Adapter 内部的 Linear 会自动作用于最后一维 Dim，所以形状没问题
                x = self.input_adapter(x)
            # -----Adapter 插入位置-----


            x = x + self.attention(self.ln_1(x)) # 将x先归一化，再送入上面的attention函数里实现注意力机制
            x = x + self.mlp(self.ln_2(x))# x归一化，再送入上面的mlp层里

            return [x, compound_rep_tokens, counter]