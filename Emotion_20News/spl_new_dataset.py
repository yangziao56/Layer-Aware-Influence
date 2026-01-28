import argparse
import logging
import os
import time
from typing import Tuple

import evaluate
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from transformers import default_data_collator

from pipeline import construct_bert, get_glue_dataset, get_ag_news_dataset, get_emotion_dataset, construct_bert_agnews, construct_bert_emotion, construct_bert_20news, get_20_news_dataset

from argparse import Namespace



VALIDATION_PERCENT = 32 / 2000.0




#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Train text classification models on GLUE datasets.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="emotion",
        help="A name of GLUE dataset.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for the evaluation dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-05,
        help="Fixed learning rate to train the model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay to train the model.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of epochs to train the model.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1004,
        help="A seed for reproducible training pipeline.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path to store the final checkpoint.",
    )

    parser.add_argument(
        "--cuda_device",
        type=str,
        default="cuda:1",
        help="选择使用的CUDA设备，例如 'cuda:0' 或 'cuda:1'。",
    )

    parser.add_argument('--method', type=str, default='Ours', help='vanilla, SPL, IP, Ghost, output_grads_dot, self_output_grads_dot')

    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args



def train(
    dataset: data.Dataset,
    val_dataset: data.Dataset,
    batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    args: Namespace,
) -> nn.Module:
    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 初始化训练数据加载器
    train_dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    # 构建并移动模型到设备
    if args.dataset_name == "agnews":
        model = construct_bert_agnews().to(device)
    elif args.dataset_name == "emotion":
        model = construct_bert_emotion().to(device)
    elif args.dataset_name == "20news":
        model = construct_bert_20news().to(device)
    #model = construct_bert().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start_time = time.time()

    for epoch in range(num_train_epochs):
        total_loss = 0.0
        sample_count = 0  # 跟踪选择的样本数量

        model.train()
        logging.info(f"Starting epoch {epoch + 1}/{num_train_epochs}")

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)

            if args.method in ['Ours'] and epoch >= 1:
                try:
                    # Step 1: 选择验证集的10%
                    all_val_data = val_dataset
                    subset_size = max(1, int(VALIDATION_PERCENT * len(all_val_data)))  # 确保至少有1个样本
                    val_indices = torch.randperm(len(all_val_data))[:subset_size]
                    val_subset = data.Subset(all_val_data, val_indices)
                    val_subset_loader = data.DataLoader(val_subset, batch_size=subset_size, shuffle=False, collate_fn=default_data_collator)

                    # Step 2: 初始化 ghost_total
                    ghost_total = torch.zeros(batch_size).to(device)  # [batch_size]

                    # Step 3: 注册 forward hooks 捕获层的输入和输出
                    layer_outputs = {}

                    def forward_hook(module, input, output):
                        # 如果输出是张量，直接调用 retain_grad
                        if isinstance(output, torch.Tensor):
                            output.retain_grad()
                            layer_outputs[str(module)] = (input, output)
                        elif isinstance(output, (tuple, list)):
                            # 如果输出是元组或列表，对每个元素检查并调用 retain_grad
                            retained_outputs = []
                            for o in output:
                                if isinstance(o, torch.Tensor):
                                    o.retain_grad()
                                    retained_outputs.append(o)
                            layer_outputs[str(module)] = (input, tuple(retained_outputs))
                        else:
                            # 如果是字典或者其他结构
                            layer_outputs[str(module)] = (input, None)



                    hooks = []
                    for name, layer in model.named_modules():
                        hooks.append(layer.register_forward_hook(forward_hook))

                    # Step 4: 拼接训练批次和验证子集批次
                    train_inputs = {
                        key: batch[key].to(device) for key in ["input_ids", "attention_mask", "token_type_ids"]
                    }
                    train_labels = batch["labels"].to(device)

                    # 获取验证子集的批次
                    val_batch = next(iter(val_subset_loader))
                    val_inputs = {
                        key: val_batch[key].to(device) for key in ["input_ids", "attention_mask", "token_type_ids"]
                    }
                    val_labels = val_batch["labels"].to(device)

                    # 拼接输入和标签
                    combined_inputs = {
                        key: torch.cat([train_inputs[key], val_inputs[key]], dim=0)
                        for key in train_inputs.keys()
                    }
                    combined_labels = torch.cat([train_labels, val_labels], dim=0)

                    # Step 5: 前向传播
                    combined_outputs = model(**combined_inputs, labels=combined_labels)
                    if not hasattr(combined_outputs, 'loss'):
                        raise AttributeError("模型输出不包含 'loss' 属性。")
                    combined_loss = combined_outputs.loss  # 标量

                    # 计算每个样本的损失
                    per_sample_loss = F.cross_entropy(combined_outputs.logits, combined_labels, reduction='none')  # [batch_size + subset_size]

                    # Step 6: 计算 loss 对输出 logits 的梯度（一次性计算）
                    logits = combined_outputs.logits  # [batch_size + val_size, num_classes]
                    loss_per_sample = per_sample_loss  # [batch_size + val_size]

                    # 计算整个批次的梯度
                    all_output_grads = torch.autograd.grad(
                        outputs=loss_per_sample.sum(),
                        inputs=logits,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )[0]  # [batch_size + val_size, num_classes]

                    # 根据 train_size 和 val_size 分割梯度
                    train_output_grads = all_output_grads[:batch_size]  # [batch_size, num_classes]
                    val_output_grads = all_output_grads[batch_size:]   # [val_size, num_classes]

                    # Step 7: 计算 ghost scores
                    ghost_total = 0  # 初始化 ghost 累加器

                    

                    # 遍历每一层的输出
                    # for layer, (layer_input, layer_output) in layer_outputs.items():
                    #     # 确保 layer_output 是张量或者包含张量的元组/列表
                    #     if isinstance(layer_output, torch.Tensor):
                    #         if layer_output.grad is None:
                    #             continue  # 跳过没有梯度的层
                    #     elif isinstance(layer_output, (tuple, list)):
                    #         # 如果 layer_output 是 tuple 或 list，提取其中第一个有 grad 的张量
                    #         layer_output = next((o for o in layer_output if isinstance(o, torch.Tensor) and o.grad is not None), None)
                    #         if layer_output is None:
                    #             continue  # 如果 tuple/list 中没有梯度，跳过
                    #     else:
                    #         continue  # 其他类型直接跳过

                    #     # 获取训练和验证部分的输入
                    #     input_train = layer_input[:batch_size].view(batch_size, -1).float()  # [batch_size, flattened_features]
                    #     input_val = layer_input[batch_size:].view(subset_size, -1).float()  # [val_size, flattened_features]
                    #     print(input_train.shape)
                    #     print(input_val.shape)

                    #     # 将训练和验证部分的输出梯度展平
                    #     output_train_grads = train_output_grads  # [batch_size, flattened_features]
                    #     output_val_grads = val_output_grads  # [val_size, flattened_features]

                    #     # 计算输入和输出的点积
                    #     grads_dot = torch.mm(output_train_grads, output_val_grads.T)  # [batch_size, val_size]
                    #     input_dot = torch.mm(input_train, input_val.T)  # [batch_size, val_size]

                    #     # 累加 ghost scores
                    #     ghost_total += (grads_dot * input_dot).mean(dim=1)  # [batch_size]

                    # 遍历每一层的输出
                    # for layer, (layer_input, layer_output) in layer_outputs.items():
                    #     print(f"Layer: {layer}")
                        
                    #     # 检查 layer_input 类型和大小
                    #     if isinstance(layer_input, torch.Tensor):
                    #         print(f"  layer_input: Tensor, shape = {layer_input.shape}")
                    #     elif isinstance(layer_input, (tuple, list)):
                    #         print(f"  layer_input: {type(layer_input)}, lengths = {len(layer_input)}")
                    #         for i, inp in enumerate(layer_input):
                    #             if isinstance(inp, torch.Tensor):
                    #                 print(f"    element {i}: Tensor, shape = {inp.shape}")
                    #             else:
                    #                 print(f"    element {i}: {type(inp)}")
                    #     else:
                    #         print(f"  layer_input: {type(layer_input)}")
                        
                    #     # 检查 layer_output 类型和大小
                    #     if isinstance(layer_output, torch.Tensor):
                    #         print(f"  layer_output: Tensor, shape = {layer_output.shape}")
                    #     elif isinstance(layer_output, (tuple, list)):
                    #         print(f"  layer_output: {type(layer_output)}, lengths = {len(layer_output)}")
                    #         for i, out in enumerate(layer_output):
                    #             if isinstance(out, torch.Tensor):
                    #                 print(f"    element {i}: Tensor, shape = {out.shape}")
                    #             else:
                    #                 print(f"    element {i}: {type(out)}")
                    #     else:
                    #         print(f"  layer_output: {type(layer_output)}")

                    # 遍历每一层的输出
                    needed_size = batch_size + subset_size  # 例如 32 + 13 = 45
                    for layer, (layer_input, layer_output) in layer_outputs.items():
                        #print(f"Processing Layer: {layer}")

                        #
                        # Step 1: 取出实际的 `layer_input` 张量
                        #
                        if isinstance(layer_input, torch.Tensor):
                            #print("layer_input_shape:", layer_input.shape)
                            # 如果行数不足 needed_size，或者第二维为 0，则跳过
                            if layer_input.shape[0] < needed_size or layer_input.shape[1] == 0:
                                #print(f"Skipping layer {layer}: shape {layer_input.shape} not compatible with needed_size={needed_size}")
                                continue
                            
                            # 切分
                            input_train = layer_input[:batch_size].view(batch_size, -1).float()
                            input_val = layer_input[batch_size:].view(subset_size, -1).float()

                        elif isinstance(layer_input, (tuple, list)):
                            # 如果是 tuple 或 list，先取出第一个元素试试
                            if len(layer_input) > 0 and isinstance(layer_input[0], torch.Tensor):
                                layer_input = layer_input[0]
                                #print("layer_input_shape:", layer_input.shape)
                                if layer_input.shape[0] < needed_size or layer_input.shape[1] == 0:
                                    #print(f"Skipping layer {layer}: shape {layer_input.shape} not compatible with needed_size={needed_size}")
                                    continue

                                # 切分
                                input_train = layer_input[:batch_size].view(batch_size, -1).float()
                                input_val = layer_input[batch_size:].view(subset_size, -1).float()
                            else:
                                #print(f"Skipping layer {layer}: no valid tensor in layer_input")
                                continue
                        else:
                            #print(f"Skipping layer {layer}: unsupported layer_input type {type(layer_input)}")
                            continue

                        #
                        # Step 2: 取出实际的 `layer_output` 张量（这里你的代码只是在用输出梯度 train_output_grads / val_output_grads，
                        # 不需要从 layer_output 中再切分，所以暂时不用检查维度。但如果需要，也可加同样的形状检查逻辑。）
                        #
                        if isinstance(layer_output, torch.Tensor):
                            output_train_grads = train_output_grads  # [batch_size, flattened_features]
                            output_val_grads = val_output_grads  # [val_size, flattened_features]
                        elif isinstance(layer_output, (tuple, list)):
                            if len(layer_output) > 0 and isinstance(layer_output[0], torch.Tensor):
                                layer_output = layer_output[0]
                                output_train_grads = train_output_grads
                                output_val_grads = val_output_grads
                            else:
                                #print(f"Skipping layer {layer}: no valid tensor in layer_output")
                                continue
                        else:
                            #print(f"Skipping layer {layer}: unsupported layer_output type {type(layer_output)}")
                            continue

                        #
                        # Step 3: 计算点积
                        #
                        grads_dot = torch.mm(output_train_grads, output_val_grads.T)  # [batch_size, val_size]
                        #print("grads_dot.shape:", grads_dot.shape)
                        #print("input_train.shape:", input_train.shape)
                        #print("input_val.shape:", input_val.shape)
                        
                        input_dot = torch.mm(input_train, input_val.T)  # [batch_size, val_size]

                        # 累加 ghost scores
                        ghost_total += (grads_dot * input_dot).mean(dim=1)  # [batch_size]

                    # 归一化 ghost scores（如果需要）
                    ghost_scores = ghost_total  # [batch_size]
                    #print(ghost_scores)


                    # 归一化 ghost scores（如果需要）
                    ghost_scores = ghost_total  # [batch_size]
                    #print(ghost_scores)




                    # Step 8: 应用阈值选择样本
                    threshold = 0  # 根据您的需求调整阈值
                    weights = (ghost_scores >= threshold).float()  # [batch_size]

                    # Step 9: 计算加权损失
                    weighted_loss = (weights * per_sample_loss[:batch_size]).mean()

                    # 检查加权损失是否有效
                    if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                        raise ValueError("加权损失为 NaN 或 Inf。")

                    # Step 10: 反向传播
                    weighted_loss.backward()

                    # Step 11: 移除 hooks
                    for hook in hooks:
                        hook.remove()

                    # Step 12: 更新优化器
                    optimizer.step()
                    total_loss += weighted_loss.item()
                    sample_count += weights.sum().item()

                    # 打印调试信息
                    if batch_idx % 10 == 0:
                        logging.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Weighted Loss = {weighted_loss.item():.4f}, Selected Samples = {weights.sum().item()}/{batch_size}")

                except Exception as e:
                    logging.error(f"在 'Ours' 方法中发生错误: {e}")
                    # 根据需求选择是否停止训练
                    raise e

            elif args.method in ['Ours_last_layer'] and epoch >= 1:
                try:
                    # Step 1: 选择验证集的10%
                    all_val_data = val_dataset
                    subset_size = max(1, int(VALIDATION_PERCENT * len(all_val_data)))  # 确保至少有1个样本
                    val_indices = torch.randperm(len(all_val_data))[:subset_size]
                    val_subset = data.Subset(all_val_data, val_indices)
                    val_subset_loader = data.DataLoader(val_subset, batch_size=subset_size, shuffle=False, collate_fn=default_data_collator)

                    # Step 2: 初始化 ghost_total
                    ghost_total = torch.zeros(batch_size).to(device)  # [batch_size]

                    # Step 3: 注册最后一层的 forward hook
                    last_layer_name = list(model.named_modules())[-1][0]  # 获取最后一层的名字
                    last_layer_outputs = {}

                    def last_layer_hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            output.retain_grad()
                            last_layer_outputs["input"] = input
                            last_layer_outputs["output"] = output
                        elif isinstance(output, (tuple, list)):
                            retained_outputs = []
                            for o in output:
                                if isinstance(o, torch.Tensor):
                                    o.retain_grad()
                                    retained_outputs.append(o)
                            last_layer_outputs["input"] = input
                            last_layer_outputs["output"] = tuple(retained_outputs)

                    # 注册 hook
                    hook = dict(model.named_modules())[last_layer_name].register_forward_hook(last_layer_hook)


                    # Step 4: 拼接训练批次和验证子集批次
                    train_inputs = {
                        key: batch[key].to(device) for key in ["input_ids", "attention_mask", "token_type_ids"]
                    }
                    train_labels = batch["labels"].to(device)

                    # 获取验证子集的批次
                    val_batch = next(iter(val_subset_loader))
                    val_inputs = {
                        key: val_batch[key].to(device) for key in ["input_ids", "attention_mask", "token_type_ids"]
                    }
                    val_labels = val_batch["labels"].to(device)

                    # 拼接输入和标签
                    combined_inputs = {
                        key: torch.cat([train_inputs[key], val_inputs[key]], dim=0)
                        for key in train_inputs.keys()
                    }
                    combined_labels = torch.cat([train_labels, val_labels], dim=0)

                    # Step 5: 前向传播
                    combined_outputs = model(**combined_inputs, labels=combined_labels)
                    if not hasattr(combined_outputs, 'loss'):
                        raise AttributeError("模型输出不包含 'loss' 属性。")
                    combined_loss = combined_outputs.loss  # 标量

                    # 计算每个样本的损失
                    per_sample_loss = F.cross_entropy(combined_outputs.logits, combined_labels, reduction='none')  # [batch_size + subset_size]

                    # Step 6: 计算 loss 对输出 logits 的梯度（一次性计算）
                    logits = combined_outputs.logits  # [batch_size + val_size, num_classes]
                    loss_per_sample = per_sample_loss  # [batch_size + val_size]

                    # 计算整个批次的梯度
                    all_output_grads = torch.autograd.grad(
                        outputs=loss_per_sample.sum(),
                        inputs=logits,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )[0]  # [batch_size + val_size, num_classes]

                    # 根据 train_size 和 val_size 分割梯度
                    train_output_grads = all_output_grads[:batch_size]  # [batch_size, num_classes]
                    val_output_grads = all_output_grads[batch_size:]   # [val_size, num_classes]

                    # Step 7: 计算最后一层的 ghost scores
                    ghost_total = 0  # 初始化 ghost 累加器
                    needed_size = batch_size + subset_size

                    layer_input = last_layer_outputs["input"]
                    layer_output = last_layer_outputs["output"]

                    # 检查并处理 layer_input 的维度
                    if isinstance(layer_input, torch.Tensor):
                        if layer_input.shape[0] >= needed_size and layer_input.shape[1] > 0:
                            input_train = layer_input[:batch_size].view(batch_size, -1).float()
                            input_val = layer_input[batch_size:].view(subset_size, -1).float()
                    elif isinstance(layer_input, (tuple, list)):
                        if len(layer_input) > 0 and isinstance(layer_input[0], torch.Tensor):
                            layer_input = layer_input[0]
                            if layer_input.shape[0] >= needed_size and layer_input.shape[1] > 0:
                                input_train = layer_input[:batch_size].view(batch_size, -1).float()
                                input_val = layer_input[batch_size:].view(subset_size, -1).float()
                    else:
                        raise ValueError("Invalid last layer input.")

                    # 使用 output_train_grads 和 output_val_grads 计算点积
                    grads_dot = torch.mm(train_output_grads, val_output_grads.T)  # [batch_size, val_size]
                    input_dot = torch.mm(input_train, input_val.T)  # [batch_size, val_size]

                    # 累加 ghost scores
                    ghost_total += (grads_dot * input_dot).mean(dim=1)  # [batch_size]

                    # 归一化 ghost scores（如果需要）
                    ghost_scores = ghost_total  # [batch_size]





                    # Step 8: 应用阈值选择样本
                    threshold = 0  # 根据您的需求调整阈值
                    weights = (ghost_scores >= threshold).float()  # [batch_size]

                    # Step 9: 计算加权损失
                    weighted_loss = (weights * per_sample_loss[:batch_size]).mean()

                    # 检查加权损失是否有效
                    if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                        raise ValueError("加权损失为 NaN 或 Inf。")

                    # Step 10: 反向传播
                    weighted_loss.backward()

                    # Step 11: 移除最后一层的 hook
                    hook.remove()


                    # Step 12: 更新优化器
                    optimizer.step()
                    total_loss += weighted_loss.item()
                    sample_count += weights.sum().item()

                    # 打印调试信息
                    if batch_idx % 10 == 0:
                        logging.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Weighted Loss = {weighted_loss.item():.4f}, Selected Samples = {weights.sum().item()}/{batch_size}")

                except Exception as e:
                    logging.error(f"在 'Ours' 方法中发生错误: {e}")
                    # 根据需求选择是否停止训练
                    raise e


            # 在训练循环中，根据 args.method 调用 Ghost 方法
            elif args.method in ['Ghost'] and epoch >= 1:
                try:
                    # -----------------------------
                    # Step 1: 从 val_dataset 中随机选取 ~10% 的数据组成 val_subset
                    # -----------------------------
                    all_val_data = val_dataset
                    subset_size = max(1, int(VALIDATION_PERCENT * len(all_val_data)))  # 确保至少有1个样本
                    val_indices = torch.randperm(len(all_val_data))[:subset_size]
                    val_subset = data.Subset(all_val_data, val_indices)
                    val_subset_loader = data.DataLoader(val_subset, 
                                                        batch_size=subset_size, 
                                                        shuffle=False, 
                                                        collate_fn=default_data_collator)

                    # 获取“训练批次”和“验证子集批次”
                    train_inputs = {
                        key: batch[key].to(device) for key in ["input_ids", "attention_mask", "token_type_ids"]
                    }
                    train_labels = batch["labels"].to(device)

                    val_batch = next(iter(val_subset_loader))
                    val_inputs = {
                        key: val_batch[key].to(device) for key in ["input_ids", "attention_mask", "token_type_ids"]
                    }
                    val_labels = val_batch["labels"].to(device)

                    # -----------------------------
                    # Step 2: 拼接训练 + 验证的输入和标签
                    # -----------------------------
                    combined_inputs = {
                        key: torch.cat([train_inputs[key], val_inputs[key]], dim=0)
                        for key in train_inputs.keys()
                    }
                    combined_labels = torch.cat([train_labels, val_labels], dim=0)

                    batch_size = train_labels.size(0)
                    val_size = val_labels.size(0)
                    needed_size = batch_size + val_size  # 用于形状检查

                    # -----------------------------
                    # Step 3: 注册 forward hook，捕获所有层的输入与输出
                    # -----------------------------
                    layer_outputs = {}  # 用来存所有层的 (input, output)

                    def forward_hook(module, module_input, module_output):
                        """
                        对每一层的输出都调用 retain_grad()，以便在一次 backward 后，
                        layer_output.grad 存储了本层输出对 loss 的梯度。
                        """
                        # 1) 若输出是单个 Tensor
                        if isinstance(module_output, torch.Tensor):
                            module_output.retain_grad()
                            layer_outputs[str(module)] = (module_input, module_output)

                        # 2) 若输出是元组/列表
                        elif isinstance(module_output, (tuple, list)):
                            retained_outs = []
                            for o in module_output:
                                if isinstance(o, torch.Tensor):
                                    o.retain_grad()
                                    retained_outs.append(o)
                            layer_outputs[str(module)] = (module_input, tuple(retained_outs))

                        # 3) 其它类型，直接存下
                        else:
                            layer_outputs[str(module)] = (module_input, None)

                    # 给 model 的所有子模块注册 hook
                    hooks = []
                    for name, submodule in model.named_modules():
                        hook_handle = submodule.register_forward_hook(forward_hook)
                        hooks.append(hook_handle)

                    # -----------------------------
                    # Step 4: 前向传播，计算总 loss
                    # -----------------------------
                    combined_outputs = model(**combined_inputs, labels=combined_labels)
                    if not hasattr(combined_outputs, 'loss'):
                        raise AttributeError("模型输出不包含 'loss' 属性。")

                    # combined_loss是标量
                    combined_loss = combined_outputs.loss  
                    
                    # 计算每个样本的交叉熵 loss，用于后续筛选
                    per_sample_loss = F.cross_entropy(
                        combined_outputs.logits, combined_labels, reduction='none'
                    )  # shape: [batch_size + val_size]

                    # -----------------------------
                    # Step 5: 对 logits 做一次性梯度计算 (可选)
                    #     如果你只想对中间层梯度进行分析，下面这一步可能不用分出 train_output_grads / val_output_grads。
                    #     不过，一般我们还是会对最终 logits 做一次 grad，可能在其它逻辑里要用。
                    # -----------------------------
                    logits = combined_outputs.logits  # [batch_size + val_size, num_classes]
                    loss_per_sample = per_sample_loss  # 只是为了代码可读性

                    # 一次性对 logits 求 jacobian
                    # 这样 all_output_grads 就是 [batch_size + val_size, num_classes]
                    all_output_grads = torch.autograd.grad(
                        outputs=loss_per_sample.sum(),
                        inputs=logits,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True
                    )[0]  
                    # 这里 all_output_grads 仅供参考，你也可以不用

                    # -----------------------------
                    # Step 6: 核心 - 对 "所有层" 的输出做一次 backward
                    # -----------------------------
                    # 直接对整个 per_sample_loss.sum() 做一次 backward，
                    # PyTorch会将梯度传播至各层的 output.grad 中。
                    total_loss_for_backprop = loss_per_sample.sum()
                    total_loss_for_backprop.backward(retain_graph=True)  
                    # 注意：如果后面不需要再做别的操作，你可以把 retain_graph=False
                    # 但如果你还想后面做别的梯度操作，需要保留计算图。

                    # -----------------------------
                    # Step 7: 计算 ghost scores
                    # -----------------------------
                    ghost_total = torch.zeros(batch_size, device=device)  # 用来累加每层的 ghost

                    for layer, (layer_input, layer_output) in layer_outputs.items():
                        # 1) 先把 layer_input 取成真正的 Tensor
                        if isinstance(layer_input, torch.Tensor):
                            # 如果 batch 尺寸不够或 shape 不合适就跳过
                            if layer_input.shape[0] < needed_size or layer_input.shape[1] == 0:
                                continue
                            input_train = layer_input[:batch_size].reshape(batch_size, -1).float()
                            input_val   = layer_input[batch_size:].reshape(val_size, -1).float()

                        elif isinstance(layer_input, (tuple, list)):
                            # 若是 tuple/list，尝试用其中的第一个 Tensor
                            if len(layer_input) == 0 or not isinstance(layer_input[0], torch.Tensor):
                                continue
                            layer_input = layer_input[0]
                            if layer_input.shape[0] < needed_size or layer_input.shape[1] == 0:
                                continue
                            input_train = layer_input[:batch_size].reshape(batch_size, -1).float()
                            input_val   = layer_input[batch_size:].reshape(val_size, -1).float()
                        else:
                            # 其它类型，跳过
                            continue

                        # 2) 获取本层的输出梯度
                        if isinstance(layer_output, torch.Tensor):
                            layer_grads = layer_output.grad  # shape: [batch_size + val_size, ...]
                        elif isinstance(layer_output, (tuple, list)):
                            # 如果是 tuple/list，拿第一个非 None 的 Tensor
                            if len(layer_output) == 0 or not isinstance(layer_output[0], torch.Tensor):
                                continue
                            layer_grads = layer_output[0].grad
                        else:
                            continue

                        # 如果这个层的 grad 是 None 或 shape不匹配，就跳过
                        if layer_grads is None or layer_grads.shape[0] < needed_size:
                            continue

                        grad_train = layer_grads[:batch_size].reshape(batch_size, -1)
                        grad_val   = layer_grads[batch_size:].reshape(val_size, -1)

                        # 3) 分别算 grads_dot 和 input_dot
                        #    grads_dot = (grad_train) dot (grad_val)
                        #    input_dot = (input_train) dot (input_val)
                        grads_dot = torch.mm(grad_train, grad_val.T)    # [batch_size, val_size]
                        input_dot = torch.mm(input_train, input_val.T)  # [batch_size, val_size]

                        # 4) 累加到 ghost_total
                        ghost_total += (grads_dot * input_dot).mean(dim=1)  # shape: [batch_size]

                    # 得到最终 ghost_scores
                    ghost_scores = ghost_total  # [batch_size]
                    
                    # -----------------------------
                    # Step 8: 根据 ghost_scores 筛选样本
                    # -----------------------------
                    threshold = 0.0
                    weights = (ghost_scores >= threshold).float()  # [batch_size]
                    
                    # -----------------------------
                    # Step 9: 计算加权后的损失并反向传播
                    # -----------------------------
                    # per_sample_loss[:batch_size] 对应“训练集”部分的 loss
                    weighted_loss = (weights * per_sample_loss[:batch_size]).mean()

                    if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                        raise ValueError("加权损失出现 NaN 或 Inf")

                    # 反向传播并更新
                    # 如果你不需要叠加梯度，这里先 optimizer.zero_grad() 或把前面那次 backward() 换成 'create_graph=True' 等方式看需求
                    optimizer.zero_grad()
                    weighted_loss.backward()
                    optimizer.step()

                    # -----------------------------
                    # Step 10: 移除所有 hooks
                    # -----------------------------
                    for hook_handle in hooks:
                        hook_handle.remove()

                    # -----------------------------
                    # 记录 / 打印当前 batch 的训练情况
                    # -----------------------------
                    total_loss += weighted_loss.item()
                    sample_count += weights.sum().item()

                    if batch_idx % 10 == 0:
                        logging.info(
                            f"Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                            f"Weighted Loss = {weighted_loss.item():.4f}, "
                            f"Selected Samples = {weights.sum().item()}/{batch_size}"
                        )

                except Exception as e:
                    logging.error(f"在 'Ghost' 方法中发生错误: {e}")
                    # 看需求决定是否要 raise
                    raise e
            

            #elif args.method in ['SPL'] and epoch >= 1:
            elif args.method in ['SPL'] and epoch >= 1:
                try:
                    # Step 1: 选择验证集的10%
                    all_val_data = val_dataset
                    subset_size = max(1, int(VALIDATION_PERCENT * len(all_val_data)))  # 确保至少有1个样本
                    val_indices = torch.randperm(len(all_val_data))[:subset_size]
                    val_subset = data.Subset(all_val_data, val_indices)
                    val_subset_loader = data.DataLoader(
                        val_subset, 
                        batch_size=subset_size, 
                        shuffle=False, 
                        collate_fn=default_data_collator
                    )

                    # Step 2: 获取验证子集的批次
                    val_batch = next(iter(val_subset_loader))
                    val_inputs = {
                        key: val_batch[key].to(device) 
                        for key in ["input_ids", "attention_mask", "token_type_ids"]
                    }
                    val_labels = val_batch["labels"].to(device)

                    # Step 3: 前向传播验证集，计算每个样本的损失
                    with torch.no_grad():
                        val_outputs = model(**val_inputs, labels=val_labels)
                        if not hasattr(val_outputs, 'loss'):
                            raise AttributeError("模型输出不包含 'loss' 属性。")
                        # 使用 'reduction="none"' 以获得每个样本的损失
                        val_loss = F.cross_entropy(val_outputs.logits, val_labels, reduction='none')  # [subset_size]

                    # # Step 4: 确定阈值（例如，验证集损失的中位数）
                    threshold = torch.median(val_loss).item()

                    # Step 5: 计算训练批次的每个样本的损失
                    train_inputs = {
                        key: batch[key].to(device) 
                        for key in ["input_ids", "attention_mask", "token_type_ids"]
                    }
                    train_labels = batch["labels"].to(device)
                    train_outputs = model(**train_inputs, labels=train_labels)
                    if not hasattr(train_outputs, 'loss'):
                        raise AttributeError("模型输出不包含 'loss' 属性。")
                    # 计算每个样本的损失
                    train_loss = F.cross_entropy(train_outputs.logits, train_labels, reduction='none')  # [batch_size]

                    # Step 6: 选择训练样本，损失低于阈值
                    weights = (train_loss <= threshold).float()  # [batch_size]

                    # Step 7: 计算加权损失
                    selected_samples = weights.sum().item()
                    if selected_samples == 0:
                        # 如果没有样本满足条件，使用所有样本
                        logging.warning("没有样本满足阈值条件。使用所有样本进行训练。")
                        weights = torch.ones_like(weights)
                        selected_samples = weights.sum().item()
                    weighted_loss = (weights * train_loss).mean()

                    # Step 8: 反向传播和优化
                    weighted_loss.backward()
                    optimizer.step()
                    total_loss += weighted_loss.item()
                    sample_count += selected_samples

                    # Step 9: 打印调试信息
                    if batch_idx % 10 == 0:
                        logging.info(
                            f"Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                            f"Weighted Loss = {weighted_loss.item():.4f}, "
                            f"Selected Samples = {selected_samples}/{batch_size}"
                        )

                except Exception as e:
                    logging.error(f"在 'SPL' 方法中发生错误: {e}")
                    # 根据需求选择是否停止训练
                    raise e


            else:
                # 标准训练，不使用 'Ours' 方法
                train_inputs = {
                    key: batch[key].to(device) for key in ["input_ids", "attention_mask", "token_type_ids"]
                }
                train_labels = batch["labels"].to(device)

                # 将 labels 传递给模型
                outputs = model(**train_inputs, labels=train_labels)
                if not hasattr(outputs, 'loss'):
                    raise AttributeError("模型输出不包含 'loss' 属性。")
                loss = outputs.loss  # 标量

                if loss is None:
                    raise ValueError("计算的损失为 None。请检查模型的输出。")

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                sample_count += batch_size

                # 打印调试信息
                if batch_idx % 10 == 0:
                    logging.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

        # 记录每个epoch的平均损失和选择的样本数量
        avg_loss = total_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch + 1} - Averaged Loss: {avg_loss:.4f}, Selected Samples: {sample_count}/{len(train_dataloader.dataset)}")
        # Step 4: 确定阈值（例如，验证集损失的中位数）
        threshold = avg_loss * 1.5  # 举例，阈值是平均损失的1.5倍

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training in {elapsed_time:.2f} seconds.")

    return model

def evaluate_model(model: nn.Module, dataset: data.Dataset, batch_size: int, args: Namespace) -> Tuple[float, float]:
    dataloader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=default_data_collator
        
    )
    DEVICE = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    model.eval()
    metric = evaluate.load("glue", "sst2")
    total_loss = 0.0
    for batch in dataloader:
        with torch.no_grad():
            logits = model(
                input_ids=batch["input_ids"].to(device=DEVICE),
                attention_mask=batch["attention_mask"].to(device=DEVICE),
                token_type_ids=batch["token_type_ids"].to(device=DEVICE),
            ).logits
            labels = batch["labels"].to(device=DEVICE)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").detach()
            predictions = logits.argmax(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=labels,
            )
    eval_metric = metric.compute()
    return total_loss.item() / len(dataloader.dataset), eval_metric["accuracy"]


def main():
    import numpy as np
    args = parse_args()
    DEVICE = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if args.seed is not None:
        set_seed(args.seed)

    if args.dataset_name in ["agnews"]:
        print("Using AG News dataset")
        train_dataset = get_ag_news_dataset(split="train")
        eval_dataset = get_ag_news_dataset(split="test")
    # train_dataset = get_glue_dataset(data_name=args.dataset_name, split="train")
    # eval_dataset = get_glue_dataset(data_name=args.dataset_name, split="valid")

        num_eval_samples = len(eval_dataset)

        # 随机划分索引
        import random
        random.seed(42)  # 固定随机种子
        indices = list(range(num_eval_samples))
        random.shuffle(indices)  # 打乱索引

        # 按比例划分验证集和测试集
        if num_eval_samples < 4000:
            split_point = num_eval_samples // 2  # 如果样本小于 4000，则验证集和测试集各占一半
        else:
            split_point = 2000  # 否则固定 2000 个样本作为验证集

        eval_indices = indices[:split_point]  # 前一半作为验证集
        test_indices = indices[split_point:]  # 后一半作为测试集

        val_dataset = get_ag_news_dataset(split="test", indices=eval_indices)
        test_dataset = get_ag_news_dataset(split="test", indices=test_indices)
    elif args.dataset_name in ["emotion"]:
        print("Using Emotion dataset")
        train_dataset = get_emotion_dataset(split="train")
        val_dataset = get_emotion_dataset(split="validation")
        test_dataset = get_emotion_dataset(split="test")
    elif args.dataset_name in ["20news"]:
        print("Using 20 News dataset")
        train_dataset = get_20_news_dataset(split="train")
        eval_dataset = get_20_news_dataset(split="test")

        num_eval_samples = len(eval_dataset)

        # 随机划分索引
        import random
        random.seed(42)  # 固定随机种子
        indices = list(range(num_eval_samples))
        random.shuffle(indices)  # 打乱索引

        # 按比例划分验证集和测试集
        if num_eval_samples < 4000:
            split_point = num_eval_samples // 2  # 如果样本小于 4000，则验证集和测试集各占一半
        else:
            split_point = 2000  # 否则固定 2000 个样本作为验证集

        eval_indices = indices[:split_point]  # 前一半作为验证集
        test_indices = indices[split_point:]  # 后一半作为测试集

        val_dataset = get_20_news_dataset(split="test", indices=eval_indices)
        test_dataset = get_20_news_dataset(split="test", indices=test_indices)


    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Sample from train dataset: {train_dataset[0]}")

    print(train_dataset['labels'][:10])


    train_losses, train_accuracies = [], []
    eval_losses, eval_accuracies = [], []

    for i in range(5):  # 训练和评估 5 次
        logger.info(f"Training iteration {i + 1}/5")

        model = train(
            dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.train_batch_size,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            args=args,
        )

        eval_train_dataset = train_dataset

        train_loss, train_acc = evaluate_model(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size, args=args)
        logger.info(f"Iteration {i + 1} - Train loss: {train_loss}, Train Accuracy: {train_acc}")

        eval_loss, eval_acc = evaluate_model(model=model, dataset=test_dataset, batch_size=args.eval_batch_size, args=args)
        logger.info(f"Iteration {i + 1} - Evaluation loss: {eval_loss}, Evaluation Accuracy: {eval_acc}")

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_acc)

    # 计算平均值和标准差
    avg_train_loss, std_train_loss = np.mean(train_losses), np.std(train_losses)
    avg_train_acc, std_train_acc = np.mean(train_accuracies), np.std(train_accuracies)
    avg_eval_loss, std_eval_loss = np.mean(eval_losses), np.std(eval_losses)
    avg_eval_acc, std_eval_acc = np.mean(eval_accuracies), np.std(eval_accuracies)

    logger.info(f"Average Train loss: {avg_train_loss} (±{std_train_loss}), Average Train Accuracy: {avg_train_acc} (±{std_train_acc})")
    logger.info(f"Average Evaluation loss: {avg_eval_loss} (±{std_eval_loss}), Average Evaluation Accuracy: {avg_eval_acc} (±{std_eval_acc})")
    # 追加记录到文件
    with open("evaluation_results_noise.txt", "a") as f:
        f.write(
            f"Dataset: {args.dataset_name}, Method: {args.method}, "
            f"Average Evaluation Accuracy: {avg_eval_acc:.4f} (±{std_eval_acc:.4f})\n"
        )


if __name__ == "__main__":
    main()
