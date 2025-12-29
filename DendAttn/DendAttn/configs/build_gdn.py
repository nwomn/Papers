# -*- coding: utf-8 -*-

"""
GatedDeltaNet 模型配置生成与参数估算器 (V2)

功能:
1. 在 `CUSTOM_CONFIG_PARAMS` 字典中自定义模型配置。
2. 根据配置创建模型，并精确计算其参数量。
3. 根据 `KEYS_TO_SAVE` 白名单，生成一个干净的、只包含指定参数的 `config.json` 文件。
"""

import torch
import json
import csv
import transformers
import os

from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

from count_params import count_and_print_parameters

# =========================================================================
# 自定义配置列表：(hidden_size, head_dim, num_heads, num_hidden_layers)
# =========================================================================
CONFIG_COMBINATIONS = [
    (32, 16, 2, 2),
    #(64, 32, 2, 1),
    (64, 32, 2, 2),
    #(64, 32, 2, 4),
    #(128, 64, 2, 1),
    (128, 64, 2, 2),
    #(128, 64, 2, 4),
    #(256, 64, 4, 1),
    (256, 64, 4, 2),
    #(256, 64, 4, 4),
    #(256, 64, 4, 6),
    #(256, 64, 4, 8),
    #(256, 64, 4, 10),
    #(256, 64, 4, 12),
    #(384, 64, 6, 2),
    (384, 64, 6, 4),
    #(384, 64, 6, 6),
    #(384, 64, 6, 8),
    #(384, 64, 6, 10),
    #(384, 64, 6, 12),
    #(512, 64, 8, 4),
    (512, 64, 8, 6),
    #(512, 64, 8, 8),
    #(512, 64, 8, 10),
    #(512, 64, 8, 12),
    #(640, 64, 10, 6),
    #(640, 64, 10, 8),
    (640, 64, 10, 10),
    #(640, 64, 10, 12),
    #(768, 64, 12, 10),
    (768, 64, 12, 12),
]

def get_param_count_in_millions(model):
    """
    计算模型参数量并返回以M为单位的整数值
    """
    total_params = sum(p.numel() for p in model.parameters())
    return round(total_params / 1_000_000), total_params

def process_single_config(hidden_size, head_dim, num_heads, num_hidden_layers):
    """
    处理单个配置组合，生成对应的config文件
    """
    # =========================================================================
    # 步骤 1: 在此字典中自定义您的模型配置
    # 注意: 这里的参数需要足够完整以成功创建模型实例。
    # 您提供的 340M config 中的所有相关 key 都已包含在此处。
    # =========================================================================
    
    CUSTOM_CONFIG_PARAMS = {
        # --- 核心架构参数 ---
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": 32000,
        "hidden_ratio": 4,
        "intermediate_size": None, # 通常由 hidden_size 和 hidden_ratio 计算，设为 None
        "max_position_embeddings": 2048,
        "hidden_act": "swish",
        "norm_eps": 1e-6,
        "initializer_range": 0.02,
        "tie_word_embeddings": True,

        # --- GatedDeltaNet/FLA 特定参数 ---
        "attn_mode": "chunk",
        "conv_size": 4,
        "expand_v": 1.,
        "use_gate": True,
        "use_short_conv": True,

        # --- Tokenizer & 其他 ---
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": None,
        "use_cache": True,
        "attn": None,
        "torch_dtype": "bfloat16", # 您可以指定数据类型

        # --- 融合优化选项 ---
        "fuse_norm": True,
        "fuse_swiglu": True,
        "fuse_cross_entropy": True,
        
        # --- 元数据 ---
        "_exp_group": "gated_deltanet",
        "_ndls": None,  # 将在后面设置
    }
    
    # =========================================================================
    # 步骤 2: 创建模型并计算参数 (此步骤逻辑不变)
    # =========================================================================
    print(f"[INFO] 正在创建模型实例 (hidden_size={hidden_size}, num_layers={num_hidden_layers}, num_heads={num_heads}, head_dim={head_dim})...")
    
    # 即使某些参数不会被保存到最终的json，也需要传入Config类以正确构建模型
    model_config = GatedDeltaNetConfig(**CUSTOM_CONFIG_PARAMS)
    model = GatedDeltaNetForCausalLM(config=model_config)
    
    # 计算参数量
    param_count_m, total_params = get_param_count_in_millions(model)
    print(f"[INFO] 模型参数量: {param_count_m}M")
    
    # 生成模型名称
    MODEL_NAME = f"gated_deltanet-{hidden_size}-{num_hidden_layers}-{param_count_m}M"
    print(f"[INFO] 模型名称: {MODEL_NAME}")
    
    # 更新 _ndls
    CUSTOM_CONFIG_PARAMS["_ndls"] = f"./{MODEL_NAME}"
    
    count_and_print_parameters(model)
    
    # =========================================================================
    # 步骤 3: 按白名单生成并保存 "干净" 的 config.json
    # =========================================================================
    
    # 定义您希望最终出现在 config.json 文件中的所有 key
    KEYS_TO_SAVE = [
        "_ndls", "architectures", "attn", "attn_mode", "bos_token_id", 
        "conv_size", "eos_token_id", "expand_v", "fuse_cross_entropy",
        "fuse_norm", "fuse_swiglu", "head_dim", "hidden_act", "hidden_ratio", 
        "hidden_size", "initializer_range", "intermediate_size", 
        "max_position_embeddings", "model_type", "norm_eps",
        "num_heads", "num_hidden_layers", 
        "tie_word_embeddings", "torch_dtype", "transformers_version", "use_beta", 
        "use_cache", "use_gate", "use_short_conv", "vocab_size"
    ]

    # 创建一个新字典，只包含我们想要的键值对
    clean_config_dict = {}
    
    for key in KEYS_TO_SAVE:
        # 对特殊 key 进行处理
        if key == "architectures":
            clean_config_dict[key] = [model.__class__.__name__]
        elif key == "model_type":
            clean_config_dict[key] = model_config.model_type
        elif key == "transformers_version":
            clean_config_dict[key] = transformers.__version__
        # 对于其他 key，直接从 CUSTOM_CONFIG_PARAMS 或 model_config 对象中获取
        # 优先从 CUSTOM_CONFIG_PARAMS 获取，以确保与用户输入完全一致
        elif key in CUSTOM_CONFIG_PARAMS:
            clean_config_dict[key] = CUSTOM_CONFIG_PARAMS[key]
        else:
            # 如果 key 不在自定义字典中，尝试从实例化的 config 对象获取
            clean_config_dict[key] = getattr(model_config, key, None)

    try:
    # 1. 创建以模型名称命名的目录
    # os.makedirs 如果目录已存在不会报错 (exist_ok=True)
        # os.makedirs(MODEL_NAME, exist_ok=True)
        EXP_GROUP = CUSTOM_CONFIG_PARAMS["_exp_group"]
        os.makedirs(os.path.join(EXP_GROUP, MODEL_NAME), exist_ok=True)
        print(f"[INFO] 确保输出目录 './{MODEL_NAME}/' 已创建。")

        # 2. 定义包含目录的完整输出路径
        # 使用 os.path.join 确保跨平台兼容性
        output_path = os.path.join(EXP_GROUP, MODEL_NAME, "config.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            # 使用 sort_keys=False 保持KEYS_TO_SAVE列表中的顺序
            json.dump(clean_config_dict, f, indent=2, ensure_ascii=False)
        
        # 3. 更新成功提示信息
        print(f"[SUCCESS] '干净' 的模型配置已成功保存至: {output_path}")
        # print("\n[INFO] 生成的JSON内容预览:")
        # print(json.dumps(clean_config_dict, indent=2))
    
    except Exception as e:
        print(f"[ERROR] 保存 config.json 文件时出错: {e}")
        
    return EXP_GROUP, MODEL_NAME, total_params


def main():
    """
    主函数：遍历所有配置组合，为每个组合生成对应的config文件
    """
    print(f"[INFO] 开始批量生成配置，共有 {len(CONFIG_COMBINATIONS)} 种配置组合")
    print("=" * 80)
    
    summary = []
    for i, (hidden_size, head_dim, num_heads, num_hidden_layers) in enumerate(CONFIG_COMBINATIONS, 1):
        print(f"\n[INFO] 处理配置组合 {i}/{len(CONFIG_COMBINATIONS)}")
        print(f"[INFO] hidden_size={hidden_size}, head_dim={head_dim}, num_heads={num_heads}, num_hidden_layers={num_hidden_layers}")
        print("-" * 60)
        
        try:
            exp_group, model_name, total_params = process_single_config(hidden_size, head_dim, num_heads, num_hidden_layers)
            summary.append({
                "model_name": model_name,
                "total_params": total_params
            })
            print(f"[SUCCESS] 配置组合 {i} 处理完成")
        except Exception as e:
            print(f"[ERROR] 配置组合 {i} 处理失败: {e}")
            continue
        
        print("-" * 60)
    
    print("\n" + "=" * 80)
    print("[INFO] 所有配置组合处理完成！")
    
    with open(os.path.join(exp_group, f"{exp_group}_config_summary.csv"), "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"[INFO] 配置摘要已保存至: {os.path.join(exp_group, 'config_summary.csv')}")


if __name__ == "__main__":
    main()