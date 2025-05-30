# hybrid_embeddings.py
import os
import numpy as np
import pickle
from sklearn.preprocessing import normalize
import time
import pandas as pd

def generate_hybrid_embeddings(model_type, data_dir="./kg_data3", model_dir="./model_outputs3", 
                              output_dir="./hybrid_embeddings3",
                              graph_weight=0.5, text_weight=0.5, normalize_vectors=True,
                              entity_type_weights=None, strategy_name=None,
                              ):
    """
    生成混合嵌入向量
    
    参数:
        data_dir: 知识图谱数据目录
        model_dir: 模型输出目录
        output_dir: 混合嵌入输出基础目录
        graph_weight: 图嵌入的权重(0-1)
        text_weight: 文本嵌入的权重(0-1)
        normalize_vectors: 是否对向量进行标准化
        entity_type_weights: 不同实体类型的权重字典(可选)
        strategy_name: 策略名称，用于文件命名(可选)
        model_type: 图嵌入模型类型，可选"transe", "transr", "rescal", "rotate", "distmult"
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定策略名称，用于文件命名
    if strategy_name is None:
        if graph_weight == 1.0 and text_weight == 0.0:
            strategy_name = "graph_only"
        elif graph_weight == 0.0 and text_weight == 1.0:
            strategy_name = "text_only"
        elif graph_weight == 0.5 and text_weight == 0.5:
            strategy_name = "balanced"
        elif graph_weight > text_weight:
            strategy_name = "graph_heavy"
        elif graph_weight < text_weight:
            strategy_name = "text_heavy"
        else:
            strategy_name = f"graph_{graph_weight}_text_{text_weight}"
    
    print(f"生成混合嵌入向量 (模型: {model_type}, 策略: {strategy_name})...")
    start_time = time.time()
    
    # 根据模型类型加载相应的图嵌入
    entity_embedding_file = get_entity_embedding_file(model_type, model_dir)
    
    with open(entity_embedding_file, 'rb') as f:
        graph_embeddings = pickle.load(f)
    
    # 加载文本嵌入
    with open(os.path.join(model_dir, 'text_embeddings.pkl'), 'rb') as f:
        text_embeddings_dict = pickle.load(f)
    
    # 加载实体元数据
    entities_df = pd.read_csv(os.path.join(data_dir, 'entities.csv'))
    
    # 获取实体类型信息
    entity_types = {}
    for _, row in entities_df.iterrows():
        entity_id = int(row['id'])  # Neo4j ID
        entity_type = row['type']
        embed_id = None
        
        # 查找嵌入ID
        with open(os.path.join(data_dir, 'entity_to_id.pkl'), 'rb') as f:
            entity_to_id = pickle.load(f)
            if entity_id in entity_to_id:
                embed_id = entity_to_id[entity_id]
        
        if embed_id is not None:
            entity_types[embed_id] = entity_type
    
    # 设置默认的实体类型权重
    if entity_type_weights is None:
        entity_type_weights = {
            'Case': 1.0,      # 案例节点权重最高
            'Charge': 0.8,    # 罪名节点权重较高
            'Article': 0.7,   # 法条节点权重也较高
            'Factor': 0.6,    # 情节节点中等权重
            'Element': 0.5,   # 要素节点中等权重
            # 'Criminal': 0.1,  # 被告人节点权重较低
            'Term': 0.4       # 判决结果节点权重中低
        }
    
    # 生成混合嵌入
    hybrid_embeddings = {}
    graph_dim = graph_embeddings.shape[1]
    text_dim = None if not text_embeddings_dict else next(iter(text_embeddings_dict.values())).shape[0]
    
    # 如果需要，对图嵌入进行标准化
    if normalize_vectors:
        graph_embeddings = normalize(graph_embeddings, axis=1)
    
    # 对每个实体生成混合嵌入
    print(f"开始混合 {len(graph_embeddings)} 个实体的嵌入向量...")
    
    for embed_id in range(len(graph_embeddings)):
        # 获取图嵌入向量
        graph_emb = graph_embeddings[embed_id]
        
        # 获取实体类型权重
        entity_type = entity_types.get(embed_id, 'Unknown')
        type_weight = entity_type_weights.get(entity_type, 0.5)
        
        # 如果是案例节点且有文本嵌入，则混合两种嵌入
        if embed_id in text_embeddings_dict:
            text_emb = text_embeddings_dict[embed_id]
            
            # 标准化文本嵌入(如果需要)
            if normalize_vectors:
                text_emb = normalize(text_emb.reshape(1, -1)).flatten()
            
            # 计算混合嵌入
            # 应用加权：graph_weight * type_weight 确保不同类型的实体权重不同
            hybrid_emb = np.concatenate([
                graph_emb * graph_weight * type_weight,
                text_emb * text_weight
            ])
            
        else:
            # 仅使用图嵌入
            hybrid_emb = graph_emb * type_weight
            
            # 如果有文本嵌入，则需要填充零向量以保持维度一致
            if text_dim is not None:
                hybrid_emb = np.concatenate([hybrid_emb, np.zeros(text_dim)])
        
        # 再次标准化整个混合向量(如果需要)
        if normalize_vectors:
            hybrid_emb = normalize(hybrid_emb.reshape(1, -1)).flatten()
        
        hybrid_embeddings[embed_id] = hybrid_emb
    
    # 将字典转换为数组形式
    hybrid_embeddings_array = np.zeros((len(hybrid_embeddings), len(next(iter(hybrid_embeddings.values())))))
    for embed_id, emb in hybrid_embeddings.items():
        hybrid_embeddings_array[embed_id] = emb
    
    # 使用模型类型和策略名称保存混合嵌入
    file_prefix = f"{model_type}_{strategy_name}"
    embedding_file = os.path.join(output_dir, f'hybrid_embeddings_{file_prefix}.pkl')
    with open(embedding_file, 'wb') as f:
        pickle.dump(hybrid_embeddings_array, f)
    
    # 保存元数据
    metadata = {
        'model_type': model_type,
        'strategy': strategy_name,
        'graph_weight': graph_weight,
        'text_weight': text_weight,
        'normalize_vectors': normalize_vectors,
        'entity_type_weights': entity_type_weights,
        'graph_dim': graph_dim,
        'text_dim': text_dim,
        'hybrid_dim': hybrid_embeddings_array.shape[1],
        'num_entities': len(hybrid_embeddings)
    }
    
    metadata_file = os.path.join(output_dir, f'hybrid_metadata_{file_prefix}.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    elapsed_time = time.time() - start_time
    print(f"混合嵌入生成完成，耗时: {elapsed_time:.2f}秒")
    print(f"混合嵌入维度: {hybrid_embeddings_array.shape[1]}")
    print(f"生成了 {len(hybrid_embeddings)} 个实体的混合嵌入向量")
    print(f"混合嵌入已保存到 {embedding_file}")
    
    return hybrid_embeddings_array, metadata

def get_entity_embedding_file(model_type, model_dir):
    """根据模型类型获取对应的实体嵌入文件路径"""
    embedding_files = {
        "transe": os.path.join(model_dir, 'entity_embeddings.pkl'),
        "transr": os.path.join(model_dir, 'transr_entity_embeddings.pkl'),
        "rescal": os.path.join(model_dir, 'rescal_entity_embeddings.pkl'),
        "rotate": os.path.join(model_dir, 'rotate_entity_embeddings.pkl'),
        "distmult": os.path.join(model_dir, 'distmult_entity_embeddings.pkl'),
        "conve": os.path.join(model_dir, 'conve_entity_embeddings.pkl'),
        "rgcn": os.path.join(model_dir, 'rgcn_entity_embeddings.pkl')
    }
    
    if model_type not in embedding_files:
        raise ValueError(f"不支持的模型类型: {model_type}，支持的类型有: {list(embedding_files.keys())}")
    
    return embedding_files[model_type]

def generate_all_hybrid_embeddings(data_dir="./kg_data3", model_dir="./model_outputs3", 
                                  output_dir="./hybrid_embeddings3", model_types=None):
    """为所有指定的模型类型生成所有混合策略的嵌入"""
    if model_types is None:
        model_types = ["transe", "transr", "rescal", "rotate", "distmult"]
    
    # 自定义实体类型权重
    custom_weights = {
        'Case': 1.0,      # 案例节点权重最高
        'Charge': 0.8,    # 罪名节点权重较高
        'Article': 0.7,   # 法条节点权重也较高
        'Factor': 0.6,    # 情节节点中等权重
        'Element': 0.5,   # 要素节点中等权重
        'Criminal': 0.1,  # 被告人节点权重较低
        'Term': 0.4       # 判决结果节点权重中低
    }
    
    # 定义混合策略
    strategies = [
        {'name': 'graph_only', 'graph_weight': 1.0, 'text_weight': 0.0},
        {'name': 'text_only', 'graph_weight': 0.0, 'text_weight': 1.0},
        {'name': 'balanced', 'graph_weight': 0.5, 'text_weight': 0.5},
        {'name': 'graph_heavy', 'graph_weight': 0.7, 'text_weight': 0.3},
        {'name': 'text_heavy', 'graph_weight': 0.3, 'text_weight': 0.7}
    ]
    
    # 创建索引记录
    all_embeddings = []
    
    # 为每种模型类型和混合策略生成嵌入
    for model_type in model_types:
        print(f"\n=== 为 {model_type} 模型生成混合嵌入 ===\n")
        
        for strategy in strategies:
            print(f"\n生成 {model_type} + {strategy['name']} 混合策略的嵌入...")
            
            try:
                # 检查文件是否存在
                entity_embedding_file = get_entity_embedding_file(model_type, model_dir)
                if not os.path.exists(entity_embedding_file):
                    print(f"警告: {model_type} 的实体嵌入文件 {entity_embedding_file} 不存在，跳过")
                    continue
                
                # 生成混合嵌入
                _, metadata = generate_hybrid_embeddings(
                    data_dir=data_dir,
                    model_dir=model_dir,
                    output_dir=output_dir,
                    graph_weight=strategy['graph_weight'], 
                    text_weight=strategy['text_weight'],
                    strategy_name=strategy['name'],
                    entity_type_weights=custom_weights,
                    model_type=model_type
                )
                
                # 记录生成的嵌入信息
                all_embeddings.append({
                    'model_type': model_type,
                    'strategy': strategy['name'],
                    'graph_weight': strategy['graph_weight'],
                    'text_weight': strategy['text_weight'],
                    'embeddings_file': f"hybrid_embeddings_{model_type}_{strategy['name']}.pkl",
                    'metadata_file': f"hybrid_metadata_{model_type}_{strategy['name']}.pkl"
                })
                
            except Exception as e:
                print(f"错误: 生成 {model_type} + {strategy['name']} 混合嵌入时出错: {str(e)}")
    
    # 保存索引文件
    index_df = pd.DataFrame(all_embeddings)
    index_df.to_csv(os.path.join(output_dir, 'all_hybrid_embeddings_index.csv'), index=False)
    
    with open(os.path.join(output_dir, 'all_hybrid_embeddings_index.pkl'), 'wb') as f:
        pickle.dump(all_embeddings, f)
    
    print(f"\n所有混合嵌入已生成并保存到 {output_dir} 目录")
    print(f"混合嵌入索引已保存到 {os.path.join(output_dir, 'all_hybrid_embeddings_index.csv')}")
    
    return all_embeddings

if __name__ == "__main__":
    # 生成所有模型的所有混合策略嵌入
    model_types = ["transe", "transr", "rescal", "rotate", "distmult", "conve", "rgcn"]
    # model_types = ["transe", "rescal", "distmult"]
    
    # 检查哪些模型的嵌入文件可用
    model_dir = "./model_outputs3"
    available_models = []
    for model_type in model_types:
        try:
            embedding_file = get_entity_embedding_file(model_type, model_dir)
            if os.path.exists(embedding_file):
                available_models.append(model_type)
                print(f"找到 {model_type} 的嵌入文件: {embedding_file}")
            else:
                print(f"未找到 {model_type} 的嵌入文件: {embedding_file}")
        except Exception as e:
            print(f"检查 {model_type} 时出错: {str(e)}")
    
    if not available_models:
        print("错误: 未找到任何可用的模型嵌入文件，请先训练模型")
        exit(1)
    
    print(f"将为以下模型生成混合嵌入: {available_models}")
    generate_all_hybrid_embeddings(model_types=available_models)