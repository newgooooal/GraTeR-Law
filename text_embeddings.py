# text_embeddings.py - 增强版
import os
import pandas as pd
import numpy as np
import pickle
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from py2neo import Graph

def generate_text_embeddings(data_dir="./kg_data3", output_dir="./model_outputs3", 
                            model_name="paraphrase-multilingual-MiniLM-L12-v2",
                            neo4j_url="neo4j://10.61.2.143:7688",
                            neo4j_auth=("neo4j", "12345678"),
                            batch_size=32):
    """
    为案例文本生成增强的SBERT嵌入向量，
    包含案例摘要、判决结果、罪名和要素信息
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载案例ID映射
    print("加载案例ID映射...")
    with open(os.path.join(data_dir, 'case_id_to_embed_id.pkl'), 'rb') as f:
        case_id_to_embed_id = pickle.load(f)
    
    # 连接Neo4j获取增强的案例信息
    print("从Neo4j获取增强的案例信息...")
    graph = Graph(neo4j_url, auth=neo4j_auth)
    
    cypher_query = """
    MATCH (c:Case)
    OPTIONAL MATCH (c)-[:RESULTS_IN]->(term:Term)
    OPTIONAL MATCH (c)-[:HAS_CHARGE]->(charge:Charge)
    OPTIONAL MATCH (c)-[:HAS_ELEMENT]->(element:Element)
    OPTIONAL MATCH (c)-[:CITES_ARTICLE]->(article:Article)
    OPTIONAL MATCH (c)-[:HAS_MITIGATING_FACTOR|HAS_AGGRAVATING_FACTOR]->(factor:Factor)
    RETURN 
        id(c) AS case_neo4j_id, 
        c.case_id AS case_id,
        c.fact_summary AS summary,
        collect(distinct term.description) AS judgments,
        collect(distinct charge.name) AS charges,
        collect(distinct element.content) AS elements,
        collect(distinct article.article_id) AS articles,
        collect(distinct factor.content) AS factors
    """
    
    result = graph.run(cypher_query).data()
    print(f"从Neo4j获取了 {len(result)} 个案例的增强信息")
    
    # 加载SBERT模型
    print(f"加载SBERT模型: {model_name}")
    model = SentenceTransformer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"模型加载完成，使用设备: {device}")
    
    # 准备文本数据
    texts = []
    embed_ids = []
    case_ids = []
    case_info = []
    
    print("构建增强文本...")
    for item in result:
        case_id = item['case_id']
        
        # 检查该案例是否在嵌入映射中
        if case_id in case_id_to_embed_id:
            embed_id = case_id_to_embed_id[case_id]
            
            # 构建增强文本
            components = []
            
            # 添加案例摘要
            if item['summary']:
                components.append(f"案情摘要: {item['summary']}")
            
            # 添加罪名信息
            if item['charges']:
                charges_text = "，".join([c for c in item['charges'] if c])
                if charges_text:
                    components.append(f"罪名: {charges_text}")
            
            # 添加要素信息
            if item['elements']:
                elements_text = "；".join([e for e in item['elements'] if e])
                if elements_text:
                    components.append(f"构成要素: {elements_text}")
            
            # 添加判决结果
            if item['judgments']:
                judgments_text = "，".join([j for j in item['judgments'] if j])
                if judgments_text:
                    components.append(f"判决结果: {judgments_text}")
            
            # 添加法条引用
            if item['articles']:
                articles_text = "，".join([str(a) for a in item['articles'] if a])
                if articles_text:
                    components.append(f"引用法条: {articles_text}")
            
            # 添加情节信息
            if item['factors']:
                factors_text = "；".join([f for f in item['factors'] if f])
                if factors_text:
                    components.append(f"案例情节: {factors_text}")
            
            # 合并所有组件
            if components:
                text = " ".join(components)
                
                texts.append(text)
                embed_ids.append(embed_id)
                case_ids.append(case_id)
                case_info.append({
                    'case_id': case_id,
                    'summary': item['summary'],
                    'charges': item['charges'],
                    'elements': item['elements'],
                    'judgments': item['judgments'],
                    'articles': item['articles'],
                    'factors': item['factors']
                })
    
    print(f"构建了 {len(texts)} 个增强文本")
    
    # 按批次生成嵌入向量
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入"):
        batch_texts = texts[i:i+batch_size]
        embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        all_embeddings.append(embeddings.cpu().numpy())
    
    # 合并所有嵌入向量
    all_embeddings = np.vstack(all_embeddings)
    
    # 创建嵌入ID到文本嵌入的映射
    text_embeddings = {}
    for i, embed_id in enumerate(embed_ids):
        text_embeddings[embed_id] = all_embeddings[i]
    
    # 保存结果
    with open(os.path.join(output_dir, 'text_embeddings.pkl'), 'wb') as f:
        pickle.dump(text_embeddings, f)
    
    # 保存案例增强文本信息(用于查看结果)
    with open(os.path.join(output_dir, 'case_enhanced_texts.pkl'), 'wb') as f:
        enhanced_texts = {case_id: text for case_id, text in zip(case_ids, texts)}
        pickle.dump(enhanced_texts, f)
    
    # 保存案例详细信息
    with open(os.path.join(output_dir, 'case_detailed_info.pkl'), 'wb') as f:
        pickle.dump(case_info, f)
    
    # 保存元数据
    metadata = {
        'model_name': model_name,
        'embedding_dim': all_embeddings.shape[1],
        'num_cases': len(embed_ids),
        'components': ['summary', 'charges', 'elements', 'judgments', 'articles', 'factors']
    }
    with open(os.path.join(output_dir, 'text_embeddings_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"增强文本嵌入向量生成完成，维度: {all_embeddings.shape[1]}")
    print(f"共为 {len(embed_ids)} 个案例生成了文本嵌入向量")
    print(f"嵌入向量已保存到 {output_dir}/text_embeddings.pkl")
    
    return text_embeddings, metadata

if __name__ == "__main__":
    generate_text_embeddings()