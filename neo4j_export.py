# neo4j_export.py (改进版)
import pandas as pd
from py2neo import Graph
import pickle
import os

def export_kg_data(output_dir="./kg_data3"):
    """从Neo4j导出知识图谱数据用于训练嵌入模型"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 连接Neo4j数据库
    graph = Graph("neo4j://10.61.2.143:7688", auth=("neo4j", "12345678"))
    
    print("正在从Neo4j导出数据...")
    
    # 使用更全面的查询获取所有节点属性
    query_entities = """
    MATCH (n) 
    RETURN DISTINCT 
        id(n) as id, 
        labels(n)[0] as type, 
        // Case属性
        n.case_id as case_id,
        n.fact as fact,
        n.fact_summary as fact_summary,
        // Charge属性
        n.name as name,
        // Article属性
        n.article_id as article_id, 
        // Element属性
        n.element_id as element_id,
        n.content as content,
        // Factor属性
        n.factor_id as factor_id,
        n.factor_type as factor_type,
        // Term属性
        n.term_id as term_id,
        n.death_penalty as death_penalty,
        n.description as description,
        n.imprisonment as imprisonment,
        n.life_imprisonment as life_imprisonment,
        n.punishment_money as punishment_money
    """
    
    entities_df = pd.DataFrame(graph.run(query_entities).data())
    print(f"导出了 {len(entities_df)} 个实体节点")
    
    # 获取所有关系 (保持不变)
    query_relations = """
    MATCH (h)-[r]->(t)
    RETURN id(h) as head_id, id(t) as tail_id, type(r) as relation, 
           labels(h)[0] as head_type, labels(t)[0] as tail_type
    """
    
    relations_df = pd.DataFrame(graph.run(query_relations).data())
    print(f"导出了 {len(relations_df)} 个关系")
    
    # 单独查询案例文本仍然保留，便于文本处理
    query_case_texts = """
    MATCH (c:Case)
    RETURN c.case_id as case_id, c.fact_summary as summary, c.fact as full_text
    """
    
    case_texts_df = pd.DataFrame(graph.run(query_case_texts).data())
    print(f"导出了 {len(case_texts_df)} 个案例文本")
    
    # 构建实体和关系映射 (其余处理逻辑不变)
    entity_to_id = {row['id']: idx for idx, row in enumerate(entities_df.to_dict('records'))}
    relation_to_id = {rel: idx for idx, rel in enumerate(relations_df['relation'].unique())}
    
    # 转换为TransE可用的三元组格式
    triples = []
    for _, row in relations_df.iterrows():
        if row['head_id'] in entity_to_id and row['tail_id'] in entity_to_id:
            h = entity_to_id[row['head_id']]
            r = relation_to_id[row['relation']]
            t = entity_to_id[row['tail_id']]
            triples.append((h, r, t))
    
    # 保存数据
    with open(os.path.join(output_dir, 'entity_to_id.pkl'), 'wb') as f:
        pickle.dump(entity_to_id, f)
    
    with open(os.path.join(output_dir, 'relation_to_id.pkl'), 'wb') as f:
        pickle.dump(relation_to_id, f)
    
    with open(os.path.join(output_dir, 'id_to_entity.pkl'), 'wb') as f:
        pickle.dump({v: k for k, v in entity_to_id.items()}, f)
    
    with open(os.path.join(output_dir, 'id_to_relation.pkl'), 'wb') as f:
        pickle.dump({v: k for k, v in relation_to_id.items()}, f)
    
    with open(os.path.join(output_dir, 'triples.pkl'), 'wb') as f:
        pickle.dump(triples, f)
    
    # 保存实体和案例文本数据
    entities_df.to_csv(os.path.join(output_dir, 'entities.csv'), index=False)
    relations_df.to_csv(os.path.join(output_dir, 'relations.csv'), index=False)
    case_texts_df.to_csv(os.path.join(output_dir, 'case_texts.csv'), index=False)
    
    # 生成 case_id ➔ 文本 的字典并保存为pkl
    case_texts_dict = {}
    for _, row in case_texts_df.iterrows():
        case_id = row['case_id']
        text = row['summary']  # 可以选择用 summary
        case_texts_dict[case_id] = text

    with open(os.path.join(output_dir, 'case_texts.pkl'), 'wb') as f:
        pickle.dump(case_texts_dict, f)

    print(f"已生成案例文本字典，共 {len(case_texts_dict)} 条")
    
    # 为方便查询，按节点类型分别保存数据
    for node_type in entities_df['type'].unique():
        type_df = entities_df[entities_df['type'] == node_type]
        type_df.to_csv(os.path.join(output_dir, f'{node_type.lower()}_nodes.csv'), index=False)
    
    # 为方便理解，保存映射表
    entity_map_df = pd.DataFrame({
        'neo4j_id': list(entity_to_id.keys()),
        'embed_id': list(entity_to_id.values())
    })
    entity_map_df.to_csv(os.path.join(output_dir, 'entity_mapping.csv'), index=False)
    
    relation_map_df = pd.DataFrame({
        'relation': list(relation_to_id.keys()),
        'relation_id': list(relation_to_id.values())
    })
    relation_map_df.to_csv(os.path.join(output_dir, 'relation_mapping.csv'), index=False)
    
    # 保存统计信息
    stats = {
        'n_entities': len(entity_to_id),
        'n_relations': len(relation_to_id),
        'n_triples': len(triples),
        'entity_types': entities_df['type'].value_counts().to_dict(),
        'relation_types': relations_df['relation'].value_counts().to_dict()
    }
    
    with open(os.path.join(output_dir, 'kg_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    # 将case_id映射到embed_id (用于后续检索)
    case_id_to_embed_id = {}
    for _, row in entities_df[entities_df['type']=='Case'].iterrows():
        if row['case_id'] and row['id'] in entity_to_id:
            case_id_to_embed_id[row['case_id']] = entity_to_id[row['id']]
    
    with open(os.path.join(output_dir, 'case_id_to_embed_id.pkl'), 'wb') as f:
        pickle.dump(case_id_to_embed_id, f)
    
    print(f"数据导出完成，已保存到 {output_dir} 目录")
    return stats

if __name__ == "__main__":
    stats = export_kg_data()
    print("\n知识图谱统计信息:")
    print(f"实体数量: {stats['n_entities']}")
    print(f"关系类型数量: {stats['n_relations']}")
    print(f"三元组数量: {stats['n_triples']}")
    print("\n实体类型分布:")
    for entity_type, count in stats['entity_types'].items():
        print(f"  {entity_type}: {count}")
    print("\n关系类型分布:")
    for relation_type, count in stats['relation_types'].items():
        print(f"  {relation_type}: {count}")