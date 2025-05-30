# evaluation.py
import os
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cosine
from tqdm import tqdm
import json
from typing import List, Dict, Tuple, Set
import random
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score, ndcg_score

class EmbeddingEvaluator:
    """嵌入向量质量评估器"""
    
    def __init__(self, data_dir="./kg_data2", model_dir="./model_outputs2", 
                hybrid_dir="./hybrid_embeddings2"):
        """
        初始化评估器
        
        Args:
            data_dir: 知识图谱数据目录
            model_dir: 模型输出目录
            hybrid_dir: 混合嵌入目录
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.hybrid_dir = hybrid_dir
        
        # 加载数据和嵌入向量
        self._load_data()
        
        print("嵌入向量评估器初始化完成")
    
    def _load_data(self):
        """加载所有必要的数据和嵌入向量"""
        print("加载数据和嵌入向量...")
        
        # 初始化 hybrid_models 字典 - 添加这一行解决问题
        self.hybrid_models = {}
        
        # 加载实体和关系数据
        self.entities_df = pd.read_csv(os.path.join(self.data_dir, 'entities.csv'))
        self.relations_df = pd.read_csv(os.path.join(self.data_dir, 'relations.csv'))
        
        # 加载案例文本
        self.case_texts_df = pd.read_csv(os.path.join(self.data_dir, 'case_texts.csv'))
        
        # 加载实体ID映射
        with open(os.path.join(self.data_dir, 'entity_to_id.pkl'), 'rb') as f:
            self.entity_to_id = pickle.load(f)
        
        with open(os.path.join(self.data_dir, 'id_to_entity.pkl'), 'rb') as f:
            self.id_to_entity = pickle.load(f)
        
        # 加载案例ID映射
        with open(os.path.join(self.data_dir, 'case_id_to_embed_id.pkl'), 'rb') as f:
            self.case_id_to_embed_id = pickle.load(f)
        
        # 创建嵌入ID到案例ID的映射
        self.embed_id_to_case_id = {v: k for k, v in self.case_id_to_embed_id.items()}
        
        # 仍需加载基础嵌入向量作为后备方案
        with open(os.path.join(self.model_dir, 'entity_embeddings.pkl'), 'rb') as f:
            self.graph_embeddings = pickle.load(f)
        
        with open(os.path.join(self.model_dir, 'text_embeddings.pkl'), 'rb') as f:
            self.text_embeddings_dict = pickle.load(f)
        
        # 加载所有混合嵌入模型
        # 检查是否存在策略索引文件
        all_index_path = os.path.join(self.hybrid_dir, 'all_hybrid_embeddings_index.pkl')
        if os.path.exists(all_index_path):
            with open(all_index_path, 'rb') as f:
                all_embeddings_index = pickle.load(f)
            
            print(f"找到混合嵌入索引，加载模型...")
            
            for item in all_embeddings_index:
                model_type = item['model_type']
                strategy = item['strategy']
                embed_key = f"{model_type}_{strategy}"
                embed_file = item['embeddings_file']
                
                try:
                    with open(os.path.join(self.hybrid_dir, embed_file), 'rb') as f:
                        self.hybrid_models[embed_key] = pickle.load(f)
                    print(f"  - 已加载 {embed_key} 模型 (图权重:{item['graph_weight']:.1f}/文本权重:{item['text_weight']:.1f})")
                except Exception as e:
                    print(f"  - 加载 {embed_key} 模型失败: {str(e)}")
        
        # 构建评估嵌入类型列表
        self.embed_types = list(self.hybrid_models.keys())
        
        if not self.embed_types:
            print("警告: 未能加载任何混合嵌入模型。请检查以下内容:")
            print(f"  1. 混合嵌入目录 '{self.hybrid_dir}' 是否存在")
            print(f"  2. 是否已运行 hybrid_embeddings.py 生成混合嵌入")
            print(f"  3. 文件命名格式是否符合预期 (hybrid_embeddings_模型类型_策略名称.pkl)")
        else:
            print(f"  - 可评估的嵌入类型: {', '.join(self.embed_types)}")
            print(f"数据加载完成:")
            print(f"  - 实体数量: {len(self.entities_df)}")
            print(f"  - 知识图谱关系数量: {len(self.relations_df)}")
            print(f"  - 案例数量: {len(self.case_texts_df)}")
            print(f"  - 混合模型数量: {len(self.hybrid_models)}")
        
        # 构建方便查询的数据结构
        self._build_query_structures()
    
    def _build_query_structures(self):
        """构建用于快速查询的数据结构"""
        print("构建查询数据结构...")
        
        # 创建案例ID到罪名的映射
        self.case_to_charges = {}
        # 创建案例ID到法条的映射
        self.case_to_articles = {}
        # 创建罪名到案例列表的映射
        self.charge_to_cases = {}
        # 创建法条到案例列表的映射
        self.article_to_cases = {}
        # 创建案例ID到情节的映射
        self.case_to_factors = {}
        
        # 处理关系数据创建映射
        for _, row in tqdm(self.relations_df.iterrows(), desc="处理关系数据", total=len(self.relations_df)):
            head_id = row['head_id']
            tail_id = row['tail_id']
            relation = row['relation']
            head_type = row['head_type']
            tail_type = row['tail_type']
            
            # 查找头节点的实体ID
            head_embed_id = None
            if head_id in self.entity_to_id:
                head_embed_id = self.entity_to_id[head_id]
            
            # 查找尾节点的实体ID
            tail_embed_id = None
            if tail_id in self.entity_to_id:
                tail_embed_id = self.entity_to_id[tail_id]
            
            # 如果头节点是Case
            if head_type == 'Case' and head_embed_id is not None:
                # 获取案例ID
                case_id = None
                for case_id_str, embed_id in self.case_id_to_embed_id.items():
                    if embed_id == head_embed_id:
                        case_id = case_id_str
                        break
                
                if case_id:
                    # 处理罪名关系
                    if relation == 'HAS_CHARGE' and tail_type == 'Charge':
                        # 获取罪名
                        charge_row = self.entities_df[(self.entities_df['id'] == tail_id) & (self.entities_df['type'] == 'Charge')]
                        if not charge_row.empty:
                            charge_name = charge_row.iloc[0]['name']
                            if charge_name:
                                # 更新案例到罪名的映射
                                if case_id not in self.case_to_charges:
                                    self.case_to_charges[case_id] = []
                                self.case_to_charges[case_id].append(charge_name)
                                
                                # 更新罪名到案例的映射
                                if charge_name not in self.charge_to_cases:
                                    self.charge_to_cases[charge_name] = []
                                self.charge_to_cases[charge_name].append(case_id)
                    
                    # 处理法条关系
                    elif relation == 'CITES_ARTICLE' and tail_type == 'Article':
                        # 获取法条ID
                        article_row = self.entities_df[(self.entities_df['id'] == tail_id) & (self.entities_df['type'] == 'Article')]
                        if not article_row.empty:
                            article_id = article_row.iloc[0]['article_id']
                            if article_id:
                                # 更新案例到法条的映射
                                if case_id not in self.case_to_articles:
                                    self.case_to_articles[case_id] = []
                                self.case_to_articles[case_id].append(article_id)
                                
                                # 更新法条到案例的映射
                                if article_id not in self.article_to_cases:
                                    self.article_to_cases[article_id] = []
                                self.article_to_cases[article_id].append(case_id)
                    
                    # 处理情节关系
                    elif (relation == 'HAS_MITIGATING_FACTOR' or relation == 'HAS_AGGRAVATING_FACTOR') and tail_type == 'Factor':
                        # 获取情节内容
                        factor_row = self.entities_df[(self.entities_df['id'] == tail_id) & (self.entities_df['type'] == 'Factor')]
                        if not factor_row.empty:
                            factor_content = factor_row.iloc[0]['content']
                            factor_type = factor_row.iloc[0]['factor_type']
                            if factor_content:
                                # 更新案例到情节的映射
                                if case_id not in self.case_to_factors:
                                    self.case_to_factors[case_id] = []
                                self.case_to_factors[case_id].append((factor_content, factor_type))
        
        print(f"查询数据结构构建完成:")
        print(f"  - 案例-罪名映射数量: {len(self.case_to_charges)}")
        print(f"  - 案例-法条映射数量: {len(self.case_to_articles)}")
        print(f"  - 罪名种类数量: {len(self.charge_to_cases)}")
        print(f"  - 法条种类数量: {len(self.article_to_cases)}")
        print(f"  - 案例-情节映射数量: {len(self.case_to_factors)}")
    
    def create_evaluation_pairs(self, num_positive=500, num_negative=500) -> Tuple[List, List]:
        """
        创建评估用的正负样本对
        
        正样本: 具有相同罪名和共享多个法条的案例对
        负样本: 具有不同罪名和不同法条的案例对
        
        Args:
            num_positive: 期望的正样本对数量
            num_negative: 期望的负样本对数量
            
        Returns:
            正样本列表和负样本列表，每个样本是一个(案例ID1, 案例ID2)元组
        """
        print(f"创建评估样本对...")
        
        # 创建正样本
        positive_pairs = []
        
        # 基于相同罪名和共享法条创建正样本
        for charge, cases in tqdm(self.charge_to_cases.items(), desc="创建正样本"):
            # 如果该罪名下的案例不足2个，跳过
            if len(cases) < 2:
                continue
            
            # 对于每个案例，找出与之共享多个法条的其他案例
            for i, case1 in enumerate(cases):
                # 获取第一个案例引用的法条
                articles1 = set(self.case_to_articles.get(case1, []))
                if not articles1:
                    continue
                
                candidates = []
                for j, case2 in enumerate(cases[i+1:], i+1):
                    # 获取第二个案例引用的法条
                    articles2 = set(self.case_to_articles.get(case2, []))
                    if not articles2:
                        continue
                    
                    # 计算共享法条数量
                    common_articles = articles1.intersection(articles2)
                    if len(common_articles) >= 1:  # 至少共享1个法条
                        # 计算相似性分数: 共享法条数量
                        similarity_score = len(common_articles)
                        candidates.append((case2, similarity_score))
                
                # 选择相似性最高的案例配对
                candidates.sort(key=lambda x: x[1], reverse=True)
                if candidates:
                    positive_pairs.append((case1, candidates[0][0]))
                
                # 如果已收集足够的正样本，停止
                if len(positive_pairs) >= num_positive:
                    break
            
            # 如果已收集足够的正样本，停止
            if len(positive_pairs) >= num_positive:
                break
        
        # 如果通过罪名和法条没有收集到足够的正样本，尝试通过共享情节创建
        if len(positive_pairs) < num_positive:
            # 基于共享情节的案例对
            all_cases = list(self.case_to_factors.keys())
            for i, case1 in enumerate(tqdm(all_cases, desc="基于情节创建正样本")):
                factors1 = set([f[0] for f in self.case_to_factors.get(case1, [])])
                if not factors1:
                    continue
                
                for case2 in all_cases[i+1:]:
                    # 检查是否已经在positive_pairs中
                    if (case1, case2) in positive_pairs or (case2, case1) in positive_pairs:
                        continue
                    
                    factors2 = set([f[0] for f in self.case_to_factors.get(case2, [])])
                    if not factors2:
                        continue
                    
                    # 计算共享情节数量
                    common_factors = factors1.intersection(factors2)
                    if len(common_factors) >= 2:  # 至少共享2个情节
                        positive_pairs.append((case1, case2))
                        
                        # 如果已收集足够的正样本，停止
                        if len(positive_pairs) >= num_positive:
                            break
                
                # 如果已收集足够的正样本，停止
                if len(positive_pairs) >= num_positive:
                    break
        
        # 裁剪正样本到指定数量
        positive_pairs = positive_pairs[:num_positive]
        
        # 创建负样本
        negative_pairs = []
        
        # 获取按罪名分组的案例
        charge_groups = {}
        for case_id, charges in self.case_to_charges.items():
            for charge in charges:
                if charge not in charge_groups:
                    charge_groups[charge] = []
                charge_groups[charge].append(case_id)
        
        # 获取所有罪名
        all_charges = list(charge_groups.keys())
        
        # 基于不同罪名创建负样本
        for _ in tqdm(range(num_negative * 2), desc="创建负样本"):  # 尝试创建更多样本，以防部分无效
            # 随机选择两个不同的罪名
            if len(all_charges) < 2:
                break
            
            charge1, charge2 = random.sample(all_charges, 2)
            
            # 确保两个罪名组都有案例
            if not charge_groups[charge1] or not charge_groups[charge2]:
                continue
            
            # 随机选择每个罪名组中的一个案例
            case1 = random.choice(charge_groups[charge1])
            case2 = random.choice(charge_groups[charge2])
            
            # 检查是否已经在negative_pairs中
            if (case1, case2) in negative_pairs or (case2, case1) in negative_pairs:
                continue
            
            # 检查是否与正样本重叠 (不应该)
            if (case1, case2) in positive_pairs or (case2, case1) in positive_pairs:
                continue
            
            # 获取第一个案例的法条和情节
            articles1 = set(self.case_to_articles.get(case1, []))
            factors1 = set([f[0] for f in self.case_to_factors.get(case1, [])])

            # 获取第二个案例的法条和情节
            articles2 = set(self.case_to_articles.get(case2, []))
            factors2 = set([f[0] for f in self.case_to_factors.get(case2, [])])

            # 如果两个案例“看起来很像”（但罪名不同），才认为是 hard negative
            if len(articles1.intersection(articles2)) >= 1 or \
            len(factors1.intersection(factors2)) >= 1:
                negative_pairs.append((case1, case2))
            
            # negative_pairs.append((case1, case2))

            
            # 如果已收集足够的负样本，停止
            if len(negative_pairs) >= num_negative:
                break
        
        # 裁剪负样本到指定数量
        negative_pairs = negative_pairs[:num_negative]
        
        print(f"创建了 {len(positive_pairs)} 个正样本对和 {len(negative_pairs)} 个负样本对")
        
        return positive_pairs, negative_pairs
    
    def compute_similarity(self, case1: str, case2: str, embed_type: str) -> float:
        """
        计算两个案例之间的余弦相似度
        
        Args:
            case1: 第一个案例ID
            case2: 第二个案例ID
            embed_type: 嵌入类型 (仅限混合模型名称)
            
        Returns:
            余弦相似度 (1表示完全相似，0表示正交，-1表示完全相反)
        """
        # 获取案例的嵌入ID
        embed_id1 = self.case_id_to_embed_id.get(case1)
        embed_id2 = self.case_id_to_embed_id.get(case2)
        
        if embed_id1 is None or embed_id2 is None:
            return 0.0
        
        # 使用混合模型中的嵌入向量
        if embed_type in self.hybrid_models:
            vec1 = self.hybrid_models[embed_type][embed_id1]
            vec2 = self.hybrid_models[embed_type][embed_id2]
        else:
            raise ValueError(f"未知的嵌入类型: {embed_type}")
        
        # 计算余弦相似度 (1 - 余弦距离)
        similarity = 1 - cosine(vec1, vec2)
        return similarity
    
    def evaluate_embeddings(self, positive_pairs, negative_pairs):
        """
        评估不同嵌入方法的效果
        
        Args:
            positive_pairs: 正样本对列表
            negative_pairs: 负样本对列表
            
        Returns:
            包含评估结果的字典
        """
        results = {}
        
        for embed_type in self.embed_types:
            print(f"\n评估 {embed_type} 嵌入...")
            
            # 计算正样本对的相似度
            pos_similarities = []
            for case1, case2 in tqdm(positive_pairs, desc=f"{embed_type}嵌入-正样本"):
                similarity = self.compute_similarity(case1, case2, embed_type)
                pos_similarities.append(similarity)
            
            # 计算负样本对的相似度
            neg_similarities = []
            for case1, case2 in tqdm(negative_pairs, desc=f"{embed_type}嵌入-负样本"):
                similarity = self.compute_similarity(case1, case2, embed_type)
                neg_similarities.append(similarity)
            
            # 计算平均相似度
            pos_mean = np.mean(pos_similarities)
            neg_mean = np.mean(neg_similarities)
            
            # 计算区分度 (正样本和负样本平均相似度之差)
            discrimination = pos_mean - neg_mean
            
            # 构造标签和分数
            y_true = [1] * len(pos_similarities) + [0] * len(neg_similarities)
            y_score = pos_similarities + neg_similarities

            # ROC AUC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            # Average Precision (AP)
            average_precision = average_precision_score(y_true, y_score)

            # NDCG@10 —— 需要把数据放进二维数组
            y_true_array = np.array([y_true])
            y_score_array = np.array([y_score])
            ndcg_at_10 = ndcg_score(y_true_array, y_score_array, k=10)
            
            # 保存结果
            results[embed_type] = {
                'positive_mean': pos_mean,
                'negative_mean': neg_mean,
                'discrimination': discrimination,
                'positive_std': np.std(pos_similarities),
                'negative_std': np.std(neg_similarities),
                'roc_auc': roc_auc,
                'average_precision': average_precision,
                'ndcg@10': ndcg_at_10,
                'positive_samples': len(pos_similarities),
                'negative_samples': len(neg_similarities)
            }
            
            print(f"  - 正样本平均相似度: {pos_mean:.4f} ± {np.std(pos_similarities):.4f}")
            print(f"  - 负样本平均相似度: {neg_mean:.4f} ± {np.std(neg_similarities):.4f}")
            print(f"  - 区分度: {discrimination:.4f}")
            print(f"  - ROC AUC: {roc_auc:.4f}")
            print(f"  - Average Precision: {average_precision:.4f}")
            print(f"  - NDCG@10: {ndcg_at_10:.4f}")
        
        return results

def run_evaluation(data_dir="./kg_data2", model_dir="./model_outputs2", 
                  hybrid_dir="./hybrid_embeddings2", output_dir="./evaluation_results2"):
    """运行完整的评估流程"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("初始化嵌入向量评估器...")
    evaluator = EmbeddingEvaluator(data_dir, model_dir, hybrid_dir)
    
    # 创建评估样本对
    positive_pairs, negative_pairs = evaluator.create_evaluation_pairs(num_positive=300, num_negative=300)
    
    # 保存评估样本对
    with open(os.path.join(output_dir, 'evaluation_pairs.json'), 'w') as f:
        json.dump({
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs
        }, f, indent=2)
    
    # 评估嵌入向量
    results = evaluator.evaluate_embeddings(positive_pairs, negative_pairs)
    
    # 保存评估结果
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        # 将numpy数组转换为列表以便JSON序列化
        for embed_type in results:
            for key in results[embed_type]:
                if isinstance(results[embed_type][key], np.ndarray):
                    results[embed_type][key] = results[embed_type][key].tolist()
        
        json.dump(results, f, indent=2)
    
    print(f"\n评估完成，结果已保存到 {output_dir} 目录")
    
    # 按roc_auc排序的结果
    sorted_methods = sorted(results.keys(), key=lambda x: results[x]['roc_auc'], reverse=True)
    
    # 打印表格形式的结果（简单格式）
    print("\n=== 嵌入方法评估结果 (按ROC AUC排序) ===")
    print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<25} {:<15}".format(
        "嵌入方法", "正样本相似度", "负样本相似度", "区分度", "ROC AUC", "average_precision", "NDCG@10"))
    print("-" * 95)
    
    for method in sorted_methods:
        result = results[method]
        pos_sim = "{:.4f} ± {:.4f}".format(result['positive_mean'], result['positive_std'])
        neg_sim = "{:.4f} ± {:.4f}".format(result['negative_mean'], result['negative_std'])
        print("{:<15} {:<25} {:<25} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            method, pos_sim, neg_sim, result['discrimination'], result['roc_auc'], result['average_precision'], result['ndcg@10']))
    
    # 找出最佳嵌入方法
    best_method = max(results, key=lambda x: results[x]['roc_auc'])
    print(f"\n最佳嵌入方法: {best_method}, roc_auc: {results[best_method]['roc_auc']:.4f}")
    
    return results

if __name__ == "__main__":
    # 可以通过命令行参数指定目录
    import argparse
    
    parser = argparse.ArgumentParser(description="评估不同嵌入方法的效果")
    parser.add_argument("--data_dir", default="./kg_data2", help="知识图谱数据目录")
    parser.add_argument("--model_dir", default="./model_outputs2", help="模型输出目录")
    parser.add_argument("--hybrid_dir", default="./hybrid_embeddings2", help="混合嵌入目录")
    parser.add_argument("--output_dir", default="./evaluation_results2", help="评估结果输出目录")
    
    args = parser.parse_args()
    
    run_evaluation(args.data_dir, args.model_dir, args.hybrid_dir, args.output_dir)