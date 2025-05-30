# faiss_evaluation.py
import os
import faiss
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from collections import defaultdict
import time
from typing import List, Dict, Set, Tuple

class FAISSRetriever:
    """基于FAISS的案例检索系统"""
    
    def __init__(self, embedding_type: str, data_dir: str = "./kg_data2", 
                model_dir: str = "./model_outputs2", hybrid_dir: str = "./hybrid_embeddings2"):
        """
        初始化检索器
        
        Args:
            embedding_type: 使用的嵌入类型，可以是以下格式:
                - 基础策略: 'graph_only', 'text_only', 'balanced', 'graph_heavy', 'text_heavy'
                - 模型特定策略: '{model_type}_{strategy}'，例如 'transe_balanced', 'rotate_graph_heavy'
            data_dir: 知识图谱数据目录
            model_dir: 模型输出目录
            hybrid_dir: 混合嵌入目录
        """
        self.embedding_type = embedding_type
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.hybrid_dir = hybrid_dir
        
        # 解析嵌入类型
        self.model_type, self.strategy = self._parse_embedding_type(embedding_type)
        
        # 加载数据
        self._load_data()
        
        # 构建索引
        self.build_index()
        
        print(f"FAISS检索器 ({embedding_type}) 初始化完成，包含 {len(self.case_ids)} 个案例")
    
    def _parse_embedding_type(self, embedding_type: str) -> Tuple[str, str]:
        """
        解析嵌入类型，提取模型类型和策略名称
        
        Args:
            embedding_type: 嵌入类型字符串

        Returns:
            (model_type, strategy): 模型类型和策略的元组
        """
        model_types = ["transe", "transr", "rescal", "rotate", "distmult"]
        strategies = ["graph_only", "text_only", "balanced", "graph_heavy", "text_heavy"]
        
        # 检查是否是模型特定策略格式 (model_type_strategy)
        parts = embedding_type.split('_', 1)
        
        if len(parts) > 1 and parts[0] in model_types:
            return parts[0], parts[1]
        
        # 检查是否是基础策略
        if embedding_type in strategies:
            return "transe", embedding_type  # 默认使用transe作为基础模型
        
        # 未识别格式，保持原样
        return "", embedding_type
    
    def _load_data(self):
        """加载数据和嵌入向量"""
        # 加载案例ID映射
        with open(os.path.join(self.data_dir, 'case_id_to_embed_id.pkl'), 'rb') as f:
            self.case_id_to_embed_id = pickle.load(f)
        
        # 创建嵌入ID到案例ID的反向映射
        self.embed_id_to_case_id = {v: k for k, v in self.case_id_to_embed_id.items()}
        
        # 构建嵌入文件路径
        if self.model_type:
            # 新格式: 模型特定策略
            emb_path = os.path.join(self.hybrid_dir, f'hybrid_embeddings_{self.model_type}_{self.strategy}.pkl')
        else:
            # 旧格式: 基础策略
            emb_path = os.path.join(self.hybrid_dir, f'hybrid_embeddings_{self.embedding_type}.pkl')
        
        # 尝试加载嵌入
        try:
            with open(emb_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"已加载 {self.embedding_type} 混合嵌入，形状: {self.embeddings.shape}")
        except FileNotFoundError:
            raise ValueError(f"找不到嵌入文件: {emb_path}")
        
        # 获取案例嵌入和ID
        self.case_ids = []
        self.case_embedding_indices = []
        
        for case_id, embed_id in self.case_id_to_embed_id.items():
            if embed_id < len(self.embeddings):
                self.case_ids.append(case_id)
                self.case_embedding_indices.append(embed_id)
        
        # 提取案例嵌入
        self.case_embeddings = self.embeddings[self.case_embedding_indices]
        
        print(f"已加载 {len(self.case_ids)} 个案例嵌入向量")
    
    def build_index(self):
        """构建FAISS索引"""
        # 选择合适的索引类型，这里使用L2距离的精确搜索
        self.dimension = self.case_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # 转换为float32格式并添加到索引
        embeddings_float32 = np.array(self.case_embeddings).astype('float32')
        self.index.add(embeddings_float32)
        
        print(f"FAISS索引构建完成，包含 {self.index.ntotal} 个向量，维度 {self.dimension}")
        
        
    def rerank(self, query_id: str, initial_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        使用 SBERT / LLM 对 Top-K 检索结果重新排序（可选加 query-candidate 相似度）
        """
        # 假设你已经加载好 case_texts[case_id]，可以查到文本内容
        with open(os.path.join(self.data_dir, 'case_texts.pkl'), 'rb') as f:
            case_texts = pickle.load(f)

        query_text = case_texts.get(query_id, "")
        if not query_text:
            return initial_results  # fallback
        
        candidates = []
        for case_id, _ in initial_results:
            text = case_texts.get(case_id, "")
            if not text:
                continue
            candidates.append((case_id, text))

        # 用 SBERT 模型做相似度打分
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("/data1/lxy/project/MODEL/text2vec-base-chinese")
        print("使用 SBERT 模型进行相似度计算...")

        texts = [text for _, text in candidates]
        embeddings = model.encode([query_text] + texts, convert_to_tensor=True)
        
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]

        similarity_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]  # shape: (N,)
        
        # 拼回 (case_id, 相似度)
        reranked = sorted(
            [(candidates[i][0], float(similarity_scores[i])) for i in range(len(candidates))],
            key=lambda x: x[1], reverse=True
        )
        
        return reranked

    
    def retrieve(self, query_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        检索与查询案例最相似的Top-K案例
        
        Args:
            query_id: 查询案例ID
            k: 返回的相似案例数量
            
        Returns:
            按相似度排序的(案例ID, 相似度得分)列表
        """
        if query_id not in self.case_id_to_embed_id:
            print(f"警告: 查询案例ID {query_id} 未找到嵌入向量")
            return []
        
        # 获取查询案例的嵌入向量
        query_embed_id = self.case_id_to_embed_id[query_id]
        query_vector = self.embeddings[query_embed_id].reshape(1, -1).astype('float32')
        
        # 执行检索 (使用L2距离，值越小越相似)
        distances, indices = self.index.search(query_vector, k+1)  # +1是因为可能包含自身
        
        # 转换为相似度得分 (1 / (1 + 距离))
        similarities = 1.0 / (1.0 + distances[0])
        
        # 构建结果，排除查询案例自身
        results = []
        for idx, sim in zip(indices[0], similarities):
            if idx < len(self.case_embedding_indices):
                result_embed_id = self.case_embedding_indices[idx]
                if result_embed_id in self.embed_id_to_case_id:
                    result_case_id = self.embed_id_to_case_id[result_embed_id]
                    if result_case_id != query_id:  # 排除自身
                        results.append((result_case_id, float(sim)))
        
        # 确保返回正确数量的结果
        return results[:k]


class RetrieverEvaluator:
    """检索系统评估器"""
    
    def __init__(self, data_dir: str = "./kg_data2", output_dir: str = "./retrieval_evaluation2", 
                term_similarity_threshold: float = 0.2, relevance_logic: str = "OR"):
        """
        初始化评估器
        
        Args:
            data_dir: 知识图谱数据目录
            output_dir: 评估结果输出目录
            term_similarity_threshold: 量刑相似度阈值（相对差异），默认0.2表示20%以内视为相似
            relevance_logic: 相关性判断逻辑，"AND"表示同时满足所有条件，"OR"表示满足任一条件
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.term_similarity_threshold = term_similarity_threshold
        self.relevance_logic = relevance_logic.upper()
        
        if self.relevance_logic not in ["AND", "OR"]:
            print(f"警告: 未知的相关性逻辑 '{relevance_logic}'，默认使用 'OR'")
            self.relevance_logic = "OR"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        self._load_data()
        
        print("检索评估器初始化完成")
        print(f"相关性判断标准: 罪名相同 {self.relevance_logic} 量刑相似(阈值={self.term_similarity_threshold*100:.0f}%)")
    
    def _load_data(self):
        """加载案例元数据(案由、量刑等)"""
        # 加载实体和关系数据
        self.entities_df = pd.read_csv(os.path.join(self.data_dir, 'entities.csv'))
        self.relations_df = pd.read_csv(os.path.join(self.data_dir, 'relations.csv'))
        
        # 构建查询所需的数据结构
        print("构建案例元数据索引...")
        
        # 构建案例到案由的映射 (使用罪名作为案由)
        self.case_to_causes = defaultdict(set)
        # 构建案例到量刑的映射 (存储刑期)
        self.case_to_terms = {}
        
        # 加载案例ID映射
        with open(os.path.join(self.data_dir, 'case_id_to_embed_id.pkl'), 'rb') as f:
            self.case_id_to_embed_id = pickle.load(f)
        
        # 创建Neo4j ID到案例ID的映射
        self.neo4j_id_to_case_id = {}
        
        # 找出所有Case节点
        case_entities = self.entities_df[self.entities_df['type'] == 'Case']
        for _, row in case_entities.iterrows():
            neo4j_id = row['id']
            case_id = row['case_id']
            if pd.notna(case_id):
                self.neo4j_id_to_case_id[neo4j_id] = case_id
        
        # 处理关系数据
        for _, row in tqdm(self.relations_df.iterrows(), desc="处理案例关系数据"):
            head_id = row['head_id']
            tail_id = row['tail_id']
            relation = row['relation']
            head_type = row['head_type']
            tail_type = row['tail_type']
            
            # 如果头节点是案例
            if head_type == 'Case' and head_id in self.neo4j_id_to_case_id:
                case_id = self.neo4j_id_to_case_id[head_id]
                
                # 处理罪名关系
                if relation == 'HAS_CHARGE' and tail_type == 'Charge':
                    charge_row = self.entities_df[(self.entities_df['id'] == tail_id) & (self.entities_df['type'] == 'Charge')]
                    if not charge_row.empty:
                        charge_name = charge_row.iloc[0]['name']
                        if pd.notna(charge_name):
                            self.case_to_causes[case_id].add(charge_name)
                
                # 处理量刑关系
                elif relation == 'RESULTS_IN' and tail_type == 'Term':
                    term_row = self.entities_df[(self.entities_df['id'] == tail_id) & (self.entities_df['type'] == 'Term')]
                    if not term_row.empty:
                        # 尝试获取不同类型的刑期信息
                        imprisonment = term_row.iloc[0].get('imprisonment')
                        life_imprisonment = term_row.iloc[0].get('life_imprisonment')
                        death_penalty = term_row.iloc[0].get('death_penalty')
                        
                        # 按优先级使用刑期信息
                        if pd.notna(imprisonment) and imprisonment > 0:
                            # 有期徒刑（月）
                            self.case_to_terms[case_id] = {'type': 'imprisonment', 'value': float(imprisonment)}
                        elif pd.notna(life_imprisonment) and life_imprisonment:
                            # 无期徒刑
                            self.case_to_terms[case_id] = {'type': 'life_imprisonment', 'value': float('inf')}
                        elif pd.notna(death_penalty) and death_penalty:
                            # 死刑
                            self.case_to_terms[case_id] = {'type': 'death_penalty', 'value': float('inf')}
        
        print(f"案例元数据构建完成:")
        print(f"  - 案例-案由映射数量: {len(self.case_to_causes)}")
        print(f"  - 案例-量刑映射数量: {len(self.case_to_terms)}")
    
    def is_causes_similar(self, query_id: str, retrieved_id: str) -> bool:
        """判断两案例的罪名是否相同"""
        query_causes = self.case_to_causes.get(query_id, set())
        retrieved_causes = self.case_to_causes.get(retrieved_id, set())
        
        return bool(query_causes and retrieved_causes and query_causes.intersection(retrieved_causes))
    
    def is_terms_similar(self, query_id: str, retrieved_id: str) -> bool:
        """判断两案例的量刑是否相似"""
        # 获取量刑信息
        query_term = self.case_to_terms.get(query_id)
        retrieved_term = self.case_to_terms.get(retrieved_id)
        
        # 如果任一案例缺少量刑信息，则认为不相似
        if not query_term or not retrieved_term:
            return False
        
        # 如果量刑类型不同，则不相似
        if query_term['type'] != retrieved_term['type']:
            return False
        
        # 对于无期徒刑或死刑，类型相同即视为相似
        if query_term['type'] in ['life_imprisonment', 'death_penalty']:
            return True
        
        # 对于有期徒刑，比较刑期差异
        if query_term['type'] == 'imprisonment':
            q_value = query_term['value']
            r_value = retrieved_term['value']
            
            # 计算相对差异
            if q_value == 0 and r_value == 0:
                return True
            elif q_value == 0 or r_value == 0:
                return False
            
            relative_diff = abs(q_value - r_value) / max(q_value, r_value)
            return relative_diff <= self.term_similarity_threshold
        
        return False
    
    def is_relevant(self, query_id: str, retrieved_id: str) -> bool:
        """
        判断检索到的案例是否相关
        
        Args:
            query_id: 查询案例ID
            retrieved_id: 检索到的案例ID
            
        Returns:
            布尔值表示是否相关
        """
        # 检查罪名相似性
        causes_similar = self.is_causes_similar(query_id, retrieved_id)
        
        # 检查量刑相似性
        terms_similar = self.is_terms_similar(query_id, retrieved_id)
        
        # 根据指定的逻辑组合结果
        if self.relevance_logic == "AND":
            return causes_similar and terms_similar
        else:  # OR
            return causes_similar or terms_similar
    
    
    def evaluate_retriever(self, retriever: FAISSRetriever, test_samples: List[str] = None, 
                          k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict:
        """
        评估检索器在不同K值下的表现
        
        Args:
            retriever: 要评估的检索器
            test_samples: 测试样本ID列表，如果为None则随机选择100个案例
            k_values: 评估的K值列表
            
        Returns:
            包含评估结果的字典
        """
        if test_samples is None:
            # 随机选择100个存在于数据集中的案例作为测试样本
            all_valid_cases = list(set(self.case_to_causes.keys()) & 
                                  set(retriever.case_id_to_embed_id.keys()))
            if len(all_valid_cases) > 100:
                test_samples = np.random.choice(all_valid_cases, 100, replace=False).tolist()
            else:
                test_samples = all_valid_cases
        
        print(f"评估检索器 {retriever.embedding_type}, 测试样本数: {len(test_samples)}")
        
        # 评估每个测试样本
        relevance_results = []
        first_relevant_ranks = []
        query_times = []
        
        for query_id in tqdm(test_samples, desc=f"评估案例检索"):
            start_time = time.time()
            # 获取检索结果 (最大K值)
            retrieved_cases = retriever.retrieve(query_id, k=max(k_values))
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # 判断每个检索结果是否相关
            relevance = []
            first_relevant_rank = float('inf')
            
            for rank, (case_id, _) in enumerate(retrieved_cases, 1):
                is_rel = self.is_relevant(query_id, case_id)
                relevance.append(is_rel)
                
                # 记录第一个相关案例的排名
                if is_rel and rank < first_relevant_rank:
                    first_relevant_rank = rank
            
            relevance_results.append(relevance)
            
            # 如果有相关案例，记录用于计算MRR
            if first_relevant_rank < float('inf'):
                first_relevant_ranks.append(first_relevant_rank)
            else:
                first_relevant_ranks.append(0)  # 无相关案例，排名为0 (对MRR贡献为0)
        
        # 计算每个K值的指标
        metrics = self.calculate_metrics(relevance_results, first_relevant_ranks, k_values)
        metrics['query_time'] = np.mean(query_times)
        
        # 保存评估结果
        result_file = os.path.join(self.output_dir, f'{retriever.embedding_type}_evaluation.pkl')
        with open(result_file, 'wb') as f:
            pickle.dump(metrics, f)
        
        # 打印总结
        self.print_summary(retriever.embedding_type, metrics, k_values)
        
        return metrics
    
    def calculate_metrics(self, relevance_results: List[List[bool]], 
                         first_relevant_ranks: List[int], 
                         k_values: List[int]) -> Dict:
        """
        计算各项指标: Hit@K, MRR
        
        Args:
            relevance_results: 每个查询的相关性判断列表
            first_relevant_ranks: 每个查询的第一个相关案例排名
            k_values: 评估的K值列表
            
        Returns:
            包含各项指标的字典
        """
        metrics = {}
        
        # 对每个K值计算指标
        for k in k_values:
            # 计算Hit@K
            hit_at_k = []
            for relevance in relevance_results:
                # 如果前K个结果中有相关案例，则命中
                hit = int(any(relevance[:k])) if len(relevance) >= k else 0
                hit_at_k.append(hit)
            
            metrics[f'hit@{k}'] = np.mean(hit_at_k)
        
        # 计算MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []
        for rank in first_relevant_ranks:
            if rank > 0:  # 如果有相关案例
                reciprocal_ranks.append(1.0 / rank)
            else:  # 如果没有相关案例
                reciprocal_ranks.append(0.0)
        
        metrics['mrr'] = np.mean(reciprocal_ranks)
        
        return metrics
    
    def print_summary(self, embedding_type: str, metrics: Dict, k_values: List[int]):
        """
        打印评估结果摘要
        
        Args:
            embedding_type: 嵌入类型
            metrics: 评估指标
            k_values: K值列表
        """
        print(f"\n=== {embedding_type} 检索系统评估结果 ===")
        print(f"平均查询时间: {metrics['query_time']*1000:.2f} 毫秒")
        print(f"MRR: {metrics['mrr']:.4f}")
        
        print("\nK值\tHit@K")
        print("-" * 16)
        for k in k_values:
            print(f"{k}\t{metrics[f'hit@{k}']:.4f}")
    
    def print_comparative_results(self, embedding_types: List[str], 
                               k_values: List[int] = [1, 3, 5, 10, 20],
                               group_by_model: bool = False):
        """
        在控制台输出不同嵌入方法的比较结果
        
        Args:
            embedding_types: 要比较的嵌入类型列表
            k_values: K值列表
            group_by_model: 是否按模型类型分组结果
        """
        results = {}
        
        # 加载每个嵌入类型的评估结果
        for emb_type in embedding_types:
            result_file = os.path.join(self.output_dir, f'{emb_type}_evaluation.pkl')
            if not os.path.exists(result_file):
                print(f"警告: {emb_type} 的评估结果文件不存在")
                continue
            
            with open(result_file, 'rb') as f:
                results[emb_type] = pickle.load(f)
        
        if not results:
            print("没有找到任何评估结果")
            return
        
        # 如果需要按模型分组
        if group_by_model and any('_' in emb_type for emb_type in results.keys()):
            # 解析模型类型
            model_groups = defaultdict(list)
            for emb_type in results.keys():
                parts = emb_type.split('_', 1)
                if len(parts) > 1 and parts[0] in ["transe", "transr", "rescal", "rotate", "distmult", "conve", "rgcn"]:
                    model_groups[parts[0]].append(emb_type)
                else:
                    model_groups['other'].append(emb_type)
            
            # 按模型类型分组打印结果
            for model_type, model_emb_types in model_groups.items():
                print(f"\n\n=== {model_type.upper()} 模型的不同策略比较 ===\n")
                self._print_hit_comparison(results, model_emb_types, k_values)
                self._print_mrr_comparison(results, model_emb_types)
        else:
            # 直接打印所有结果
            self._print_hit_comparison(results, results.keys(), k_values)
            self._print_mrr_comparison(results, results.keys())
    
    def _print_hit_comparison(self, results: Dict, embedding_types: List[str], k_values: List[int]):
        """打印Hit@K比较结果"""
        print(f"=== 不同嵌入方法的 Hit@K 比较 ===\n")
        
       # 确定嵌入方法列的宽度 (至少20个字符)
        method_width = max(20, max([len(emb) for emb in embedding_types]) + 2)
        
        # 创建表头格式
        header_format = "{:<" + str(method_width) + "}"
        for _ in k_values:
            header_format += " {:<8}"
        
        # 打印表头
        header_values = ["嵌入方法"]
        for k in k_values:
            header_values.append(f"Hit@{k}")
        print(header_format.format(*header_values))
        
        # 打印分隔线
        print("-" * (method_width + 9 * len(k_values)))
        
        # 打印每个嵌入方法的结果
        for emb_type in embedding_types:
            if emb_type in results:
                row_values = [emb_type]
                for k in k_values:
                    row_values.append(f"{results[emb_type][f'hit@{k}']:.4f}")
                print(header_format.format(*row_values))
    
    def _print_mrr_comparison(self, results: Dict, embedding_types: List[str]):
        """打印MRR比较结果"""
        print(f"\n=== 不同嵌入方法的 MRR 比较 ===\n")
        
        # 确定嵌入方法列的宽度 (至少20个字符)
        method_width = max(20, max([len(emb) for emb in embedding_types]) + 2)
        
        # 创建格式字符串
        format_str = "{:<" + str(method_width) + "} {:<10}"
        
        # 打印表头
        print(format_str.format("嵌入方法", "MRR"))
        print("-" * (method_width + 12))
        
        # 按MRR降序排序
        sorted_types = sorted(
            [emb for emb in embedding_types if emb in results],
            key=lambda x: results[x]['mrr'], 
            reverse=True
        )
        
        for emb_type in sorted_types:
            print(format_str.format(emb_type, f"{results[emb_type]['mrr']:.4f}"))


def discover_available_embeddings(hybrid_dir="./hybrid_embeddings2"):
    """
    发现可用的混合嵌入类型
    
    Args:
        hybrid_dir: 混合嵌入目录
        
    Returns:
        可用的嵌入类型列表
    """
    available_embeddings = []
    
    if not os.path.exists(hybrid_dir):
        print(f"警告: 混合嵌入目录 {hybrid_dir} 不存在")
        return available_embeddings
    
    # 检查是否有索引文件
    index_file = os.path.join(hybrid_dir, 'all_hybrid_embeddings_index.csv')
    if os.path.exists(index_file):
        try:
            # 从索引文件加载嵌入类型
            df = pd.read_csv(index_file)
            for _, row in df.iterrows():
                emb_file = row['embeddings_file']
                # 提取嵌入类型
                if emb_file.startswith('hybrid_embeddings_') and emb_file.endswith('.pkl'):
                    emb_type = emb_file[len('hybrid_embeddings_'):-len('.pkl')]
                    # 检查文件是否存在
                    if os.path.exists(os.path.join(hybrid_dir, emb_file)):
                        available_embeddings.append(emb_type)
            print(f"从索引文件找到 {len(available_embeddings)} 种嵌入类型")
            return available_embeddings
        except Exception as e:
            print(f"读取索引文件出错: {str(e)}")
    
    # 直接扫描目录中的文件
    print("直接扫描目录查找嵌入文件...")
    prefix = 'hybrid_embeddings_'
    suffix = '.pkl'
    
    for file in os.listdir(hybrid_dir):
        if file.startswith(prefix) and file.endswith(suffix):
            embedding_type = file[len(prefix):-len(suffix)]
            available_embeddings.append(embedding_type)
    
    print(f"通过扫描目录找到 {len(available_embeddings)} 种嵌入类型")
    return available_embeddings


def run_retrieval_evaluation(data_dir="./kg_data2", 
                           model_dir="./model_outputs2", 
                           hybrid_dir="./hybrid_embeddings2",
                           output_dir="./retrieval_evaluation2",
                           embedding_types=None,
                           k_values=None,
                           test_size=100,
                           group_by_model=True,
                           term_similarity_threshold=0.2,
                           relevance_logic="OR"):
    """
    运行完整的检索系统评估
    
    Args:
        data_dir: 知识图谱数据目录
        model_dir: 模型输出目录
        hybrid_dir: 混合嵌入目录
        output_dir: 评估结果输出目录
        embedding_types: 要评估的嵌入类型列表
        k_values: 评估的K值列表
        test_size: 测试样本数量
        group_by_model: 是否按模型类型分组结果
        term_similarity_threshold: 量刑相似度阈值（相对差异）
        relevance_logic: 相关性判断逻辑，"AND"表示同时满足所有条件，"OR"表示满足任一条件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置默认K值
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]
    
    # 如果未指定嵌入类型，则自动发现
    if embedding_types is None:
        print("未指定嵌入类型，自动发现可用的嵌入...")
        embedding_types = discover_available_embeddings(hybrid_dir)
        
        if not embedding_types:
            # 使用默认嵌入类型
            embedding_types = ['graph_only', 'text_only', 'balanced', 'graph_heavy', 'text_heavy']
            print(f"未找到任何嵌入，将使用默认嵌入类型: {embedding_types}")
    
    print(f"将评估以下嵌入类型: {embedding_types}")
    
    # 初始化评估器，传入新增参数
    evaluator = RetrieverEvaluator(
        data_dir=data_dir, 
        output_dir=output_dir,
        term_similarity_threshold=term_similarity_threshold,
        relevance_logic=relevance_logic
    )
    
    # 先获取所有案例ID
    all_case_ids = list(evaluator.case_to_causes.keys())
    
    # 评估每种嵌入方法
    for embedding_type in embedding_types:
        print(f"\n===== 评估 {embedding_type} 嵌入 =====")
        
        try:
            # 初始化检索器
            retriever = FAISSRetriever(embedding_type, data_dir, model_dir, hybrid_dir)
            
            # 找出同时在评估器和检索器中的案例ID
            valid_cases = list(set(all_case_ids) & set(retriever.case_id_to_embed_id.keys()))
            
            # 随机选择测试样本
            if len(valid_cases) > test_size:
                test_samples = np.random.choice(valid_cases, test_size, replace=False).tolist()
            else:
                test_samples = valid_cases
                
            print(f"随机选择了 {len(test_samples)} 个测试样本")
            
            # 评估检索效果
            evaluator.evaluate_retriever(retriever, test_samples, k_values)
        except Exception as e:
            print(f"评估 {embedding_type} 嵌入时出错: {str(e)}")
    
    # 在控制台打印比较结果
    print("\n\n========== 嵌入方法比较结果 ==========\n")
    
    # 比较Hit@K和MRR，可选按模型分组
    evaluator.print_comparative_results(embedding_types, k_values, group_by_model=group_by_model)
    
    print("\n所有评估完成!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估基于FAISS的法律案例检索系统")
    parser.add_argument("--data_dir", default="./kg_data2", help="知识图谱数据目录")
    parser.add_argument("--model_dir", default="./evaluation_results2", help="模型输出目录")
    parser.add_argument("--hybrid_dir", default="./hybrid_embeddings2", help="混合嵌入目录")
    parser.add_argument("--output_dir", default="./retrieval_evaluation2", help="评估结果输出目录")
    parser.add_argument("--test_size", type=int, default=100, help="测试样本数量")
    parser.add_argument("--embedding_types", nargs='+', help="要评估的嵌入类型列表")
    parser.add_argument("--no_group", action="store_true", help="不按模型类型分组结果")
    parser.add_argument("--term_threshold", type=float, default=0.4, help="量刑相似度阈值（0-1），默认0.2表示20%")
    parser.add_argument("--relevance_logic", choices=["AND", "OR"], default="AND", help="相关性判断逻辑：'AND'表示同时满足所有条件，'OR'表示满足任一条件")
    
    args = parser.parse_args()
    
    run_retrieval_evaluation(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        hybrid_dir=args.hybrid_dir,
        output_dir=args.output_dir,
        embedding_types=args.embedding_types,
        test_size=args.test_size,
        group_by_model=not args.no_group,
        term_similarity_threshold=args.term_threshold,
        relevance_logic=args.relevance_logic
    )