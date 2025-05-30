# distmult_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import time
from tqdm import tqdm

class KGDataset(Dataset):
    def __init__(self, triples, n_entities, n_relations, neg_samples=1):
        self.triples = triples
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.neg_samples = neg_samples
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        pos_triple = torch.LongTensor([head, relation, tail])
        
        neg_triples = []
        for _ in range(self.neg_samples):
            # 负采样 - 随机替换头实体或尾实体
            if np.random.random() < 0.5:
                # 替换头实体
                corrupt_head = np.random.randint(self.n_entities)
                while corrupt_head == head:
                    corrupt_head = np.random.randint(self.n_entities)
                neg_triples.append([corrupt_head, relation, tail])
            else:
                # 替换尾实体
                corrupt_tail = np.random.randint(self.n_entities)
                while corrupt_tail == tail:
                    corrupt_tail = np.random.randint(self.n_entities)
                neg_triples.append([head, relation, corrupt_tail])
        
        neg_triples = torch.LongTensor(neg_triples)
        return pos_triple, neg_triples

class DistMult(nn.Module):
    def __init__(self, n_entities, n_relations, dim=100, margin=1.0, reg_lambda=0.0):
        """
        DistMult模型，使用实体向量和关系向量的三元积表示知识图谱
        
        参数：
            n_entities: 实体数量
            n_relations: 关系数量
            dim: 嵌入维度
            margin: 损失函数的边界值
            reg_lambda: L2正则化系数
        """
        super(DistMult, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.margin = margin
        self.reg_lambda = reg_lambda
        
        # 实体嵌入
        self.entity_emb = nn.Embedding(n_entities, dim)
        # 关系嵌入
        self.relation_emb = nn.Embedding(n_relations, dim)
        
        # 初始化嵌入
        nn.init.xavier_uniform_(self.entity_emb.weight.data)
        nn.init.xavier_uniform_(self.relation_emb.weight.data)
        
        # 正则化实体嵌入（避免过拟合）
        self.normalize_embeddings()
        
    def normalize_embeddings(self):
        """归一化实体嵌入向量"""
        self.entity_emb.weight.data = torch.renorm(
            self.entity_emb.weight.data, p=2, dim=0, maxnorm=1.0
        )
    
    def _calc_score(self, head_emb, rel_emb, tail_emb):
        """
        计算DistMult得分: <h,r,t> = sum(h * r * t)
        
        参数:
            head_emb: 头实体嵌入
            rel_emb: 关系嵌入
            tail_emb: 尾实体嵌入
            
        返回:
            score: 三元组得分，越大表示越可能为真
        """
        # 按元素乘法 (h * r * t)，然后求和
        score = torch.sum(head_emb * rel_emb * tail_emb, dim=-1)
        return score
    
    def forward(self, pos_triples, neg_triples):
        """
        前向传播，计算正负三元组的得分和损失
        
        参数:
            pos_triples: 正三元组
            neg_triples: 负三元组
            
        返回:
            loss: 计算得到的损失值
        """
        # 正三元组
        pos_heads = self.entity_emb(pos_triples[:, 0])
        pos_relations = self.relation_emb(pos_triples[:, 1])
        pos_tails = self.entity_emb(pos_triples[:, 2])
        
        # 计算正三元组得分
        pos_score = self._calc_score(pos_heads, pos_relations, pos_tails)
        
        batch_size = len(pos_triples)
        neg_size = len(neg_triples[0])
        
        # 负三元组 - 处理多个负样本
        neg_heads = self.entity_emb(neg_triples[:, :, 0]).view(batch_size * neg_size, self.dim)
        neg_relations = self.relation_emb(neg_triples[:, :, 1]).view(batch_size * neg_size, self.dim)
        neg_tails = self.entity_emb(neg_triples[:, :, 2]).view(batch_size * neg_size, self.dim)
        
        # 计算负三元组得分
        neg_score = self._calc_score(neg_heads, neg_relations, neg_tails)
        neg_score = neg_score.view(batch_size, neg_size)
        
        # 对每个正样本，取所有负样本中最好的那个(最大得分)
        best_neg_score = torch.max(neg_score, dim=1)[0]
        
        # 计算软边界损失
        loss = torch.mean(torch.relu(self.margin - pos_score + best_neg_score))
        
        # 添加L2正则化（如果需要）
        if self.reg_lambda > 0:
            l2_reg = torch.mean(pos_heads.norm(p=2, dim=1)**2 + 
                               pos_relations.norm(p=2, dim=1)**2 + 
                               pos_tails.norm(p=2, dim=1)**2)
            loss = loss + self.reg_lambda * l2_reg
        
        return loss
    
    def predict(self, triples):
        """为给定的三元组预测分数"""
        heads = self.entity_emb(triples[:, 0])
        relations = self.relation_emb(triples[:, 1])
        tails = self.entity_emb(triples[:, 2])
        
        scores = self._calc_score(heads, relations, tails)
        return scores
    
    def get_embeddings(self):
        """获取所有实体和关系的嵌入向量"""
        return self.entity_emb.weight.data.cpu().numpy(), self.relation_emb.weight.data.cpu().numpy()

def train_distmult(data_dir="./kg_data3", output_dir="./model_outputs3", 
                  dim=100, epochs=100, batch_size=2048, lr=0.001, 
                  margin=1.0, neg_samples=5, reg_lambda=0.0,
                  save_every=20, eval_every=10):
    """训练DistMult模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("加载知识图谱数据...")
    with open(os.path.join(data_dir, 'triples.pkl'), 'rb') as f:
        triples = pickle.load(f)
    
    with open(os.path.join(data_dir, 'kg_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    
    n_entities = stats['n_entities']
    n_relations = stats['n_relations']
    
    print(f"实体数量: {n_entities}")
    print(f"关系类型数量: {n_relations}")
    print(f"三元组数量: {len(triples)}")
    
    # 数据划分 - 训练集和验证集
    np.random.shuffle(triples)
    val_size = int(0.1 * len(triples))
    train_triples = triples[:-val_size]
    val_triples = triples[-val_size:]
    
    print(f"训练集大小: {len(train_triples)}")
    print(f"验证集大小: {len(val_triples)}")
    
    # 创建数据集和数据加载器
    train_dataset = KGDataset(train_triples, n_entities, n_relations, neg_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_dataset = KGDataset(val_triples, n_entities, n_relations, neg_samples)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistMult(n_entities, n_relations, dim=dim, margin=margin, reg_lambda=reg_lambda).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"开始训练DistMult模型，维度={dim}，批大小={batch_size}，学习率={lr}，边际值={margin}")
    print(f"设备: {device}")
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for pos_triples, neg_triples in pbar:
            pos_triples = pos_triples.to(device)
            neg_triples = neg_triples.to(device)
            
            optimizer.zero_grad()
            loss = model(pos_triples, neg_triples)
            loss.backward()
            optimizer.step()
            
            # 归一化实体嵌入
            model.normalize_embeddings()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        if epoch % eval_every == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for pos_triples, neg_triples in val_loader:
                    pos_triples = pos_triples.to(device)
                    neg_triples = neg_triples.to(device)
                    loss = model(pos_triples, neg_triples)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch}/{epochs} - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(output_dir, 'distmult_best.pt'))
                print(f"Epoch {epoch} - 保存了最佳模型，验证损失: {val_loss:.4f}")
        
        # 定期保存模型
        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss if epoch % eval_every == 0 else None,
            }, os.path.join(output_dir, f'distmult_epoch_{epoch}.pt'))
    
    # 训练完成后保存最终模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_losses[-1] if val_losses else None,
    }, os.path.join(output_dir, 'distmult_final.pt'))
    
    # 导出嵌入向量
    model.eval()
    entity_embeddings, relation_embeddings = model.get_embeddings()
    
    with open(os.path.join(output_dir, 'distmult_entity_embeddings.pkl'), 'wb') as f:
        pickle.dump(entity_embeddings, f)
    
    with open(os.path.join(output_dir, 'distmult_relation_embeddings.pkl'), 'wb') as f:
        pickle.dump(relation_embeddings, f)
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"训练完成，总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    print(f"嵌入向量已保存到 {output_dir} 目录")
    
    return entity_embeddings, relation_embeddings, {'train_losses': train_losses, 'val_losses': val_losses}

if __name__ == "__main__":
    # 可以根据需要调整参数
    train_distmult(dim=100, epochs=100, batch_size=2048, lr=0.001, margin=1.0, neg_samples=5)