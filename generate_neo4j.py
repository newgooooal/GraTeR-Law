# 增加了embed_id属性，使得embed_id=case_id
import os
import time
import json
from openai import OpenAI
from py2neo import Graph, Node, Relationship
import logging
import traceback
from tqdm import tqdm  # 导入tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kg_build2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 断点恢复文件
CHECKPOINT_FILE = "kg_checkpoint2.txt"

# 声明全局变量
BATCH_SIZE = None # 批处理大小
MAX_LINES = None # 最大处理行数 (设置为None则处理全部)
RESTART = None # 是否从头开始处理(True: 从第一行开始, False: 从检查点继续)


# 连接Neo4j数据库
graph = Graph("neo4j://10.61.2.143:7689", auth=("neo4j", "12345678"))

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-6bb64f2976ac476c8b355713c651362a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# JSON输出模板
JSONmodel = """
{
    "fact_summary": "案件关键事实摘要",
    "key_elements": ["关键行为1", "关键行为2", "关键行为3"],
    "mitigating_factors": ["从轻情节1", "从轻情节2"...],
    "aggravating_factors": ["从重情节1", "从重情节2"...]
}
"""

# JSON 文件路径
json_file_path = "/data1/lxy/project/5050.json"

# 获取文件总行数
def get_total_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

# 创建索引以提高查询性能
def create_indices():
    try:
        # 获取所有索引
        indexes = graph.run("SHOW INDEXES").data()
        if not indexes:
            logger.info("正在创建索引...")
            graph.run("CREATE INDEX FOR (c:Case) ON (c.case_id)")
            graph.run("CREATE INDEX FOR (ch:Charge) ON (ch.name)")
            graph.run("CREATE INDEX FOR (a:Article) ON (a.article_id)")
            graph.run("CREATE INDEX FOR (cr:Criminal) ON (cr.name)")
            logger.info("索引创建完成")
        else:
            logger.info("索引已存在，跳过创建")
    except Exception as e:
        logger.warning(f"创建索引时出现异常: {e}")

# 读取检查点，确定从哪一行开始处理
def read_checkpoint():
    global RESTART
    if RESTART:
        # 如果设置了从头开始，删除检查点文件并返回1
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logger.info("检查点文件已删除，将从第1行开始处理")
        return 1
    
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 1
    return 1

# 写入检查点
def write_checkpoint(line_num):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(line_num))

# 批量处理数据并构建知识图谱
def build_knowledge_graph():
    global BATCH_SIZE, MAX_LINES, RESTART
    # 创建索引
    create_indices()
    
    # 读取检查点
    start_line = read_checkpoint()
    logger.info(f"从第 {start_line} 行开始处理")
    
    # 获取文件总行数
    total_lines = get_total_lines(json_file_path)
    if MAX_LINES:
        total_lines = min(MAX_LINES, total_lines)
    
    # 计算还需处理的行数
    remaining_lines = total_lines - start_line + 1
    if remaining_lines <= 0:
        logger.info("已经处理完所有数据，无需再次处理")
        return
    
    logger.info(f"总共需处理 {remaining_lines} 行数据")
    
    # 记录开始时间
    start_time = time.time()
    
    # 当前批次中的操作计数
    batch_operations = 0
    
    # 事务对象
    tx = None
    
    try:
        # 打开文件并跳过已处理的行
        with open(json_file_path, 'r', encoding='utf-8') as f:
            # 跳过已处理的行
            for _ in range(start_line - 1):
                next(f, None)
            
            line_num = start_line
            
            # 开始批处理
            tx = graph.begin()
            
            # 创建tqdm进度条
            pbar = tqdm(total=remaining_lines, initial=0, desc="处理进度", 
                       unit="例", ncols=100, position=0, leave=True)
            
            for line in f:
                if MAX_LINES and line_num > MAX_LINES:
                    break
                
                if not line.strip():
                    line_num += 1
                    pbar.update(1)  # 更新进度条
                    continue
                    
                try:
                    # 解析原始数据
                    data = json.loads(line)
                    fact_text = data.get("fact", "").strip()
                    meta_data = data.get("meta", {})
                    
                    if not fact_text or not meta_data:
                        logger.warning(f"第{line_num}行数据不完整，跳过。")
                        line_num += 1
                        pbar.update(1)  # 更新进度条
                        continue
                    
                    # 使用tqdm.set_description更新当前处理行
                    pbar.set_description(f"处理第 {line_num} 行")
                    
                    # 提取显式实体
                    case_id = f"case_{line_num}"
                    criminals = meta_data.get("criminals", [])
                    charges = meta_data.get("accusation", [])
                    articles = meta_data.get("relevant_articles", [])
                    term_info = meta_data.get("term_of_imprisonment", {})
                    punishment_money = meta_data.get("punish_of_money", 0)
                    
                    try:
                        # 检查案例是否已存在
                        query = "MATCH (c:Case {case_id: $case_id}) RETURN count(c) > 0 AS exists"
                        case_exists = graph.run(query, case_id=case_id).data()[0]["exists"]
                        if case_exists:
                            logger.warning(f"案例 {case_id} 已存在，跳过处理")
                            line_num += 1
                            pbar.update(1)  # 更新进度条
                            continue
                    except Exception as e:
                        logger.error(f"检查案例是否存在时出错: {e}")
                        # 继续处理，最坏情况是创建重复节点
                    
                    # 提取隐式实体(LLM调用)
                    prompt = f"""
                    请从以下刑事案件事实描述中提取:

                    1. fact_summary: 案件关键事实摘要,用100字以内概括案件关键事实
                    2. key_elements: 列出3-5个关于案件的关键行为
                    3. mitigating_factors: 可能导致判决较轻的因素，包括但不限于：
                    - 自首/投案自首
                    - 坦白/如实供述/认罪认罚
                    - 当庭认罪态度好
                    - 未遂/中止犯罪
                    - 积极退赃/退赔/赔偿损失
                    - 取得被害人谅解
                    - 被胁迫/从犯/胁从犯
                    - 初犯/偶犯
                    - 未成年人犯罪
                    - 有立功表现
                    4. aggravating_factors: 可能导致判决较重的因素，包括但不限于：
                    - 累犯/再犯
                    - 主犯
                    - 情节恶劣/手段残忍
                    - 后果严重
                    - 屡教不改
                    - 共同犯罪
                    - 犯罪数额巨大
                    - 前科劣迹
                    - 拒不认罪
                    - 逃避/抗拒执法

                    以下是JSON格式的参考输出:

                    {JSONmodel}

                    注意:
                    1. 请确保输出为有效的JSON格式，只输出JSON内容，不要有其他说明性文字
                    2. 如果某类信息在文本中不存在，请返回空数组[]
                    3. 每个情节用简洁词组表达，不要添加文本中未明确提及的信息
                    4. 在提取信息时，请严格排除日期、真实或虚构的人名（包括但不限于 "王某某""赵某某" 等）、地名等具体信息

                    案例事实描述:
                    {fact_text}
                    """
                    
                    try:
                        # 更新进度条描述为当前状态
                        pbar.set_description(f"第 {line_num} 行: LLM处理中")
                        
                        completion = client.chat.completions.create(
                            model="qwen-max",
                            messages=[
                                {'role': 'system', 'content': 'You are a helpful assistant.'},
                                {'role': 'user', 'content': prompt}
                            ]
                        )
                        
                        # 提取LLM结果
                        llm_result_text = completion.choices[0].message.content.strip()
                        
                        # 解析LLM返回的JSON
                        llm_result = json.loads(llm_result_text)
                        
                        # 更新进度条描述为图谱构建
                        pbar.set_description(f"第 {line_num} 行: 构建图谱")
                        
                        # 构建知识图谱节点和关系
                        # 创建案件节点
                        case_node = Node("Case", 
                                        case_id=case_id, 
                                        embed_id=line_num,
                                        fact=fact_text,
                                        fact_summary=llm_result.get("fact_summary", ""))
                        tx.create(case_node)
                        batch_operations += 1
                        
                        # 创建并关联被告人节点
                        criminal_node = Node("Criminal", name=criminals)
                        tx.create(criminal_node)
                        rel = Relationship(case_node, "HAS_DEFENDANT", criminal_node)
                        tx.create(rel)
                        batch_operations += 2
                        
                        # 创建并关联罪名节点
                        for charge in charges:
                            charge_node = Node("Charge", name=charge)
                            tx.merge(charge_node, "Charge", "name")
                            rel = Relationship(case_node, "HAS_CHARGE", charge_node)
                            tx.create(rel)
                            batch_operations += 2
                        
                        # 创建并关联法条节点
                        for article in articles:
                            article_node = Node("Article", article_id=article)
                            tx.merge(article_node, "Article", "article_id")
                            rel = Relationship(case_node, "CITES_ARTICLE", article_node)
                            tx.create(rel)
                            batch_operations += 2
                        
                        # 创建判决结果节点
                        imprisonment = term_info.get("imprisonment", 0)
                        life_imprisonment = term_info.get("life_imprisonment", False)
                        death_penalty = term_info.get("death_penalty", False)
                        
                        term_description = ""
                        if death_penalty:
                            term_description = "死刑"
                        elif life_imprisonment:
                            term_description = "无期徒刑"
                        else:
                            term_description = f"{imprisonment}个月有期徒刑"
                        
                        term_node = Node("Term", 
                                       term_id=f"term_{case_id}",
                                       imprisonment=imprisonment,
                                       life_imprisonment=life_imprisonment,
                                       death_penalty=death_penalty,
                                       punishment_money=punishment_money,
                                       description=term_description)
                        tx.create(term_node)
                        rel = Relationship(case_node, "RESULTS_IN", term_node)
                        tx.create(rel)
                        batch_operations += 2
                        
                        # 处理关键行为
                        for i, element in enumerate(llm_result.get("key_elements", [])):
                            element_node = Node("Element", 
                                              element_id=f"element_{case_id}_{i}",
                                              content=element)
                            tx.create(element_node)
                            rel = Relationship(case_node, "HAS_ELEMENT", element_node)
                            tx.create(rel)
                            batch_operations += 2
                        
                        # 处理从轻情节
                        for i, factor in enumerate(llm_result.get("mitigating_factors", [])):
                            factor_node = Node("Factor", 
                                             factor_id=f"mitigating_{case_id}_{i}",
                                             content=factor,
                                             factor_type="mitigating")
                            tx.create(factor_node)
                            rel = Relationship(case_node, "HAS_MITIGATING_FACTOR", factor_node)
                            tx.create(rel)
                            batch_operations += 2
                        
                        # 处理从重情节
                        for i, factor in enumerate(llm_result.get("aggravating_factors", [])):
                            factor_node = Node("Factor", 
                                             factor_id=f"aggravating_{case_id}_{i}",
                                             content=factor,
                                             factor_type="aggravating")
                            tx.create(factor_node)
                            rel = Relationship(case_node, "HAS_AGGRAVATING_FACTOR", factor_node)
                            tx.create(rel)
                            batch_operations += 2
                        
                        logger.info(f"第{line_num}条数据处理成功")
                        
                    except json.JSONDecodeError:
                        logger.error(f"第{line_num}条LLM返回结果不是有效JSON: {llm_result_text}")
                    except Exception as e:
                        logger.error(f"处理LLM结果出错: {e}")
                        logger.debug(traceback.format_exc())
                    
                    # 检查是否需要提交事务
                    if batch_operations >= BATCH_SIZE:
                        # 更新进度条描述为提交事务
                        pbar.set_description(f"提交事务 (第 {line_num} 行)")
                        
                        logger.info(f"批处理达到 {BATCH_SIZE} 次操作，正在提交事务...")
                        tx.commit()
                        # 更新检查点
                        write_checkpoint(line_num)
                        logger.info(f"事务已提交，检查点已更新到第 {line_num} 行")
                        
                        # 重新开始新事务
                        tx = graph.begin()
                        batch_operations = 0
                
                except json.JSONDecodeError as e:
                    logger.error(f"第{line_num}行 JSON 解析失败: {e}")
                except Exception as e:
                    logger.error(f"处理第{line_num}行时发生错误: {e}")
                    logger.debug(traceback.format_exc())
                
                # 更新进度条
                pbar.update(1)
                line_num += 1
            
            # 关闭进度条
            pbar.close()
            
            # 提交最后的事务
            if batch_operations > 0:
                logger.info(f"提交最后的事务，包含 {batch_operations} 次操作...")
                tx.commit()
                write_checkpoint(line_num)
                logger.info(f"最后的事务已提交，检查点已更新到第 {line_num} 行")
    
    except Exception as e:
        logger.critical(f"发生严重错误: {e}")
        logger.debug(traceback.format_exc())
        # 如果有活跃事务，尝试提交以保存已处理的数据
        if tx and batch_operations > 0:
            try:
                tx.commit()
                logger.info("尝试保存已处理的数据")
            except:
                logger.error("无法保存已处理的数据")
    
    finally:
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"程序运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        print(f"\n总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

# 主程序入口
if __name__ == "__main__":
    # 批处理大小
    BATCH_SIZE = 150

    # 最大处理行数 (设置为None则处理全部)
    MAX_LINES = None
    
    RESTART = False
    
    # 如果想从头开始，取消下面这行的注释
    RESTART = True
    
    # 清空 Neo4j 中所有的节点和关系，保留标签和索引结构
    # MATCH (n) DETACH DELETE n
    
    logger.info(f"开始处理，最大行数: {MAX_LINES}, 批处理大小: {BATCH_SIZE}, 从头开始: {RESTART}")
    build_knowledge_graph()