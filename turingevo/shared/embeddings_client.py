import requests
from typing import List, Dict, Any, Optional

import torch
import numpy as np

##
# 结果与 /media/wmx/ws1/workspace/ai/Qwen3-Embedding/examples/qwen3_embedding_transformers.py  一致
##


class EmbeddingClient:
    def __init__(self, cfg={"model_server": "http://localhost:8000"}):
        self.base_url = cfg.get("model_server")

    def get_embeddings(self, texts: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        获取批量文本的嵌入向量
        
        :param texts: 文本列表
        :return: 包含索引和嵌入向量的字典列表
        """
        url = f"{self.base_url}/embedding"
        payload = {
            "content": texts,
            }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"嵌入请求失败: {e}")
            return None
        
    def get_last_token_embeddings(self, response: List[Dict[str, Any]]) -> Optional[List[float]]:
        """
        --pooling last
        从响应中获取最后一个 token 的嵌入向量
        """
        pooled_embeddings = []
        for item in response:
            embeddings = item["embedding"]  # 所有 token 的嵌入
            last_token_emb = embeddings[-1]  # 取最后一个 token 的嵌入
            pooled_embeddings.append(last_token_emb)
        return pooled_embeddings
    

    def get_last_embedding(self,  texts: List[str] ) -> Optional[List[float]]:
        """
        --pooling last
        获取文本的最后一个 token 的嵌入向量
        
        :param texts: 文本字符串
        :return: 嵌入向量
        """
        response = self.get_embeddings(texts)
        if response:
            return self.get_last_token_embeddings(response)
        return None
    

if __name__ == "__main__":
    client = EmbeddingClient()

    queries = ['What is the capital of China?', 'Explain gravity']
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    dim = 1024

    # query_rsp = client.get_embeddings(queries)
    # doc_rsp = client.get_embeddings(documents)
    # if query_rsp:
    #     for item in query_rsp:
    #         index = item["index"]
    #         embeddings = item["embedding"]
    #         print(f"\nIndex: {index}")
    #         print(f"Number of tokens: {len(embeddings)}")
    #         print(f"1 token dim : {len(embeddings[0])}")


    query_outputs = client.get_last_embedding(queries)
    doc_outputs = client.get_last_embedding(documents)

    
    query_outputs  = torch.tensor(query_outputs)
    doc_outputs = torch.tensor(doc_outputs)
    print('query outputs size : ', query_outputs.size(), '\ndoc outputs size : ', doc_outputs.size())
    print('query outputs', query_outputs)
    print('doc outputs', doc_outputs)

  # 小数 转为 100 分制
    scores = (query_outputs @ doc_outputs.T) * 100

    print('scores  size : ', scores.size())
    print(f"scores.tolist() : \n {scores.tolist()}")

    # 打印每个配对分数
    print("配对分数结果:")
    for i, query in enumerate(queries):
        for j, doc in enumerate(documents):
            score = scores[i][j].item()  # 转换为Python float
            print(f"配对 {i+1}-{j+1}:")
            print(f"  Query: {query}")
            print(f"  Document: {doc[:50]}...")  # 截断长文本
            print(f"  Score: {score:.4f}")  # 保留4位小数
            print()