import requests
from typing import List, Dict, Any, Optional


class ReRankerClient:
    def __init__(self, cfg={"model_server": "http://localhost:8001/v1", "model": "RNK"}):
        self.base_url = cfg.get("model_server")
        self.model_name = cfg.get("model")

    def rerank_documents(self, query: str, documents: List[str]) -> Optional[List[Dict]]:
        """
        对文档进行重排序
        
        :param query: 查询文本
        :param documents: 文档列表
        :return: 排序后的结果列表
        """
        url = f"{self.base_url}/rerank"
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents
        }

        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            results = response.json().get("results", [])
            sorted_result = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
            return sorted_result
        except requests.exceptions.RequestException as e:
            print(f"重排请求失败: {e}")
            return None

    def run_test_cases(self, test_cases: list):
        """
        执行多组测试用例并输出结果
        :param test_cases: 包含查询和文档列表的测试用例列表
        """
        for i, case in enumerate(test_cases, start=1):
            print(f"\n=== 测试用例 #{i} ===")
            print(f"查询: {case['query']}")
            sorted_result = self.rerank_documents(case["query"], case["documents"])

            if sorted_result:
                for doc in sorted_result:
                    print(
                        f"文档 {doc['index']+1}: 得分={doc['relevance_score']:.4f}, 内容={case['documents'][doc['index']]}"
                    )
            else:
                print("未获取到有效结果")

# 多组测试用例定义
TEST_CASES = [
    {
        "query": "人工智能应用",
        "documents": [
            "机器学习在医疗诊断中的应用",
            "深度学习在自动驾驶中的最新进展",
            "自然语言处理技术发展史",
            "人工智能伦理问题探讨"
        ]
    },
    {
        "query": "量子计算的最新突破",
        "documents": [
            "量子计算在密码学中的应用",
            "超导量子比特的稳定性研究",
            "经典计算机与量子计算机对比分析",
            "量子纠缠现象的实验验证"
        ]
    },
    {
        "query": "糖尿病治疗的新方法",
        "documents": [
            "胰岛素泵治疗糖尿病的优势",
            "干细胞疗法在糖尿病治疗中的潜力",
            "糖尿病患者的饮食管理指南",
            "新型口服降糖药物的临床试验结果"
        ]
    }
]

if __name__ == "__main__":
    # 初始化 ReRankerClient 实例
    reranker = ReRankerClient()

    # 运行测试用例
    reranker.run_test_cases(TEST_CASES)