- [Self Positioning](self-positioning.md)


### 联邦学习
Federated Learning，隐私保护的技术方法 → "怎么安全地协作训练"，类似于数据隔离的分布式并行训练，在多个设备上分别使用各自的数据获得更新量，将总更新量聚合成一个全局更新量，然后分发给每个设备，以此实现联邦学习。

```python
class FederatedLearning:
    def __init__(self):
        self.hospitals = {
            "北京协和": LocalModel(data="肺部CT，不出院"),
            "上海瑞金": LocalModel(data="心脏病例，不出院"), 
            "广州中山": LocalModel(data="脑部扫描，不出院")
        }
        self.global_model = GlobalModel()
    
    def train_round(self):
        # 1. 各医院本地训练（数据不动）
        local_updates = []
        for hospital in self.hospitals:
            update = hospital.train_locally()  # 只训练，不传数据
            local_updates.append(update)  # 传"心得"，不传数据
        
        # 2. 聚合更新（联邦平均）
        self.global_model.aggregate(local_updates)
        
        # 3. 分发全局模型
        for hospital in self.hospitals:
            hospital.update_model(self.global_model)
```