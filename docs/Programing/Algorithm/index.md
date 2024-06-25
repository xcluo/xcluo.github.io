## 图

### 最短路径算法

#### Dijkstra
由荷兰计算机科学家Dijkstra提出的从一个顶点到其余各顶点的最短路径算法，解决的是有权图中最短路径问题。Dijkstra算法主从起始点开始，采用贪心算法的策略，流程如下

1. $start\_node \rightarrow S$, $V\text{-}S \rightarrow T$, `distance=[inf]*N, distance[start_node]=0`
2. 遍历 $T$ 中的顶点，选择距 $start\_node$ 最近的节点$v$
3. $S\text{+}v \rightarrow S$, $T\text{-}v \rightarrow T$
4. 更新纳入节点$v$后的 `distance` 数组（保留最短路径），重复step 2.
=== "Python"
    ```python
    def dijkstra(matrix, start_node):
        N = len(matrix)                     # |V|
        used_node = [False] * N             # S = [i for i, v in enumerate(used_node) if v]
                                            # T = [i for i, v in enumerate(used_node) if not v]
        distance = [math.inf] * N           # start_node至各节点的最短路径距离数组
        distance[start_node] = 0            # start_node至自身的最短路径为0
        pre_node = list(range(N))           # 各节点至start_node的路径的上一节点，初始化为自身

        while used_node.count(False):
            min_value = math.inf
            min_value_index = -1

            # 遍历找到start_node至T中距离最近的节点v
            for index in range(N):
                if not used_node[index] and distance[index] < min_value:
                    min_value = distance[index]
                    min_value_index = index

            # S + v -> S
            used_node[min_value_index] = True

            # dp更新distance数组
            # distance[index] = min(distance[index], distance[min_value_index] + matrix[min_value_index][index])
            for index in range(N):
                if distance[min_value_index] + matrix[min_value_index][index] < distance[index]:
                    distance[index] = distance[min_value_index] + matrix[min_value_index][index]
                    pre_node[index] = min_value_index

        # concat route and distance
        routes = []
        for i in range(N):
            ret = [i]
            pre_n = pre_node[i]
            while pre_n != start_node:
                ret.append(pre_n)
                pre_n = pre_node[pre_n]
            ret.append(start_node)
            routes.append({'distance': distance[i], 'route': ret[::-1]})
            
        return routes
    ```

!!! info ""
    - 是一个dp的贪心算法
    - 单源(start_node)全局(至其它顶点)最优路径，但不保证是最小生成树
#### Prim
#### Kruskal
#### Floyd