## 图

### 最短路径算法

#### Dijkstra
由荷兰计算机科学家Dijkstra提出的**单源最短路径算法**，解决的是有权图中最短路径问题。Dijkstra算法主从起始点开始，采用贪心算法的策略，流程如下

1. $start\_node \rightarrow S$, $V\text{-}S \rightarrow T$, `distance=[inf]*N, distance[start_node]=0`
2. 遍历 $T$ 中的顶点，选择距 `start_node` 最近的新顶点 $v$
3. $S\text{+}v \rightarrow S$, $T\text{-}v \rightarrow T$
4. 更新新增节点$v$后的 `distance` 数组（保留最短路径），重复step 2直到 `T==∅`
=== "Python"
    ```python
    def dijkstra(graph, start_node):
        N = len(graph)                      # |V|
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

            # distance[v] 表示 `start_node -> v` 的最短距离
            # distance[index] = min(distance[index], distance[min_value_index] + graph[min_value_index][index])
            for index in range(N):
                if distance[min_value_index] + graph[min_value_index][index] < distance[index]:
                    distance[index] = distance[min_value_index] + graph[min_value_index][index]
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
    - 要求所有边的权重非负
    - 返回指定单源(start_node)至其它顶点的最短路径(和路径)，但不保证是最小生成树

#### Floyd
由1978年图灵奖获得者、斯坦福大学计算机科学系教授Robert Floyd提出的**多源最短路径算法**，稠密图效果最佳，流程如下：


1. 插入顶点 $v_i$
2. 更新插入该顶点后所有 `start_node` 到 `end_node` 的最短距离
3. 重复step 1 直到插入了所有顶点

=== "python"
    ```python
    def floyd(graph):
        # graph[i][j] 表示start_node `i` 至 end_node `j` 的最短距离
        # min(Dis(i,j), Dis(i,k) + Dis(k,j))
        N = len(graph)
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    if graph[i][j] > graph[i][k] + graph[k][j]:
                        graph[i][j] = graph[i][k] + graph[k][j]
        return graph
    ```

!!! info ""
    - 是一个dp算法，本质上是从一个个最短子片段图到全连通图的过程
    - 边的权重可为负数，但如果存在负权重的环，算法可能无法正确工作
    - 返回所有单源(start_node)至其它顶点的最短距离，但不保证是最小生成树
  
#### Prim
1. $start\_node \rightarrow S$, $V\text{-}S \rightarrow T$, 空图 $G$
2. 遍历 $T$ 中的顶点，选择距 $S$ 最近的边$e$ 以及新顶点 $v$
3. $S\text{+}v \rightarrow S$, $T\text{-}v \rightarrow T$, $G\text{+}e\rightarrow G$
4. 重复step 2直到 `T==∅`

!!! info ""
    - 与`dijkstra`区别在于后者是dp算法，传递因子是到`start_node`的最短距离，该方法是新边`e`到子图的最短距离

#### Kruskal

1. 边信息 $E$，空图 $G$，初始化各顶点并查集`union_set`为自身
2. 选取权重最小的边 $e \in E$，要求$e$两端顶点不能同时处一并查集（子图）中
3. 将 $e$ 加入 $G$、基于两端顶点更新并查集`union_set`，重复 step 2 直到所有顶点连通
