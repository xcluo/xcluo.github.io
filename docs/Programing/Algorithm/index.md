## 图

### 最短路径算法

#### Dijkstra
由荷兰计算机科学家Dijkstra提出的从一个顶点到其余各顶点的最短路径算法，解决的是有权图中最短路径问题。Dijkstra算法主从起始点开始，采用贪心算法的策略，流程如下

1. $v_0 \rightarrow S$, $V\text{-}S \rightarrow T$
2. 遍历 $T$ 中的顶点，，选择对应的顶点$v_i \in T$ 满足与 $S$ 中的顶点连接边权值最小
3. $S\text{+}v_i \rightarrow S$, $T\text{-}v_i \rightarrow T$，重复step.2
#### Prim
#### Kruskal