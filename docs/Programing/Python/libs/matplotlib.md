```python
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator   # 导入坐标轴控制库

plt.rcParams['font.sans-serif']=['SimHei']      # 显示中文不乱码
plt.rcParams['axes.unicode_minus']=False        # 显示负号不乱码
```

### 绘图
#### 折线图 `plot`
```python
plot(
    *args,
    scalex=True,
    scaley=True,
    data=None,
    **kwargs 
) 
```
#### 散点图 `sactter`
#### 直方图 `hist`
长方形编辑表示频数
#### 条形图 `bar`
高度表示频数
#### 子图 `subplots`

### 图形界面
#### 画布 `figure`
#### 文本 `text`
#### 网格线 `grid`
#### 图例 `legned`
```python
legend(
    *args,
    **kwargs    # handles，isinstanceof(handles, Iterable)，元素为绘制的图表artist
                # labels，isinstanceof(handles, Iterable)，元素绘制的图表artist的自定义文本解释
                    #> 与handlers中元素一一对应，未指定handlers则对应artist绘制顺序
                # loc in {'best', 'upper left', 'upper center', 'upper right', 'lower left', 'lower center', 'lower right', 'center left', 'center right', 'center', (x, y)}
                    #> best表示自动安置在图表最少的位置
                    #> (x, y)表示图表内位置，x, y ∈ [0, 1]
)
```

### 坐标轴
#### 图片标题 `title`
#### 横、纵坐标轴标签 `xlabel/ylabel`
#### 横、纵坐标轴刻度间隔 `MultipleLocator`
#### 横、纵坐标轴刻度文本 `xticks/yticks`