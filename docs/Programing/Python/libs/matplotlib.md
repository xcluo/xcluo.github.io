```python
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator   # 导入坐标轴控制库

plt.rcParams['font.sans-serif']=['SimHei']      # 显示中文不乱码
plt.rcParams['axes.unicode_minus']=False        # 显示负号不乱码
```

### 绘图

#### 折线图 `plot`

```python
def plot(
        *args,
        scalex=True,
        scaley=True,
        data=None,
        **kwargs 
) 
```

#### 散点图 `sactter`

```python
def scatter(
        x,              # 横坐标序列
        y,              # 对应的纵坐标序列
        s=None,
        c=None,
        marker=None,
)
```

#### 直方图 `hist`

对一维序列或二维数列频数进行统计，纵坐标表示频数

```python
def hist(
        x,              # 一维序列或者2维序列 seq([int,seq])
        bins=None,      # 直方图条形个数，缺省为 10
        range=None,     # 横坐标显示范围，（left, right)，缺省为x中的最大值和最小值
        log=False,      # 纵坐标统计值是否取对数
        color=None, density=False, weights=None, cumulative=False, bottom=None, 
        histtype='bar', align='mid', orientation='vertical', rwidth=None, 
        label=None, stacked=False, *, data=None, **kwargs):
```

#### 条形图 `bar`

```python
def bar( 
        x,              # 横坐标序列
        height,         # 对应的纵坐标序列
        width=0.8,      # 条形宽度
        bottom=0,       # 纵坐标`offset`值
        *, 
        align="center", # 条形对齐方式 {'center': 刻度居中, 'edge': 刻度居左}
        **kwargs):
```
> 可以理解为选择性的 `hist` 图

#### 子图 `subplots`

### 图形界面

#### 画布 `figure`

#### 文本 `text`

#### 网格线 `grid`

#### 图例 `legned`

```python
def legend(
        *args,
        **kwargs    
        # handles，isinstanceof(handles, Iterable)，元素为绘制的图表artist
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