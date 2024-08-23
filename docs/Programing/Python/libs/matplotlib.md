```python
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator   # 导入坐标轴控制库

plt.rcParams['font.sans-serif']=['SimHei']      # 显示中文不乱码
plt.rcParams['axes.unicode_minus']=False        # 显示负号不乱码
```

### 绘图
```python
# **kwargs
linewidth=3             # 线条粗度，也可使用 `lw` 表示
linestyle='solid'       # 线条格式，也可使用 `ls` 表示 {'solid', 'dashed', 'dashdot', 'dotted'}
color='red'             # 线条颜色，也可用rgb形式表示，如 '#1f77b4'
marker='.'              # 点的格式，{'.', ',', 'o', '*', '+'}
alpha=0.5               # 不透明度
```

#### 折线图 `plot`

代码格式：`plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)`
    ```python
    def plot(
            *args,          # [x], y, [fmt]
                            # 只有`y`时，对应`x = range(len(y))`  
                            # `fmt : str`，绘图,缺省为 'b-'
            scalex=True,
            scaley=True,
            data=None,
            **kwargs):
    ```

#### 散点图 `sactter`

```python
def scatter(
        x,              # 横坐标序列
        y,              # 对应的纵坐标序列
        s=None,         # 散点面积，序列长度与 `x` 一致
        c=None,         # 散点颜色，缺省为 'b'
        marker=None,    # 散点标记样式，缺省为 'o'
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
        **kwargs)
```
> 可以理解为选择性的 `hist` 图



### 图形界面

#### 画布
``` python
def figure(
        figsize=(24,8), # 宽，长，单位为寸inch
        dpi=300)        # 分辨率
```

#### 子图
1. `plt.subplot`，自动切换子图
2. `plt.subplots`，手动访问子图

#### 标题




#### 图例
依次为绘制的图增加图例说ing
```python
''' plt.legend(['hello', 'world']) '''
def legend(
        *args,
        **kwargs)   
        # handles，isinstanceof(handles, Iterable)，元素为绘制的图表artist
        # labels，isinstanceof(handles, Iterable)，元素绘制的图表artist的自定义文本解释
            #> 与handlers中元素一一对应，未指定handlers则对应artist绘制顺序
        # loc in {'best', 'upper left', 'upper center', 'upper right', 'lower left', 'lower center', 'lower right', 'center left', 'center right', 'center', (x, y)}
            #> best表示自动安置在图表最少的位置
            #> (x, y)表示图表内位置，x, y ∈ [0, 1]
```

#### 网格线
```python
def grid(
        b=None,         # bool值, 是否显示网格
        which='major',  # str, {major, minor, both}, 分别表示显示主要值处、不显示、全显示网格
        axis='both',    # str, {both, x, y}
        **kwargs)
```     

#### 文本 `text`


### 坐标轴
#### 坐标轴
获取当前坐标轴 (get current axes)
```python
ax = plt.gca()
```

#### 坐标轴文本 `xlabel/ylabel`
#### 坐标轴刻度文本 `xticks/yticks`

#### 坐标轴刻度范围
```python
# 设置 xlim, ylim
plt.xlim(xmin, xmax)                            # or plt.xlim([xmin, xmax])
plt.xlim(bottom=1)
plt.xlim(top=2)
# 获取 xlim, ylim
plt.xlim()                                      # (bottom, top)
```

#### 坐标轴刻度间隔
```python
x_major_locator=MultipleLocator(1)              # x轴刻度值间隔，为1的倍数
y_major_locator=MultipleLocator(2)              # y轴刻度值间隔，为2的倍数
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)     # x轴刻设置应用
ax.yaxis.set_major_locator(y_major_locator)     # y轴刻设置应用
```

### 通用组件

#### `font`

#### `marker`

#### `c`