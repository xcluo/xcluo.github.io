### argparse

#### 基本用法

```python
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(...)

args = parser.parse_args()      # 通过`args.var_name` 调用相应参数
```

#### ArgumentParser

```python
# 格式
def add_argument(
    'var_name',                 # 代码传参时不能通过key指定，只能位置匹配，不建议使用                                                          
    '--var_name',               # 代码var_name传参时key可用`var_name`指定
    '-var_abbr',                # 代码var_name传参时key可用简写`var_abbr`指定

    type,                       # 指定参数var_name的类型
    default,                    # 指定参数var_name的缺省值
    required=False,             # 参数var_name是否必须【传参或有初始值】
    help,                       # type(help) == str, 参数var_name的描述信息
)

## int
parser.add_argument('-v', '--var', type=int, default=0, required=False, help='var description')

## float
parser.add_argument('-v', '--var', type=float, default=0, required=False, help='var description')

## Iterable, 输入tuple，如： --var 1 2 3
parser.add_argument('-v', '--var', type=int, nargs=3, required=False, help='var description')    # nargs指定输入的元素个数

## str
parser.add_argument('-v', '--var', type=str, default='', required=False, help='var description')

## bool, stroe_true传参时只需要传入动作：【-v/--var】 即可实现 var=action.split('_')[-1]
parser.add_argument('-v', '--var', action='store_true', required=False, help='var description')
```

### tf.flags

#### 基本用法

```python
import tensorflow as tf

flags = tf.flags

flag.DEFINE_integer/float/string/boole(...)

FLAGS = flags.FLAGS             # 通过FLAGS.var_name 调用相应参数
```

#### flags方法

```python
# 格式
def DEFINE_integer/float/string/bool(
    'var_name',                 # 代码传参时可通过key指定，等价于argparse的 --var_name
    'var_value',                # var_name的缺省值
    'var_description',          # var_name的描述信息
)


## int
flags.DEFINE_integer('var_name', 0, 'var description')

## float
flag.DEFINE_float('var_name', 0.0, 'var description')

## str
flag.DEFINE_string('var_name', '', 'var description')

## bool
flag.DEFINE_bool('var_name', False, 'var description')


## 其他特性：是否必须传参
flags.mark_flag_as_required("var_name")
flags.mark_flags_as_requred(["var_name1", "var_name2", ...])
```