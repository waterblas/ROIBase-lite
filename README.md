# Requirements

numpy>=1.14.0
scipy>=1.0.0
jieba==0.39
protobuf>=3.6.1

Download `loc.vec.bin` [file](https://pan.baidu.com/s/1YJe6xjObzyRaV0PmX-OoZg)
put `loc.vec.bin` in `./recognise_district/data/`

# Usage

## recognise_district

To use (with caution), simply do::

```
import jieba
from recognise_district import recog



rd= recog.init()
text = """
陕西五兄妹打遗产官司未有结果，多数拆迁赔偿已被分给一人泾阳县的杨金梅四姐妹和她们的亲兄弟杨文一，因为家产问题，已经在法院“缠斗”十余年。
"""

words = jieba.lcut(text)
path = rd.detect(words)

print(path)
print(rd.path2name(path))
print(rd.path2str(path))

# output fullname directly
rd.detect_for_fullname(words)

```

output:
```
[610423, 610400, 610000]
['泾阳县', '咸阳市', '陕西省']
泾阳县 咸阳市 陕西省
```

根据推断上下文推断不确定的区域(实验功能，不准确)
```
rd.path2name(rd.detect_with_infer(['广东', '腾讯']))
rd.path2name(rd.detect_with_infer(['广东', '腾讯'], step=2))
```
output:
```
['', '深圳市', '广东省']
['南山区', '深圳市', '广东省']
```

