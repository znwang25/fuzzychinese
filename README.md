fuzzychinese
=====
形近词中文模糊匹配

A simple tool to fuzzy match chinese words, particular useful for proper name matching and address matching. 

一个可以模糊匹配形近字词的小工具。对于专有名词，地址的匹配尤其有用。

## 安装说明
```
pip install fuzzychinese
```

## 使用说明
首先使用想要匹配的字典对模型进行训练。

然后用`FuzzyChineseMatch.transform(raw_words, n)` 来快速查找与`raw_words`的词最相近的前n个词。

训练模型时有三种分析方式可以选择，笔划分析(`stroke`)，部首分析(`radical`)，和单字分析(`char`)。也可以通过调整`ngram_range`的值来提高模型性能。

匹配完成后返回相似度分数，匹配的相近词语及其原有索引号。

```python
    import pandas as pd
    from fuzzychinese import FuzzyChineseMatch
    test_dict =  pd.Series(['长白朝鲜族自治县','长阳土家族自治县','城步苗族自治县','达尔罕茂明安联合旗','汨罗市'])
    raw_word = pd.Series(['达茂联合旗','长阳县','汩罗市'])
    assert('汩罗市'!='汨罗市') # They are not the same!

    fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer='stroke')
    fcm.fit(test_dict)
    top2_similar = fcm.transform(raw_word, n=2)
    res = pd.concat([
        raw_word,
        pd.DataFrame(top2_similar, columns=['top1', 'top2']),
        pd.DataFrame(
            fcm.get_similarity_score(),
            columns=['top1_score', 'top2_score']),
        pd.DataFrame(
            fcm.get_index(),
            columns=['top1_index', 'top2_index'])],
                    axis=1)
```

|            | top1               | top2             | top1_score | top2_score | top1_index | top2_index |
| ---------- | ------------------ | ---------------- | ---------- | ---------- | ---------- | ---------- |
| 达茂联合旗 | 达尔罕茂明安联合旗 | 长白朝鲜族自治县 | 0.824751   | 0.287237   | 3          | 0          |
| 长阳县     | 长阳土家族自治县   | 长白朝鲜族自治县 | 0.610285   | 0.475000   | 1          | 0          |
| 汩罗市     | 汨罗市             | 长白朝鲜族自治县 | 1.000000   | 0.152093   | 4          | 0          |

## 其他功能
+ 直接使用`Stroke`, `Radical`进行汉字分解。
    ```python
    stroke = Stroke()
    radical = Radical()
    print("像", stroke.get_stroke("像"))
    print("像", radical.get_radical("像"))
    ```
    ```
    像 ㇒〡㇒㇇〡㇕一㇒㇁㇒㇒㇒㇏
    像 人象
    ```
+ 使用`FuzzyChineseMatch.compare_two_columns(X, Y)`对每一行的两个词进行比较，获得相似度分数。

+ 详情请参见[说明文档](http://znwang.me/fuzzychinese/).

## 致谢

拆字数据来自于 [漢語拆字字典](https://github.com/kfcd/chaizi) by [開放詞典網](http://kaifangcidian.com/)。

## Installation
```
pip install fuzzychinese
```

## Quickstart

First train a model with the target list of words you want to match to. 

Then use `FuzzyChineseMatch.transform(raw_words, n)` to find top n most similar words in the target for your `raw_words` . 

There are three analyzers to choose from when training a model: `stroke`, `radical`, and `char`. You can also change `ngram_range` to fine-tune the model.

After the matching, similarity score, matched words and its corresponding index are returned. 

```python
    from fuzzychinese import FuzzyChineseMatch
    test_dict =  pd.Series(['长白朝鲜族自治县','长阳土家族自治县','城步苗族自治县','达尔罕茂明安联合旗','汨罗市'])
    raw_word = pd.Series(['达茂联合旗','长阳县','汩罗市'])
    assert('汩罗市'!='汨罗市') # They are not the same!

    fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer='stroke')
    fcm.fit(test_dict)
    top2_similar = fcm.transform(raw_word, n=2)
    res = pd.concat([
        raw_word,
        pd.DataFrame(top2_similar, columns=['top1', 'top2']),
        pd.DataFrame(
            fcm.get_similarity_score(),
            columns=['top1_score', 'top2_score']),
        pd.DataFrame(
            fcm.get_index(),
            columns=['top1_index', 'top2_index'])],
                    axis=1)
```

|            | top1               | top2             | top1_score | top2_score | top1_index | top2_index |
| ---------- | ------------------ | ---------------- | ---------- | ---------- | ---------- | ---------- |
| 达茂联合旗 | 达尔罕茂明安联合旗 | 长白朝鲜族自治县 | 0.824751   | 0.287237   | 3          | 0          |
| 长阳县     | 长阳土家族自治县   | 长白朝鲜族自治县 | 0.610285   | 0.475000   | 1          | 0          |
| 汩罗市     | 汨罗市             | 长白朝鲜族自治县 | 1.000000   | 0.152093   | 4          | 0          |

## Other use
+ Directly use `Stroke`, `Radical` to decompose Chinese character into strokes or radicals.
    ```python
    stroke = Stroke()
    radical = Radical()
    print("像", stroke.get_stroke("像"))
    print("像", radical.get_radical("像"))
    ```
    ```
    像 ㇒〡㇒㇇〡㇕一㇒㇁㇒㇒㇒㇏
    像 人象
    ```
+ Use `FuzzyChineseMatch.compare_two_columns(X, Y)` to compare the pair of words in each row to get similarity score.

+ See [documentation](http://znwang.me/fuzzychinese/) for details.


## Credits

Data for Chinese radicals are from [漢語拆字字典](https://github.com/kfcd/chaizi) by [開放詞典網](http://kaifangcidian.com/).