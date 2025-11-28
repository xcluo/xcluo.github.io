```bash
pip install pytest
```
pytest对标于java中的JUnit

#### pytest.mark
- `@pytest.mark.mark_name` 测试方法标记
- `pytest.mark.parametrize`
- `pytest.mark.skip`
- `pytest.mark.skipif`
- `pytest.mark.xfail`
#### pytest.fixture

#### 常用方法
- `pytest` 运行所有测试
- `pytest test_file.py` 运行特定文件测试
- `pytest test_file.py::test_case` 运行特定文件中的特定测试用例
- `pytest -v/-q` 显示详情/简略
- `pytest -x` 失败时停止
- `pytest -s` 显示打印输出
- `pytest --junitxml=report.xml` 生成报告
- `pytest -m mark_name` 运行特定标记的测试用例，多个`mark_name` 可通过`and, or, not`连接，用`""`包裹
- `pytest -k substring` 测试用例名带有`substring`字串的用例