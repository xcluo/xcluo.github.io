#### JSON

```Java title="JSONObject"
// json.loads_map，左式指定{键: 值}类型
Map<String, String> map = JSONObject.parseObject(line, Map.class)

// json.loads_list，方法中指定list中元素类型
List<Integer> list = JSONObject.parseArray(a, Integer.class)

// json.dumps
JSONObject.toJSONString(object)
```