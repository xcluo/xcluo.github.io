#### IOUtils
```java
// 文件读取，IOUtiles.readLines一次性读取了所有行数据
List<String> samples = new LinkedList<>();
InputStream is = new FileInputStream(path);
for (String line : IOUtils.readLines(is, "utf8")) {
    samples.add(JSON.parseObject(line, CorpusTextSample.class));
}
```

#### BufferedReader
```java
// BufferedReader每次读一行数据
String path = "";
String inFileName = "";
String outFileName = "";
String line;
try (InputStream in = new FileInputStream(path + inFileName);
     BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
     OutputStream out = new FileOutputStream(path + outFileName);
     Writer writer = new OutputStreamWriter(out, StandardCharsets.UTF_8)) {
    while ((line = reader.readLine()) != null) {
    
    }
}
```
