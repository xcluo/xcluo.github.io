

### 数据类型

```javascript
// 原始类型（Primitive Types），支持python中类型注释，例如：str: string
const str = "Hello";           // String
const num = 42;                // Number
const big = 9007199254740991n; // BigInt
const bool = true;             // Boolean
const nothing = null;          // Null
const notDefined = undefined;  // Undefined
const sym = Symbol("id");      // Symbol

// 引用类型（Reference Types），类似于
const obj = { name: "Alice", age: 25 };  // Object
const arr = [1, 2, 3, 4, 5];             // Array
const fn = function() {};               // Function
```

### 变量声明

```javascript
// var - 函数作用域，可重复声明（不推荐）
var name = "Alice";
var name = "Bob"; // 不会报错

// let - 块级作用域，可重新赋值
let age = 25;
age = 26; // ✅ 可以
// let age = 30; // ❌ 错误：不能重复声明

// const - 块级作用域，不可重新赋值
const PI = 3.14159;
// PI = 3.14; // ❌ 错误：不能重新赋值
// 注意：const 对象的内容可以修改
const person = { name: "Alice" };
person.name = "Bob"; // ✅ 可以修改属性
// person = {}; // ❌ 错误：不能重新赋值整个对象
```

#### 字符串拼接

```javascript
const name = "Alice";
const age = 25;

// 旧方式（字符串拼接），类python的`+`运算符
const oldWay = "My name is " + name + " and I am " + age + " years old.";

// 新方式（模板字符串），类python的`f-string`
const newWay = `My name is ${name} and I am ${age} years old.`;

// 支持多行（换行自动插入回车）
const multiLine = `
  Name: ${name}
  Age: ${age}
`;
```

#### 解构赋值

```javascript
const numbers = [1, 2, 3];

/* 数组解构，类似于 a, b, c = numbers */
const [first, second, third] = numbers;
/* 跳过元素，类似于 a, _, c = numbers */
const [a, , c] = numbers;           // a=1, c=3
/* 剩余元素，类似于 a, *tail = numbers */
const [head, ...tail] = numbers;    // head=1, tail=[2,3]

/* 对象解构，获取对象中指定键的值 */
const user = { name: "Alice", age: 25, city: "Beijing" };
const { name, age } = user;         //name="Alice", age=25

/* 重构对象，获取对象中指定键的值并赋给指定变量 */
const { name: userName, age: userAge } = user;      // userName="Alice", userAge=25

/* 重构对象默认赋值情况，name: name */
const { name, role = "user" } = { name: "Alice" }; // name="Alice", role="user"

/* 嵌套解构 */
const person = {
  name: "Alice",
  address: { city: "Beijing", zip: "100000" }
};
const { address: { city } } = person; // city="Beijing"，将address的值解构到city变量
```

#### 展开运算符

```javascript
/* 数组展开，类似于 c = *a, *b */
const arr1 = [1, 2, 3];
const arr2 = [4, 5, 6];
const combined = [...arr1, ...arr2];          // [1,2,3,4,5,6]

/* 对象展开，类似于 dict.update */
const defaults = { theme: "dark", lang: "en" };
const settings = { ...defaults, lang: "zh" }; // { theme: "dark", lang: "zh" }

/* 函数参数，类似于 *args */
function sum(...numbers) {
  return numbers.reduce((a, b) => a + b, 0);
}
sum(1, 2, 3, 4); // 10
```

### 函数相关

```javascript
/* 传统函数 */
function add(a, b) {
  return a + b;
}

/* 箭头函数，类似于 lambda */
const add = (a, b) => a + b;
const square = x => x * x;            // 单参数可省略括号

/* 多行箭头函数，需要花括号和 return */
const greet = name => {
  const message = `Hello, ${name}!`;
  return message;
};

/* this 绑定差异（重要！） */
const obj = {
  name: "Alice",
  /* 传统函数：this 指向调用者 */
  traditional: function() {
    console.log(this.name); // "Alice"
  },
  /* 箭头函数：this 指向调用者外层作用域 */
  arrow: () => {
    console.log(this.name); // undefined（继承全局 this）
  }
};

obj.traditional();          // "Alice"
obj.arrow();                // undefined
```
