# 第二章 变量和基本类型

## 1.基本内置类型

### 算术类型

除去字符型和其他扩展的字符型之外，其他整型可以分为带符号的(signed)和无符号（unsigned），带符号的可以表示正数、负数、0，无符号的仅能表示**大于等于0**的值。

unsigned int可以表示int

字符型可以分为:char、signed char和unsigned char，但实际表现形式只有两种：有符号的和无符号的，char具体表示哪种由编译器决定。

### 类型转换

当一个表达式中既有无符号数又有int值时，int会自动转换为无符号数

```c++
unsigned u= 10;
int i=-42;
std::cout<< i + i << std::endl; //输出-84
std::cout<< u + i << std::endl; //如果int占32位，输出4294967264
//实际上等于负数加无符号数的模

```

无符号数减去一个值，无论是否为无符号数，都要确保结果不是负值

```C++
unsigned int a = 42, b = 10 ;
std::cout<< a - b << std::endl; // 正确，输出32
std::cout<< b - a << std::endl; // 正确，输出的为取模后的值
```

**切勿混用带符号类型和无符号类型**

## 2.变量

### 变量声明与定义的关系

C++支持**分离式编译**，允许将程序分割为多个文件，每个文件可被独立编译。

声明（declaration）使得名字为程序所知，定义（definition）负责创建与名字关联的实体。

声明规定了变量的类型和名字，在这一点上定义与之相同，但除此之外，**定义还申请存储空间，也可能为变量赋一个初始值**。

```c++
extern int i // 声明i
int j // 声明并定义j
extern double pi = 3.14 // 定义
```

**变量能且仅能被定义依次，但可以被多次声明**

## 3.复合类型

### 引用

引用为对象起了另外一个名字

```c++
int ival=1024;
int &refval=ival; // refval 指向 ival （是 ival 的另一个名字 ）
int &refval2; // 报错：引用必须被初始化
```

**引用必须初始化**

**引用即别名**  引用并非对象，它只是为一个已经存在的对象所起的另外一个名字

**引用本身不是一个对象，所以不能定义引用的引用**

```c++
int i1 = 1024 , i2 = 2048 ; // i1，i2都是int
int &r = i , r2 = i2 ; // r是引用 r2是int
int i3 = 1024 , &ri = i3 ; // i3是int ri是引用
int &r3 = i3 , &r4 = i2 ; // r3 r4都是int
```

### 指针

**赋值永远改变的是等号左侧的对象**

void*指针可以存放任意对象的地址

```;
double obj = 3.14 ,*pd = &obj ;
void *pv = &obj ;
pv = pd ;
```

## 4.const限定符



**指向常量的指针也没有规定其所指的对象必须是一个常量，仅要求不能通过该指针改变对象的值，而没有规定不能通过其他途经改变**

### 顶层const

用名词顶层const表示指针本身是一个常量，而用底层const表示指针所指的对象是一个常量

```c++
int i = 0 ;
int *const p1 = &i ; //不能改变p1的值，这是顶层const
const int ci = 42 ; //不能改变ci的值，这是顶层const
const int * p2 = &ci ; //可以改变p2，这是底层const
const int *const p3 = p2 ; //右边的是顶层const，左边的是底层const
const int &r = ci ; //用于声明的const都是底层const
```

## 5.处理类型

**auto一般会忽略顶层const，保留底层const**

```c++
const int ci = i , &cr = ci ;
auto b = ci; //b是一个整数
auto c = cr; //c是一个整数
auto d = &i; //d是一个整型指针
auto e = &ci;//e是一个指向整数常量的指针（对常量对象取地址是一种底层const）

const auto f = ci; //f是const int
```



### decltype类型指示符

decltype的作用是选择并返回操作数的数据类型，但并不执行此函数

```c++
decltype(f()) sum = x ;
```

**如果decltype使用的表达式不是一个变量，则decltype返回表达式结果对应的类型**

```c++
int i = 42 , *p = &i , &r = i;
decltype( r + 0 ) b; // b是一个int
decltype(*p) c ; // 错误，c是int&，引用必须初始化
```



**如果变量名加上一对或多对括号，则结果与不加括号时结果不同，编译器将会把他当作一个表达式，此是会得到引用类型**

decltype((i))的结果永远是引用，而decltype(i)的结果只有当i是引用时才会是引用。

## 自定义数据结构



# 第三章 字符串、向量和数组

## 1. 命名空间的using声明



## 2. 标准库类型string

string::size_type是一个无符号类型的值

size函数返回的是一个无符号整型数，因此不能在表达式中混用带符号数和size

处理一个string对象中的每个字符

```c++
string  str("hello world");
for( auto c : str )
	cout << c << endl;
```

## 3. 标准库类型vector

```c++
vector<T> v1 
//下面两行含义相同：v2中包含了v1所有元素的副本
vector<T> v2 (v1)
vector<T> v2 = v1 
//v3包含了n个元素，每个元素的值都是val
vector<T> v3 ( n , val )
//v4包含n个重复的值初始化的对象
vector<T> v4 ( n )
//下面两行含义相同： v5 包含了初始值个数的元素，每个元素被赋予相应的初始值
vector<T> v5 { a , b , c }
vector<T> v5 = { a , b , c }
```



