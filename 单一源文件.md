



## 单一源文件

源文件为main.c



```cmake
cmake_minimum_required (VERSION 2.8)
#所要求的cmake最低版本
project (demo)
#工程名
add_executable(main main.c)
#main表示生成的可执行文件名 main.c表示使用的源文件为main.c
```



## 同一目录下多个源文件

在之前的目录下添加2个文件，testFunc.c和testFunc.h

```cmake
cmake_minimum_required (VERSION 2.8)

project (demo)

add_executable(main main.c testFunc.c)

```

add_executable 需要添加testFunc.c

当所需要的源文件较多时，cmake提供了一个命令可以把指定目录下所有的源文件存储在一个变量中，这个命令就是 **aux_source_directory(dir var)**  第一个参数dir是指定目录，第二个参数var是用于存放源文件列表的变量

在main.c所在目录下再添加2个文件，testFunc1.c和testFunc1.h

```cmake
cmake_minimum_required (VERSION 2.8)

project (demo)

aux_source_directory(. SRC_LIST)

add_executable(main ${SRC_LIST})
```

使用aux_source_directory把当前目录下的源文件存列表存放到变量SRC_LIST里，然后在add_executable里调用SRC_LIST

## 不同目录下多个源文件

| main.c          |             |      |
| --------------- | ----------- | ---- |
| --------------  | testFunc    |      |
|                 | testFunc.c  |      |
|                 | testFunc.h  |      |
| --------------- | testFunc1   |      |
|                 | testFunc1.c |      |
|                 | testFunc1.h |      |

```cmake
cmake_minimum_required (VERSION 2.8)

project (demo)

include_directories (test_func test_func1)
#用来向工程添加多个指定头文件的搜索路径，路径之间用空格分隔
aux_source_directory (test_func SRC_LIST)
aux_source_directory (test_func1 SRC_LIST1)

add_executable (main main.c ${SRC_LIST} ${SRC_LIST1})
```

