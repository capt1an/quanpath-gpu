cmake_minimum_required(VERSION 3.10)  # 根据你的 CMake 版本调整

project(MyCUDAProject)

# 设置 CUDA 语言支持
enable_language(CUDA)

# 查找 CUDA
find_package(CUDA REQUIRED)

# 从环境变量获取 cuQuantum 根目录
if(NOT DEFINED ENV{CUQUANTUM_ROOT})
    message(FATAL_ERROR "环境变量 CUQUANTUM_ROOT 未定义。请设置该变量指向 cuQuantum 的安装路径。")
endif()

set(CUQUANTUM_ROOT $ENV{CUQUANTUM_ROOT})

# 设置 cuQuantum 包含路径和库路径
set(CUQUANTUM_INCLUDE_DIR ${CUQUANTUM_ROOT}/include/libcuquantum/12)
set(CUQUANTUM_LIBRARY_DIR ${CUQUANTUM_ROOT}/lib/x86_64-linux-gnu/libcuquantum/12)

# 添加 cuQuantum 包含路径
include_directories(${CUQUANTUM_INCLUDE_DIR})

# 添加 cuQuantum 库路径
link_directories(${CUQUANTUM_LIBRARY_DIR})

# 获取所有源文件 (假设你的 CUDA 源文件以 .cu 结尾)
file(GLOB SOURCES *.cu)

# 添加可执行文件
add_executable(MyCUDAProject ${SOURCES})

# 设置编译选项
target_compile_options(MyCUDAProject PRIVATE
    -g  # 生成调试信息
    -G  # 生成调试信息并嵌入到目标文件中
)

# 链接 custatevec 和 cutensornet 库
target_link_libraries(MyCUDAProject custatevec cutensornet)

# 链接 CUDA 运行时库
target_link_libraries(MyCUDAProject ${CUDA_LIBRARIES})