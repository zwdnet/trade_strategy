# coding:utf-8
# 将程序上传到服务器上执行
import os
import sys
from functools import wraps
import time


# 两个常用的工具函数，装饰器
# 工具函数，在上传到服务器上运行时改变当前目录
def change_dir(func):
    @wraps(func)
    def change(*args, **kwargs):
        oldpath = os.getcwd()
        newpath = "/home/code/"
        os.chdir(newpath)
        r = func(*args, **kwargs)
        os.chdir(oldpath)
        return r
    return change
    
    
# 工具函数，计算函数运行时间    
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{}的运行时间为 : {}秒'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper


# 运行代码前准备
def before_run(user, server):
    # 创建输出目录
    s = "ssh " + user + "@" + server + " \"sudo mkdir ~/code/output\""
    # print("测试4", s)
    os.system(s)
    # 创建数据目录
    s = "ssh " + user + "@" + server + " \"sudo mkdir ~/code/datas\""
    # print("测试4", s)
    os.system(s)
    # 更改目录权限
    s = "ssh root@" + server +  " -p 2222 \"chown -R 1000:1000 /home/code/output\""
    os.system(s)
    s = "ssh root@" + server +  " -p 2222 \"chown -R 1000:1000 /home/code/datas\""
    os.system(s)
    # 将本地目录所有文件上传至容器
    s = "scp ./* " + user + "@" + server + ":~/code"
    # print("测试3", s)
    os.system(s)
    # 删除md文件，避免影响博客更新
    s = "ssh root@" + server +  " -p 2222 \"rm /home/code/README.md\""
    os.system(s)
    # 更改服务器容器里的当前目录
    s = "ssh root@" + server +  " -p 2222 \"cd /home/code\""
    os.system(s)
    
    
# 运行完成后的后续操作
def after_run(user, server):
    # 看本地是否有输出目录，没有则新建
    if os.path.exists("./output") == False:
        s = "mkdir ./output"
        os.system(s)
    # 看本地是否有数据目录，没有则新建
    if os.path.exists("./datas") == False:
        s = "mkdir ./datas"
        os.system(s)
    # 改变文件所有者
    s = "ssh root@" + server +  " -p 2222 \"chown 1000:1000 -R /home/code/*"
    # 将代码目录里所有输出文件传回
    s = "scp -r " + user +"@" + server + ":~/code/output/* ./output/"
    os.system(s)
    # 将代码目录里所有数据文件传回
    # s = "scp -r " + user +"@" + server + ":~/code/datas/* ./datas/"
    # print("测试5", s)
    # os.system(s)


# 上传代码至服务器并运行
def run(gpus, user, server):
    # 上传本目录所有文件再执行指定文件
    if gpus == "all":
        before_run(user, server)
        # 运行指定代码
        s = "ssh root@" + server +  " -p 2222 \"python -u /home/code/" + sys.argv[2] + "\""
        # print("测试4", s)
        print("正在运行代码……\n")
        os.system(s)
        after_run(user, server)
    elif gpus == "copy":
        if sys.argv[2] == "up":
            before_run(user, server)
        elif sys.argv[2] == "down":
            after_run(user, server)
        elif sys.argv[2] == "up_data":
            s = "scp ./datas/* " + user + "@" + server + ":~/code/datas"
            os.system(s)
        elif sys.argv[2] == "down_data":
            s = "scp " + user + "@" + server + ":~/code/datas/* ./datas/"
            os.system(s)
        else:
            print("输入错误，copy后面要跟up、down、up_data或down_data。")
    # 用pytest执行单元测试
    elif gpus == "test":
        before_run(user, server)
        # 运行指定代码
        arg_len = len(sys.argv)
        if arg_len == 3:
            s = "ssh root@" + server +  " -p 2222 \"/opt/conda/bin/pytest -v /home/code/" + sys.argv[2] + "\""
        # elif arg_len == 2:
        #     s = "ssh root@" + server +  " -p 2222 \"/opt/conda/bin/pytest -v\""
        else:
            print("输入有误!")
            return
        print("测试4", s)
        print("正在测试代码……\n")
        os.system(s)
        after_run(user, server)
    elif gpus == "clean":
        # 清除服务器代码目录所有文件
        s = "ssh root@" + server +  " -p 2222 \"rm -rf /home/code/*\""
        # print("测试1", s)
        print("清除服务器代码目录上的文件……\n")
        os.system(s)
    else:
        print("输入有误，格式: python run.py all/copy/test filename.py 其中filename.py为要运行/测试的源文件。")
        
        
# 主函数
def main():
    gpus = sys.argv[1]
    # 读取服务器IP地址，自己编辑serverIP.txt去
    with open("serverIP.txt", "rt") as f:
        server_list = f.readlines()
    for s in server_list:
        s = s.replace('\n', '').replace('\r', '')
        # print(s)
        if s[0] != "#":
            res = s.split("@")
            username = res[0]
            server = res[1]
            # print("测试", username, server)
            # input("按任意键继续")
            run(gpus, username, server)
            return    


if __name__ == "__main__":
    main()
        
    
