import sys


def parse_args():
    # 获取命令行参数
    args = sys.argv[1:]  # 忽略第一个参数（脚本名）

    # 初始化默认值
    id_value = None
    order_value = None
    from_value = None

    # 遍历参数来寻找 id 和 order
    for arg in args:
        if arg.startswith("id="):
            id_value = arg.split("=")[1]
        elif arg.startswith("from="):
            from_value = arg.split("=")[1]

    return id_value, from_value
