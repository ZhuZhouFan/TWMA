import os
import shutil
import joblib
from pathlib import Path

def delete_item(path):
    """
    删除单个文件或文件夹。

    :param path: 文件或文件夹的路径
    """
    try:
        if os.path.isfile(path):
            os.remove(path)  # 删除文件
        elif os.path.isdir(path):
            shutil.rmtree(path)  # 删除文件夹及其内容
    except Exception as e:
        print(f"Error deleting {path}: {e}")

def get_all_items(folder_path):
    """
    获取目标文件夹中的所有文件和子文件夹。

    :param folder_path: 目标文件夹的路径
    :return: 文件和子文件夹的完整路径列表
    """
    return [str(item) for item in Path(folder_path).iterdir()]

def delete_items_in_parallel(folder_path, n_jobs=50):
    """
    使用多进程删除目标文件夹中的所有文件和子文件夹。

    :param folder_path: 目标文件夹的路径
    :param n_jobs: 使用的CPU核心数，默认为-1（使用所有核心）
    """
    all_items = get_all_items(folder_path)
    
    if not all_items:
        print("No files or directories found in the specified directory.")
        return
    
    # 使用joblib的Parallel和delayed方法进行多进程删除
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(delete_item)(item) for item in all_items
    )

if __name__ == "__main__":
    # 设置要删除的文件夹路径
    folder_to_delete = '/home/zfzhu/Documents/TWMA_image'
    
    # 执行删除操作
    delete_items_in_parallel(folder_to_delete)
