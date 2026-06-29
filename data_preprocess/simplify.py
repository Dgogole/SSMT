import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
from multiprocessing import Pool
import trimesh
import vedo
import argparse
import copy
import pyfqmr
import csv
from tqdm import tqdm

def manifold(obj_path,dataroot,output_root_manifold):
    if os.path.exists(output_root_manifold + '/' + obj_path.name):
        return
    mesh = vedo.Mesh(str(obj_path))
    oface_number = len(mesh.cells)
    mface_number = oface_number * 1.2
    # mface_number = 20000
    commandm = '../Manifold/build/manifold ' + str(
        obj_path) + ' ' + output_root_manifold + '/' + obj_path.name + ' ' + str(int(mface_number)) + ' >/dev/null 2>&1'
    try:
        status1 = os.system(commandm)
    except:
        if status1 != 0:
            raise Exception('wrong, command=%s, status=%s' % (commandm, status1))

def simplify(obj_path,output_root_simplify,output_root_manifold):
    commands = '../Manifold/build/simplify -i ' + output_root_manifold + '/' + obj_path.name + ' -o ' + output_root_simplify + '/' + obj_path.name + ' -m -f ' + str(
                256) + ' >/dev/null 2>&1'
    try:
        status = os.system(commands)
    except:
        if status != 0:
            raise Exception('wrong, command=%s, status=%s' % (commands, status))

def quad_simplify(obj_path,output_root_simplify,output_root_manifold):
    if os.path.exists(os.path.join(output_root_simplify,obj_path.name)):
        return
    try:
        mesh = trimesh.load_mesh(os.path.join(output_root_manifold,obj_path.name))
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
        mesh_simplifier.simplify_mesh(target_count = 256, aggressiveness=7, preserve_border=True, verbose=0)
        vertices, faces, normals = mesh_simplifier.getMesh()
        mesh.vertices = vertices
        mesh.faces = faces
        mesh.export(os.path.join(output_root_simplify,obj_path.name))
    except Exception as e:
        raise Exception(f"Failed to process {obj_path.name}: {str(e)}")

def process_file(args):
    """处理单个文件 - 用于多进程"""
    obj_path, output_root_simplify, output_root_manifold = args
    try:
        quad_simplify(obj_path, output_root_simplify, output_root_manifold)
        return (obj_path.name, True, None)
    except Exception as e:
        return (obj_path.name, False, str(e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--manifold_before', type=str, required=True)
    parser.add_argument('--simplify_before', type=str, required=True)
    args = parser.parse_args()

    root = args.dataroot
    dataroot = Path(root)
    output_root_manifold = args.manifold_before
    output_root_simplify = args.simplify_before
    if not os.path.exists(output_root_manifold):
        os.mkdir(output_root_manifold)
    if not os.path.exists(output_root_simplify):
        os.mkdir(output_root_simplify)

    # 获取所有需要处理的文件
    all_files = [f for f in dataroot.iterdir() if f.is_file()]
    print(f"找到 {len(all_files)} 个文件待处理")

    # 过滤已存在的文件
    files_to_process = [f for f in all_files if not os.path.exists(os.path.join(output_root_simplify, f.name))]
    print(f"需要处理 {len(files_to_process)} 个新文件")

    if len(files_to_process) == 0:
        print("所有文件都已处理完成!")
    else:
        # 使用 imap_unordered 以便与 tqdm 配合
        pool = Pool(processes=16)

        # 准备参数列表
        args_list = [(f, output_root_simplify, output_root_manifold) for f in files_to_process]

        # 记录失败的文件
        failed_files = []

        # 使用 tqdm 显示进度
        for filename, success, error in tqdm(pool.imap_unordered(process_file, args_list), total=len(files_to_process), desc="Simplify处理进度"):
            if not success:
                failed_files.append((filename, error))

        pool.close()
        pool.join()

        print(f"\n处理完成! 成功: {len(files_to_process) - len(failed_files)}/{len(files_to_process)}")

        if failed_files:
            print(f"\n失败的文件数量: {len(failed_files)}")
            print("前10个失败文件及错误:")
            for filename, error in failed_files[:10]:
                print(f"  - {filename}: {error}")
            if len(failed_files) > 10:
                print(f"  ... 还有 {len(failed_files) - 10} 个文件失败")
