import torch
import hashlib


def simple_state_dict_hash(state_dict):
    """
    对 state_dict 中每个键对应的参数（或值）生成 MD5 哈希值。
    如果 value 是 torch.Tensor，则将其转为 bytes；否则将其转换为字符串后编码。
    """
    md5 = hashlib.md5()
    # 对所有键排序，确保顺序一致
    for key, value in sorted(state_dict.items()):
        md5.update(key.encode('utf-8'))
        if isinstance(value, torch.Tensor):
            md5.update(value.cpu().numpy().tobytes())
        else:
            md5.update(str(value).encode('utf-8'))
    return md5.hexdigest()


def compare_models(path1, path2):
    # 加载两个模型或 checkpoint
    checkpoint1 = torch.load(path1, map_location='cpu')
    checkpoint2 = torch.load(path2, map_location='cpu')

    # 如果加载的是包含 'state_dict' 的字典，则取出 state_dict，否则认为加载的就是 state_dict
    if isinstance(checkpoint1, dict) and "state_dict" in checkpoint1:
        state_dict1 = checkpoint1["state_dict"]
    else:
        state_dict1 = checkpoint1

    if isinstance(checkpoint2, dict) and "state_dict" in checkpoint2:
        state_dict2 = checkpoint2["state_dict"]
    else:
        state_dict2 = checkpoint2

    # 计算两个 state_dict 的哈希值
    hash1 = simple_state_dict_hash(state_dict1)
    hash2 = simple_state_dict_hash(state_dict2)

    print("Model 1 hash:", hash1)
    print("Model 2 hash:", hash2)
    if hash1 == hash2:
        print("两个模型完全一致。")
    else:
        print("两个模型不同。")


if __name__ == '__main__':
    # 修改以下路径为你的模型文件路径
    path1 = "../../scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/sac/02111230/agent1_sac_20000.pt"
    path2 = "../../scripts/results/SingleCombat/1v1/NoWeapon/Selfplay/sac/02111230/agent1_sac_60000.pt"
    compare_models(path1, path2)
