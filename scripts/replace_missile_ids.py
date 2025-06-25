import re
import argparse


def replace_missile_ids(input_file, output_file):
    # 正则表达式匹配导弹 ID，例如 A0200M2, B0100M2
    missile_id_pattern = re.compile(r'([A-Z]\d{4})M(\d)')

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 查找并替换导弹 ID
            new_line = missile_id_pattern.sub(r'\g<1>0\2', line)
            outfile.write(new_line)

    print(f"处理完成！新文件已保存到 {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Replace missile IDs in ACMI file (e.g., A0200M2 -> A020002)")
    parser.add_argument('input_file', help="输入的 ACMI 文件路径")
    parser.add_argument('output_file', help="输出的新 ACMI 文件路径")
    args = parser.parse_args()

    replace_missile_ids(args.input_file, args.output_file)


if __name__ == "__main__":
    main()