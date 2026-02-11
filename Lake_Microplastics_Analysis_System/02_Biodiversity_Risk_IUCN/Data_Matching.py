import pandas as pd
import os

# --- 文件路径配置 ---
# 使用您提供的绝对路径
# 注意：在Python中，最好使用正斜杠 / 或者双反斜杠 \\ 来表示路径
small_file_path = r"E:\lake-MP-W\dataset\IUCN\bio\iucn_threatened_species_attributes.csv"
large_file_path = r"E:\lake-MP-W\dataset\IUCN\bio\all_freshwater_species_attributes.csv"
output_filename = r"E:\lake-MP-W\dataset\IUCN\bio\matched_species_output.csv"


# --------------------

def match_species_files(small_path, large_path, output_path):
    """
    加载两个物种属性CSV文件，根据科学名称进行匹配，并将结果保存到新文件。
    """
    # 检查文件是否存在
    if not os.path.exists(small_path):
        print(f"错误：找不到小文件，请检查路径: '{small_path}'")
        return
    if not os.path.exists(large_path):
        print(f"错误：找不到大文件，请检查路径: '{large_path}'")
        return

    try:
        print("步骤 1: 开始加载文件...")
        # 加载文件时使用 engine='python' 以增加对复杂CSV格式的兼容性
        # on_bad_lines='skip' 会跳过格式错误的行
        small_df = pd.read_csv(small_path, engine='python', on_bad_lines='skip')
        print(f"  - 成功加载小文件 (共 {len(small_df)} 行)")

        large_df = pd.read_csv(large_path, engine='python', on_bad_lines='skip')
        print(f"  - 成功加载大文件 (共 {len(large_df)} 行)")
        print("-" * 30)

        # 步骤 2: 准备用于匹配的列
        # 小文件中的列名为 'SCI_NAME', 大文件中为 'sci_name'
        print("步骤 2: 准备用于匹配的列...")
        if 'SCI_NAME' not in small_df.columns:
            print(f"错误: 小文件中未找到 'SCI_NAME' 列。可用列: {small_df.columns.tolist()}")
            return
        if 'sci_name' not in large_df.columns:
            print(f"错误: 大文件中未找到 'sci_name' 列。可用列: {large_df.columns.tolist()}")
            return

        small_df_renamed = small_df.rename(columns={'SCI_NAME': 'sci_name'})
        print("  - 已将小文件中的 'SCI_NAME' 列重命名为 'sci_name' 以进行匹配。")
        print("-" * 30)

        # 步骤 3: 统一关键列的数据类型为字符串，这是避免合并错误的关键
        print("步骤 3: 统一 'sci_name' 列的数据类型为字符串...")
        small_df_renamed['sci_name'] = small_df_renamed['sci_name'].astype(str)
        large_df['sci_name'] = large_df['sci_name'].astype(str)
        print("  - 数据类型已统一。")
        print("-" * 30)

        # 步骤 4: 执行匹配操作 (inner join)
        print("步骤 4: 开始匹配两个文件中的物种名称...")
        # 使用 sufrefinees 参数来区分来自不同文件的同名列
        matched_df = pd.merge(large_df, small_df_renamed, on='sci_name', how='inner', suffixes=('_large', '_small'))
        print("  - 匹配完成。")
        print("-" * 30)

        # 步骤 5: 保存并报告结果
        print("步骤 5: 保存结果并生成报告...")
        # 使用 utf-8-sig 编码以确保在Excel中正确显示中文等字符
        matched_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print("\n🎉 操作成功完成！ 🎉")
        if not matched_df.empty:
            print(f"在两个文件中找到了 {len(matched_df)} 条完全匹配的物种记录。")
            print(f"详细匹配结果已保存到文件: '{output_path}'")
        else:
            print("操作完成，但在两个文件之间未找到任何共同的物种名称。")

    except Exception as e:
        print(f"处理过程中发生了一个未预料的错误: {e}")
        print("请检查您的CSV文件内容和格式是否正确。")


# --- 执行主函数 ---
if __name__ == "__main__":
    match_species_files(small_file_path, large_file_path, output_filename)