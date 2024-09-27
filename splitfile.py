def split_large_file(input_filename, output_prefix, chunk_size):
    """
    将大文件分割成多个小文件。
    
    :param input_filename: 输入文件的完整路径
    :param output_prefix: 输出文件的前缀
    :param chunk_size: 每个输出文件的最大字节大小
    """
    chunk_number = 1
    with open(input_filename, 'r', encoding='utf-8') as infile:
        while True:
            # 使用当前chunk_number创建输出文件名
            output_filename = f"{output_prefix}_{chunk_number}.txt"
            with open(output_filename, 'w', encoding='utf-8') as outfile:
                # 读取并写入下一个chunk_size大小的数据到输出文件
                data_chunk = infile.read(chunk_size)
                if not data_chunk:
                    break  # 如果没有更多的数据可读，则退出循环
                outfile.write(data_chunk)
                print(f"已写入文件：{output_filename}")
            chunk_number += 1


if __name__ == '__main__':
    # 500MB
    split_large_file('G:\\nlp\zhwiki_simplified.txt','zhwiki_simplified',200*1024*1024)