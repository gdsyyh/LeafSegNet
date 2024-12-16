import subprocess

# 设置.ipynb文件和.py文件的路径
input_ipynb = 'LeafOnlySAM.ipynb'
output_py = 'LeafOnlySAM.py'

# 使用nbconvert命令行工具进行转换
command = f'jupyter nbconvert --to script {input_ipynb} --output {output_py}'

# 执行命令
subprocess.run(command, shell=True)
