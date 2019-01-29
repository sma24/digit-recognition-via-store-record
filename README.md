# read me
设有两个文件夹：
1. mnist：包含提取mnist图片集的代码，tensorflow识别数字的两个模型代码以及保存下来的convolutional模型的ckpt文件
2. 成果图：项目成功运行的截图，共8张
被隐藏的文件夹： data：空文件夹，只是开docker service时候用到的
其他文件:
main.py + requirements.txt + Dockerfile -->这三个文件用于docker build镜像
docker-compose.yaml  --->用于开docker容器并开启service
外加一个项目report

注意：
整个项目代码以main.py文件为主
