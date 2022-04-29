###运行相关系数匹配+最小二乘平差：
1.解压Match.zip到Match文件夹，下载opencv4.5.4替换Match中的opencv4.5.4文件夹。
2.运行程序请先基于Match.cpp构建整体工程sln并配置opencv c++(include及lib，dll)
说明：Match.cpp为包含主函数的文件，
	由于opencv4.5.4及相关编译文件太大因此只提交了cpp文件
###运行Sift
1.解压Sift.zip到Sift文件
2.请在虚拟环境中配置opencv，numpy，matplotlib，time(pip install)
3.在命令行中执行python main.py运行特征匹配及特征提取
说明：由于没有运用CUDA加速，运算速度非常慢，需要十几分钟左右，请耐心等待结果
	如果需要查看中间结果如特征点提取结果，可以去掉相关cv2.imshow的注释
