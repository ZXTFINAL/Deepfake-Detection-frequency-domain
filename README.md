# Deepfake-Detection

Detection solution for Deepfake images
在频率角度对深度gan换脸和deepfake的图片进行特征提取并检测
![detector_network](https://user-images.githubusercontent.com/53009474/203906791-0dc8936e-6934-425c-b697-e9ac36a0049a.png)

依赖参考requirements.txt

测试环境为 Nvidia A30 ubuntu 22.02

python 3.8

cuda 11.7 加速


提供切人脸图片的工具 脚本extract_face_img.py

Examples here came from Google public datasets
示例样本来于谷歌公开数据集
![000001](https://user-images.githubusercontent.com/53009474/203890656-6d835a2b-8f09-4afd-a172-5bd9bbacdaa4.png)
![000005](https://user-images.githubusercontent.com/53009474/203890675-a696d92a-a605-4648-be31-91041612c527.png)

Get fourier-transform features and high-pass filtering features
获取傅里叶频谱图和高通滤波
![image](https://user-images.githubusercontent.com/53009474/203890416-f91469fb-e3f7-4312-9ef8-f7be341e856a.png)

Definition of examples for training should be various.
训练样本尽量包含各种清晰度的样本

![cut_1real](https://user-images.githubusercontent.com/53009474/203890540-3cce811e-e548-495c-b3f7-6fa486be3c7c.jpg)
![cut_66](https://user-images.githubusercontent.com/53009474/203890575-e8accd23-b82d-4e6a-a837-ac475b53ef8e.jpg)



