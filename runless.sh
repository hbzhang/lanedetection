#!/bin/bash
#cd /content/drive/MyDrive/Colab-Notebooks/TuSimple/LSTR
conda create --name lstr2 --file environment.txt --force
source activate lstr2
conda install -c intel mkl_fft --force
pip install -r requirements.txt --ignore-installed
python train.py LSTR --iter 30000

python test.py LSTR --testiter 30000 --modality images --image_root ./ --debug

scp -P 13689 root@2.tcp.ngrok.io:/content/drive/MyDrive/Colab-Notebooks/TuSimple/LSTR/IMG.zip /home/hbzhang/Downloads/

scp -P 18820 images.py root@8.tcp.ngrok.io:/content/drive/MyDrive/Colab-Notebooks/TuSimple/LSTR/test

ssh root@2.tcp.ngrok.io -p 13689
#https://stackoverflow.com/questions/34932288/how-to-pass-a-numpy-ndarray-as-a-color-in-opencv
#https://stackoverflow.com/questions/54560488/python-opencv-obtain-the-region-of-interest-on-an-rgb-image
#https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region
