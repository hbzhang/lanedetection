#!/bin/bash
bash Anaconda3-5.2.0-Linux-x86_64.sh -bfp /usr/local
conda install -y -q -c conda-forge -c omnia/label/cuda100 -c omnia openmm python=3.6 --force
#cd /content/drive/MyDrive/Colab-Notebooks/TuSimple/LSTR
conda create --name lstr1 --file environment.txt --force
source activate lstr1
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
