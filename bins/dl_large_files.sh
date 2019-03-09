#!/bin/bash


fileid="1ZG7DsldifePgE3vnQ6klW7VKrL0TRUWx"
filename="mobile_image.tar.gz"
curl -c ./cookie-"${fileid}" -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie-"${fileid}" "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie-"${fileid}"`&id=${fileid}" -o ${filename}


fileid="13acmJdLbNdij8c12dmlVeFLiodIEDYJJ"
filename="fashion_image.tar.gz"
curl -c ./cookie-"${fileid}" -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie-"${fileid}" "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie-"${fileid}"`&id=${fileid}" -o ${filename}


fileid="1fkvJtFEoE_XLqGXmHRdhRdahq8oaeVXJ"
filename="beauty_image.tar.gz"
curl -c ./cookie-"${fileid}" -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie-"${fileid}" "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie-"${fileid}"`&id=${fileid}" -o ${filename}

rm cookie-*
