#### 登录命令
```
lftp <user>:<pwd>@<ip>:<port>

lftp nisp:nisp163@sa-dianxin-ftp.hz.163.org
```


#### 远端命令
```
ls          # 显示远端文件列表
cd          # 切换远端目录
pwd         # 显示远端目录     
mv          # 移动远端文件(也可进行远端文件改名)
mrm         # 批量删除多个远端文件(支持通配符*)
rm          # 删除远端文件
mkdir       # 新建远端目录
rmdir       # 删除远端目录
du          # 计算远端目录大小
```

#### 本地命令
```
!ls         # 显示本地文件列表
lcd         # 切换本地目录 
lpwd        # 显示本地目录 
```


#### 下载命令
```
get         # 下载远端文件
mget        # 批量下载远端文件(支持通配符*)，完全兼容get
pget        # 使用多个线程来下载远端文件, 预设为五个。 
mirror      # 下载整个目录
```

#### 上传命令
```
put         # 上传文件至远端
mput        # 批量上传多个文件(支持通配符*)
mirror -R   # 上传整个目录
```

#### 退出命令
```
bye
exit
```