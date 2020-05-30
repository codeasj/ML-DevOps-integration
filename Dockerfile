FROM centos

RUN yum install -y python36 

RUN pip3 install numpy

RUN pip3 install keras==2.3.1

RUN yum install -y python36-devel

RUN yum install -y epel-release

RUN pip3 install --upgrade pip

RUN yum groupinstall -y 'development tools'

RUN pip3  install  pillow

RUN pip3 install opencv-python

RUN pip3 install pandas

RUN pip3 install setuptools

RUN pip3 install tensorflow==2.0.0

CMD ["python3","/mlops_task_project/cnn_code.py"]
