#!/bin/bash

#sudo python3 -m pip install wheel pillow pandas pyarrow s3fs fsspec keras pyspark dm-tree
# To fix exception TypeError: Cannot convert numpy.ndarray to numpy.ndarray
#sudo python3 -m pip install --upgrade 'numpy<2.0' 'pandas>=2.2'
sudo python3 -m pip install wheel==0.44.0 pillow==11.0.0 pandas==2.2.3 pyarrow==18.0.0 s3fs fsspec keras==3.6.0 pyspark==3.5.3 dm-tree numpy==1.26.4 tensorflow==2.18.0
