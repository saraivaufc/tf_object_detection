# tf_object_detection

Get gdal development libraries:

```
$ sudo apt-add-repository ppa:ubuntugis/ubuntugis-unstable
$ sudo apt-get update
$ sudo apt-get install libgdal-dev
$ sudo apt-get install python-dev
$ sudo apt-get install gdal-bin python-gdal python3-gdal
$ sudo apt-get install python3-tk
```

Create and activate a virtual environment:

```
$ virtualenv env
$ source env/bin/activate
```

Install GDAL:

```
(env) $ pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
```

Install Others Requirements

```
(env) $ pip install -r requirements.txt
```

To use TensorFlow with GPU, install:

```
sudo apt-get install libcupti-dev
```
