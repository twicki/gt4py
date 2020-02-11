FROM jdahm/dawn-gcc9-env AS gt4py-env
RUN apt update && apt install -y --no-install-recommends python3-pip libpython-dev && apt clean
RUN python -m pip install --upgrade pip

FROM gt4py-env AS gt4py
COPY . /usr/src/gt4py
RUN pip install -e /usr/src/gt4py
# Dawn still requires the v1.0.4 GridTools release
RUN cd /usr/src/gt4py && python setup.py install_gt_sources --git-branch=v1.0.4
