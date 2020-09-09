#!/usr/bin/env bash
if [[ $HOSTNAME =~ .*daint.* ]]
then
    module load /project/d107/install/modulefiles/boost/1_74_0
else
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-10.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
fi
python3 -m pip install --user virtualenv
python3 -m venv gt4pyenv
source gt4pyenv/bin/activate
# pip3 install --upgrade wheel # might need this later
git clone https://github.com/VulcanClimateModeling/gt4py.git gt4py
pip3 install cupy-cuda101==7.7.0
pip3 install --no-cache-dir -e gt4py[cupy-cuda101]
python3 -m gt4py.gt_src_manager install
python3 -m pytest gt4py
