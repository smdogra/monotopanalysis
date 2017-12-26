# monotopanalysis on LXPLUS
# cmssw releases

cmsrel CMSSW_8_0_26_patch1

cd CMSSW_8_0_26_patch1/src

cmsenv

git clone -b git@github.com:smdogra/monotopanalysis.git

mv monotopanalysis/*  .

-->install numpy_root in your home directory
cd -

git clone git://github.com/rootpy/root_numpy.git

cd root_numpy

python setup.py install --user

Compile the packages, go back to your Panda's cloned directory.

cd PandaCore

./bin/genDict

cd ..

scram b -j 10

cd PandaAnalysis/MonoX/fitting

python makefittingforest.py --region bjet1 #(bjet1--> is  the selection and is defined in the  PandaAnalyzer/MonoX/python/selection.py)
