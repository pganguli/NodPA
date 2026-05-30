python3 -m pip install --user --upgrade pip setuptools
python3 -m pip install -r requirements-base.txt

python3 dnn-models\transform.py --data-output-dir build-windows --target msp432 --hawaii har-dnp
cmake -B build-windows -G "Visual Studio 17 2022"
cd build-windows
msbuild intermittent-cnn.sln
cd ..
.\build-windows\Debug\intermittent-cnn.exe
