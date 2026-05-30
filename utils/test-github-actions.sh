python3 -m pip install --user --upgrade pip setuptools
python3 -m pip install -r requirements-base.txt

python3 dnn-models/transform.py --target msp432 --hawaii har-dnp
cmake -B build
make -C build
./build/intermittent-cnn
