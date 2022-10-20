# tf-training-inference-benchmark

Scripts for benchmarking various models available on TF
Benchmark would include training and inference
This benchmark aims at measuring GPU performance on various DL models
First with CUDA enabled Tensorflow
And then add TensorRT for Inference test

This repo is tested on Ubuntu 20.04

For quick environment setup, run the setup script
First give it permission to the script
```
chmod +x setup.sh
```
Then run the script (still in beta, read the script carefully and make sure)
```
./setup.sh
```
Let it run and install required libraries
After that, run the benchmark script
```
python resnet50_benchmark.py --batch_size=32 --num_epochs=20 --use_fp16
```

Flags available:
* batch_size (better leave it at default)
* num_epochs (better leave it at default)
* use_fp16 : Use FP16 operation (Make sure your GPU supports it)
* use_bfloat16 : Use BFloat16 operation (untested feature)
