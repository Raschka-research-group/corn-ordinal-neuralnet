
There are multiple flavors of conditional approaches we tried. 

The one that ultimately corresponds to the CORN method corresponds to [`resnet34_conditional-v3.py`](resnet34_conditional-v3.py).



Run

```
python xxx.py --help
```

to see the available command line options.

For example, to do a test run, you can try the following:

```
python resnet34_xentr.py --outpath ~/Desktop/output --seed 123 --numworkers 0 --save_models false --learningrate 0.005 --batchsize 128 --epochs 2 --optimizer adam --scheduler false --dataset mnist
```

