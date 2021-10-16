This is the code we used internally for our experiments. Simpler-to-read scripts (recommended) for understanding the main model runs are available under [../simple-scripts](../simple-scripts)

There are multiple flavors of conditional approaches we tried. 

The one that ultimately corresponds to the CORN method corresponds to [`resnet34_conditional-v3.py`](resnet34_conditional-v3.py).





Example:

```
python resnet34_coral.py --cuda 1 --seed 0 --epochs 200 --dataset morph2 --outpath runs/morph2/resnet34_coral-imp0-seed0-epochs200_best-rmse
```

Note that for MNIST, the number of workers needs to be set to 0 or 1. I think that's because of the small dataset size and the fast loading, which results in too many open files. I can also be related to the sampler, but I believe this has worked with multiple workers in the past.

E.g., 

```python
python resnet34_coral.py --cuda 1 --seed 0 --epochs 2 --dataset mnist --outpath runs/mnist/resnet34_coral-imp0-seed0-epochs5_best-rmse --numworkers 0
```

