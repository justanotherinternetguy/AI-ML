# Why?

* "vanilla" NNs and CNNs can only work with predetermined sizes (fixed-size input --> fixed-size output)
* RNNs allow for **variable length sequences**

# Usages
* Machine translation
* Sentiment analysis

# How does this work?
* Consider "many-to-many" RNN with inputs [x0, x1, ..., xn] that wants to produce outputs [y0, y1, ..., yn]
  * xi and yi are vectors and can have arbitrary dimensions

* RNNs iteratively update a hidden state h, which is a vector that can also have an arbitrary dimension.

* At any given step t,
  1. the next hidden state ht is calculated using the previous hidden state ht-1 and the next input xt
  2. the next output yt is calculated using ht

* RNNs use the **same weight each step**
  * Wxh for all xt --> ht links
  * Whh for all ht-1 --> ht links
  * Why for all ht --> yt links

  * Bh added when calculating ht
  * By added when calcuating yt

* Weights: matrices
* Biases: vectors

* ht = tanh(Wxh * xt + Whh * ht-1 + Bh)
* yt = Why * h1 + By