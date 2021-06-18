# Code for Robust deep hedging

## Eva Lutkebohmert, Thorsten Schmidt, Julian Sester

# Abstract

We study pricing and hedging under parameter uncertainty for a class of Markov
processes which we call generalized affine processes and which includes the Black-
Scholes model as well as the constant elasticity of variance (CEV) model as special
cases. Based on a general dynamic programming principle, we are able to link the
associated nonlinear expectation to a variational form of the Kolmogorov equation
which opens the door for fast numerical pricing in the robust framework.
The main novelty of the paper is that we propose a deep hedging approach which
efficiently solves the hedging problem under parameter uncertainty. We numerically
evaluate this method on simulated and real data and show that the robust deep hedging
outperforms existing hedging approaches, in particular in highly volatile periods.


# Content

The Examples are provided as seperate jupyter notebooks.

The file Functions.py contains the Pyhton-code that is employed to train the hedging strategies.

## Data
Note that the data for the S&P 500 examples cannot be provided for legal reasons.

# License

MIT License

Copyright (c) 2021 Julian Sester

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
