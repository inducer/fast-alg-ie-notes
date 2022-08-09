#! /usr/bin/env python3

import matplotlib.pyplot as pt
import numpy as np

x = np.linspace(-1, 1, 1000)
alpha = 100
pt.rc("font", size=20)
pt.plot(x, (np.exp(1j*alpha*x)).real, label="$f$")
pt.plot(x, (1j*alpha*np.exp(1j*alpha*x)).real, label="$f'$")
pt.ylim([-alpha-5, alpha+5])
pt.legend()
pt.savefig("diff-complex-exp.pdf")
