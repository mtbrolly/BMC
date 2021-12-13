"""
Minimal demonstration of model.
"""

from model import Model
from inits import JMcW
from plots import run_and_plot

m = Model(beta=10.)

m.zk = JMcW(m)
run_and_plot(m)
