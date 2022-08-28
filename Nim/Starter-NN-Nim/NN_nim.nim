# Neural Network in Nim only using the standard library

import std/[complex, math, fenv]

proc sigmoid(x: float): float = 
  return 1 / (1 + pow(E, float(-x)))
