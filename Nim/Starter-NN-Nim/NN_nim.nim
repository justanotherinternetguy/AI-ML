# Neural Network in Nim only using the standard library

import std/[complex, math, fenv]

proc sigmoid(x: int): float