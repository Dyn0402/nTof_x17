#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 04 2:58 PM 2020
Created in PyCharm
Created as QGP_Scripts/Measure.py

@author: Dylan Neff, dylan

Updated version copied from UCLA_Nuclear on 3/16/21
"""

import math


class Measure:
    def __init__(self, val=0, err=0):
        self._val = val
        self._err = err

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val

    @val.deleter
    def val(self):
        del self._val

    @property
    def err(self):
        return self._err

    @err.setter
    def err(self, err):
        self._err = err

    @err.deleter
    def err(self):
        del self._err

    def __neg__(self):
        return Measure(-self.val, self.err)

    def __add__(self, o):
        result = Measure()
        if type(o) == Measure:
            result.val = self.val + o.val
            result.err = (self.err**2 + o.err**2)**0.5
        else:
            try:
                c = float(o)
                result.val = self.val + c
                result.err = self.err
            except ValueError:
                result = NotImplemented
        return result

    def __radd__(self, o):
        return self.__add__(o)

    def __sub__(self, o):
        result = Measure()
        if type(o) == Measure:
            result.val = self.val - o.val
            result.err = (self.err**2 + o.err**2)**0.5
        else:
            try:
                c = float(o)
                result.val = self.val - c
                result.err = self.err
            except ValueError:
                result = NotImplemented
        return result

    def __rsub__(self, o):
        return -self.__sub__(o)

    def __mul__(self, o):  # What if self.val or o.val == 0?
        result = Measure()
        if type(o) == Measure:
            result.val = self.val * o.val
            result.err = ((self.err * o.val) ** 2 + (o.err * self.val) ** 2) ** 0.5
        else:
            try:
                c = float(o)
                result.val = self.val * c
                result.err = abs(c) * self.err
            except ValueError:
                result = NotImplemented
        return result

    def __rmul__(self, o):
        return self.__mul__(o)

    def __truediv__(self, o):
        if o == 0:
            return Measure(float('nan'), float('nan'))
        result = Measure()
        if type(o) == Measure:
            result.val = self.val / o.val
            result.err = ((self.err / o.val) ** 2 + (o.err * self.val / o.val ** 2) ** 2) ** 0.5
        else:
            try:
                c = float(o)
                result.val = self.val / c
                result.err = self.err / abs(c)
            except ValueError:
                result = NotImplemented
        return result

    def __rtruediv__(self, o):  # Might lead to an error when inverting with zeros, maybe better to write out explicitly
        return self.__truediv__(o)**-1

    def __pow__(self, o):
        result = Measure()
        if type(o) == Measure:
            result.val = self.val ** o.val
            result.err = abs(result.val) * ((o.val / self.val * self.err)**2 + (math.log(self.val) * o.err)**2)**0.5
        else:
            try:
                c = float(o)
                result.val = self.val ** c
                result.err = abs(result.val * c * self.err / self.val)
            except ValueError:
                result = NotImplemented
        return result

    def sqrt(self):
        return self**0.5

    def __rpow__(self, o):  # If o is Measure then o.__pow__ should run, so assume o is not Measure
        result = Measure()
        try:
            c = float(o)
            result.val = c**self.val
            result.err = abs(result.val * math.log(c) * self.err)
        except ValueError:
            result = NotImplemented
        return result

    def __abs__(self):
        return Measure(abs(self.val), self.err)

    def __eq__(self, o):
        if type(o) == Measure:
            return self.val == o.val
        try:
            return self.val == float(o)
        except ValueError:
            return NotImplemented

    def __ne__(self, o):
        if type(o) == Measure:
            return self.val != o.val
        try:
            return self.val != float(o)
        except ValueError:
            return NotImplemented

    def __lt__(self, o):
        if type(o) == Measure:
            return self.val < o.val
        try:
            return self.val < float(o)
        except ValueError:
            return NotImplemented

    def __le__(self, o):
        if type(o) == Measure:
            return self.val <= o.val
        try:
            return self.val <= float(o)
        except ValueError:
            return NotImplemented

    def __gt__(self, o):
        if type(o) == Measure:
            return self.val > o.val
        try:
            return self.val > float(o)
        except ValueError:
            return NotImplemented

    def __ge__(self, o):
        if type(o) == Measure:
            return self.val >= o.val
        try:
            return self.val >= float(o)
        except ValueError:
            return NotImplemented

    def conjugate(self):
        return self

    def __str__(self):
        dec = err_dec(self.err)
        return f'{self.val:.{dec}f} ± {self.err:.{dec}f}'

    def __repr__(self):
        return str(self)


def err_dec(x, prec=2):
    if math.isinf(x) or math.isnan(x):
        return 0
    dec = 0
    while int(x) < 10**(prec - 1) and x != 0:
        x *= 10
        dec += 1
    return dec


def log(x, base=math.e):
    try:
        float(base)
    except ValueError:
        return NotImplemented

    if type(x) == Measure:
        val = math.log(x.val, base)
        err = abs(x.err / x.val)
        return Measure(val, err)
    else:
        try:
            return math.log(float(x), base)
        except ValueError:
            return NotImplemented
