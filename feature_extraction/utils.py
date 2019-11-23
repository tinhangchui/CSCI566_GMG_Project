class ParamDict(dict):
    __setattr__ = dict.__setitem__
    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)
    def __getstate__(self): return self
    def __setstate__(self, d): self = d
