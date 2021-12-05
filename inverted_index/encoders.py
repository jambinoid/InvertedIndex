from abc import abstractclassmethod
from math import log


log2 = lambda x: log(x, 2)


class BinaryEncoder:
    @abstractclassmethod
    def encode(self):
        pass

    def decode(self):
        pass


class GammaEncoder(BinaryEncoder):
    @staticmethod 
    def _unary(n: int) -> str:
        return '0' * n + '1'
    
    @staticmethod
    def _binary(b: int, n: int = 1) -> str:
        s = '{0:0%db}' % n
        return s.format(b)
        
    def encode(self, x: int) -> str:
        if (x == 0): 
            return '0'
        if (x == 1): 
            return '1'
    
        n = int(log2(x))
        b = x - 2 ** n
    
        return self._unary(n) + self._binary(b, n)

    def decode(self, s: str) -> int:
        if (s == '0'): 
            return 0
        if (s == '1'): 
            return 1
        
        u, _, b = s.partition('1')
        return 2 ** len(u) + int(b, 2)


class DeltaEncoder(GammaEncoder):
    def __init__(self):
        self.gamma = GammaEncoder()

    def encode(self, x: int) -> str:
        if (x == 0): 
            return '0'
        if (x == 1): 
            return '1'

        m = int(log2(x)) + 1
        b = x - 2 ** (m - 1)

        return self.gamma.encode(m) + self._binary(b, m - 1)

    def decode(self, s: str) -> int:
        if (s == '0'): 
            return 0
        if (s == '1'): 
            return 1

        u, _, b = s.partition('1')
        m = len(u)
        l = 2 ** m + int(b[:m], 2)
        return 2 ** (l - 1) + int(b[m:m+l-1], 2)


if __name__ == "__main__":
    n = 10
    print(f"Integer number: {n}")
    gamma = GammaEncoder()
    de = gamma.encode(n)
    print(f"Gamma encoding: {de}")
    dd = gamma.decode(de)
    print(f"Gamma decoding: {dd}")
    delta = DeltaEncoder()
    de = delta.encode(n)
    print(f"Gamma encoding: {de}")
    dd = delta.decode(de)
    print(f"Gamma decoding: {dd}")