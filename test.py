import numpy as np


def main():
    r = 0.5 - 0.5 * np.cos(2 * np.pi * 1 / (30 - 1))
    print(r)

    r = np.hanning(30)

    print(tuple(r.tolist()))


if __name__ == '__main__':
    main()
