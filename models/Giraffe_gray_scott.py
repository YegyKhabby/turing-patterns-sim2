# Giraffe (Gierer-Meinhardt) reaction terms
def giraffe_f(u, v, p):
    a, b = p[0], p[1]
    return (u**2 / (v + 1e-6)) - a * u + b  # small number avoids division by zero

def giraffe_g(u, v, p):
    c = p[2]
    return u**2 - c * v
