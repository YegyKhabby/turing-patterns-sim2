# Gray-Scott reaction terms
def gray_scott_f(u, v, p):
    alpha = p[0]
    return -u * v**2 + alpha * (1 - u)

def gray_scott_g(u, v, p):
    alpha, beta = p[0], p[1]
    return u * v**2 - (alpha + beta) * v
