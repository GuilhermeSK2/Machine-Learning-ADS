def somaEntrada():
    quantidadeDeEntradas = 3
    sinaisDeEntrada = [1, 2, 3]
    pesos = [4, 5, 6]
    somaTotal = 0
    indice = 0

    for sinal in sinaisDeEntrada:
        somaTotal = somaTotal + (sinal * pesos[indice])
        indice = indice + 1

    return somaTotal

somaEntrada()
