# FADD - Ferramenta para Análise de Dados da Dengue

Esse repositório contém uma ferramenta em linha de comanda para a análise de dados sobre a progressão da dengue. 

Esses dados foram obtidos através da inicitiva open data da cidade de Recife.

# Instalação

Para utilizar a ferramente é necessário Python versão 3. As dependencias são descritas no arquivo environment.yml para a distribuição Anaconda.

Para criação do ambiente virtual de desenvolvimento é necessário executar o seguinte comando:

```
conda env create -f environment.yml
```
# Utilização

Para utilizar, execute o arquivo Fadd.py.

As possíveis opções são as seguintes, disponível pela opção --help:

```
Usage: Fadd.py [OPTIONS]

Options:
  --file TEXT                     Path of the file to used in the training of
                                  the model
  -s, --save / -ns, --no-save     Serialize the model
  -l, --load / -nl, --no-load     Loads an existing model
  --feature-score, --load / --no-load
                                  Loads an existing model
  --help                          Show this message and exit.
```

# Autor

Bernardo Botelho
