# Documentação do Notebook: PySpark

## Introdução
Este notebook aborda a utilização do PySpark para análise e manipulação de dados em grande escala. O PySpark é uma interface do Python para Apache Spark, permitindo realizar operações de processamento de dados de forma eficiente.

## Configuração do Ambiente

### Instalação do PySpark
Para utilizar o PySpark, certifique-se de que o pacote está instalado:
```python
!pip install pyspark
```

### Importação de Bibliotecas
Importamos as bibliotecas necessárias para manipulação de dados e criação de sessões Spark.
```python
from pyspark.sql import SparkSession
```

### Inicialização do Spark
Criamos uma sessão Spark que servirá como ponto de entrada para as operações.
```python
spark = SparkSession.builder.appName("Exemplo PySpark").getOrCreate()
```

## Carregamento de Dados
Carregamos os dados a partir de um arquivo CSV.
```python
df = spark.read.csv("caminho/do/arquivo.csv", header=True, inferSchema=True)
df.show()
```

## Manipulação de Dados
Realizamos algumas operações básicas, como seleção e filtragem de colunas.
```python
df_filtered = df.filter(df['coluna'] > valor)
df_filtered.select("coluna1", "coluna2").show()
```

## Conclusão
Este notebook fornece uma visão geral básica de como trabalhar com PySpark, incluindo a configuração do ambiente e operações iniciais de manipulação de dados.

---

# Documentação do Notebook: PySpark2

## Introdução
Este notebook amplia a análise anterior com PySpark, adicionando operações mais complexas e técnicas de manipulação de dados.

## Carregamento de Dados
Assim como no notebook anterior, começamos com o carregamento de dados.
```python
df = spark.read.csv("caminho/do/arquivo.csv", header=True, inferSchema=True)
```

## Análise Exploratória
Realizamos uma análise exploratória dos dados, verificando estatísticas descritivas e a contagem de valores nulos.
```python
df.describe().show()
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
```

## Manipulação Avançada
Neste notebook, aplicamos operações como agrupamento e agregação.
```python
df_grouped = df.groupBy("coluna").agg({"outra_coluna": "mean"})
df_grouped.show()
```

## Conclusão
Este notebook demonstrou técnicas avançadas de análise e manipulação de dados com PySpark, incluindo análise exploratória e operações de agregação.

---

# Documentação do Notebook: Regressão Linear com PySpark

## Introdução
Este notebook demonstra como aplicar um modelo de regressão linear utilizando PySpark para prever valores com base em um conjunto de dados.

## Configuração do Ambiente

### Importação de Bibliotecas
Importamos as bibliotecas necessárias para a análise.
```python
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
```

### Inicialização do Spark
Criamos uma sessão Spark.
```python
spark = SparkSession.builder.appName("Regressao Linear").getOrCreate()
```

## Carregamento de Dados
Carregamos os dados para o modelo.
```python
data = spark.read.csv("caminho/do/arquivo.csv", header=True, inferSchema=True)
data.show()
```

## Preparação dos Dados
Preparação dos dados para a regressão linear, incluindo a montagem de vetores de características.
```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["coluna1", "coluna2"], outputCol="features")
data = assembler.transform(data)
```

## Divisão dos Dados
Dividimos os dados em conjuntos de treino e teste.
```python
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)
```

## Criação do Modelo
Treinamos o modelo de regressão linear.
```python
lr = LinearRegression(featuresCol="features", labelCol="label")
model = lr.fit(train_data)
```

## Avaliação do Modelo
Realizamos previsões e avaliamos o modelo.
```python
predictions = model.transform(test_data)
predictions.select("prediction", "label").show()
```

## Conclusão
Neste notebook, aplicamos um modelo de regressão linear com PySpark e avaliamos seu desempenho.

---

# Documentação do Notebook: Classificação com PySpark

## Introdução
Este notebook demonstra como utilizar o PySpark para realizar tarefas de classificação. Utilizaremos um conjunto de dados para treinar um modelo e prever classes com base em variáveis de entrada.

## Configuração do Ambiente

### Instalação do PySpark
Para usar o PySpark, você deve garantir que o PySpark esteja instalado e corretamente configurado no ambiente.
```python
!pip install pyspark
```

### Importação de Bibliotecas
Importamos as bibliotecas necessárias para a análise e manipulação de dados.
```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
```

### Inicialização do Spark
Criamos uma sessão Spark para realizar as operações.
```python
spark = SparkSession.builder.appName("Classificacao").getOrCreate()
```

## Carregamento de Dados
Carregamos o conjunto de dados a partir de um arquivo CSV.
```python
data = spark.read.csv("caminho/do/arquivo.csv", header=True, inferSchema=True)
data.show()
```

## Pré-processamento de Dados
Realizamos a preparação dos dados, incluindo a seleção de características e a montagem de um vetor de características.
```python
assembler = VectorAssembler(inputCols=["coluna1", "coluna2", "coluna3"], outputCol="features")
data = assembler.transform(data)
```

## Divisão dos Dados
Dividimos os dados em conjuntos de treino e teste.
```python
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)
```

## Criação do Modelo
Treinamos um modelo de árvore de decisão.
```python
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
model = dt.fit(train_data)
```

## Avaliação do Modelo
Realizamos previsões no conjunto de teste e avaliamos o desempenho do modelo.
```python
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Acurácia: {accuracy}")
```

## Conclusão
Neste notebook, realizamos uma classificação usando PySpark. A árvore de decisão foi utilizada como modelo, e o desempenho foi avaliado com base na acurácia.
