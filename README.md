# Classificação de Estágios do Sono com o Uso de Representações Latentes de EEG, Alinhamento Euclidiano e Mecanismos de Atenção
Este trabalho visa realizar testes com modelos disponibilizados no site do [braindecode](https://braindecode.org/stable/api.html#models) com redes transformers como 
o EEGConformer para classificar estágios do sono. Para tanto, usamos técnicas de busca de parâmetros, alinhamento euclidiano 
e amostragem sequencial.

# Descrição do Projeto
O projeto teve como base três instantes com a implementação de códigos diferentes em cada parte com algumas partes em comum:
(Comum a todos): Utilizados os 100 primeiros sujeitos mais saudáveis, podendo ou não usarmos o alinhamento euclidiano para
carregarlos ao modificar o euclid_alignment para verdadeiro no arquivo **shhs_dataset.py**.

(i) - Primeiramente realizamos testes que são salvos por meio do mlflow com k-fold = 5 resultando em 5 testes por modelo
modificando a parte usada em cada um **Arquivo main_1_kfold_5.py**.

(ii) - Em seguida realizamos busca de parâmetros por meio da implementação do código de grid_search com o k-fold de 2 do braindecode e com o
modelo em sua forma parcial **Arquivo main_2_busca_param.py**.


(iii) - E por fim implementamos a amostragem sequencial, transformando os dados carregados de 30 (s) de leitura em 150 (s) para serem alimentados
no modelo e para comparar as predições com a realidade para conseguir os resultados de acurácia, usamos a label do central **Arquivo main_3_amostragem_seq.py**.

## Dica
Vale lembrar que todos os caminhos para a entrada ou saída dos dados tem que ser modificados para o o computador da pessoa que deseja utilizar-lo.

# Pré-requisito para reprodução.
Instale a biblioteca Conda, recomendamos o [tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Crie uma variável de sistema no conda, conforme instruções abaixo:

```bash
conda create --name experimento python=3.10 pip --yes
conda activate experimento
```


Depois baixos os requerimentos descritos no arquivo que está no repositório:
```bash
pip install -r requirements.txt
```

Especialmente para a biblioteca do braindecode, é necessário baixar da seguinte maneira:
```bash
pip install -U https://api.github.com/repos/braindecode/braindecode/zipball/master#egg=braindecode
python -m "import braindecode; braindecode.__version__"
```
Deveria devolver um valor 0.8, 0.9 ou maior.

# Como conseguir os dados usados para o treino:
Em princípio é necessário que se entre no site do [Sleep Heart Health Study(SHHS)](https://sleepdata.org/datasets/shhs) para que se crie uma conta e peça a permissão aos dados do estudo. E por fim deve-se passar os dados por um pré-processamento disponível no github do [mulEEG](https://github.com/likith012/mulEEG/tree/main/preprocessing/shhs).


# Os parâmetros e resultados dos experimentos:
Tanto os parâmetros como os seus respectivos resultados estão disponíveis na [planilha](https://docs.google.com/spreadsheets/d/1q3-GsbcvHWdnrMp_I3fvmrjWTIUWpZhL2g6lhM-65GM/edit?usp=sharing) para serem visualizados.
