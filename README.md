# Classificação de Estágios do Sono com o Uso de Representações Latentes de EEG, Alinhamento Euclidiano e Mecanismos de Atenção
Este trabalho visa realizar testes com modelos disponibilizados no site do braindecode com redes transformers como 
o EEGConformer para classificar estágios do sono. Para tanto, usamos técnicas de busca de parâmetros, alinhamento euclidiano 
e amostragem sequencial.
# Descrição do Projeto
O projeto teve como base três instantes com a implementação de códigos diferentes em cada parte com algumas partes em comum:
(Comum a todos): Utilizados os 100 primeiros sujeitos mais saudáveis, podendo ou não usarmos o alinhamento euclidiano para
carregarlos.

(i) - Primeiramente realizamos testes que são salvos por meio do mlflow com k-fold = 5 resultando em 5 testes por modelo
modificando a parte usada em cada um.
(ii) - Em seguida realizamos busca de parâmetros por meio da implementação do código de grid_search com o k-fold de 2 do braindecode e com o
modelo em sua forma parcial.

(iii) - E por fim implementamos a amostragem sequencial, transformando os dados carregados de 30 (s) de leitura em 150 (s) para serem alimentados
no modelo e para comparar as predições com a realidade para conseguir os resultados de acurácia, usamos a label do central.
