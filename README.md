Prueba Nynsus   
---   
En la carpeta `scripts` se encuentran los archvos de código:   
- `preprocess_data.py` contiene un método para limpiar los textos del csv y convertir los tweets en vectores a partir de modelos preentrenados basados en transformers
- `train_DNN.py` contiene un model simple escrito en torch y el método para entrenarlo
- `grid_train.ipynb` es un jupyter notebook que permite lanzar varios entrenamientos seguidos conectado para poder encontrar los parámetros más adecuados para el problema     
    
En la carpeta `scripts/runs` sen encuentran los logs de tensorboad con las métricas de los entrenamientos
   
   
En la carpeta `scripts/models` se encuentran los modelos entrenados desde el jupyter notebook   
   
En la carpeta `data` encontramos los csv del reto de Kaggle, dentro de la carpeta `data/arrays` están las arrays de numpy resultado de pasar los tweets por un modelo basado en transformers
   
   

