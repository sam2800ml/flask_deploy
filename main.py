from utils import Utils
from models import Models

#pip install -r requirements.txt --> Descargar las dependencias
#python -m virtualenv entorno --> Para crear el entorno virtual desde cero
#./entorno/Scripts/activate -->Acceder al entorno virtual


if __name__ == '__main__':

    utils = Utils()
    models = Models()

    data = utils.load_csv('in/felicidad_b0b50c6d-41dd-4ea8-a4f0-92a8068d4d3e.csv')

    X,y = utils.features_target(data,['score','rank','country'],['score'])

    models.grid_training(X,y)



    