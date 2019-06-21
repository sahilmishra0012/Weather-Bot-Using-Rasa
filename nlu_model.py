#script for model training

from rasa.nlu.training_data import load_data
from rasa.nlu import config
from rasa.nlu.model import Trainer

def train_nlu(data, configs, model_dir):
	t_data = load_data(data)
	trainer = Trainer(config.load(configs))
	trainer.train(t_data)
	model_directory = trainer.persist(model_dir, fixed_model_name = 'weather_nlu')



if __name__ == '__main__':
	train_nlu('./data/data.json', 'config_spacy.json', './models/nlu')
