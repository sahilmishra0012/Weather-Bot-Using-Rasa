from rasa.nlu.model import Metadata, Interpreter


def run_nlu():
	interpreter = Interpreter.load('./models/nlu/weather_nlu')
	print(interpreter.parse(u"I am planning my holiday to Lithuania. I wonder what is the weather out there."))

if __name__ == '__main__':
    run_nlu()
