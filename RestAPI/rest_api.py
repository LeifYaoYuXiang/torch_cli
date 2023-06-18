from flask import Flask, request
from service_streamer import ThreadedStreamer
from utils import *
model = None
use_thread_stemmer = False

app = Flask(__name__)


def load_model(filepath, use_gpu=True):
    model: torch.nn.Module = read_data_from_pickle(filepath)
    model.eval()
    if use_gpu:
        model.cuda()
    return model


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if use_thread_stemmer:
            streamer = ThreadedStreamer(model.predict, batch_size=64, max_latency=0.1)
        else:
            pass


if __name__ == '__main__':
    model_filepath = ''
    model = load_model(model_filepath)
    app.run()
