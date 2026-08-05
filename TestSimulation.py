import os
import argparse

print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import base64
from io import BytesIO

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from PIL import Image
from keras.models import load_model

from src.data_preprocessing import preprocess

sio = socketio.Server()
app = Flask(__name__)

model = None
MAX_SPEED = 10.0
MIN_SPEED = 4.0
speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    global speed_limit

    if not data:
        sio.emit('manual', data={}, skip_sid=True)
        return

    try:
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)

        # Exactly the same preprocessing used in training.
        image = preprocess(image)
        image = np.array([image], dtype=np.float32)

        # model.predict returns shape (1, 1). float() on an ndim>0 array is a
        # hard TypeError on numpy >= 2.4, so index down to a scalar first.
        steering = float(model.predict(image, verbose=0)[0][0])

        # Ease off the throttle on sharp turns so the car does not oversteer.
        if speed > speed_limit:
            speed_limit = MIN_SPEED
        else:
            speed_limit = MAX_SPEED
        throttle = 1.0 - (steering ** 2) - (speed / speed_limit) ** 2
        throttle = float(np.clip(throttle, -1.0, 1.0))

        print(f'steering={steering:+.4f}  throttle={throttle:+.4f}  speed={speed:.2f}')
        sendControl(steering, throttle)
    except Exception as exc:
        # eventlet swallows greenlet exceptions, which makes a crashing
        # handler look identical to "the car just sits there".
        import traceback
        traceback.print_exc()
        print(f'telemetry handler failed: {exc}')


@sio.on('connect')
def connect(sid, environ):
    print('Connected:', sid)
    sendControl(0.0, 0.0)


@sio.on('disconnect')
def disconnect(sid):
    print('Disconnected:', sid)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle': str(throttle),
    })


def main():
    global model

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model/model.h5', help='path to the trained .h5 model')
    parser.add_argument('--port', type=int, default=4567)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f'Model file not found: {args.model}')

    print(f'Loading model from {args.model} ...')
    model = load_model(args.model, compile=False)

    # Warm up so the first real telemetry frame is not delayed by graph tracing.
    model.predict(np.zeros((1, 66, 200, 3), dtype=np.float32), verbose=0)
    print('Model loaded. Start the simulator in Autonomous Mode.')

    wsgi_app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', args.port)), wsgi_app)


if __name__ == '__main__':
    main()