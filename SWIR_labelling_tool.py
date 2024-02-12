from dash_canvas import DashCanvas
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
import numpy as np
from dash_canvas.utils import (parse_jsonstring, array_to_data_url)
from scipy import ndimage
import cv2
import h5py
from dash.exceptions import PreventUpdate
import os

# Read the files in a directory
directory="D:/Hyperspectral data/SWIR_pro_18March/UD_3/" #New folder/
l = os.listdir(directory)
L = []
for i in l:
    if ".h5" in i:  ### modify to read npy #".hdf5"
        L.append(i)
    else:
        pass


def fft(im, frac=0.85):
    '''Fourier transformation and fraction removal'''
    im_fft2 = np.fft.rfft2(im)
    # Set r and c to be the number of rows and columns of the array.
    r, c = im_fft2.shape
    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[int(r * frac):int(r * (1 - frac))] = 0
    # Similarly with the columns:
    #im_fft2[:, int(c * frac):int(c * (1 - frac))] = 0
    im_fft2[:, int(c * frac):] = 0
    im_new = np.fft.irfft2(im_fft2)
    return im_new

K_median = 5 # Kernel for Median filter
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) # Kernel for sharpening the image

def img_transf(im):
    '''Median filter + rotation 90 degrees + Fast Fourier transform'''
    im1 = ndimage.median_filter(im, K_median, mode="nearest")
    im1 = np.rot90(im1)
    im1 = fft(im1)
    return im1

def normalize(x):
    '''Normalization min-max'''
    res = (x - np.min(x)) / (np.max(x) - np.min(x))
    return res.astype("float64")

def img(n,w1=60,w2=75,w3=150):
    '''Synthetic image generation from 3 different wavebands'''
    filename = L[n]
    f = h5py.File(directory+filename, 'r')
    n1 = f.get("image")
    n2 = n1.get("image")
    n2 = np.array(n2)
    n2 = np.array(n2, dtype='int32')

    im = n2[w1, :, :]
    im1 = img_transf(im)
    im1 = normalize(im1)

    im = n2[w2, :, :] #162
    im2 = img_transf(im)
    im2 = normalize(im2)

    im = n2[w3, :, :]
    im3 = img_transf(im)
    im3 = normalize(im3)

    im4 = np.stack((im1, im2, im3), axis=2)
    #im4 = im1
    #im5 = np.divide(im4, im4.max(), dtype="float64")
    im5 = normalize(im4)
    im5 = np.round(im5 * 255)

    return im5.astype('uint8'), filename

im5 = img(0,60,75,150)[0]
im5 = cv2.filter2D(im5, -1, kernel)
shape = im5.shape[0:2]


class Shape:
    var1 = 1

    @classmethod
    def update(cls, value):
        cls.var1 = value

    def __init__(self, value):
        self.value = value
        self.update(value)

Shape.update(shape)


# Canvas width of the images
canvas_width = 3700



##################################################################################
# Browser app starts here
app = dash.Dash(__name__)

# Layouts
app.layout = html.Div([
        html.Div([
            html.H6(children=['Next Image']),
            dcc.Input(
                placeholder='Enter a value...',
                id='next',
                type='number',
                value='1',
                max='71'
            ),
            html.Button('Submit', id='button'),
            html.H6(children=[L[0]], id="filename"),
            ]),
        html.Div([
            html.H6(children=['1st waveband']),
            dcc.Input(
                placeholder='Enter a value...',
                id='wave1',
                type='number',
                value='60',
                max='200'
            ),
            html.H6(children=['2nd']),
            dcc.Input(
                placeholder='Enter a value...',
                id='wave2',
                type='number',
                value='75',
                max='200'
            ),
            html.H6(children=['3d']),
            dcc.Input(
                placeholder='Enter a value...',
                id='wave3',
                type='number',
                value='150',
                max='200'
            ),
            html.Button('Submit', id='button2')
            ]),
        html.Div([
            html.H6(children=['Images']),
            DashCanvas(
                id='original',
                width=canvas_width,
                image_content=array_to_data_url(im5),
                hide_buttons=['line', 'rectangle']
                ),
            html.Img(
                id='segmented',
                width=canvas_width,
                ),
                ], className="six columns"),
        html.Div([
            html.H6(children=['Brush width']),
            dcc.Slider(
                id='bg-width-slider',
                min=2,
                max=40,
                marks={i: '{}'.format(i) for i in range(39)},
                step=1,
                value=5
            ),
            daq.ColorPicker(
                id='color-picker',
                label='Brush color',
                value=dict(hex='#119DFF')
            )
        ], className="three columns"),

    ])


##### Callbacks #####

@app.callback(Output('segmented', 'src'),
              [Input('original', 'json_data')])
def update_data(string):
    if string:
        mask = parse_jsonstring(string, Shape.var1)
    else:
        raise PreventUpdate
    return array_to_data_url((255 * mask).astype(np.uint8))

@app.callback(Output('original', 'lineWidth'),
            [Input('bg-width-slider', 'value')])
def update_canvas_linewidth(value):
    return value

## waveband changer
@app.callback(Output('original', 'image_content'),
                [Input('button2', 'n_clicks')],
                state=[State('wave1', 'value'),
                State('wave2', 'value'),State('wave3', 'value')])
def update_wave(n_clicks, wave1, wave2, wave3):
    print((wave1,wave2,wave3))
    im5 = img(0, int(wave1),int(wave2),int(wave3))[0]
    im5 = cv2.filter2D(im5, -1, kernel)
    print(im5.shape)
    print(type(im5))
    shape = im5.shape[0:2]
    Shape.update(shape)
    return array_to_data_url(im5)


# Next image
@app.callback([Output('original', 'image_content'), Output('filename','children')],
                [Input('button', 'n_clicks')],
                [State('next', 'value')])
def update(n_clicks, next):
    N = int(next)
    I = img(N-1)
    im5 = I[0]
    im5 = cv2.filter2D(im5, -1, kernel)
    shape = im5.shape[0:2]
    Shape.update(shape)
    return array_to_data_url(im5), I[1]


if __name__ == '__main__':
    app.run_server(debug=True)
