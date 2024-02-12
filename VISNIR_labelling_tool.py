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
from dash.exceptions import PreventUpdate
import os

directory = "D:/Hyperspectral data/VIS-NIR/visnir_18_march/selected/"
l = os.listdir(directory)
L = []
for i in l:
    if ".png" in i:
        L.append(i)
    else:
        pass

def img(n):
    filename = L[n]
    im = cv2.imread(directory + filename)
    im2 = cv2.resize(im, (512, 512), interpolation=cv2.INTER_NEAREST)
    im2 = np.flip(im2,axis=2)

    return im2.astype('uint8'), filename

image = img(0)[0]

canvas_width = 4000

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
                    html.H6(children=['Brush width']),
                    daq.ColorPicker(
                                    id='color-picker',
                                    label='Brush color',
                                    value=dict(hex='#119DFF')
                                ),
                    dcc.Slider(
                            id='bg-width-slider',
                            min=2,
                            max=40,
                            marks={i: '{}'.format(i) for i in range(39)},
                            step=1,
                            value=5
                        ),

                    ], className="three columns"),
        html.Div([
            html.H6(children=['Images']),
            DashCanvas(
                id='original',
                width=canvas_width,
                image_content=array_to_data_url(image),
                hide_buttons=['line', 'rectangle', 'select']
                ),
            html.Img(
                id='segmented',
                width=canvas_width,
                ),
                ], className="six columns"),


    ])

##### Callbacks #####

@app.callback(Output('segmented', 'src'),
              [Input('original', 'json_data')])
def update_data(string):
    if string:
        mask = parse_jsonstring(string, image.shape[0:2]) #
    else:
        raise PreventUpdate
    return array_to_data_url((255 * mask).astype(np.uint8))

@app.callback(Output('original', 'lineWidth'),
            [Input('bg-width-slider', 'value')])
def update_canvas_linewidth(value):
    return value

@app.callback(Output('original', 'lineColor'),
            [Input('color-picker', 'value')])
def update_canvas_linewidth(value):
    if isinstance(value, dict):
        return value['hex']
    else:
        return value


# Next image
@app.callback([Output('original', 'image_content'), Output('filename','children')],
                [Input('button', 'n_clicks')],
                [State('next', 'value')])
def update(n_clicks, next):
    N = int(next)
    I = img(N-1)
    image = I[0]
    #image = cv2.filter2D(image, -1, kernel)

    return array_to_data_url(image), I[1]


if __name__ == '__main__':
    app.run_server(debug=False)
