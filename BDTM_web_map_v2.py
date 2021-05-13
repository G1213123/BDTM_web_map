# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:33:52 2019

@author: Andrew.WF.Ng
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 22:52:34 2019

@author: ngwin
"""

import numpy as np
import pandas as pd
import networkx as nx
import os
import geopandas
from shapely.geometry import Point
import kmb

from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.filedialog import asksaveasfilename

import pyproj

pyproj.Proj("+init=epsg:4326")
# %%
route = '277X'
bound = 1

network_ref_points = {
    'HK2': [['5062', '6006'], [22.264011, 114.235025], [22.289856, 114.194122]],
    'NTE1': [['52420', '52304'], [22.504201, 114.130619], [22.492023, 114.140654]],
    'K2': [['3778', '3183'], [22.350376, 114.200268], [22.295410, 114.240054]],
    'K1': [['1434', '1541'], [22.337744, 114.136352], [22.303662, 114.189310]],
    'NTW1': [['75009', '75100'], [22.377718, 113.966747], [22.476290, 114.057366]],
    'NTW2': [['88192', '99497'], [22.367426, 114.062772], [22.331403, 114.136068]],
    'NTW3': [['92550', '91602'], [22.228062, 113.891104], [22.335974, 114.031643]]
}
global REF
REF = network_ref_points['NTE1']


# %%
class get_file:

    def __init__(self, path):
        self.file_path = path
        self.file_name = os.path.basename(path).split('.')[0]
        self.load_dat()

    def load_dat(self):
        # Load network file line by line, store in df
        if self.file_path is None:
            Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
            self.file_path = askopenfilename(title="Select dat", filetypes=(
                ("dat", "*.dat"),
                ("all files", "*.*")))  # show an "Open" dialog box and return the path to the selected file
        f = open(self.file_path)
        lines = f.read().splitlines()
        self.data = pd.DataFrame(columns=['a'])
        self.data['a'] = lines


class get_data:

    def __init__(self, df, file_path):
        self.file_path = file_path
        self.df = df
        self.read_111()
        self.read_555()
        self.read_222()
        self.read_333()
        self.extrapolate()
        self.read_666()
        # self.anode_list=self.anode()

    # card reading
    def read(self, card, space=[5, 5, 5, 5, 5, 5, 5, 5, 10]):
        # Locate the start and the end of section
        end_card = '99999'
        start = self.df.loc[self.df['a'] == card].index[0]
        end = [i for i in self.df[self.df['a'] == end_card].index if i > start][0]
        # read card
        col_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        data = pd.read_fwf(self.file_path, sep=r'\s', skiprows=start + 1, widths=space, names=col_names)
        # handle $include tag
        data_row = end - start
        data = data.head(data_row - 1)
        if len(data) > 0:
            if '$INCL' in str(data['a'][0]):
                file_path = os.path.join(os.path.dirname(self.file_path),
                                         self.df['a'][start + 1].replace('$INCLUDE ', ''))
                data = pd.read_fwf(file_path, sep=r'\s', widths=space, names=col_names)
        return data

    def read_111(self):
        card = '11111'
        self.nodes = self.read(card)
        self.nodes.columns = ['id', 'anode', 'lanes', 'd', 'e', 'f', 'g', 'h', 'i']
        self.nodes = self.nodes[self.nodes.id != '*****']
        self.nodes['id'] = self.nodes['id'].apply(lambda a: a if len(str(a)) > 3 else np.nan)
        self.nodes['id'].fillna(method='ffill', inplace=True)
        # retrieve desired node details

    def read_222(self):
        card = '22222'
        self.zones = self.read(card)
        self.zones.columns = ['id', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']
        self.zones = pd.melt(self.zones, id_vars=['id']).sort_values('id')
        self.zones.columns = ['id', 'variable', 'anode']
        self.zones.dropna(subset=['anode'], inplace=True)
        self.zones.drop(['variable'], axis=1, inplace=True)
        self.zones['id'] = 'C' + self.zones['id']

    def read_333(self):
        card = '33333'
        card3 = self.read(card)
        card3.columns = ['c1', 'id1', 'c2', 'id2', 'n4', 'n5', 'n6', 'n7', 'n8']
        card3['c1'] = card3['c1'].fillna('')
        card3['c2'] = card3['c2'].fillna('')
        card3['id1'] = card3['c1'] + card3['id1']
        card3['id2'] = card3['c2'] + card3['id2']
        card3['id1'], card3['id2'] = np.where(card3['c2'] == 'C', (card3['id2'], card3['id1']),
                                              (card3['id1'], card3['id2']))
        card3.columns = ['c1', 'id', 'c2', 'anode', 'n4', 'n5', 'n6', 'n7', 'n8']
        df_list = [self.zones, card3[['id', 'anode']]]
        self.zones = pd.concat(df_list, ignore_index=True)

    def read_555(self):
        # read card 55555
        card = '55555'
        self.nodes_loc = self.read(card)
        self.nodes_loc.drop(['e', 'f', 'g', 'h', 'i'], inplace=True, axis=1)
        self.nodes_loc.columns = ['a', 'id', 'x', 'y']
        self.nodes_loc = self.nodes_loc[self.nodes_loc.a != '*****']
        self.nodes_loc = self.nodes_loc[self.nodes_loc.a != '*']
        self.nodes_loc['id'] = self.nodes_loc['a'].fillna('') + self.nodes_loc['id']
        self.nodes_loc['x'] = pd.to_numeric(self.nodes_loc['x'])
        self.nodes_loc['y'] = pd.to_numeric(self.nodes_loc['y'])

    def read_666(self):
        # read card 66666
        card = '66666'
        space = [5, 5, 5, 10, 10, 10, 10, 10, 10]
        self.bus_route = self.read(card, space)
        # Construct bus_route dataframe from card 66666
        self.bus_route['route'] = self.bus_route.b + self.bus_route.c + self.bus_route.d
        self.bus_route['route'] = self.bus_route['route'].str.extract('\(([^)]+)')
        lst = ['-1', '-2', 'Kmb']
        self.bus_route['route'] = self.bus_route['route'].replace({w: "" for w in lst}, regex=True)
        self.bus_route['route'] = self.bus_route['route'].fillna(method='ffill')

    def extrapolate(self):
        missing = pd.DataFrame(list(set(self.nodes.id) - set(self.nodes_loc.id)), columns=['id'])
        while len(missing) > 0:
            missing = pd.DataFrame(list(set(self.nodes.id) - set(self.nodes_loc.id)), columns=['id'])
            missing['x'] = np.nan
            missing['y'] = np.nan

            for index, row in missing.iterrows():
                EP_source = self.anode_candidate(row.id)
                EP_source = EP_source.merge(self.nodes_loc[['id', 'x', 'y']])
                if len(EP_source) == 1:
                    EP_source.x += 1
                    EP_source.y += 1
                missing.at[index, 'x'] = EP_source.x.mean()
                missing.at[index, 'y'] = EP_source.y.mean()
            missing = missing.dropna()
            self.nodes_loc = self.nodes_loc.append(missing, ignore_index=True)

            # node=52312

    def anode_candidate(self, node):
        test_node = node
        # test_node='51434'
        test_nodes_loc = self.nodes.loc[self.nodes['id'] == str(test_node)]
        test_anode = test_nodes_loc[['anode', 'lanes']]
        test_anode.columns = ['id', 'lanes']
        test_anode = test_anode.tail(len(test_anode) - 1)
        test_anode = test_anode.dropna()
        test_anode.lanes = test_anode.lanes.str.extract('(\d+)')
        return test_anode

    def anode(self):
        anode = pd.DataFrame(columns=['id', 'anode'])
        for i in self.nodes_loc['id']:
            print(i)
            add_anode = self.anode_candidate(i)['id']
            if len(add_anode) > 0:
                anode = anode.append(pd.DataFrame({'id': [i] * len(add_anode), 'anode': list(add_anode)}))
        return anode


def to_Saturn_coord(lat, lon, nodes_loc,
                    nodes=REF[0], node1_gps=REF[1], node2_gps=REF[2]):
    # 52420 lat = 22.504201, lon = 114.130619
    i = np.where(nodes_loc['id'] == nodes[0])
    x_52420 = float(nodes_loc.iloc[i]['x'])
    y_52420 = float(nodes_loc.iloc[i]['y'])
    lat_52420 = node1_gps[0]
    lon_52420 = node1_gps[1]

    i = np.where(nodes_loc['id'] == nodes[1])
    x_52304 = float(nodes_loc.iloc[i]['x'])
    y_52304 = float(nodes_loc.iloc[i]['y'])
    lat_52304 = node2_gps[0]
    lon_52304 = node2_gps[1]

    de = x_52304 - x_52420
    dn = y_52304 - y_52420

    dLat = lat_52304 - lat_52420
    dLon = lon_52304 - lon_52420

    Rn = dn / dLat
    Re = de / dLon

    New_y = (lat - lat_52420) * Rn + y_52420
    New_x = (lon - lon_52420) * Re + x_52420

    return New_x, New_y


# to_Saturn_coord(lat_52420,lon_52420)

def to_Earth_coord(x, y, nodes_loc, nodes=REF[0], node1_gps=REF[1], node2_gps=REF[2]):
    i = np.where(nodes_loc['id'] == nodes[0])
    x_52420 = float(nodes_loc.iloc[i]['x'])
    y_52420 = float(nodes_loc.iloc[i]['y'])
    lat_52420 = node1_gps[0]
    lon_52420 = node1_gps[1]

    i = np.where(nodes_loc['id'] == nodes[1])
    x_52304 = float(nodes_loc.iloc[i]['x'])
    y_52304 = float(nodes_loc.iloc[i]['y'])
    lat_52304 = node2_gps[0]
    lon_52304 = node2_gps[1]

    de = x_52304 - x_52420
    dn = y_52304 - y_52420

    dLat = lat_52304 - lat_52420
    dLon = lon_52304 - lon_52420

    Rn = dn / dLat
    Re = de / dLon

    New_y = (y - y_52420) / Rn + lat_52420
    New_x = (x - x_52420) / Re + lon_52420

    return New_y, New_x


# to_World_coord(x_52420,y_52420)

def wgs84_to_web_mercator(df, lon="lon", lat="lat"):
    k = 6378137
    df["MX"] = df[lon] * (k * np.pi / 180.0)
    df["MY"] = np.log(np.tan((90 + df[lat]) * np.pi / 360.0)) * k

    return df


# %%

def main(file_path=None):
    my_dat = get_file(file_path)
    for k, v in network_ref_points.items():
        if k in my_dat.file_name.upper():
            REF = {k: v}
            break
    my_data = get_data(my_dat.data, my_dat.file_path)
    m = my_data.bus_route
    n = my_data.nodes_loc
    l = my_data.nodes
    o = my_data.anode()
    p = my_data.zones
    model_ref = list(REF.values())[0]
    n['lat'], n['lon'] = to_Earth_coord(n['x'], n['y'], n, nodes=model_ref[0], node1_gps=model_ref[1],
                                        node2_gps=model_ref[2])

    bokeh_plot(n, o.append(p), my_dat.file_path.replace('dat', 'html'))


def bokeh_plot(node, link, name='NetworkMap'):
    from bokeh.plotting import figure, from_networkx, save
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.io import export_png
    from bokeh.models import CustomJS, TextInput, CustomJSFilter, CDSView, TapTool
    from bokeh.layouts import column
    from bokeh.plotting import output_file, show
    from bokeh.tile_providers import get_provider, Vendors
    from bokeh.models import Circle, MultiLine, LabelSet, Toggle, CheckboxGroup
    from bokeh.models.graphs import NodesAndLinkedEdges

    text_input = TextInput(value="", title="Filter Nodes:")

    wgs84_to_web_mercator(node)

    node_source_data = ColumnDataSource(data=dict(x=node['MX'], y=node['MY'], desc=node['id']))

    # link
    G = nx.from_pandas_edgelist(link, source='id', target='anode')
    nx.set_node_attributes(G, dict(zip(link.id, link.id)), 'desc')
    n_loc = {k: (x, y) for k, x, y in zip(node['id'], node['MX'], node['MY'])}
    nx.set_node_attributes(G, n_loc, 'pos')
    n_color = {k: 'orange' if 'C' in k else 'green' for k in node['id']}
    nx.set_node_attributes(G, n_color, 'color')
    n_alpha = {k: 1 if 'C' in k else 0 for k in node['id']}
    nx.set_node_attributes(G, n_alpha, 'alpha')
    e_color = {(s, t): 'red' if 'C' in s else 'black' for s, t in zip(link['id'], link['anode'])}
    nx.set_edge_attributes(G, e_color, 'color')
    e_line_type = {(s, t): 'dashed' if 'C' in s else 'solid' for s, t in zip(link['id'], link['anode'])}
    nx.set_edge_attributes(G, e_line_type, 'line_type')

    tile_provider = get_provider(Vendors.CARTODBPOSITRON)

    bokeh_plot = figure(title="%s network map" % name.split('/')[-1].split('.')[0], sizing_mode="scale_height",
                        plot_width=1300, x_range=(min(node['MX']), max(node['MX'])),
                        tools='pan,wheel_zoom', active_drag="pan", active_scroll="wheel_zoom")
    bokeh_plot.add_tile(tile_provider)

    # This callback is crucial, otherwise the filter will not be triggered when the slider changes
    callback = CustomJS(args=dict(source=node_source_data), code="""
        source.change.emit();
    """)
    text_input.js_on_change('value', callback)

    # Define the custom filter to return the indices from 0 to the desired percentage of total data rows. You could
    # also compare against values in source.data
    js_filter = CustomJSFilter(args=dict(text_input=text_input), code=f"""
    const z = source.data['desc'];
    var indices = ((() => {{
      var result = [];
      for (let i = 0, end = source.get_length(), asc = 0 <= end; asc ? i < end : i > end; asc ? i++ : i--) {{
        if (z[i].includes(text_input.value.toString(10))) {{
          result.push(i);
        }}
      }}
      return result;
    }})());
    return indices;""")

    # Use the filter in a view
    view = CDSView(source=node_source_data, filters=[js_filter])

    callback2 = CustomJS(args=dict(x_range=bokeh_plot.x_range, y_range=bokeh_plot.y_range, text_input=text_input,
                                   source=node_source_data),
                         code=f"""
    const z = source.data['desc'];
    const x = source.data['x'];
    const y = source.data['y'];
    var result = [];
    for (let i = 0, end = source.get_length(), asc = 0 <= end; asc ? i < end : i > end; asc ? i++ : i--) {{
      if (z[i].includes(text_input.value.toString(10))) {{
        result.push(i);
      }}
    }}
    var indices = result[0];
    var Xstart = x[indices];
    var Ystart = y[indices];
    y_range.setv({{"start": Ystart-280, "end": Ystart+280}});
    x_range.setv({{"start": Xstart-500, "end": Xstart+500}});
    x_range.change.emit();
    y_range.change.emit();
    """)

    text_input.js_on_change('value', callback2)

    graph = from_networkx(G, nx.get_node_attributes(G, 'pos'), scale=2, center=(0, 0))
    graph.node_renderer.glyph = Circle(radius=15, fill_color='color', fill_alpha='alpha')
    graph.node_renderer.hover_glyph = Circle(radius=15, fill_color='red')

    graph.edge_renderer.glyph = MultiLine(line_alpha=1, line_color='color', line_width=1,
                                          line_dash='line_type')  # zero line alpha
    graph.edge_renderer.hover_glyph = MultiLine(line_color='#abdda4', line_width=5)
    graph.inspection_policy = NodesAndLinkedEdges()

    bokeh_plot.circle('x', 'y', source=node_source_data, radius=10, color='green', alpha=0.7, view=view)

    labels = LabelSet(x='x', y='y', text='desc', text_font_size="8pt", text_color='black',
                      x_offset=5, y_offset=5, source=node_source_data, render_mode='canvas')

    code = '''\
    if (toggle.active) {
        box.text_alpha = 0.0;
        console.log('enabling box');
    } else {
        box.text_alpha = 1.0;
        console.log('disabling box');
    }
    '''
    callback3 = CustomJS(code=code, args={})
    toggle = Toggle(label="Annotation", button_type="success")
    toggle.js_on_click(callback3)
    callback3.args = {'toggle': toggle, 'box': labels}

    bokeh_plot.add_tools(HoverTool(tooltips=[("id", "@desc")]), TapTool())
    # Output filepath
    bokeh_plot.renderers.append(graph)
    bokeh_plot.add_layout(labels)
    layout = column(toggle, text_input, bokeh_plot)

    # export_png(p, filename="plot.png")
    output_file(name)
    show(layout)


# %%

'''
import networkx as nx
conc = nodes[nodes['connect_node']>0]
conc= pd.merge(conc[['id','connect_node']], nodes_loc[['id','x','y']], how='left')
G = nx.from_pandas_edgelist(df=conc, source='id', target='connect_node')
nx.set_node_attributes(G, pd.Series(conc.x, index=nodes.id).to_dict(), 'x')
nx.set_node_attributes(G, pd.Series(conc.y, index=nodes.id).to_dict(), 'y')
G.add_nodes_from(nodes_for_adding=conc.id.tolist(),pos=(conc.x,conc.y))
plt.figure(3,figsize=(25,25)) 
nx.draw(G,node_size=0.01,font_size=8)
plt.savefig(fname='1')
'''
if __name__ == "__main__":
    Tk().withdraw()
    directory = askdirectory(title="Select folder")
    for f in os.listdir(directory):
        if f.endswith('.dat'):
            main(os.path.join(directory,f))
# %% Node sekect base on bus route API corrdinate
# if __name__ == '__main__':
