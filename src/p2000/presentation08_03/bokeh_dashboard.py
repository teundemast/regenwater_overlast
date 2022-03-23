from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import row, column
from bokeh.models.tools import HoverTool
from bokeh.models import FactorRange, Select, LinearAxis, Range1d, GeoJSONDataSource, LinearColorMapper, ColorBar, Slider
from bokeh.io import curdoc, show, output_file
from bokeh.palettes import cividis
import pandas as pd
import json
import datetime
import random 

start_date = datetime.datetime(2021, 3, 31)
end_date = datetime.datetime(2022, 2, 1)

df_rain = pd.read_csv("p2000/presentation08_03/neerslaggeg_DE-BILT_550.txt")
format = "%Y%m%d"
df_rain["YYYYMMDD"] = pd.to_datetime(df_rain.YYYYMMDD, format = format)
df_rain = df_rain[start_date < df_rain["YYYYMMDD"]]
df_rain = df_rain[df_rain["YYYYMMDD"] < end_date]

df_p2000 = pd.read_csv("p2000/presentation08_03/calls_by_date_parsed.csv")
df_p2000["date"] = pd.to_datetime(df_p2000["date"])
df_p2000 = df_p2000[start_date < df_p2000["date"]]
df_p2000 = df_p2000[df_p2000["date"] < end_date]

p = figure(title="24-uur som van de neerslag in tiende millimeters in De Bilt vs p2000 'wateroverlast' meldingen", width=1210, height=400,
           x_axis_type="datetime", tools='save,pan,wheel_zoom,box_zoom,reset', y_range = (0,300))
p.yaxis.axis_label = "Neerslag in tiende millimeters"
randomlist = random.sample(range(10,500), 306)
p.add_layout(LinearAxis(y_range_name="foo",
             axis_label="Totaal aantal p2000 'wateroverlast' meldingen"), 'right')

p.extra_y_ranges = {"foo": Range1d(0, 160)}
df_rain["RD"] = df_rain["RD"].str.strip()
df_rain["RD"] = pd.to_numeric(df_rain.RD)

p.line(df_rain["YYYYMMDD"], df_rain["RD"], color='navy', alpha=0.5, legend_label="Neerslag")
p.line(df_p2000['date'], df_p2000['n_calls'], color='red',
       alpha=0.5, y_range_name="foo", legend_label="Aantal 'wateroverlast' meldingen")

p.legend.location = "top_left"
p.toolbar.logo = None

df_province = pd.read_csv("p2000/presentation08_03/calls_by_province_parsed.csv")
df_province = df_province.sort_values("n_calls", ascending=False)
df_province["call_per_house"] = df_province["n_calls"] / df_province["woningen"]
source = ColumnDataSource(df_province)

s = figure(
    title="Aantal p2000 'wateroverlast' meldingen per provincie",
    x_range=df_province["province"],
    tools="box_zoom,reset,save",
    height=400
)

s.vbar(x="province", top="n_calls", width=0.5, bottom=0, source = source,
       line_color="white", fill_color=(173, 216, 230), fill_alpha=0.9)

s.xaxis.axis_label = "Provincie"
s.yaxis.axis_label = "Aantal p2000 'wateroverlast' meldingen"

hover = HoverTool()
hover.tooltips = [("Provincie", "@province"),
                  ("Aantal", "@n_calls")]
s.add_tools(hover)

s.toolbar.logo = None
s.y_range.start = 0
s.xgrid.grid_line_color = None

drop_s = Select(title="Dimensie van de Y-as",
                options=["Aantal meldingen", "Aantal meldingen per woning"],
                value="Aantal meldingen)",
                width=225)

def update_bar_chart(attrname, old, new):
    if drop_s.value == "Aantal meldingen per woning":
        title = "Aantal p2000 'wateroverlast' meldingen per woning per provincie"
        y_label = "Aantal p2000 'wateroverlast' meldingen per woning"
        top = "call_per_house"
        df_province = pd.read_csv("p2000/presentation08_03/calls_by_province_parsed.csv")
        df_province["call_per_house"] = df_province["n_calls"] / df_province["woningen"]
        df_province = df_province.sort_values("call_per_house", ascending=False)
    else: 
        title = "Aantal p2000 'wateroverlast' meldingen per provincie"
        y_label = "Aantal p2000 'wateroverlast' meldingen"
        top = "n_calls"
        df_province = pd.read_csv("p2000/presentation08_03/calls_by_province_parsed.csv")
        df_province = df_province.sort_values("n_calls", ascending=False)

    source = ColumnDataSource(df_province)

    s = figure(
    title=title,
    x_range=df_province["province"],
    tools="box_zoom,reset,save",
    height=400
    )

    s.vbar(x="province", top=top, width=0.5, bottom=0, source = source,
        line_color="white", fill_color=(173, 216, 230), fill_alpha=0.9)
    
    s.xaxis.axis_label = "Provincie"
    s.yaxis.axis_label = y_label

    hover = HoverTool()
    hover.tooltips = [("Provincie", "@province"),
                    ("Aantal", f"@{top}")]
    s.add_tools(hover)

    s.toolbar.logo = None
    s.y_range.start = 0
    s.xgrid.grid_line_color = None

    layout_with_widgets.children[1].children[1]= s
    
drop_s.on_change("value", update_bar_chart)

layout_with_widgets = column(p, column(drop_s, s))
curdoc().add_root(layout_with_widgets)
