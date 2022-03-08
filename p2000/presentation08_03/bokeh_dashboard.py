from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Range1d

bosch_logo = "p2000/presentation08_03/test.png"
logo_src = ColumnDataSource(dict(url = [bosch_logo]))

page_logo = figure(plot_width = 500, plot_height = 500, title="")
page_logo.toolbar.logo = None
page_logo.toolbar_location = None
page_logo.x_range=Range1d(start=0, end=1)
page_logo.y_range=Range1d(start=0, end=1)
page_logo.xaxis.visible = False
page_logo.yaxis.visible = False
page_logo.xgrid.grid_line_color = None
page_logo.ygrid.grid_line_color = None
page_logo.image_url(url='url', x=0.05, y = 0.85, h=0.7, w=0.9, source=logo_src)
page_logo.outline_line_alpha = 0 
curdoc().add_root(page_logo)