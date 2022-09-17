import networkx as nx
import pandas as pd
import numpy as np
import matplotlib as mpl
from IPython.display import display_html
import random


def highlight_max(s, props=""):
    return np.where(s == np.nanmax(s.values), props, "")


def local_view(graph, v, to_highlight=None, axis=1):
    right = graph.vertex_to_neighbours_right[v]
    left = graph.vertex_to_neighbours_left[v]
    df = pd.DataFrame(
        graph.vertex_to_squares[v], index=pd.Index(left), columns=pd.Index(right, name="")
    )
    s = df.style.format("{:.0f}")
    cell_hover = {  # for row hover use <tr> instead of <td>
        "selector": "td:hover",
        "props": [("background-color", "#99ccff")],
    }
    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: darkgrey; font-weight:normal;",
    }
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #000066; color: white;",
    }
    s.set_table_styles([cell_hover, index_names, headers])
    s.set_table_styles(
        [
            {"selector": "th.col_heading", "props": "text-align: center;"},
            {"selector": "th.col_heading.level0", "props": "font-size: 1.0em;"},
            {"selector": "td", "props": "text-align: center; font-weight: bold;"},
        ],
        overwrite=False,
    )
    s.set_table_styles(
        [
            {
                "selector": "th:not(.index_name)",
                "props": [("background-color", "#ffd6cc"), ("color", "black")],
            },
            {
                "selector": "th.row_heading",
                "props": [("background-color", "#cce6ff"), ("color", "black")],
            },
        ],
        overwrite=False,
    )
    if to_highlight != None:
        idx = pd.IndexSlice
        if axis == 1:
            slice_ = idx[to_highlight, :]
        else:
            slice_ = idx[:, to_highlight]
        s.apply(highlight_max, axis=1, subset=slice_).set_properties(
            **{"background-color": "#ffffb3"}, subset=slice_
        )
    s.set_caption(str(v) + " local view")
    return s


def show_common(graph, v1, v2, side):
    if side == "left":
        axis = 1
    else:
        axis = 0
    df1 = local_view(graph, v1, v2, axis)
    df2 = local_view(graph, v2, v1, axis)

    df1_styler = df1.set_table_attributes("style='display:inline'").set_caption(
        str(v1) + " local view"
    )
    df2_styler = df2.set_table_attributes("style='display:inline'").set_caption(
        str(v2) + " local view"
    )

    display_html(df1._repr_html_() + df2._repr_html_(), raw=True)


def local_view_in_word(graph, word, v, to_highlight=None, axis=1):
    values = graph.local_codeword_on_vertex(v, word)
    squares = graph.vertex_to_squares[v]
    right = graph.vertex_to_neighbours_right[v]
    left = graph.vertex_to_neighbours_left[v]
    df = pd.DataFrame(values, index=pd.Index(left), columns=pd.Index(right, name=""))
    s = df.style.format("{:.0f}")
    cell_hover = {  # for row hover use <tr> instead of <td>
        "selector": "td:hover",
        "props": [("background-color", "#99ccff")],
    }
    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: darkgrey; font-weight:normal;",
    }
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #000066; color: white;",
    }
    s.set_table_styles([cell_hover, index_names, headers])
    s.set_table_styles(
        [
            {"selector": "th.col_heading", "props": "text-align: center;"},
            {"selector": "th.col_heading.level0", "props": "font-size: 1.0em;"},
            {"selector": "td", "props": "text-align: center; font-weight: bold;"},
        ],
        overwrite=False,
    )
    s.set_table_styles(
        [
            {
                "selector": "th:not(.index_name)",
                "props": [("background-color", "#ffd6cc"), ("color", "black")],
            },
            {
                "selector": "th.row_heading",
                "props": [("background-color", "#cce6ff"), ("color", "black")],
            },
        ],
        overwrite=False,
    )
    tooltips_df = pd.DataFrame(
        squares, index=pd.Index(left), columns=pd.Index(right, name="")
    )
    s.set_tooltips(
        tooltips_df,
        css_class="tt-add",
        props="visibility:hidden; position:absolute; z-index:1;transform: translate(-20px, -20px); background-color: #b3ffd9",
    )
    if to_highlight != None:
        idx = pd.IndexSlice
        if axis == 1:
            slice_ = idx[to_highlight, :]
        else:
            slice_ = idx[:, to_highlight]
        s.apply(highlight_max, axis=1, subset=slice_).set_properties(
            **{"background-color": "#ffffb3"}, subset=slice_
        )
    s.set_caption(str(v) + " local view")
    return s


def show_common_views_in_word(c3ltc, word, v1, v2, side):
    if side == "left":
        axis = 1
    else:
        axis = 0
    df1 = local_view_in_word(c3ltc, word, v1, v2, axis)
    df2 = local_view_in_word(c3ltc, word, v2, v1, axis)

    df1_styler = df1.set_table_attributes("style='display:inline'").set_caption(
        str(v1) + " local view"
    )
    df2_styler = df2.set_table_attributes("style='display:inline'").set_caption(
        str(v2) + " local view"
    )

    display_html(df1_styler._repr_html_() + df2_styler._repr_html_(), raw=True)


def show_square(graph, square):
    df = pd.DataFrame(
        [graph.square_to_vertices[square]],
        index=pd.Index([str(square)]),
        columns=pd.Index(["vertex", "left", "right", "closing"], name=""),
    )
    s = df.style
    cell_hover = {  # for row hover use <tr> instead of <td>
        "selector": "td:hover",
        "props": [("background-color", "#e0e0eb")],
    }
    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: darkgrey; font-weight:normal;",
    }
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #52527a; color: white;",
    }
    s.set_table_styles([cell_hover, index_names, headers])
    s.set_table_styles(
        [
            {"selector": "td", "props": "text-align: center; font-weight: bold;"},
        ],
        overwrite=False,
    )
    return s
