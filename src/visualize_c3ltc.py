import networkx as nx
import pandas as pd
import numpy as np
import matplotlib as mpl
from IPython.display import display_html
import random


def highlight_max(s, props=""):
    return np.where(s == np.nanmax(s.values), props, "")


def local_view(graph, v, to_highlight=None, axis=1):
    right = graph.vertex_to_neighbours_B[v]
    left = graph.vertex_to_neighbours_A[v]
    df = pd.DataFrame(
        graph.vertex_to_squares[v], index=pd.Index(left), columns=pd.Index(right, name="")
    )
    s = df.style.format("{:.0f}")
    cell_hover = {
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
    if side == "A":
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
    right = graph.vertex_to_neighbours_B[v]
    left = graph.vertex_to_neighbours_A[v]
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
    if side == "A":
        axis = 1
    elif side == "B":
        axis = 0
    else:
        exit(1)
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
        columns=pd.Index(["g", "ag", "gb", "agb"], name=""),
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


def get_good_view(c3ltc):
    for g in c3ltc.G:
        s = set([])
        s.add(g)
        for a in c3ltc.A:
            s.add(a * g)
            for b in c3ltc.B:
                s.add(g * b)
                s.add(a * g * b)
        if len(s) == (len(c3ltc.A) + 1) * (len(c3ltc.B) + 1):
            at_leat_one = g
            return g


def show_graph(c3ltc, special_vertex=None, special_vertices_set=None):
    if not special_vertex:
        special_vertex = get_good_view(c3ltc)

    if special_vertex: 
        nxg = nx.Graph()
        add_node(nxg, special_vertex, list(c3ltc.G).index(special_vertex), special_vertices_set, int(100), int(100))
        for i, a in enumerate(c3ltc.A):
            add_node(nxg, a * special_vertex, list(c3ltc.G).index(a * special_vertex), special_vertices_set,
                     int(100 + i * 5), int(100 * (2 + i) + i * 5), color="#5692FC")
            for j, b in enumerate(c3ltc.B):
                add_node(nxg, special_vertex * b, list(c3ltc.G).index(special_vertex * b), special_vertices_set,
                         int(100 * (2 + j) + j * 5), int(100 + j * 5), color="#FC6956")
                add_node(nxg, a * special_vertex * b, list(c3ltc.G).index(a * special_vertex * b), special_vertices_set,
                         int(100 * (2 + j) + (j + 1) * 15), int(100 * (2 + i) + (i + 1) * 15))

    for g in c3ltc.G:
        if str(g) not in nxg.nodes:
            add_node(nxg, g, list(c3ltc.G).index(g), special_vertices_set, random.randint(5000 / 5, 15000 / 5),
                     random.randint(5000 / 5, 15000 / 5))
    for g in c3ltc.G:
        for a in c3ltc.A:
            nxg.add_edge(
                str(a * g), str(g), color="#cce6ff"
            )
        for b in c3ltc.B:
            nxg.add_edge(str(g), str(g * b), color="#ffd6cc")
    return nxg


def add_node(nxg, g, index, special_vertices_set, x, y, color="#6C7076"):
    if special_vertices_set and index in special_vertices_set:
        nxg.add_node(
            str(g),
            label=str(g),
            x=x,
            y=y,
            color="#68F67D"
        )
    else:
        nxg.add_node(
            str(g),
            label=str(g),
            x=x,
            y=y,
            color=color
        )
