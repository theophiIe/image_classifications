import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

import dashboard1
import dashboard2


app = Dash(external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Dashboards template"

nav_contents = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("Dashboard 1", href="/dashboard1", active="exact")),
        dbc.NavItem(dbc.NavLink("DashBoard 2", href="/dashboard2", active="exact")),
    ],
    className="ms-auto",
    navbar=True
)

app.layout = html.Div(
    [
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                            [
                                dbc.Col(dbc.NavbarBrand("Dashboard", className="ms-2")),
                                dbc.Col(dbc.Nav(nav_contents, className="ms-2"), )
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="https://plotly.com",
                        style={"textDecoration": "none"},
                    ),
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                    # dbc.Nav(nav_contents),
                ]
            ),
            color="dark",
            dark=True,
        ),

        dcc.Location(id="url"),
        dbc.Container(id="page-content", className="pt-4", fluid=True),
    ]
)


@app.callback(Output("page-content", "children"),
              Input("url", "pathname"))
def render_page_content(pathname):
    if pathname == "/":
        return dashboard1.serve_layout()
    elif pathname == "/dashboard1":
        return dashboard1.serve_layout()
    elif pathname == "/dashboard2":
        return dashboard2.serve_layout()

    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=True)
