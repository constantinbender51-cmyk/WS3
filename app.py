# ... (Previous code remains the same)

# -----------------------------------------------------------------------------
# 5. Dash Web Server
# -----------------------------------------------------------------------------
app = Dash(__name__)
server = app.server  # <--- ADD THIS LINE (Required for Gunicorn/WSGI)

app.layout = html.Div([
    html.H1("Algorithmic Trading Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.H3("Strategy Performance & Indicators"),
        dcc.Graph(figure=fig_main)
    ]),
    
    html.Div([
        html.H3("Event Study: 30-Day Price Trajectory"),
        html.P("Average cumulative return starting from the moment a signal is triggered."),
        dcc.Graph(figure=fig_event)
    ], style={'marginTop': '50px'})
])

if __name__ == '__main__':
    print("Starting server on port 8080...")
    app.run_server(debug=True, port=8080, use_reloader=False)
