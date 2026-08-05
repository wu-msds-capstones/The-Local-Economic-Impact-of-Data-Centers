
# define map layout function
def apply_map_layout_and_controls(fig, title_text, years):
    """
    Applies the carto-positron map style, playback menus, and time sliders
    to an existing Plotly animation figure.
    """
    fig.update_layout(
        title=title_text,
        map=dict(
            style="carto-positron",
            center=dict(lat=37.8, lon=-96),
            zoom=3
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=0.95,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 700, "redraw": True}, "transition": {"duration": 300}, "fromcurrent": True}]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"}]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Year: "},
                pad={"t": 30},
                steps=[
                    dict(
                        label=str(int(y)),
                        method="animate",
                        args=[[str(int(y))], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}]
                    )
                    for y in years
                ]
            )
        ]
    )
    return fig