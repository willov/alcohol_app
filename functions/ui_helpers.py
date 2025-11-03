import streamlit as st


def prune_session_keys(page, prefix, indices, keys):
    """Remove session_state keys for given indices and key name patterns.

    - page: page id string (e.g., '02')
    - prefix: base prefix like 'drink' or 'meal'
    - indices: iterable of integer indices to prune
    - keys: list of suffix patterns (e.g., ["time", "time_locked", "length0"]) - will be formatted
    """
    for i in indices:
        for k in keys:
            full = f"{k.format(prefix=prefix, page=page, i=i)}"
            if full in st.session_state:
                del st.session_state[full]


def ensure_prev_counter(page, name, default=0):
    key = f"_prev_{name}_{page}"
    if key not in st.session_state:
        st.session_state[key] = default
    return key


def seed_new_items(page, name, n, default_start, step, seed_key_template, lock_key_template, key_prefix=None):
    """Seed newly added items for a list field.

    - seed_key_template e.g. 'drink_time_{page}_{i}' (pass with braces for format)
    - lock_key_template e.g. 'drink_time_locked_{page}_{i}'
    Returns nothing; writes to st.session_state.
    """
    # Determine a stable prefix to use in generated keys (allow plural or singular names)
    prefix = key_prefix if key_prefix is not None else (name[:-1] if isinstance(name, str) and name.endswith('s') else name)

    # Determine which prev counter key is present or should be used. Support variants for backwards compatibility.
    candidates = [f"_prev_{name}_{page}", f"_prev_n_{name}_{page}", f"_prev_{prefix}_{page}"]
    prev_key = next((c for c in candidates if c in st.session_state), candidates[0])
    prev = st.session_state.get(prev_key, 0)
    if n < prev:
        # prune removed indices
        prune_session_keys(page, prefix, range(n, prev), [seed_key_template, lock_key_template, f"{prefix}_length{{i}}", f"{prefix}_concentrations{{i}}", f"{prefix}_volumes{{i}}", f"{prefix}_kcals{{i}}"])
        st.session_state[prev_key] = n
        return
    if n == prev:
        return
    last_index = prev - 1
    if last_index >= 0:
        last_time = st.session_state.get(seed_key_template.format(prefix=prefix, page=page, i=last_index))
    else:
        last_time = None
    if last_time is None:
        last_time = default_start + (last_index if last_index >= 0 else 0) * step
    for i in range(prev, n):
        st.session_state.setdefault(seed_key_template.format(prefix=prefix, page=page, i=i), last_time + (i - last_index) * step if last_index >= 0 else default_start + i * step)
        st.session_state.setdefault(lock_key_template.format(prefix=prefix, page=page, i=i), False)
    st.session_state[prev_key] = n


def lock_all(page, what, n, locked=True):
    """Set all lock flags for items on a page."""
    for i in range(n):
        st.session_state[f"{what}_time_locked_{page}_{i}"] = locked


def on_change_time_propagate(page, what, index, n, step):
    """Propagate time changes forward for an item list (drinks or meals).

    - what: 'drink' or 'meal'
    - index: int index that changed
    - n: total count
    - step: hours to add per subsequent item
    """
    base_key = f"{what}_time_{page}_{index}"
    base_time = st.session_state.get(base_key, None)
    if base_time is None:
        return
    for j in range(index + 1, n):
        lock_key = f"{what}_time_locked_{page}_{j}"
        if not st.session_state.get(lock_key, False):
            st.session_state[f"{what}_time_{page}_{j}"] = base_time + (j - index) * step


def enforce_minimum_time(page, what, index, n, min_gap=None):
    """Enforce that an item's time is not before the previous item's time + min_gap.
    
    If the current time is less than the previous item's time + min_gap,
    reset it to the previous time + min_gap (or min_gap if index 0).
    
    - page: page id string (e.g., '02')
    - what: 'drink' or 'meal'
    - index: int index of the item to check
    - n: total count
    - min_gap: minimum gap from previous item. If None, uses duration of previous item.
               For drinks, this is drink_length (in minutes), for meals it defaults to 0.0.
    """
    current_key = f"{what}_time_{page}_{index}"
    current_time = st.session_state.get(current_key, 0.0)
    
    if index == 0:
        # First item must be >= 0
        if current_time < 0.0:
            st.session_state[current_key] = 0.0
    else:
        # Subsequent items must be >= previous item's time + min_gap
        prev_key = f"{what}_time_{page}_{index - 1}"
        prev_time = st.session_state.get(prev_key, 0.0)
        
        # Compute min_gap from previous item's duration if not provided
        if min_gap is None:
            # Get the duration of the previous item in hours
            prev_length_key = f"{what}_length{index - 1}"
            prev_length_minutes = st.session_state.get(prev_length_key, 0.0)
            min_gap = prev_length_minutes / 60.0  # Convert minutes to hours
        
        min_allowed = prev_time + min_gap
        if current_time < min_allowed:
            st.session_state[current_key] = min_allowed

def draw_drink_timeline_plotly(sim_df, feature, drink_starts, drink_lengths, title=None, uncert_time=None, uncert_min=None, uncert_max=None, uncert_color='rgba(200,200,200,0.25)'):
    """Return a plotly Figure showing the simulation line and drink-duration rectangles at the bottom.

    - sim_df: pandas DataFrame with 'Time' column (hours) and the feature column
    - feature: column name to plot
    - drink_starts: list of start times (hours)
    - drink_lengths: list of durations (minutes)
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        raise

    x = sim_df['Time'].tolist()
    y = sim_df[feature].tolist() if feature in sim_df.columns else [0] * len(x)

    fig = go.Figure()
    # optional uncertainty band
    if uncert_time is not None and uncert_min is not None and uncert_max is not None:
        fig.add_trace(
            go.Scatter(
                x=list(uncert_time) + list(reversed(uncert_time)), y=list(uncert_max) + list(reversed(uncert_min)),
                fill='toself', fillcolor=uncert_color, line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=True, name='Uncertainty'
            )
        )

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Simulation', line=dict(color='blue')))

    # Determine y-range and space for timeline
    y_min = min(y) if y else 0
    y_max = max(y) if y else 1
    yrange = y_max - y_min if (y_max - y_min) != 0 else 1.0

    # timeline occupies bottom 8% of y-range
    timeline_height = 0.04 * yrange
    y0 = y_min - 0.02 * yrange
    y1 = y0 + timeline_height

    # Add colored rectangles for each drink
    colors = ['rgba(255,127,0,0.4)', 'rgba(127,0,255,0.35)', 'rgba(0,127,255,0.35)', 'rgba(0,200,0,0.35)']
    for i, start in enumerate(drink_starts):
        dur_min = drink_lengths[i] if i < len(drink_lengths) else 0
        end = start + (dur_min / 60.0)
        color = colors[i % len(colors)]
        fig.add_shape(type='rect', x0=start, x1=end, y0=y0, y1=y1, fillcolor=color, line=dict(width=0))
        # add an invisible wide line trace spanning the rectangle so hovering anywhere over the bar shows details
        hover_text = f"Drink {i+1}<br>Start: {start:.2f} h<br>Duration: {dur_min} min<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=[start, end],
                y=[y0 + timeline_height / 2.0, y0 + timeline_height / 2.0],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=20),
                hovertemplate=hover_text,
                showlegend=False,
                hoverlabel=dict(bgcolor=color)
            )
        )

    # Update layout
    # Use 'closest' hovermode so when the cursor is between drink rectangles the simulation trace (which spans all x)
    # is chosen as the closest trace and drink traces don't appear.
    fig.update_layout(margin=dict(l=40, r=20, t=80, b=60), hovermode='closest')
    fig.update_xaxes(title_text='Time (h)')
    fig.update_yaxes(title_text=feature)
    if title:
        fig.update_layout(title=title)

    return fig

