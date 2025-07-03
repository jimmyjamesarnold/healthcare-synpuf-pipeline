import dash
from dash import dcc, html, dash_table, Input, Output
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

# Load data
df = pd.read_csv("data/processed/claims_2009.csv")
df["CLM_FROM_DT"] = pd.to_datetime(df["CLM_FROM_DT"])
# Add week and month columns
df["Week"] = df["CLM_FROM_DT"].dt.isocalendar().week
df["Month"] = df["CLM_FROM_DT"].dt.to_period("M").astype(str)
# Clean primary_dx: convert to string, strip whitespace, remove leading zeros
df["primary_dx"] = df["primary_dx"].astype(str).str.strip().str.lstrip("0")

# Load ICD-9 mapping file
icd9_map = pd.read_csv("data/raw/CMS32_DESC_LONG_DX.csv", sep="\t")
icd9_map = icd9_map.rename(columns={"ICD9_CD": "primary_dx", "ICD9_DESC_LONG": "description"})  # Adjust column names if needed
# Clean ICD-9 codes: convert to string, strip whitespace, remove leading zeros
icd9_map["primary_dx"] = icd9_map["primary_dx"].astype(str).str.strip().str.lstrip("0")

# Create fixed week and month DataFrames for imputation
weeks = pd.DataFrame(
    [(week, file_type) for week in range(1, 53) for file_type in ["inpatient", "outpatient", "carrier"]],
    columns=["Week", "file_type"]
)
months = pd.DataFrame(
    [(f"2009-{str(i).zfill(2)}", file_type) for i in range(1, 13) for file_type in ["inpatient", "outpatient", "carrier"]],
    columns=["Month", "file_type"]
)

app = dash.Dash(__name__)

# Beneficiary table data
ben_df = df.groupby("DESYNPUF_ID").agg({
    "claim_amt": "sum",
    "CLM_ID": "count"
}).reset_index().rename(columns={"claim_amt": "Total Payment", "CLM_ID": "Claim Count"})
ben_df = ben_df.sort_values("Total Payment", ascending=False)

# Diagnosis table data (default: entire population)
dx_df = df.groupby("primary_dx").agg({
    "claim_amt": "sum",
    "CLM_ID": "count"
}).reset_index().rename(columns={"claim_amt": "Total Payment", "CLM_ID": "Claim Count"})
dx_df = dx_df.merge(icd9_map[["primary_dx", "description"]], on="primary_dx", how="left")
dx_df["description"] = dx_df["description"].fillna(dx_df["primary_dx"])  # Fallback to code if description missing
dx_df = dx_df.sort_values("Total Payment", ascending=False).head(10)

# Layout
app.layout = html.Div([
    html.H1("SynPUF Claims Dashboard (2009)", style={"textAlign": "center", "fontSize": "24px", "margin": "10px"}),
    html.Div([
        # Top-Left: Beneficiary Table
        html.Div([
            html.H2("Top Beneficiaries", style={"fontSize": "18px", "margin": "5px"}),
            dash_table.DataTable(
                id="ben-table",
                columns=[
                    {"name": "DESYNPUF_ID", "id": "DESYNPUF_ID"},
                    {"name": "Total Payment", "id": "Total Payment"},
                    {"name": "Claim Count", "id": "Claim Count"}
                ],
                data=ben_df.to_dict("records"),
                page_size=10,
                sort_action="native",
                row_selectable="single",
                style_table={"overflowX": "auto", "maxHeight": "40vh", "overflowY": "auto"},
                style_cell={"textAlign": "left", "padding": "5px", "fontSize": "12px"},
                style_header={"fontWeight": "bold", "fontSize": "12px"},
                style_data_conditional=[
                    {
                        "if": {"state": "selected"},
                        "backgroundColor": "#e6f3ff",
                        "fontWeight": "bold"
                    }
                ]
            )
        ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),
        
        # Top-Right: Diagnosis Table
        html.Div([
            html.H2("Top 10 Diagnoses", style={"fontSize": "18px", "margin": "5px"}),
            dash_table.DataTable(
                id="dx-table",
                columns=[
                    {"name": "Diagnosis Description", "id": "description"},
                    {"name": "Total Payment", "id": "Total Payment"},
                    {"name": "Claim Count", "id": "Claim Count"}
                ],
                data=dx_df.to_dict("records"),
                sort_action="native",
                row_selectable="single",
                style_table={"overflowX": "auto", "maxHeight": "40vh", "overflowY": "auto"},
                style_cell={"textAlign": "left", "padding": "5px", "fontSize": "12px"},
                style_header={"fontWeight": "bold", "fontSize": "12px"},
                style_cell_conditional=[
                    {
                        "if": {"column_id": "description"},
                        "width": "300px",
                        "maxWidth": "300px",
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis"
                    },
                    {
                        "if": {"column_id": "Total Payment"},
                        "width": "100px",
                        "textAlign": "right"
                    },
                    {
                        "if": {"column_id": "Claim Count"},
                        "width": "100px",
                        "textAlign": "right"
                    }
                ],
                style_data_conditional=[
                    {
                        "if": {"state": "selected"},
                        "backgroundColor": "#e6f3ff",
                        "fontWeight": "bold"
                    }
                ],
                tooltip_data=[
                    {
                        "description": {"value": row["description"], "type": "text"}
                    } for row in dx_df.to_dict("records")
                ],
                tooltip_delay=0,
                tooltip_duration=None
            )
        ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"})
    ], style={"display": "flex", "height": "50vh"}),
    
    # Bottom Half: Stacked Bar Chart
    html.Div([
        html.H2("Payments Over Time by File Type", style={"fontSize": "18px", "margin": "5px"}),
        html.Button("Reset Filters", id="reset-button", n_clicks=0, style={"fontSize": "14px", "margin": "5px"}),
        html.Label("Click 'Reset Filters' to revert to total population", style={"fontSize": "12px", "margin": "5px", "color": "#666"}),
        html.Label("Aggregation Period:", style={"fontSize": "14px", "marginLeft": "10px"}),
        dcc.Dropdown(
            id="time-agg-dropdown",
            options=[
                {"label": "Weekly", "value": "Week"},
                {"label": "Monthly", "value": "Month"}
            ],
            value="Week",
            style={"width": "200px", "marginBottom": "10px", "fontSize": "14px", "display": "inline-block"}
        ),
        dcc.Graph(id="payment-chart", style={"height": "40vh"})
    ], style={"padding": "10px"})
], style={"maxWidth": "100vw", "height": "95vh", "margin": "0", "padding": "0"})

# Callbacks
@app.callback(
    [
        Output("dx-table", "data"),
        Output("payment-chart", "figure")
    ],
    [
        Input("ben-table", "selected_rows"),
        Input("dx-table", "selected_rows"),
        Input("time-agg-dropdown", "value"),
        Input("reset-button", "n_clicks")
    ]
)
def update_dashboard(selected_ben_row, selected_dx_row, time_agg, reset_clicks):
    # Initialize variables
    filtered_df = df
    selected_id = None
    selected_dx = None
    
    # Handle reset button
    ctx = dash.callback_context
    if ctx.triggered_id == "reset-button":
        filtered_df = df
    else:
        if selected_ben_row and selected_ben_row[0] is not None:
            selected_id = ben_df.iloc[selected_ben_row[0]]["DESYNPUF_ID"]
            filtered_df = df[df["DESYNPUF_ID"] == selected_id]
        elif selected_dx_row and selected_dx_row[0] is not None:
            selected_dx = dx_df.iloc[selected_dx_row[0]]["primary_dx"]
            filtered_df = df[df["primary_dx"] == selected_dx]
    
    # Update diagnosis table
    dx_data = filtered_df.groupby("primary_dx").agg({
        "claim_amt": "sum",
        "CLM_ID": "count"
    }).reset_index().rename(columns={"claim_amt": "Total Payment", "CLM_ID": "Claim Count"})
    dx_data = dx_data.merge(icd9_map[["primary_dx", "description"]], on="primary_dx", how="left")
    dx_data["description"] = dx_data["description"].fillna(dx_data["primary_dx"])  # Fallback to code if description missing
    dx_data = dx_data.sort_values("Total Payment", ascending=False).head(10)
    
    # Update payment chart
    time_col = "Week" if time_agg == "Week" else "Month"
    chart_data = filtered_df.groupby([time_col, "file_type"])["claim_amt"].sum().reset_index()
    
    # Impute missing periods with 0
    if time_agg == "Week":
        base_df = weeks.copy()
        chart_data = base_df.merge(chart_data, on=["Week", "file_type"], how="left").fillna({"claim_amt": 0})
        chart_data["Week"] = chart_data["Week"].astype(int)  # Ensure numerical sorting
        chart_data = chart_data.sort_values("Week")
        tick_vals = list(range(1, 53))
        tick_text = [str(i) for i in range(1, 53)]
    else:
        base_df = months.copy()
        chart_data = base_df.merge(chart_data, on=["Month", "file_type"], how="left").fillna({"claim_amt": 0})
        chart_data = chart_data.sort_values("Month")
        tick_vals = [f"2009-{str(i).zfill(2)}" for i in range(1, 13)]
        tick_text = tick_vals
    
    fig = px.bar(
        chart_data,
        x=time_col,
        y="claim_amt",
        color="file_type",
        title=f"Payments by {time_agg} (2009)" + (f" - ID: {selected_id}" if selected_id else f" - DX: {selected_dx}" if selected_dx else ""),
        labels={"claim_amt": "Total Payment", time_col: time_agg}
    )
    fig.update_layout(
        barmode="stack",
        xaxis={"tickangle": 45, "tickmode": "array", "tickvals": tick_vals, "ticktext": tick_text},
        height=400
    )
    
    return dx_data.to_dict("records"), fig

if __name__ == "__main__":
    app.run_server(debug=True)