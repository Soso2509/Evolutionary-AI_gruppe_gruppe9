import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

# Last inn dataene fra CSV-filen
def load_data(file_path):
    try:
        returns_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        return returns_df
    except FileNotFoundError:
        print(f"Filen {file_path} ble ikke funnet. Sjekk filstien.")
        return None
    except Exception as e:
        print(f"En feil oppstod under innlasting av data: {e}")
        return None

# Funksjon for å lage interaktivt linjediagram med Plotly
def plot_interactive_line_chart(returns_df, output_dir):
    if returns_df is not None:
        fig = px.line(returns_df, title="Månedlige Avkastninger for Alle Selskaper")
        fig.update_layout(xaxis_title="Dato", yaxis_title="Avkastning")
        chart_path = os.path.join(output_dir, "line_chart.html")
        pio.write_html(fig, file=chart_path, auto_open=False)
        return chart_path
    else:
        print("Ingen data tilgjengelig for interaktivt linjediagram.")
        return None

# Funksjon for å lage interaktivt heatmap med Plotly
def plot_interactive_heatmap(returns_df, output_dir):
    if returns_df is not None:
        fig = px.imshow(returns_df.T, title="Heatmap av Månedlige Avkastninger for Alle Selskaper", aspect="auto", color_continuous_scale='RdYlGn')
        fig.update_layout(xaxis_title="Dato", yaxis_title="Selskaper")
        heatmap_path = os.path.join(output_dir, "heatmap.html")
        pio.write_html(fig, file=heatmap_path, auto_open=False)
        return heatmap_path
    else:
        print("Ingen data tilgjengelig for interaktivt heatmap.")
        return None

# Funksjon for å generere HTML-side
def generate_html(returns_df, line_chart_path, heatmap_path, output_dir):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Aksjeavkastning</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .container {{ width: 80%; margin: 0 auto; }}
            h1, h2 {{ text-align: center; }}
            iframe {{ width: 100%; height: 500px; border: none; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Aksjeavkastning</h1>
            <h2>Tabell med Månedlige Avkastninger</h2>
            {table}
            <h2>Interaktivt Linjediagram</h2>
            <iframe src="{line_chart}" title="Linjediagram over månedlige avkastninger"></iframe>
            <h2>Interaktivt Heatmap</h2>
            <iframe src="{heatmap}" title="Heatmap av månedlige avkastninger"></iframe>
        </div>
    </body>
    </html>
    """.format(
        table=returns_df.to_html(classes='table table-striped', index=True),
        line_chart=os.path.basename(line_chart_path),
        heatmap=os.path.basename(heatmap_path)
    )
    
    html_file_path = os.path.join(output_dir, "aksje_avkastning_interaktiv.html")
    with open(html_file_path, "w") as file:
        file.write(html_content)
    
    print(f"Interaktiv HTML-side generert: {html_file_path}")

# Hoveddel: Kjør visualiseringene og generer HTML-side
if __name__ == '__main__':
    # Definer stier
    file_path = 'mnd_avk_aksjer.csv'
    output_dir = 'output'

    # Lag output-mappen hvis den ikke eksisterer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Last inn data
    returns_df = load_data(file_path)

    if returns_df is not None:
        # Generer interaktive visualiseringer og lagre som HTML-filer
        line_chart_path = plot_interactive_line_chart(returns_df, output_dir)
        heatmap_path = plot_interactive_heatmap(returns_df, output_dir)

        # Generer HTML-side
        generate_html(returns_df, line_chart_path, heatmap_path, output_dir)
