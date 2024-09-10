import pandas as pd
import os

# Last inn dataene fra hver CSV-fil i en mappe og returner en liste med dataframes og filnavn
def load_data_from_folder(folder_path):
    all_data = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                # Les CSV-filen uten å anta at første kolonne er datoer
                df = pd.read_csv(file_path)
                print(f"Innholdet i filen '{filename}':")
                print(df)  # Skriv ut innholdet i filen
                all_data.append((filename, df))  # Lagre filnavnet og dataen
                print(f"Lastet inn data fra fil: {file_path}")
            except Exception as e:
                print(f"En feil oppstod ved lasting av {file_path}: {e}")
    
    if all_data:
        return all_data
    else:
        print(f"Ingen CSV-filer funnet i {folder_path}.")
        return None

# Funksjon for å generere HTML-side med en tabell for hver CSV-fil
def generate_html_with_sortable_tables(data_list, output_dir):
    tables_html = ""
    
    for filename, df in data_list:
        # Generer headers for tabellen
        headers = ''.join(f'<th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{col}</th>' for col in df.columns)
        
        # Generer rader for tabellen
        rows = []
        for _, row in df.iterrows():
            cells = ''.join(f'<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{value}</td>' for value in row)
            rows.append(f'<tr>{cells}</tr>')
        
        table_body = ''.join(rows)
        
        table_html = f"""
        <div class="my-8 bg-white shadow-md rounded-lg overflow-hidden">
            <h2 class="text-2xl font-bold mb-4 p-4 bg-gray-100">{filename}</h2>
            <p class="text-sm text-gray-600 mb-2 px-4">Viser data fra filen {filename}</p>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            {headers}
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        {table_body}
                    </tbody>
                </table>
            </div>
        </div>
        """
        tables_html += table_html

    # HTML-struktur med integrert DataTables for sortering og TailwindCSS for styling
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Aksjeavkastning - Sortérbare og responsive tabeller</title>
        <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
        <script src="https://cdn.datatables.net/1.10.22/js/jquery.dataTables.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/1.10.22/css/jquery.dataTables.min.css" rel="stylesheet">
        <style>
            .dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, .dataTables_wrapper .dataTables_info, .dataTables_wrapper .dataTables_processing, .dataTables_wrapper .dataTables_paginate {{
                color: #4a5568;
                padding: 1rem;
            }}
            .dataTables_wrapper .dataTables_paginate .paginate_button {{
                padding: 0.5rem 1rem;
                margin-left: 0.25rem;
                border-radius: 0.25rem;
                border: 1px solid #e2e8f0;
            }}
            .dataTables_wrapper .dataTables_paginate .paginate_button.current {{
                background-color: #4299e1;
                color: white !important;
            }}
        </style>
        <script>
            $(document).ready(function() {{
                $('table').DataTable({{
                    "pageLength": 15,
                    "lengthMenu": [[15, 30, 50, -1], [15, 30, 50, "All"]],
                    "responsive": true,
                    "language": {{
                        "search": "Søk:",
                        "lengthMenu": "Vis _MENU_ rader per side",
                        "info": "Viser _START_ til _END_ av _TOTAL_ rader",
                        "infoEmpty": "Viser 0 til 0 av 0 rader",
                        "infoFiltered": "(filtrert fra _MAX_ totale rader)",
                        "paginate": {{
                            "first": "Første",
                            "last": "Siste",
                            "next": "Neste",
                            "previous": "Forrige"
                        }}
                    }}
                }});
            }});
        </script>
    </head>
    <body class="bg-gray-100 text-gray-900">
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-4xl font-bold mb-10 text-center text-gray-800">Aksjeavkastning - Tabeller per CSV-fil</h1>
            {tables_html}
        </div>
    </body>
    </html>
    """
    
    # Lagre den genererte HTML-filen
    html_file_path = os.path.join(output_dir, "aksje_avkastning_sorterbar.html")
    with open(html_file_path, "w", encoding="utf-8") as file:
        file.write(html_content)
    
    print(f"Interaktiv HTML-side generert og lagret: {html_file_path}")
    return html_file_path

# Hoveddel: Kjør visualiseringen og generer HTML-side
if __name__ == '__main__':
    # Hent prosjektets rotmappe (der skriptet kjører fra)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gå ett nivå opp fra 4.prob2_visual
    
    # Definer relative stier for 'output'-mappen der HTML-filen blir lagret
    output_dir = os.path.join(project_root, '4.prob2_visual')
    print(f"HTML-filen vil bli lagret i: {output_dir}")

    # Definer relative stier til mappen der CSV-filene er lagret
    folder_path = os.path.join(project_root, '3.prob2_output')
    print(f"Laster CSV-filer fra: {folder_path}")

    # Last inn data fra alle CSV-filer i mappen
    data_list = load_data_from_folder(folder_path)

    if data_list is not None:
        # Generer HTML-side med en tabell for hver CSV-fil
        generate_html_with_sortable_tables(data_list, output_dir)
