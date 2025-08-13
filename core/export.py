import pandas as pd

def export_to_excel(filename, **sheets):
    with pd.ExcelWriter(filename) as writer:
        for name, df in sheets.items():
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=name[:31], index=False)
