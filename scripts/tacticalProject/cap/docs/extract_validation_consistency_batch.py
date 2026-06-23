from pathlib import Path
import pandas as pd
ROOT = Path(r'd:\Pycharm\LAG\scripts\tacticalProject\cap_results\Chapter6_validation')
rows = []
for csv_path in ROOT.glob('ALL_*/comparison_tables/scenario_comparison.csv'):
    df = pd.read_csv(csv_path)
    df['batch'] = csv_path.parents[1].name
    rows.append(df)
all_df = pd.concat(rows, ignore_index=True)
all_df['win_flag'] = (all_df['enemy_kill_count'] > all_df['friendly_loss_count']).astype(int)
result = all_df.groupby('scenario_id')['win_flag'].agg(['mean', 'std', 'count']).reset_index()
print(result.to_string(index=False))
