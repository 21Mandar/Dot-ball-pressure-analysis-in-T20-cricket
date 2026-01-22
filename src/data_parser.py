# Data parser code for Cricsheet JSON Files
# Converts JSON Data into CSV data

import json
import os
import pandas as pd
from glob import glob
from tqdm import tqdm

def cricsheet_parse(file_path):
    with open(file_path,'r') as f:
        match_data = json.load(f)

    records = []
    match_info = match_data.get('info',{})

    #Extracting match metadata
    match_id = f"{match_info.get('dates',['unknown'])[0]}_{match_info.get('venue','unknown')}"
    venue = match_info.get('venue','unknown')
    date = match_info.get('dates',['unknown'])[0]

    if 'event' in match_info:
        competition = match_info['event'].get('name','unknown')
    else:
        competition = 'unknown'

    #Parse the innings
    for innings_num, innings in enumerate(match_data.get('innings',[])):
        team = innings.get('team','unknown')

        for over_data in innings.get('overs',[]):
            over_num = over_data.get('over',0)

            for delivery in over_data.get('deliveries',[]):
                batter = delivery.get('batter','unknown')
                bowler = delivery.get('bowler','unknown')

                # Runs information
                runs_info = delivery.get('runs',{})
                runs_scored = runs_info.get('batter',0)
                total_runs = runs_info.get('total',0)
                extras = runs_info.get('extras',0)

                # Wickets information
                wicket = 0
                dismissal_kind = None
                dismissed_batter = None

                if "wickets" in delivery:
                    wicket = 1
                    wicket_info = delivery['wickets'][0]
                    dismissal_kind = wicket_info.get('kind','unknown')
                    dismissed_batter = wicket_info.get('player_out',batter)

                records.append({
                    'match_id': match_id,
                    'venue': venue,
                    'date': date,
                    'competition': competition,
                    'innings_num': innings_num + 1,
                    'batting_team': team,
                    'over': over_num,
                    'batter': batter,
                    'bowler': bowler,
                    'runs_scored': runs_scored,
                    'total_runs': total_runs,
                    'extras': extras,
                    'wicket': wicket,
                    'dismissal_kind': dismissal_kind,
                    'dismissed_batter': dismissed_batter
                 })
    return records

def parse_all_matches(data_folder,competition_name):

    all_records = []
    json_files = glob(os.path.join(data_folder, '*json'))

    print(f"Processing{competition_name}: Found {len(json_files)} match files")

    for file_path in tqdm(json_files, desc=f"Parsing {competition_name}"):
        try:
            records = cricsheet_parse(file_path)

            for record in records:
                record['competition_type'] = competition_name
            all_records.extend(records)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

    df = pd.DataFrame(all_records)
    print(f"{competition_name} complete: {len(df)} delivers parsed \n")
    return df

def main():
    print("=" * 60)
    print("CRICSHEET DATA PARSER")
    print("=" * 60 + "\n")

    ipl_df = parse_all_matches('/Users/mandar/Library/Mobile Documents/com~apple~CloudDocs/NEU/NEU Content/Cricket Project/DOT_Pressure_Analysis/data/raw/ipl_json','IPL')
    t20i_df = parse_all_matches('/Users/mandar/Library/Mobile Documents/com~apple~CloudDocs/NEU/NEU Content/Cricket Project/DOT_Pressure_Analysis/data/raw/t20s_json','International Games')
    bbl_df = parse_all_matches('/Users/mandar/Library/Mobile Documents/com~apple~CloudDocs/NEU/NEU Content/Cricket Project/DOT_Pressure_Analysis/data/raw/bbl_json','BBL')

    print ("Combining all competitions...")
    all_data = pd.concat([ipl_df,t20i_df,bbl_df], ignore_index=True)

    output_path = '/Users/mandar/Library/Mobile Documents/com~apple~CloudDocs/NEU/NEU Content/Cricket Project/DOT_Pressure_Analysis/data/processed/raw_ballbyball_data.csv'
    all_data.to_csv(output_path, index= False)

    print("\n" + "=" * 60)
    print("PARSING COMPLETE")
    print("=" * 60)
    print(f"Total deliveries: {len(all_data):,}")
    print(f"Unique matches: {all_data['match_id'].nunique():,}")
    print(f"Unique batters: {all_data['batter'].nunique():,}")
    print(f"Output saved to: {output_path}")
    print("\nBreakdown by competition:")
    print(all_data['competition_type'].value_counts())


if __name__ == "__main__":
    main()
