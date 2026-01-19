from scoundrel.scoundrel import Scoundrel

from datetime import datetime
import json
import os
import pandas as pd
import sys
from typing import List, Dict, Any


HIGH_SCORES_FILE_PATH = os.path.abspath('high_scores.json')


def update_scores(score: int):

    username = input("\nEnter username:")

    user_score = {
        "name": username,
        "score": score,
        "timestamp": datetime.now().isoformat()
    }

    try:
        with open(HIGH_SCORES_FILE_PATH, 'r') as file:
            data: List[Dict[str, Any]] = json.load(file)
        
        if not data:
            data = []

        data.append(user_score)
        
    except Exception as e:
        print(f'could not read score data. {e}')
        return 

    try:
        with open(HIGH_SCORES_FILE_PATH, 'w') as file:
            json.dump(data, file, indent=4) 

    except Exception as e:
        print(f'could not write score data. {e}')
        return 


def view_high_scores():
    df = pd.read_json(HIGH_SCORES_FILE_PATH)
    if not df.empty:
        df_sorted = df.sort_values(by='score', ascending=False)
        print(df_sorted.head())
    else:
        print('no scores to display')
    return


def main():
    while True:
        action = input("\nwhat do? \nPlay Scoundrel (p). View high scores (v). Quit (q).\n")

        if not action in ['p', 'v', 'q']:
            print('invalid action.\n')

        elif action == 'p':
            scoundrel = Scoundrel()
            score = scoundrel.play()
            update_scores(score=score)

        elif action == 'v':
            view_high_scores()

        elif action == 'q':
            sys.exit(0)

        continue

if __name__ == "__main__":
    main()
