import numpy as np
import pandas as pd

import numpy as np
import copy
import pandas as pd
import os

import numpy as np
import pandas as pd
import os

import numpy as np
import pandas as pd
import os
import copy

class EloRating:
    def __init__(self, k_factor=32, initial_rating=400, meta='mmlu', topic='high_school_physics', student_model='Mistral-7B-Instruct-v0.2', load_data=True):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}
        self.match_history = pd.DataFrame(columns=["player_a_evaluator", "player_a_card_format", "player_a_iterative_method", "player_a_epoch",
                                                   "player_b_evaluator", "player_b_card_format", "player_b_iterative_method", "player_b_epoch",
                                                   "wins_a", "wins_b"])
        self.folder_name = f'elo_rating/{meta}/{topic}/{student_model}/'


        # make directory if it doesn't exist
        if load_data:
            self.load_data()
        os.makedirs(self.folder_name, exist_ok=True)    


    def add_player(self, player):
        player_tuple = (player['evaluator'], player['card_format'], player['iterative_method'], player['epoch'])
        if player_tuple not in self.ratings:
            self.ratings[player_tuple] = self.initial_rating

    def get_rating(self, player):
        player_tuple = (player['evaluator'], player['card_format'], player['iterative_method'], player['epoch'])
        if player_tuple not in self.ratings:
            self.add_player(player)
        return self.ratings[player_tuple]

    def update_ratings_batch(self, matches):
        players = set()
        for match in matches:
            player_a, player_b, _ = match
            players.add((player_a['evaluator'], player_a['card_format'], player_a['iterative_method'], player_a['epoch']))
            players.add((player_b['evaluator'], player_b['card_format'], player_b['iterative_method'], player_b['epoch']))

        for player in players:
            self.add_player({'evaluator': player[0], 'card_format': player[1], 'iterative_method': player[2], 'epoch': player[3]})

        player_indices = {player: index for index, player in enumerate(players)}
        num_players = len(players)

        expected_scores = np.zeros((num_players, num_players))
        actual_scores = np.zeros((num_players, num_players))

        for player_a, player_b, results in matches:
            player_a_tuple = (player_a['evaluator'], player_a['card_format'], player_a['iterative_method'], player_a['epoch'])
            player_b_tuple = (player_b['evaluator'], player_b['card_format'], player_b['iterative_method'], player_b['epoch'])

            index_a = player_indices[player_a_tuple]
            index_b = player_indices[player_b_tuple]

            rating_a = self.ratings[player_a_tuple]
            rating_b = self.ratings[player_b_tuple]

            expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
            expected_b = 1 - expected_a

            wins_a, wins_b = results
            actual_a = wins_a / (wins_a + wins_b)
            actual_b = 1 - actual_a

            expected_scores[index_a, index_b] = expected_a
            expected_scores[index_b, index_a] = expected_b
            actual_scores[index_a, index_b] = actual_a
            actual_scores[index_b, index_a] = actual_b

            match_details = {
                "player_a_evaluator": player_a['evaluator'],
                "player_a_card_format": player_a['card_format'],
                "player_a_iterative_method": player_a['iterative_method'],
                "player_a_epoch": player_a['epoch'],
                "player_b_evaluator": player_b['evaluator'],
                "player_b_card_format": player_b['card_format'],
                "player_b_iterative_method": player_b['iterative_method'],
                "player_b_epoch": player_b['epoch'],
                "wins_a": wins_a,
                "wins_b": wins_b
            }
            self.match_history = pd.concat([self.match_history, pd.DataFrame([match_details])], ignore_index=True)

        ratings = np.array([self.ratings[player] for player in players])
        new_ratings = ratings + self.k_factor * (actual_scores.sum(axis=1) - expected_scores.sum(axis=1))

        for player, new_rating in zip(players, new_ratings):
            self.ratings[player] = new_rating

        self.save_data()

    def get_expected_score(self, player_a, player_b):
        player_a_tuple = (player_a['evaluator'], player_a['card_format'], player_a['iterative_method'], player_a['epoch'])
        player_b_tuple = (player_b['evaluator'], player_b['card_format'], player_b['iterative_method'], player_b['epoch'])
        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def save_data(self):
        os.makedirs(self.folder_name, exist_ok=True)
        
        # Convert each rating in self.ratings to a dict
        rating_temp = copy.deepcopy(self.ratings)
        for key, value in rating_temp.items():
            rating_temp[key] = {'rating': value}
        
        ratings_df = pd.DataFrame.from_dict(rating_temp, orient="index")
        ratings_df.index.names = ["evaluator", "card_format", "iterative_method", "epoch"]
        ratings_df.to_csv(f'{self.folder_name}/ratings.csv')
        self.match_history.to_csv(f'{self.folder_name}/match_history.csv', index=False)

    def load_data(self):
        try:
            ratings_df = pd.read_csv(f'{self.folder_name}/ratings.csv', index_col=["evaluator", "card_format", "iterative_method", "epoch"])
            self.ratings = {tuple(index): row["rating"] for index, row in ratings_df.iterrows()}
            self.match_history = pd.read_csv(f'{self.folder_name}/match_history.csv')
        except FileNotFoundError:
            print("No saved data found. Starting with empty ratings and match history.")

if __name__ == '__main__':
    elo = EloRating()
    matches = [
    ({'evaluator': 'gpt' ,'card_format': 'str', 'iterative_method': 'one-pass', 'epoch': '0'}, 
     {'evaluator': 'gpt' ,'card_format': 'dict', 'iterative_method': 'prog-reg', 'epoch': '4'}, [1, 1]),
    # Add more matches as needed
    ]

    elo.update_ratings_batch(matches)
    for player, rating in elo.ratings.items():
        print(f"{player}: {rating}")
