CONCEPTS = {"Breakout": ["balls", "paddle", "bricks_top_left", "bricks_top_mid", "bricks_top_right", "bricks_bottom_left", "bricks_bottom_mid", "bricks_bottom_right"], \
			 "Amidar": ["tiles", "player", "enemies", "score", "lives"]}
SAVE_DIR = "./saliency_maps/experiments/results/"
INTERVENTIONS = {"balls": ["intervention_move_ball", "intervention_ball_speed"], \
                "paddle": ["intervention_move_paddle"], \
                "bricks": ["intervention_flip_bricks", "intervention_remove_bricks", "intervention_remove_rows"]}
