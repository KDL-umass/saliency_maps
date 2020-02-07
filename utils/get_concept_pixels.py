from toybox.interventions.amidar import *

def get_concept_pixels_breakout(concept, state_json, size):
    pixels = []

    if concept == "balls":
        ball_pos = (int(state_json[concept][0]['position']['x']), int(state_json[concept][0]['position']['y']))
        ball_radius = int(state_json['ball_radius'])
        ball_top_left = (ball_pos[0]-ball_radius, ball_pos[1]-ball_radius)

        for x in range(ball_radius*2):
            for y in range(ball_radius*2):
                pixels += [(ball_top_left[0]+x, ball_top_left[1]+y)]
    elif concept == "paddle":
        paddle_pos = (int(state_json[concept]['position']['x']), int(state_json[concept]['position']['y']))
        paddle_width = int(state_json['paddle_width']) 

        for i in range(int(paddle_width/2)+1):
            left_pos = (paddle_pos[0] - i, paddle_pos[1])
            right_pos = (paddle_pos[0] + i, paddle_pos[1])
            lleft_pos = (paddle_pos[0] - i, paddle_pos[1] + 1)
            lright_pos = (paddle_pos[0] + i, paddle_pos[1] + 1)
            llleft_pos = (paddle_pos[0] - i, paddle_pos[1] + 2)
            llright_pos = (paddle_pos[0] + i, paddle_pos[1] + 2)
            if i == 0:
                pixels += [left_pos, lleft_pos, lright_pos, llleft_pos, llright_pos]
            else:
                pixels += [left_pos, right_pos, lleft_pos, lright_pos, llleft_pos, llright_pos]
    elif concept == "bricks_top_left":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + x, upper_left_corner[1] + y)]
    elif concept == "bricks_top_mid":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + x_partition + x, upper_left_corner[1] + y)]
    elif concept == "bricks_top_right":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + 2*x_partition + x, upper_left_corner[1] + y)]
    elif concept == "bricks_bottom_left":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + x, upper_left_corner[1] + y_partition + y)]
    elif concept == "bricks_bottom_mid":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + x_partition + x, upper_left_corner[1] + y_partition + y)]
    elif concept == "bricks_bottom_right":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + 2*x_partition + x, upper_left_corner[1] + y_partition + y)]
    elif concept == "bricks":
        bricks = state_json["bricks"]
        for brick in bricks:
            if brick['alive']:
                brick_size = (int(brick['size']['x']), int(brick['size']['y']))
                brick_pos = (int(brick['position']['x']), int(brick['position']['y']))
                pix_brick = []
                for x in range(brick_size[0]):
                    for y in range(brick_size[1]):
                        pix_brick.append((brick_pos[0]+x, brick_pos[1]+y))
                pixels.append(pix_brick)
    elif concept == "lives":
        lives_size = (24,12)
        lives_pos = (148,1)
        for x in range(lives_size[0]):
            for y in range(lives_size[1]):
                pixels += [(lives_pos[0]+x, lives_pos[1]+y)]
    elif concept == "score":
        score_size = (80,12)
        score_pos = (50,1)
        for x in range(score_size[0]):
            for y in range(score_size[1]):
                pixels += [(score_pos[0]+x, score_pos[1]+y)]

    #ensure that pixels are not out of scope
    if concept != "bricks":
        for pixel in pixels:
            if (pixel[0] >= size[0] or pixel[0] <= 0) or (pixel[1] >= size[1] or pixel[1] <= 0):
                pixels.remove(pixel)

    return pixels

def get_concept_pixels_amidar(concept, state_json, size, tb):
    pixels = []
    board_width = size[0]
    board_length = size[1]
    print("concept: ", concept)
    ["tiles", "player", "enemies", "score", "lives"]

    if concept == "tiles":
        return []
    elif concept == "player":
        player_index = (state_json[concept]['position']['x'], state_json[concept]['position']['y']) #world pos
        player_pos = world_to_pixels(player_index, tb) #get pixel pos
        #get pixels of tile the player is sitting on
        for x in range(5):
            for y in range(6):
                pixels += [(player_pos[0]+x, player_pos[1]+y)]
        #get pixels outside of tile the player is sitting on
        above_tile = (player_pos[0], player_pos[1]-1)
        below_tile = (player_pos[0], player_pos[1]+1)
        left_tile = (player_pos[0]-1, player_pos[1])
        right_tile = (player_pos[0]+1, player_pos[1])
        pixels += [above_tile, below_tile, left_tile, right_tile]
    elif concept == "enemies":
        for enemy in state_json[concept]:
            enemy_pix = []
            enemy_index = (enemy['position']['x'], enemy['position']['y']) #world pos
            enemy_pos = world_to_pixels(enemy_index, tb) #get pixel pos

            #get pixels of tile the enemy is sitting on
            for x in range(5):
                for y in range(6):
                    enemy_pix += [(enemy_pos[0]+x, enemy_pos[1]+y)]

            #get pixels outside of tile the enemy is sitting on
            above_tile = (enemy_pos[0], enemy_pos[1]-1)
            below_tile = (enemy_pos[0], enemy_pos[1]+1)
            left_tile = (enemy_pos[0]-1, enemy_pos[1])
            right_tile = (enemy_pos[0]+1, enemy_pos[1])
            enemy_pix += [above_tile, below_tile, left_tile, right_tile]

            pixels += [enemy_pix]
    elif concept == "score":
        for x in range(80,105):
            for y in range(195,210):
                pixels += [(x,y)]
    elif concept == "lives":
        for x in range(115,150):
            for y in range(195,210):
                pixels += [(x,y)]

    return pixels

def world_to_pixels(world_pos, tb):
    tile_pos = (0, 0)
    with AmidarIntervention(tb) as intervention:
        tile_pos = intervention.world_to_tile(world_pos[0], world_pos[1])
    pixel_pos = (tile_pos['tx']*4 + 16, tile_pos['ty']*5 + 37)

    return pixel_pos
    