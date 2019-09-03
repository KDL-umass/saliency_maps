RESOURCES_PATH = './saliency_maps/object_saliency/resources'

AMIDAR_OBJECTS = {'player': RESOURCES_PATH+'/amidar'+'/player_l1.png',
					'enemy': RESOURCES_PATH+'/amidar'+'/enemy_l1.png',
					'tile_painted': RESOURCES_PATH+'/amidar'+'/block_tile_painted_l1.png',
					'tile_unpainted': RESOURCES_PATH+'/amidar'+'/block_tile_unpainted_l1.png',
					'tile_box_painted': RESOURCES_PATH+'/amidar'+'/painted_box_bar.png',
					'player_chase': RESOURCES_PATH+'/amidar'+'/enemy_chase_l1.png'}
AMIDAR_MOVING_OBJ = ['player', 'enemy', 'player_chase']

BREAKOUT_OBJECTS = {}
BREAKOUT_MOVING_OBJ = ['balls', 'paddle']