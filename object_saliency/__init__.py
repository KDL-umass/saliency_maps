RESOURCES_PATH = './saliency_maps/object_saliency/resources'

AMIDAR_OBJECT_KEYS = ['player', 'enemy', 'tile_painted', 'tile_unpainted', 'tile_box_painted']#,'player_chase']
AMIDAR_OBJECT_TEMPLATES = {'player': RESOURCES_PATH+'/amidar'+'/player_l1.png',
						'enemy': RESOURCES_PATH+'/amidar'+'/enemy_l1.png',
						'tile_painted': RESOURCES_PATH+'/amidar'+'/block_tile_painted_l1.png',
						'tile_unpainted': RESOURCES_PATH+'/amidar'+'/block_tile_unpainted_l1.png',
						'tile_box_painted': RESOURCES_PATH+'/amidar'+'/painted_box_bar.png'}
						# 'player_chase': RESOURCES_PATH+'/amidar'+'/enemy_chase_l1.png'}

BREAKOUT_OBJECT_KEYS = []
BREAKOUT_OBJECT_TEMPLATES = {}
