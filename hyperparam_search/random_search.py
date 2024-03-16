import logging
import random
class RandomSearch:
    def __init__(self, params_dict, retries):
        self.params_dict = params_dict
        self.retries = retries
        self.visited = set()
    
    def grid_search(self):
        logger = logging.getLogger()
        # logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

        params = {}
        flag = True
        for _ in range(self.retries):
            for i in self.params_dict.keys():
                ind = random.randint(0, len(self.params_dict[i])-1)
                params[i] = self.params_dict[i][ind]
            if str(params) not in self.visited:
                self.visited.add(str(params))
                print(self.visited)
                logger.info(f'new param {params}')
                return params
            
        logger.info('Reached maximum retry count')
        return None
        

# if __name__ == '__main__':
#     gs = RandomSearch({'drop': [0.3,0.5,0.7], 'bn': [True, False]}, 3)
#     for i in range(30):
#         a = gs.grid_search()
#         print(a)